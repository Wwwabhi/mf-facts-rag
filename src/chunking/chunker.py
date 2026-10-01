from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import tiktoken

from src.data.source_registry import SourceRecord, load_registry


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_DIR = ROOT / "data" / "chunks"
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"
CSV_PATH = ROOT / "data" / "mf_rag_sources.csv"
TARGET_TOKENS = 500
MAX_TOKENS = 650
OVERLAP_TOKENS = 75
TOKENIZER_NAME = "cl100k_base"
ENCODING = tiktoken.get_encoding(TOKENIZER_NAME)

MARKDOWN_HEADING = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
HTML_HEADING = re.compile(r"^\s*H([1-6]):\s*(.+?)\s*$", re.IGNORECASE)
QUESTION_LINE = re.compile(r"^(?:(?:Q|Question)\s*:\s*)?.+\?\s*$", re.IGNORECASE)
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}.*\|\s*:?-{3,}.*\|?\s*$")


@dataclass(frozen=True)
class ContentItem:
    kind: str
    text: str


def _token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def _normalized_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def _normalized_content(content: str) -> str:
    normalized = unicodedata.normalize("NFKC", content).replace("\r\n", "\n")
    return " ".join(normalized.split())


def _source_lookup(records: list[SourceRecord]) -> dict[str, SourceRecord]:
    return {_normalized_url(record.source_url): record for record in records}


def _heading(line: str) -> tuple[int, str] | None:
    match = MARKDOWN_HEADING.match(line)
    if match:
        level, title = len(match.group(1)), match.group(2).strip()
    else:
        match = HTML_HEADING.match(line)
        if not match:
            return None
        level, title = int(match.group(1)), match.group(2).strip()
    if len(title) > 200 or _token_count(title) > 32:
        return None
    return level, title


def _content_items(lines: list[str]) -> list[ContentItem]:
    items: list[ContentItem] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            text = " ".join(part.strip() for part in paragraph if part.strip())
            if text:
                items.append(ContentItem("paragraph", text))
            paragraph.clear()

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush_paragraph()
            index += 1
            continue

        if line.startswith("Table:") or line.startswith("|"):
            flush_paragraph()
            table_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index].strip()
                if not next_line or _heading(next_line):
                    break
                if "|" not in next_line:
                    break
                table_lines.append(next_line)
                index += 1
            items.append(ContentItem("table", "\n".join(table_lines)))
            continue

        flush_if_heading = _heading(line)
        if flush_if_heading:
            flush_paragraph()
            items.append(ContentItem("heading", line))
        else:
            paragraph.append(line)
        index += 1

    flush_paragraph()
    return items


def _sentence_atoms(text: str, limit: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if not sentences:
        return []

    atoms: list[str] = []
    for sentence in sentences:
        if _token_count(sentence) <= limit:
            atoms.append(sentence)
            continue

        words = sentence.split()
        current: list[str] = []
        for word in words:
            if _token_count(word) > limit:
                if current:
                    atoms.append(" ".join(current))
                    current = []
                start = 0
                while start < len(word):
                    low = start + 1
                    high = len(word)
                    best = low
                    while low <= high:
                        middle = (low + high) // 2
                        if _token_count(word[start:middle]) <= limit:
                            best = middle
                            low = middle + 1
                        else:
                            high = middle - 1
                    atoms.append(word[start:best])
                    start = best
                continue
            candidate = " ".join((*current, word))
            if current and _token_count(candidate) > limit:
                atoms.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            atoms.append(" ".join(current))

    return atoms


def _breadcrumb(source_title: str, section_path: tuple[str, ...]) -> str:
    path = list(section_path)
    if not path or path[0].casefold() != source_title.casefold():
        path.insert(0, source_title)
    return " > ".join(part for part in path if part)


def _overlap_tail(text: str, limit: int) -> str:
    tokens = ENCODING.encode(text)
    return ENCODING.decode(tokens[-limit:]).strip() if tokens else ""


def _chunk_text_body(
    text: str,
    section_path: tuple[str, ...],
    source_title: str,
    *,
    repeat_prefix: str = "",
) -> list[tuple[str, str]]:
    breadcrumb = _breadcrumb(source_title, section_path)
    prefix = f"Section: {breadcrumb}\n"
    if repeat_prefix:
        prefix += f"{repeat_prefix}\n"
    prefix_tokens = _token_count(prefix)
    body_target = max(1, TARGET_TOKENS - prefix_tokens)
    body_maximum = max(1, MAX_TOKENS - prefix_tokens)
    atoms = _sentence_atoms(text, body_target)
    chunks: list[tuple[str, str]] = []
    current: list[str] = []

    def emit(parts: list[str]) -> None:
        body = " ".join(part for part in parts if part).strip()
        if body:
            chunks.append((f"{prefix}{body}", breadcrumb))

    for atom in atoms:
        candidate = " ".join((*current, atom)).strip()
        if current and _token_count(candidate) > body_target:
            previous = " ".join(current)
            emit(current)
            overlap = _overlap_tail(previous, OVERLAP_TOKENS)
            current = [overlap, atom] if overlap else [atom]
            while _token_count(" ".join(current)) > body_maximum and overlap:
                overlap_tokens = ENCODING.encode(overlap)
                overlap = ENCODING.decode(overlap_tokens[1:]).strip()
                current = [overlap, atom] if overlap else [atom]
        else:
            current.append(atom)
    emit(current)
    return chunks


def _table_chunks(
    text: str,
    section_path: tuple[str, ...],
    source_title: str,
    table_index: int,
) -> list[tuple[str, str, dict[str, Any]]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    breadcrumb = _breadcrumb(source_title, section_path)
    prefix = f"Section: {breadcrumb}\n"
    prefix_tokens = _token_count(prefix)
    body_target = max(1, TARGET_TOKENS - prefix_tokens)
    body_maximum = max(1, MAX_TOKENS - prefix_tokens)

    if _token_count(prefix + text) <= MAX_TOKENS:
        return [(prefix + text, breadcrumb, {"table_index": table_index})]

    caption = ""
    if lines and lines[0].startswith("Table:"):
        caption = lines.pop(0)
    header: list[str] = []
    rows = lines
    if rows:
        header = [rows.pop(0)]
        if rows and TABLE_SEPARATOR.match(rows[0]):
            header.append(rows.pop(0))
    rows = [row for row in rows if not TABLE_SEPARATOR.match(row)]
    header_text = "\n".join(([caption] if caption else []) + header)
    if _token_count(prefix + header_text) > TARGET_TOKENS:
        rows = [*header, *rows]
        header = []
        header_text = caption
    if not header_text:
        header_text = "Table"

    chunks: list[tuple[str, str, dict[str, Any]]] = []
    current_rows: list[str] = []

    def emit_rows(row_group: list[str]) -> None:
        table_text = "\n".join(part for part in (header_text, *row_group) if part)
        chunks.append((prefix + table_text, breadcrumb, {"table_index": table_index}))

    for row in rows:
        single_row = prefix + "\n".join(part for part in (header_text, row) if part)
        if _token_count(single_row) > MAX_TOKENS:
            if current_rows:
                emit_rows(current_rows)
                current_rows = []
            continued_prefix = "Table row (continued): "
            available = max(1, body_maximum - _token_count(header_text) - _token_count(continued_prefix) - 16)
            for fragment in _sentence_atoms(row, available):
                emit_rows([f"{continued_prefix}{fragment}"])
            continue

        candidate_rows = [*current_rows, row]
        candidate = prefix + "\n".join(part for part in (header_text, *candidate_rows) if part)
        if current_rows and _token_count(candidate) > body_target + prefix_tokens:
            emit_rows(current_rows)
            current_rows = [row]
        else:
            current_rows.append(row)
    if current_rows:
        emit_rows(current_rows)
    if not chunks:
        for fragment in _sentence_atoms(text, body_maximum):
            chunks.append((prefix + fragment, breadcrumb, {"table_index": table_index}))
    return chunks


def _section_chunks(
    items: list[ContentItem],
    section_path: tuple[str, ...],
    source_title: str,
    table_start: int,
) -> tuple[list[tuple[str, str, str, dict[str, Any]]], int]:
    chunks: list[tuple[str, str, str, dict[str, Any]]] = []
    normal_paragraphs: list[str] = []
    question: str | None = None
    answer: list[str] = []
    table_index = table_start

    def flush_normal() -> None:
        if normal_paragraphs:
            for content, breadcrumb in _chunk_text_body("\n\n".join(normal_paragraphs), section_path, source_title):
                chunks.append((content, breadcrumb, "text", {}))
            normal_paragraphs.clear()

    def flush_faq() -> None:
        nonlocal question, answer
        if question is None:
            return
        faq_text = "\n".join([question, *answer]).strip()
        breadcrumb = _breadcrumb(source_title, section_path)
        prefix = f"Section: {breadcrumb}\n"
        if _token_count(prefix + faq_text) <= MAX_TOKENS:
            chunks.append((prefix + faq_text, breadcrumb, "faq", {"faq_question": question}))
        else:
            for content, chunk_breadcrumb in _chunk_text_body(
                " ".join(answer), section_path, source_title, repeat_prefix=f"Question: {question}"
            ):
                chunks.append((content, chunk_breadcrumb, "faq", {"faq_question": question}))
        question = None
        answer = []

    for item in items:
        if item.kind == "table":
            flush_faq()
            flush_normal()
            table_index += 1
            for content, breadcrumb, extra in _table_chunks(item.text, section_path, source_title, table_index):
                chunks.append((content, breadcrumb, "table", extra))
            continue

        if QUESTION_LINE.match(item.text):
            if question is not None:
                flush_faq()
            else:
                flush_normal()
            question = item.text
            continue

        if question is not None:
            answer.append(item.text)
        else:
            normal_paragraphs.append(item.text)

    flush_faq()
    flush_normal()
    return chunks, table_index


def _document_chunks(document: dict[str, Any], source: SourceRecord) -> list[dict[str, Any]]:
    content = str(document.get("content", "")).replace("\r\n", "\n")
    source_title = str(document.get("title") or source.title)
    items = _content_items(content.splitlines())
    heading_stack: list[tuple[int, str]] = []
    section_items: list[ContentItem] = []
    chunks: list[tuple[str, str, str, dict[str, Any]]] = []
    table_index = 0

    def flush_section() -> None:
        nonlocal table_index
        if section_items:
            path = tuple(text for _, text in heading_stack)
            generated, table_index = _section_chunks(section_items, path, source_title, table_index)
            chunks.extend(generated)
            section_items.clear()

    for item in items:
        heading = _heading(item.text) if item.kind == "heading" else None
        if heading:
            flush_section()
            level, title = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            section_items.append(item)
    flush_section()

    source_hash = hashlib.sha256(_normalized_content(content).encode("utf-8")).hexdigest()
    result: list[dict[str, Any]] = []
    for index, (chunk_text, breadcrumb, kind, extra) in enumerate(chunks):
        token_count = _token_count(chunk_text)
        if token_count > MAX_TOKENS:
            raise ValueError(f"chunk {index} exceeds {MAX_TOKENS} tokens ({token_count})")
        chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        record = {
            "chunk_id": f"{source.id}:{index}:{chunk_hash[:12]}",
            "chunk_index": index,
            "content": chunk_text,
            "token_count": token_count,
            "tokenizer": TOKENIZER_NAME,
            "section_breadcrumb": breadcrumb,
            "chunk_type": kind,
            "source_id": source.id,
            "product": source.product,
            "publisher": source.publisher,
            "scheme": source.scheme,
            "document_type": source.document_type,
            "title": source.title,
            "source_url": source.source_url,
            "source_date": source.source_date,
            "verified_on": source.verified_on,
            "date": source.date,
            "use_in_rag": source.use_in_rag,
            "raw_file": document.get("raw_file"),
            "document_hash": source_hash,
            "page_number": None,
        }
        record.update(extra)
        result.append(record)
    return result


def run_chunking_pipeline(
    processed_dir: Path = PROCESSED_DIR,
    chunks_path: Path = CHUNKS_PATH,
    csv_path: Path = CSV_PATH,
) -> dict[str, Any]:
    source_by_url = _source_lookup(load_registry(csv_path))
    seen_documents: set[tuple[str, str]] = set()
    all_chunks: list[dict[str, Any]] = []
    source_count = 0
    duplicates_skipped = 0
    errors: list[str] = []

    for path in sorted(processed_dir.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            source_url = str(document.get("source_url", ""))
            source = source_by_url.get(_normalized_url(source_url))
            if source is None:
                raise ValueError(f"no registry record matches source URL: {source_url}")
            content = str(document.get("content", ""))
            if not content.strip():
                raise ValueError("document content is empty")

            content_hash = hashlib.sha256(_normalized_content(content).encode("utf-8")).hexdigest()
            document_key = (_normalized_url(source_url), content_hash)
            if document_key in seen_documents:
                duplicates_skipped += 1
                continue
            seen_documents.add(document_key)

            generated = _document_chunks(document, source)
            if not generated:
                raise ValueError("document produced no chunks")
            all_chunks.extend(generated)
            source_count += 1
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as output:
        for chunk in all_chunks:
            output.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return {
        "sources_processed": source_count,
        "chunks_written": len(all_chunks),
        "duplicates_skipped": duplicates_skipped,
        "errors": errors,
        "output_path": str(chunks_path),
    }


def main() -> int:
    report = run_chunking_pipeline()
    print(f"Sources processed: {report['sources_processed']}")
    print(f"Total chunks: {report['chunks_written']}")
    print(f"Duplicates skipped: {report['duplicates_skipped']}")
    print(f"Processing errors: {len(report['errors'])}")
    print(f"Output: {report['output_path']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())