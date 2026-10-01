from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.data.source_registry import load_registry


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            chunk_id = str(record.get("chunk_id", "")).strip()
            content = record.get("content")
            if not chunk_id or not isinstance(content, str) or not content.strip():
                raise ValueError(f"Invalid chunk record on line {line_number}")
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id on line {line_number}: {chunk_id}")
            seen_ids.add(chunk_id)
            chunks.append(record)
    return chunks


def _normalized_source_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for chunk in chunks:
        source_url = str(chunk.get("source_url", ""))
        normalized_content = " ".join(
            unicodedata.normalize("NFKC", str(chunk.get("content", ""))).split()
        )
        key = (_normalized_source_url(source_url), normalized_content)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique, len(chunks) - len(unique)


def validate_registered_sources(chunks: list[dict[str, Any]], registry_path: str | Path) -> None:
    registered_urls = {
        _normalized_source_url(record.source_url)
        for record in load_registry(registry_path)
        if record.source_url
    }
    unknown_urls = {
        str(chunk.get("source_url", ""))
        for chunk in chunks
        if _normalized_source_url(str(chunk.get("source_url", ""))) not in registered_urls
    }
    if unknown_urls:
        raise ValueError(f"Chunks reference {len(unknown_urls)} source URL(s) not listed in the registry")


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {}
    for key, value in chunk.items():
        if key == "content" or value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return metadata