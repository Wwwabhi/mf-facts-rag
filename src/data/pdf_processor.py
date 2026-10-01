from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from docling.document_converter import DocumentConverter

from src.data.source_registry import SourceRecord


def fetch_pdf(record: SourceRecord, raw_dir: Path) -> tuple[bool, Path | None, bytes | None]:
    url = record.source_url
    file_name = f"{record.id}_{record.scheme.lower().replace(' ', '_')}_{record.document_type.lower().replace(' ', '_')}.pdf"
    raw_path = raw_dir / file_name

    try:
        response = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        raw_path.write_bytes(response.content)
        return True, raw_path, response.content
    except Exception as exc:  # pragma: no cover - operational guard
        return False, raw_path, str(exc).encode("utf-8")


def process_pdf_source(record: SourceRecord, raw_dir: Path, processed_dir: Path) -> bool:
    success, raw_path, _ = fetch_pdf(record, raw_dir)
    if not success or raw_path is None:
        return False

    try:
        converter = DocumentConverter()
        result = converter.convert(str(raw_path), raises_on_error=False)
        status = getattr(result, "status", None)
        status_name = getattr(status, "name", str(status)).lower()
        if status_name and "success" not in status_name and "done" not in status_name:
            return False

        if not hasattr(result, "document"):
            return False

        markdown_text = result.document.export_to_markdown()
    except Exception:
        return False

    payload: dict[str, Any] = {
        "source_url": record.source_url,
        "scheme": record.scheme,
        "publisher": record.publisher,
        "document_type": record.document_type,
        "title": record.title,
        "date": record.date,
        "source_date": record.source_date,
        "verified_on": record.verified_on,
        "raw_file": str(raw_path),
        "content": markdown_text,
        "status": "success",
    }

    out_file = processed_dir / f"{record.id}_{record.scheme.lower().replace(' ', '_')}_{record.document_type.lower().replace(' ', '_')}.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return True
