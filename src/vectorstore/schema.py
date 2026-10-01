from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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