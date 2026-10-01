from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import settings
from src.vectorstore.chroma_client import get_persistent_collection
from src.vectorstore.embed_chunks import EMBEDDING_DIMENSION, MODEL_NAME
from src.vectorstore.index_documents import CHUNKS_PATH, PREVIEW_PATH, index_chunks


def ensure_index(
    chunks_path: str | Path = CHUNKS_PATH,
    persist_directory: str | Path = settings.CHROMA_DB_PATH,
    preview_path: str | Path = PREVIEW_PATH,
) -> dict[str, Any]:
    collection = get_persistent_collection(persist_directory, MODEL_NAME, EMBEDDING_DIMENSION)
    existing_count = int(collection.count())
    if existing_count:
        return {
            "built": False,
            "chunks_processed": 0,
            "vectors_stored": existing_count,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "persistence_path": str(Path(persist_directory).resolve()),
        }

    result = index_chunks(
        chunks_path=chunks_path,
        persist_directory=persist_directory,
        preview_path=preview_path,
    )
    return {**result, "built": True}