from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import settings
from src.retrieval.retrieve import clear_collection_cache
from src.vectorstore.chroma_client import (
    COLLECTION_NAME,
    FINGERPRINT_METADATA_KEY,
    compute_index_fingerprint,
    get_persistent_collection,
)
from src.vectorstore.embed_chunks import EMBEDDING_DIMENSION, MODEL_NAME
from src.vectorstore.index_documents import CHUNKS_PATH, PREVIEW_PATH, index_chunks


def ensure_index(
    chunks_path: str | Path = CHUNKS_PATH,
    persist_directory: str | Path = settings.CHROMA_DB_PATH,
    preview_path: str | Path = PREVIEW_PATH,
) -> dict[str, Any]:
    expected_fingerprint = compute_index_fingerprint(
        chunks_path=chunks_path,
        model_name=MODEL_NAME,
        dimension=EMBEDDING_DIMENSION,
    )
    collection = get_persistent_collection(persist_directory, MODEL_NAME, EMBEDDING_DIMENSION)
    existing_count = int(collection.count())

    is_compatible = True
    raw_metadata = getattr(collection, "metadata", None)
    meta = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    existing_dim = meta.get("embedding_dimension")
    existing_model = meta.get("embedding_model")
    if existing_dim is not None and int(existing_dim) != int(EMBEDDING_DIMENSION):
        is_compatible = False
    if existing_model is not None and str(existing_model) != str(MODEL_NAME):
        is_compatible = False

    existing_fingerprint = meta.get(FINGERPRINT_METADATA_KEY)

    if existing_count > 0 and is_compatible and existing_fingerprint == expected_fingerprint:
        return {
            "built": False,
            "chunks_processed": 0,
            "vectors_stored": existing_count,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "persistence_path": str(Path(persist_directory).resolve()),
            "fingerprint": expected_fingerprint,
        }

    if not is_compatible:
        import chromadb

        client = chromadb.PersistentClient(path=str(Path(persist_directory).resolve()))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        clear_collection_cache(persist_directory)

    result = index_chunks(
        chunks_path=chunks_path,
        persist_directory=persist_directory,
        preview_path=preview_path,
        fingerprint=expected_fingerprint,
    )
    return {**result, "built": True}