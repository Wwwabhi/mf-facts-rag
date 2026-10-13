from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.vectorstore.embed_chunks import EMBEDDING_DIMENSION, MODEL_NAME

COLLECTION_NAME = "mutual_fund_facts"
FINGERPRINT_METADATA_KEY = "fingerprint"
DEFAULT_DISTANCE_SPACE = "cosine"
ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"


def compute_index_fingerprint(
    chunks_path: str | Path = CHUNKS_PATH,
    model_name: str = MODEL_NAME,
    dimension: int = EMBEDDING_DIMENSION,
    distance_space: str = DEFAULT_DISTANCE_SPACE,
) -> str:
    path = Path(chunks_path)
    if not path.is_file():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    hasher = hashlib.sha256()
    hasher.update(f"{model_name}:{dimension}:{distance_space}\n".encode("utf-8"))
    with path.open("rb") as handle:
        while block := handle.read(65536):
            hasher.update(block)
    return hasher.hexdigest()


def record_collection_fingerprint(
    collection: Any,
    fingerprint: str,
    model_name: str = MODEL_NAME,
    dimension: int = EMBEDDING_DIMENSION,
) -> None:
    raw_metadata = getattr(collection, "metadata", None)
    current_metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    current_metadata.setdefault("embedding_model", model_name)
    current_metadata.setdefault("embedding_dimension", dimension)
    modify_metadata = {k: v for k, v in current_metadata.items() if not k.startswith("hnsw:")}
    modify_metadata[FINGERPRINT_METADATA_KEY] = fingerprint
    if hasattr(collection, "modify"):
        collection.modify(metadata=modify_metadata)


def get_persistent_collection(
    persist_directory: str | Path,
    embedding_model: str,
    embedding_dimension: int,
) -> Any:
    import chromadb

    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": DEFAULT_DISTANCE_SPACE,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
        },
    )