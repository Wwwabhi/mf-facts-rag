from __future__ import annotations

from pathlib import Path
from typing import Any


COLLECTION_NAME = "mutual_fund_facts"


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
            "hnsw:space": "cosine",
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
        },
    )