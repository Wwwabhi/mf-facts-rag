from __future__ import annotations

from threading import Lock
from pathlib import Path
from typing import Any

from src.vectorstore.chroma_client import COLLECTION_NAME
from src.vectorstore.embed_chunks import MODEL_NAME, load_embedding_model
from src.retrieval.rerank import rerank_chunks
from src.config import settings


ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIRECTORY = settings.CHROMA_DB_PATH
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = Lock()
_COLLECTION_CACHE: dict[str, Any] = {}
_COLLECTION_LOCK = Lock()


def _shared_embedding_model(model_name: str = MODEL_NAME) -> Any:
    with _MODEL_LOCK:
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = load_embedding_model(model_name)
        return _MODEL_CACHE[model_name]


def _load_collection(persist_directory: str | Path = PERSIST_DIRECTORY) -> Any:
    path = str(Path(persist_directory).resolve())
    with _COLLECTION_LOCK:
        if path not in _COLLECTION_CACHE:
            import chromadb

            client = chromadb.PersistentClient(path=path)
            _COLLECTION_CACHE[path] = client.get_collection(COLLECTION_NAME)
        return _COLLECTION_CACHE[path]


def retrieve_chunks(
    question: str,
    top_k: int = 5,
    *,
    model: Any | None = None,
    collection: Any | None = None,
    persist_directory: str | Path = PERSIST_DIRECTORY,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not question.strip():
        raise ValueError("question must not be empty")

    model = model if model is not None else _shared_embedding_model(MODEL_NAME)
    collection = collection if collection is not None else _load_collection(persist_directory)
    count = int(collection.count())
    if count == 0:
        return []

    candidate_k = min(count, max(top_k, top_k * 3))
    vector = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = vector.tolist() if hasattr(vector, "tolist") else vector
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(result["ids"][0]):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "content": result["documents"][0][index] or "",
                "metadata": result["metadatas"][0][index] or {},
                "distance": float(result["distances"][0][index]),
            }
        )
    return rerank_chunks(question, candidates, top_k)