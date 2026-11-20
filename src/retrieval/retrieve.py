from __future__ import annotations

from threading import Lock
from pathlib import Path
from typing import Any

from src.vectorstore.chroma_client import COLLECTION_NAME
from src.vectorstore.embed_chunks import MODEL_NAME, load_embedding_model
from src.retrieval.rerank import (
    RISKOMETER_QUERY,
    SCHEME_PAGE_QUERY,
    extract_mentioned_schemes,
    rerank_chunks,
)
from src.config import settings


ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIRECTORY = settings.CHROMA_DB_PATH
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = Lock()
_COLLECTION_CACHE: dict[str, Any] = {}
_COLLECTION_LOCK = Lock()


def clear_collection_cache(persist_directory: str | Path | None = None) -> None:
    with _COLLECTION_LOCK:
        if persist_directory is None:
            _COLLECTION_CACHE.clear()
        else:
            path = str(Path(persist_directory).resolve())
            _COLLECTION_CACHE.pop(path, None)


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

    active_model: Any = model if model is not None else _shared_embedding_model(MODEL_NAME)
    active_collection: Any = collection if collection is not None else _load_collection(persist_directory)
    count = int(active_collection.count())
    if count == 0:
        return []

    mentioned_schemes = extract_mentioned_schemes(question)
    effective_top_k = max(top_k, 8) if len(mentioned_schemes) >= 2 else top_k

    candidate_k = min(count, max(50, effective_top_k * 3))
    vector = active_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = vector.tolist() if hasattr(vector, "tolist") else vector
    result = active_collection.query(
        query_embeddings=query_embedding,
        n_results=candidate_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates: list[dict[str, Any]] = []
    existing_cids: set[str] = set()
    for index, chunk_id in enumerate(result["ids"][0]):
        existing_cids.add(chunk_id)
        candidates.append(
            {
                "chunk_id": chunk_id,
                "content": result["documents"][0][index] or "",
                "metadata": result["metadatas"][0][index] or {},
                "distance": float(result["distances"][0][index]),
            }
        )

    # Multi-scheme sub-queries: retrieve dedicated partition chunks for each mentioned scheme
    if len(mentioned_schemes) >= 2:
        for scheme in mentioned_schemes:
            try:
                sub_vector = active_model.encode([f"{scheme} {question}"], convert_to_numpy=True, normalize_embeddings=True)
                sub_embedding = sub_vector.tolist() if hasattr(sub_vector, "tolist") else sub_vector
                sub_res = active_collection.query(
                    query_embeddings=sub_embedding,
                    n_results=min(count, 15),
                    where={"scheme": scheme},
                    include=["documents", "metadatas", "distances"],
                )
                for index, chunk_id in enumerate(sub_res["ids"][0]):
                    if chunk_id not in existing_cids:
                        existing_cids.add(chunk_id)
                        candidates.append(
                            {
                                "chunk_id": chunk_id,
                                "content": sub_res["documents"][0][index] or "",
                                "metadata": sub_res["metadatas"][0][index] or {},
                                "distance": float(sub_res["distances"][0][index]),
                            }
                        )
            except Exception:
                pass

    if RISKOMETER_QUERY.search(question):
        try:
            extra_res = active_collection.get(
                where={"document_type": "Scheme Page"},
                include=["documents", "metadatas"],
            )
            for cid, doc, meta in zip(extra_res["ids"], extra_res["documents"], extra_res["metadatas"]):
                if cid not in existing_cids and "riskometer" in (doc or "").casefold():
                    candidates.append(
                        {
                            "chunk_id": cid,
                            "content": doc or "",
                            "metadata": meta or {},
                            "distance": 0.35,
                        }
                    )
                    existing_cids.add(cid)
        except Exception:
            pass

    if SCHEME_PAGE_QUERY.search(question):
        try:
            extra_res = active_collection.get(
                where={"document_type": "Scheme Page"},
                include=["documents", "metadatas"],
            )
            for cid, doc, meta in zip(extra_res["ids"], extra_res["documents"], extra_res["metadatas"]):
                if cid not in existing_cids:
                    candidates.append(
                        {
                            "chunk_id": cid,
                            "content": doc or "",
                            "metadata": meta or {},
                            "distance": 0.25,
                        }
                    )
                    existing_cids.add(cid)
        except Exception:
            pass

    return rerank_chunks(question, candidates, effective_top_k)
