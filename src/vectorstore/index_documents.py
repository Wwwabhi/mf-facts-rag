from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from src.vectorstore.chroma_client import get_persistent_collection
from src.vectorstore.embed_chunks import (
    MODEL_NAME,
    embed_texts,
    load_embedding_model,
    write_embeddings_preview,
)
from src.vectorstore.schema import chroma_metadata, deduplicate_chunks, load_chunks, validate_registered_sources
from src.config import settings


ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "chunks" / "chunks.jsonl"
CSV_PATH = ROOT / "data" / "mf_rag_sources.csv"
PREVIEW_PATH = ROOT / "data" / "embeddings_preview.txt"
PERSIST_DIRECTORY = settings.CHROMA_DB_PATH
BATCH_SIZE = 128


def _remove_stale_vectors(collection: Any, chunk_ids: set[str]) -> None:
    existing = collection.get(include=["metadatas"])
    stale_ids = [chunk_id for chunk_id in existing["ids"] if chunk_id not in chunk_ids]
    for start in range(0, len(stale_ids), BATCH_SIZE):
        collection.delete(ids=stale_ids[start : start + BATCH_SIZE])


def index_chunks(
    chunks_path: str | Path = CHUNKS_PATH,
    persist_directory: str | Path = PERSIST_DIRECTORY,
    preview_path: str | Path = PREVIEW_PATH,
) -> dict[str, Any]:
    chunks, duplicates_removed = deduplicate_chunks(load_chunks(chunks_path))
    validate_registered_sources(chunks, CSV_PATH)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    model = load_embedding_model()
    dimension_method = getattr(model, "get_embedding_dimension", None)
    if dimension_method is None:
        dimension_method = model.get_sentence_embedding_dimension
    dimension = int(dimension_method())
    embeddings = embed_texts(model, [chunk["content"] for chunk in chunks])
    if embeddings.shape != (len(chunks), dimension):
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
    write_embeddings_preview(chunks, embeddings, preview_path)

    collection = get_persistent_collection(persist_directory, MODEL_NAME, dimension)
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    _remove_stale_vectors(collection, chunk_ids)

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            embeddings=embeddings[start : start + len(batch)].tolist(),
            documents=[chunk["content"] for chunk in batch],
            metadatas=[chroma_metadata(chunk) for chunk in batch],
        )

    vectors_stored = int(collection.count())
    if vectors_stored != len(chunks):
        raise RuntimeError(f"Expected {len(chunks)} vectors, found {vectors_stored}")

    return {
        "chunks_processed": len(chunks),
        "duplicates_removed": duplicates_removed,
        "vectors_stored": vectors_stored,
        "embedding_dimension": dimension,
        "persistence_path": str(Path(persist_directory).resolve()),
        "preview_path": str(Path(preview_path).resolve()),
    }


def main() -> int:
    report = index_chunks()
    print(f"Chunks processed: {report['chunks_processed']}")
    print(f"Duplicate chunks removed: {report['duplicates_removed']}")
    print(f"Vectors stored: {report['vectors_stored']}")
    print(f"Embedding dimension: {report['embedding_dimension']}")
    print(f"ChromaDB persistence path: {report['persistence_path']}")
    print(f"Embedding preview: {report['preview_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())