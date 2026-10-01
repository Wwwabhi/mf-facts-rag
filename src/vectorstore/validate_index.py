from __future__ import annotations

from pathlib import Path

from src.vectorstore.chroma_client import COLLECTION_NAME


ROOT = Path(__file__).resolve().parents[2]
PERSIST_DIRECTORY = ROOT / "vectorstore" / "chroma_db"


def main() -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(PERSIST_DIRECTORY))
    collection = client.get_collection(COLLECTION_NAME)
    sample = collection.get(limit=1, include=["embeddings", "metadatas"])
    embeddings = sample["embeddings"]
    dimension = len(embeddings[0]) if len(embeddings) else 0
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vectors: {collection.count()}")
    print(f"Sample embedding dimension: {dimension}")
    print(f"Persistence path: {PERSIST_DIRECTORY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())