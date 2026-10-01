from __future__ import annotations

from pathlib import Path
from typing import Any


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def load_embedding_model(model_name: str = MODEL_NAME) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(model: Any, texts: list[str], batch_size: int = 64) -> Any:
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def write_embeddings_preview(
    chunks: list[dict[str, Any]],
    embeddings: Any,
    output_path: str | Path,
    model_name: str = MODEL_NAME,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dimension = int(embeddings.shape[1])
    lines = [
        f"Model: {model_name}",
        f"Embedding dimension: {dimension}",
        "First 10 dimensions for the first 5 chunks:",
    ]
    for index, (chunk, vector) in enumerate(zip(chunks[:5], embeddings[:5]), start=1):
        values = ", ".join(f"{float(value):.8f}" for value in vector[:10])
        lines.append(f"{index}. {chunk['chunk_id']}: [{values}]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")