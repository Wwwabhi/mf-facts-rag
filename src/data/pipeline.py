from __future__ import annotations

from pathlib import Path

from src.data.html_extractor import process_html_source
from src.data.pdf_processor import process_pdf_source
from src.data.source_registry import load_registry


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CSV_PATH = ROOT / "data" / "mf_rag_sources.csv"


def is_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".pdf") or "pdf" in lowered


def run_pipeline() -> tuple[int, int, list[dict[str, object]]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sources = load_registry(CSV_PATH)
    results: list[dict[str, object]] = []
    successful = 0

    for source in sources:
        if is_pdf_url(source.source_url):
            success = process_pdf_source(source, RAW_DIR, PROCESSED_DIR)
        else:
            success = process_html_source(source, RAW_DIR, PROCESSED_DIR)

        if success:
            successful += 1

        results.append(
            {
                "id": source.id,
                "title": source.title,
                "source_url": source.source_url,
                "success": success,
            }
        )

    print(f"Processed {successful}/{len(sources)} sources successfully.")
    return successful, len(sources), results


if __name__ == "__main__":
    run_pipeline()
