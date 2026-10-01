from __future__ import annotations

import argparse
import sys

from src.retrieval.query_pipeline import QueryResponse, answer_question


def _print_result(response: QueryResponse) -> None:
    print(f"Status: {response.decision.category}")
    if response.retrieved_chunks:
        print(f"Retrieved chunks: {len(response.retrieved_chunks)}")
        for index, chunk in enumerate(response.retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})
            print(
                f"\n[{index}] distance={chunk.get('distance', 0.0):.4f} "
                f"source={metadata.get('source_id', 'unknown')} "
                f"scheme={metadata.get('scheme', 'General')} "
                f"type={metadata.get('document_type', 'Unknown')}"
            )
            section = str(metadata.get("section_breadcrumb", "Unknown"))
            print(f"Section: {section}")
            content = str(chunk.get("content", ""))
            breadcrumb = f"Section: {section}\n"
            if content.startswith(breadcrumb):
                content = content[len(breadcrumb) :].lstrip()
            print(content)
    if response.answer:
        print(f"\nAnswer: {response.answer}")
    if response.source_url:
        print(f"Source URL: {response.source_url}")
    if response.source_date:
        print(f"Source date: {response.source_date}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Query the mutual fund facts corpus.")
    parser.add_argument("question", nargs="+", help="Factual question to test")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks (default: 5)")
    parser.add_argument("--retrieve-only", action="store_true", help="Show Chroma results without calling Groq")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    try:
        response = answer_question(" ".join(args.question), args.top_k, retrieve_only=args.retrieve_only)
    except Exception as exc:
        print(f"Query failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _print_result(response)
    return 0 if response.decision.allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())