from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlsplit
from typing import Callable

from src.retrieval.query_pipeline import QueryResponse, answer_question


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "sample_qa.md"
OFFICIAL_HOSTS = ("hdfcfund.com", "sebi.gov.in", "amfiindia.com")
QUESTIONS = (
    ("Expense Ratio", "What were the actual expense ratios for HDFC Flexi Cap Fund for the financial year ended March 31, 2025?"),
    ("Exit Load", "What is the exit load for HDFC Flexi Cap Fund?"),
    ("ELSS Lock-In", "What is the lock-in period for HDFC ELSS Tax Saver Fund?"),
    ("Minimum SIP", "What is the minimum SIP amount for HDFC Mid Cap Fund?"),
    ("Benchmark and Riskometer", "What is the benchmark and riskometer level for HDFC Large Cap Fund?"),
    ("Capital-Gains Statement", "How can I download a capital-gains statement?"),
    ("Investment Advice", "Should I invest in HDFC Flexi Cap Fund?"),
    ("Returns and Performance", "What are the 5-year returns of HDFC Flexi Cap Fund?"),
    ("PII", "My PAN is ABCDE1234F. What is my HDFC Flexi Cap Fund balance?"),
    ("Unsupported Topic", "What will the weather be in Mumbai tomorrow?"),
)
URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _is_official_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_HOSTS)


def _render_case(index: int, title: str, question: str, response: QueryResponse) -> str:
    answer = (response.answer or "").strip()
    source_url = (response.source_url or "").strip()
    source_date = (response.source_date or "").strip()
    if not answer or not source_url or not source_date:
        raise ValueError(f"Question {index} did not return an answer with source URL and source date")
    if not _is_official_url(source_url):
        raise ValueError(f"Question {index} returned a non-approved source URL")
    if URL_PATTERN.search(answer):
        raise ValueError(f"Question {index} answer body contains an unstructured URL")
    if len(SENTENCE_BOUNDARY.split(answer)) > 3:
        raise ValueError(f"Question {index} answer exceeds three sentences")

    result = "Factual answer" if response.decision.allowed else f"Refusal ({response.decision.category})"
    return "\n".join(
        [
            f"## {index}. {title}",
            "",
            f"**Question:** {question}",
            "",
            f"**Result:** {result}",
            "",
            f"**Answer:** {answer}",
            "",
            f"**Source URL:** {source_url}",
            "",
            f"**Last updated from sources:** {source_date}",
        ]
    )


def generate_sample_qa(
    output_path: str | Path = OUTPUT_PATH,
    query_runner: Callable[[str], QueryResponse] = answer_question,
) -> Path:
    sections = ["# Sample End-to-End QA", "", "Generated from real query-pipeline responses. Answers, source URLs, and dates are taken directly from the pipeline output."]
    for index, (title, question) in enumerate(QUESTIONS, start=1):
        response = query_runner(question)
        sections.extend(["", _render_case(index, title, question, response)])

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    temporary_path.replace(destination)
    return destination


def main() -> int:
    try:
        output_path = generate_sample_qa()
    except Exception as exc:
        print(f"Sample QA generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Generated {len(QUESTIONS)} real pipeline responses in {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())