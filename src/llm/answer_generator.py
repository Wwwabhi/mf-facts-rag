from __future__ import annotations

import re
from typing import Any

from src.guardrails.answer_policy import _is_official_url
from src.llm.groq_client import create_groq_client
from src.llm.response_formatter import format_answer


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("metadata", item)
    return value if isinstance(value, dict) else {}


def _source_date(metadata: dict[str, Any]) -> str:
    source_date = str(metadata.get("source_date", "")).strip()
    if source_date and source_date.casefold() not in {"live/current page", "unknown"}:
        return source_date
    return str(metadata.get("verified_on") or metadata.get("date") or "unknown")


def _format_context(evidence: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for index, item in enumerate(evidence, start=1):
        metadata = _metadata(item)
        content = str(item.get("content", "")).strip()
        sections.append(
            "\n".join(
                [
                    f"[Evidence {index}]",
                    f"Scheme: {metadata.get('scheme', 'General')}",
                    f"Document type: {metadata.get('document_type', 'Unknown')}",
                    f"Section: {metadata.get('section_breadcrumb', 'Unknown')}",
                    f"Source date: {_source_date(metadata)}",
                    f"Content: {content}",
                ]
            )
        )
    return "\n\n".join(sections)


def _citation_source(evidence: list[dict[str, Any]]) -> tuple[str, str]:
    for item in evidence:
        metadata = _metadata(item)
        source_url = str(metadata.get("source_url", "")).strip()
        if source_url and _is_official_url(source_url):
            return source_url, _source_date(metadata)
    raise ValueError("Retrieved evidence contains no approved source URL")


def generate_answer(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    client: Any | None = None,
    model_name: str | None = None,
) -> str:
    from src.config import settings

    model = (model_name or settings.GROQ_MODEL).strip()
    if not model:
        raise RuntimeError("GROQ_MODEL is not set. Add it to the project-root .env file.")
    if client is None:
        client = create_groq_client()
    source_url, source_date = _citation_source(evidence)

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a facts-only assistant for the supported HDFC mutual fund corpus. "
                    "Answer using only the supplied evidence. If it does not support the answer, "
                    "say that the available approved sources cannot verify it. Do not give investment "
                    "advice, recommendations, performance or returns analysis, or account-specific help. "
                    "Write no more than three concise sentences. Return answer text only; do not include "
                    "URLs, citations, source dates, or headings."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved evidence:\n{_format_context(evidence)}",
            },
        ],
    )
    answer_text = completion.choices[0].message.content or ""
    answer_text = answer_text.strip()
    if not answer_text:
        raise ValueError("Groq returned an empty answer")
    if len(SENTENCE_BOUNDARY.split(answer_text)) > 3:
        raise ValueError("Groq answer exceeded the three-sentence limit")
    return format_answer(answer_text, source_url, source_date)