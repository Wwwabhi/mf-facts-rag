from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from src.guardrails.answer_policy import GuardrailDecision, _is_official_url, classify_question
from src.guardrails.refusal_templates import REFUSAL_MESSAGES
from src.guardrails.validation import run_guarded_query
from src.llm.answer_generator import generate_answer
from src.llm.response_formatter import parse_formatted_answer
from src.retrieval.retrieve import retrieve_chunks
from src.data.source_registry import SourceRecord, load_registry


@dataclass(frozen=True)
class QueryResponse:
    answer: str | None
    source_url: str | None
    source_date: str | None
    retrieved_chunks: list[dict[str, Any]]
    decision: GuardrailDecision


ROOT = Path(__file__).resolve().parents[2]
SOURCE_REGISTRY_PATH = ROOT / "data" / "mf_rag_sources.csv"


def _cannot_verify(reason: str) -> GuardrailDecision:
    return GuardrailDecision(False, "cannot_verify", reason, REFUSAL_MESSAGES["cannot_verify"])


def _source_date(source_date: str, verified_on: str, date: str = "") -> str:
    source_date = source_date.strip()
    if source_date and source_date.casefold() not in {"live/current page", "unknown"}:
        return source_date
    return verified_on.strip() or date.strip() or "unknown"


def _normalized_terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _matches_scheme(question: str, scheme: str) -> bool:
    question_terms = _normalized_terms(question)
    scheme_terms = _normalized_terms(scheme) - {"hdfc", "fund"}
    if scheme_terms and scheme_terms.issubset(question_terms):
        return True
    return {"tax", "saver"}.issubset(scheme_terms) and {"tax", "saver"}.issubset(question_terms)


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip().rstrip(".,;"))
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/") or "/", parts.query, ""))


def _refusal_source(
    question: str,
    category: str,
    evidence: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    if category == "cannot_verify":
        for chunk in evidence:
            metadata = chunk.get("metadata", {})
            source_url = str(metadata.get("source_url", "")).strip()
            if source_url and _is_official_url(source_url):
                return source_url, _source_date(
                    str(metadata.get("source_date", "")),
                    str(metadata.get("verified_on", "")),
                    str(metadata.get("date", "")),
                )

    records = load_registry(SOURCE_REGISTRY_PATH)
    if category == "performance_returns":
        factsheets = [record for record in records if record.document_type.casefold() == "factsheet"]
        matching = [record for record in factsheets if _matches_scheme(question, record.scheme)]
        if matching:
            record = matching[0]
            return record.source_url, _source_date(record.source_date, record.verified_on, record.date)

    educational_source = next(
        (record for record in records if record.document_type == "Investor FAQs"),
        None,
    )
    if educational_source is None or not _is_official_url(educational_source.source_url):
        return None, None
    return educational_source.source_url, _source_date(
        educational_source.source_date,
        educational_source.verified_on,
        educational_source.date,
    )


def _metadata_date_for_url(evidence: list[dict[str, Any]], source_url: str) -> str | None:
    canonical_source = _canonical_url(source_url)
    for chunk in evidence:
        metadata = chunk.get("metadata", {})
        metadata_url = str(metadata.get("source_url", "")).strip()
        if metadata_url and _canonical_url(metadata_url) == canonical_source:
            return _source_date(
                str(metadata.get("source_date", "")),
                str(metadata.get("verified_on", "")),
                str(metadata.get("date", "")),
            )
    return None


def answer_question(
    question: str,
    top_k: int = 5,
    *,
    retrieve_only: bool = False,
    retriever: Any | None = None,
    generator: Any | None = None,
) -> QueryResponse:
    retrieve_fn = retriever or retrieve_chunks
    retrieved_chunks: list[dict[str, Any]] = []

    def retrieve(query: str) -> list[dict[str, Any]]:
        retrieved_chunks.extend(retrieve_fn(query, top_k=top_k))
        return retrieved_chunks

    decision = classify_question(question)
    if not decision.allowed:
        source_url, source_date = _refusal_source(question, decision.category, retrieved_chunks)
        return QueryResponse(decision.message, source_url, source_date, retrieved_chunks, decision)

    if retrieve_only:
        retrieve(question)
        if not retrieved_chunks:
            decision = _cannot_verify("No chunks matched the factual question.")
            source_url, source_date = _refusal_source(question, decision.category, retrieved_chunks)
            return QueryResponse(decision.message, source_url, source_date, retrieved_chunks, decision)
        return QueryResponse(None, None, None, retrieved_chunks, decision)

    def generate(query: str, evidence: list[dict[str, Any]]) -> str:
        if generator is not None:
            return generator(query, evidence)
        return generate_answer(query, evidence)

    guarded = run_guarded_query(question, retrieve, generate)
    if not guarded.decision.allowed or not guarded.answer:
        source_url, source_date = _refusal_source(question, guarded.decision.category, retrieved_chunks)
        return QueryResponse(guarded.answer or guarded.decision.message, source_url, source_date, retrieved_chunks, guarded.decision)

    formatted = parse_formatted_answer(guarded.answer)
    if formatted is None:
        decision = _cannot_verify("The generated response did not match the required answer/source/date format.")
        source_url, source_date = _refusal_source(question, decision.category, retrieved_chunks)
        return QueryResponse(decision.message, source_url, source_date, retrieved_chunks, decision)
    metadata_date = _metadata_date_for_url(retrieved_chunks, formatted.source_url)
    if metadata_date is None or metadata_date != formatted.source_date:
        decision = _cannot_verify("The generated source date did not match the retrieved source metadata.")
        source_url, source_date = _refusal_source(question, decision.category, retrieved_chunks)
        return QueryResponse(decision.message, source_url, source_date, retrieved_chunks, decision)
    return QueryResponse(formatted.answer, formatted.source_url, formatted.source_date, retrieved_chunks, guarded.decision)