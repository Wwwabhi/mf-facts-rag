from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.guardrails.answer_policy import GuardrailDecision, classify_question
from src.guardrails.refusal_templates import REFUSAL_MESSAGES
from src.guardrails.validation import run_guarded_query
from src.llm.answer_generator import generate_answer
from src.llm.response_formatter import parse_formatted_answer
from src.retrieval.retrieve import retrieve_chunks


@dataclass(frozen=True)
class QueryResponse:
    answer: str | None
    source_url: str | None
    source_date: str | None
    retrieved_chunks: list[dict[str, Any]]
    decision: GuardrailDecision


def _cannot_verify(reason: str) -> GuardrailDecision:
    return GuardrailDecision(False, "cannot_verify", reason, REFUSAL_MESSAGES["cannot_verify"])


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
        return QueryResponse(decision.message, None, None, retrieved_chunks, decision)

    if retrieve_only:
        retrieve(question)
        if not retrieved_chunks:
            decision = _cannot_verify("No chunks matched the factual question.")
            return QueryResponse(decision.message, None, None, retrieved_chunks, decision)
        return QueryResponse(None, None, None, retrieved_chunks, decision)

    def generate(query: str, evidence: list[dict[str, Any]]) -> str:
        if generator is not None:
            return generator(query, evidence)
        return generate_answer(query, evidence)

    guarded = run_guarded_query(question, retrieve, generate)
    if not guarded.decision.allowed or not guarded.answer:
        return QueryResponse(guarded.answer or guarded.decision.message, None, None, retrieved_chunks, guarded.decision)

    formatted = parse_formatted_answer(guarded.answer)
    if formatted is None:
        decision = _cannot_verify("The generated response did not match the required answer/source/date format.")
        return QueryResponse(decision.message, None, None, retrieved_chunks, decision)
    return QueryResponse(formatted.answer, formatted.source_url, formatted.source_date, retrieved_chunks, guarded.decision)