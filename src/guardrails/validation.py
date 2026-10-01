from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.guardrails.answer_policy import (
    GuardrailDecision,
    answer_has_supported_citations,
    classify_question,
    has_approved_evidence,
)
from src.guardrails.refusal_templates import REFUSAL_MESSAGES


@dataclass(frozen=True)
class GuardedResult:
    decision: GuardrailDecision
    answer: str | None


def _metadata_records(evidence: Any) -> list[Mapping[str, Any]]:
    if isinstance(evidence, Mapping):
        metadatas = evidence.get("metadatas")
        if isinstance(metadatas, Sequence):
            return [item for group in metadatas for item in (group or []) if isinstance(item, Mapping)]
        return [evidence]
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
        records: list[Mapping[str, Any]] = []
        for item in evidence:
            if not isinstance(item, Mapping):
                continue
            metadata = item.get("metadata", item)
            if isinstance(metadata, Mapping):
                records.append(metadata)
        return records
    return []


def _cannot_verify(reason: str) -> GuardrailDecision:
    return GuardrailDecision(False, "cannot_verify", reason, REFUSAL_MESSAGES["cannot_verify"])


def run_guarded_query(
    question: str,
    retrieve: Callable[[str], Any],
    generate: Callable[[str, Any], str],
) -> GuardedResult:
    decision = classify_question(question)
    if not decision.allowed:
        return GuardedResult(decision, decision.message)

    evidence = retrieve(question)
    metadata = _metadata_records(evidence)
    evidence_urls = [str(item.get("source_url", "")) for item in metadata]
    if not has_approved_evidence(evidence_urls):
        blocked = _cannot_verify("No approved official source was returned for the factual question.")
        return GuardedResult(blocked, blocked.message)

    answer = generate(question, evidence)
    if not answer_has_supported_citations(answer, evidence_urls):
        blocked = _cannot_verify("The generated answer did not cite an approved source from retrieved evidence.")
        return GuardedResult(blocked, blocked.message)
    return GuardedResult(decision, answer)