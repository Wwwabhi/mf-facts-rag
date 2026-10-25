"""RAG Pipeline execution runner for evaluation.

Executes questions against either:
1. Direct in-process RAG pipeline (src.retrieval.query_pipeline & src.llm.answer_generator),
   accurately capturing retrieval latency, generation latency, total latency,
   token usage, and retrieved chunks.
2. An external HTTP REST endpoint with configurable request payload and response JSON mapping.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Sourced from official published provider pricing (USD per 1,000,000 tokens)
# Only models explicitly present here will have costs calculated.
# If a model is not listed, cost is set to None. Never invent pricing!
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "qwen/qwen3.8-27b": {"input_per_million": 0.20, "output_per_million": 0.20},
    "llama-3.3-70b-versatile": {"input_per_million": 0.59, "output_per_million": 0.79},
    "llama-3.1-8b-instant": {"input_per_million": 0.05, "output_per_million": 0.08},
    "mixtral-8x7b-32768": {"input_per_million": 0.24, "output_per_million": 0.24},
    "gpt-4o-mini": {"input_per_million": 0.15, "output_per_million": 0.60},
    "gpt-4o": {"input_per_million": 2.50, "output_per_million": 10.00},
}


@dataclass
class PipelineExecutionResult:
    sample_id: str
    question: str
    category: str
    ground_truth: str
    actual_output: str
    retrieval_context: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    source_url: str | None = None
    source_date: str | None = None
    decision_allowed: bool = True
    decision_category: str = "factual"
    decision_reason: str = ""
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    status: str = "SUCCESS"  # SUCCESS, FAILED, ERROR
    error_message: str | None = None
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "category": self.category,
            "ground_truth": self.ground_truth,
            "actual_output": self.actual_output,
            "retrieval_context": list(self.retrieval_context),
            "source_url": self.source_url,
            "source_date": self.source_date,
            "decision_allowed": self.decision_allowed,
            "decision_category": self.decision_category,
            "decision_reason": self.decision_reason,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "error_message": self.error_message,
            "retries": self.retries,
        }


def calculate_cost(
    model_name: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing_table: dict[str, dict[str, float]] | None = None,
) -> float | None:
    if prompt_tokens is None or completion_tokens is None:
        return None
    table = pricing_table if pricing_table is not None else DEFAULT_MODEL_PRICING
    model_key = model_name.strip()
    if model_key not in table:
        return None
    rates = table[model_key]
    in_rate = rates.get("input_per_million")
    out_rate = rates.get("output_per_million")
    if in_rate is None or out_rate is None:
        return None
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0


class DirectPipelineRunner:
    """Executes queries directly through the local Python RAG pipeline with full instrumentation."""

    def __init__(
        self,
        top_k: int = 5,
        model_name: str | None = None,
        pricing_table: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.top_k = top_k
        self.model_name = model_name or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.pricing_table = pricing_table or DEFAULT_MODEL_PRICING

    def run_sample(
        self,
        sample_id: str,
        question: str,
        ground_truth: str,
        category: str,
        max_retries: int = 2,
    ) -> PipelineExecutionResult:
        from src.guardrails.answer_policy import (
            _is_official_url,
            answer_has_supported_citations,
            classify_question,
            has_approved_evidence,
        )
        from src.guardrails.refusal_templates import REFUSAL_MESSAGES
        from src.llm.answer_generator import _citation_source, _format_context, SENTENCE_BOUNDARY
        from src.llm.groq_client import create_groq_client, format_groq_error
        from src.llm.response_formatter import format_answer, parse_formatted_answer
        from src.retrieval.query_pipeline import _metadata_date_for_url, _refusal_source
        from src.retrieval.retrieve import retrieve_chunks

        t_start = time.perf_counter()
        retries = 0

        # 1. Guardrail classification
        decision = classify_question(question)
        if not decision.allowed:
            source_url, source_date = _refusal_source(question, decision.category, [])
            total_lat = (time.perf_counter() - t_start) * 1000.0
            return PipelineExecutionResult(
                sample_id=sample_id,
                question=question,
                category=category,
                ground_truth=ground_truth,
                actual_output=decision.message or "",
                retrieval_context=[],
                retrieved_chunks=[],
                source_url=source_url,
                source_date=source_date,
                decision_allowed=False,
                decision_category=decision.category,
                decision_reason=decision.reason,
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                total_latency_ms=total_lat,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                status="SUCCESS",
            )

        # 2. Retrieval with timing
        t_ret_start = time.perf_counter()
        retrieved_chunks: list[dict[str, Any]] = []
        try:
            retrieved_chunks = retrieve_chunks(question, top_k=self.top_k)
        except Exception as exc:
            total_lat = (time.perf_counter() - t_start) * 1000.0
            return PipelineExecutionResult(
                sample_id=sample_id,
                question=question,
                category=category,
                ground_truth=ground_truth,
                actual_output="",
                total_latency_ms=total_lat,
                status="ERROR",
                error_message=f"Retrieval error: {exc}",
            )
        t_ret_end = time.perf_counter()
        retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000.0

        # Extract context strings
        retrieval_context = [
            f"{c.get('metadata', {}).get('scheme', '')}: {c.get('content', '')}".strip()
            for c in retrieved_chunks
            if c.get("content")
        ]

        # 3. Check evidence guardrail
        evidence_urls = [str(c.get("metadata", {}).get("source_url", "")) for c in retrieved_chunks]
        if not has_approved_evidence(evidence_urls):
            source_url, source_date = _refusal_source(question, "cannot_verify", retrieved_chunks)
            refusal_msg = REFUSAL_MESSAGES["cannot_verify"]
            total_lat = (time.perf_counter() - t_start) * 1000.0
            return PipelineExecutionResult(
                sample_id=sample_id,
                question=question,
                category=category,
                ground_truth=ground_truth,
                actual_output=refusal_msg,
                retrieval_context=retrieval_context,
                retrieved_chunks=retrieved_chunks,
                source_url=source_url,
                source_date=source_date,
                decision_allowed=False,
                decision_category="cannot_verify",
                decision_reason="No approved official source returned in retrieved evidence",
                retrieval_latency_ms=retrieval_latency_ms,
                generation_latency_ms=0.0,
                total_latency_ms=total_lat,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                status="SUCCESS",
            )

        # 4. LLM Generation with timing & token capture
        client = create_groq_client()
        source_url, source_date = _citation_source(retrieved_chunks)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                t_gen_start = time.perf_counter()
                completion = client.chat.completions.create(
                    model=self.model_name,
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
                            "content": f"Question:\n{question}\n\nRetrieved evidence:\n{_format_context(retrieved_chunks)}",
                        },
                    ],
                )
                t_gen_end = time.perf_counter()
                generation_latency_ms = (t_gen_end - t_gen_start) * 1000.0

                answer_text = (completion.choices[0].message.content or "").strip()
                if not answer_text:
                    raise ValueError("Groq returned empty text")
                if len(SENTENCE_BOUNDARY.split(answer_text)) > 3:
                    raise ValueError("Groq answer exceeded three-sentence limit")

                formatted_answer_str = format_answer(answer_text, source_url, source_date)

                # Citation check
                if not answer_has_supported_citations(formatted_answer_str, evidence_urls):
                    raise ValueError("Answer cited unapproved source")

                parsed = parse_formatted_answer(formatted_answer_str)
                if parsed is None:
                    raise ValueError("Failed to parse formatted answer")

                metadata_date = _metadata_date_for_url(retrieved_chunks, parsed.source_url)
                if metadata_date is None or metadata_date != parsed.source_date:
                    raise ValueError("Metadata date mismatch")

                # Tokens & cost
                usage = getattr(completion, "usage", None)
                p_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                c_tokens = getattr(usage, "completion_tokens", None) if usage else None
                tot_tokens = getattr(usage, "total_tokens", None) if usage else None
                cost = calculate_cost(self.model_name, p_tokens, c_tokens, self.pricing_table)

                total_lat = (time.perf_counter() - t_start) * 1000.0

                return PipelineExecutionResult(
                    sample_id=sample_id,
                    question=question,
                    category=category,
                    ground_truth=ground_truth,
                    actual_output=parsed.answer,
                    retrieval_context=retrieval_context,
                    retrieved_chunks=retrieved_chunks,
                    source_url=parsed.source_url,
                    source_date=parsed.source_date,
                    decision_allowed=True,
                    decision_category="factual",
                    decision_reason="Generated with verified evidence",
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    total_latency_ms=total_lat,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=tot_tokens,
                    cost_usd=cost,
                    status="SUCCESS",
                    retries=attempt,
                )
            except Exception as exc:
                last_error = exc
                retries = attempt + 1
                time.sleep(1.0 * (2**attempt))

        # If generation retries exhausted
        source_url_ref, source_date_ref = _refusal_source(question, "cannot_verify", retrieved_chunks)
        total_lat = (time.perf_counter() - t_start) * 1000.0
        return PipelineExecutionResult(
            sample_id=sample_id,
            question=question,
            category=category,
            ground_truth=ground_truth,
            actual_output=REFUSAL_MESSAGES["cannot_verify"],
            retrieval_context=retrieval_context,
            retrieved_chunks=retrieved_chunks,
            source_url=source_url_ref,
            source_date=source_date_ref,
            decision_allowed=False,
            decision_category="cannot_verify",
            decision_reason=f"Generation failed after {retries} retries: {last_error}",
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=None,
            total_latency_ms=total_lat,
            status="FAILED",
            error_message=str(last_error),
            retries=retries,
        )


class HttpApiRunner:
    """Executes queries against an external HTTP API endpoint with configurable schema mapping."""

    def __init__(
        self,
        endpoint_url: str,
        headers: dict[str, str] | None = None,
        request_mapping: Callable[[str], dict[str, Any]] | None = None,
        response_mapping: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.headers = headers or {"Content-Type": "application/json"}
        # Default request payload mapping: {"question": question}
        self.request_mapping = request_mapping or (lambda q: {"question": q})
        # Default response keys mapping:
        self.response_mapping = response_mapping or {
            "answer": "answer",
            "retrieval_context": "retrieval_context",
            "source_url": "source_url",
            "source_date": "source_date",
        }
        self.timeout_seconds = timeout_seconds

    def _extract_field(self, data: dict[str, Any], key_path: str) -> Any:
        # Supports dotted keys like "data.result.answer"
        curr = data
        for part in key_path.split("."):
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr

    def run_sample(
        self,
        sample_id: str,
        question: str,
        ground_truth: str,
        category: str,
        max_retries: int = 2,
    ) -> PipelineExecutionResult:
        import requests

        payload = self.request_mapping(question)
        t_start = time.perf_counter()
        retries = 0
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout_seconds,
                )
                total_lat = (time.perf_counter() - t_start) * 1000.0

                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

                data = resp.json()
                answer_key = self.response_mapping.get("answer", "answer")
                actual_output = str(self._extract_field(data, answer_key) or "")

                ctx_key = self.response_mapping.get("retrieval_context", "retrieval_context")
                raw_ctx = self._extract_field(data, ctx_key)
                if isinstance(raw_ctx, list):
                    retrieval_context = [str(x) for x in raw_ctx]
                elif isinstance(raw_ctx, str) and raw_ctx:
                    retrieval_context = [raw_ctx]
                else:
                    retrieval_context = []

                source_url = self._extract_field(data, self.response_mapping.get("source_url", "source_url"))
                source_date = self._extract_field(data, self.response_mapping.get("source_date", "source_date"))

                return PipelineExecutionResult(
                    sample_id=sample_id,
                    question=question,
                    category=category,
                    ground_truth=ground_truth,
                    actual_output=actual_output,
                    retrieval_context=retrieval_context,
                    source_url=str(source_url) if source_url else None,
                    source_date=str(source_date) if source_date else None,
                    total_latency_ms=total_lat,
                    status="SUCCESS",
                    retries=attempt,
                )
            except Exception as exc:
                last_error = exc
                retries = attempt + 1
                time.sleep(1.0 * (2**attempt))

        total_lat = (time.perf_counter() - t_start) * 1000.0
        return PipelineExecutionResult(
            sample_id=sample_id,
            question=question,
            category=category,
            ground_truth=ground_truth,
            actual_output="",
            total_latency_ms=total_lat,
            status="FAILED",
            error_message=str(last_error),
            retries=retries,
        )
