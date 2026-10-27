"""Evaluation orchestrator using DeepEval metrics with appropriate unanswerable handling.

Calculates:
  - Answer Relevancy (AnswerRelevancyMetric)
  - Faithfulness (FaithfulnessMetric)
  - Contextual Recall (ContextualRecallMetric)

Key Design Rules:
1. Clearly distinguishes unavailable metrics (None / N/A) from true zero scores (0.0).
2. Unanswerable / refusal queries (where reference_contexts or retrieval_context is empty)
   do not receive artificial zero scores for Faithfulness or Contextual Recall;
   they are marked as None (N/A) with an explanatory reason.
3. Supports custom GroqJudge or OpenAIJudge.
4. Includes a heuristic fallback evaluator if DeepEval is not installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from evaluation.dataset_validator import EvaluationSample
from evaluation.pipeline_runner import PipelineExecutionResult

REFUSAL_KEYWORDS = (
    "cannot verify",
    "cannot be verified",
    "cannot answer",
    "do not provide",
    "does not provide",
    "does not contain",
    "do not contain",
    "not contain",
    "not enough information",
    "cannot determine",
    "cannot give",
    "outside the supported",
    "outside that scope",
    "please do not share",
    "cannot access",
)


@dataclass
class EvaluatedSampleResult:
    sample_id: str
    question: str
    category: str
    ground_truth: str
    actual_output: str
    retrieval_context: list[str]
    source_url: str | None
    source_date: str | None
    decision_allowed: bool
    decision_category: str
    # Metric scores and reasons
    answer_relevancy_score: float | None
    answer_relevancy_reason: str | None
    faithfulness_score: float | None
    faithfulness_reason: str | None
    contextual_recall_score: float | None
    contextual_recall_reason: str | None
    # Latencies (ms)
    retrieval_latency_ms: float | None
    generation_latency_ms: float | None
    total_latency_ms: float
    # Tokens & Cost
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    status: str
    error_message: str | None
    retries: int

    def to_csv_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "category": self.category,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "actual_output": self.actual_output,
            "source_url": self.source_url or "",
            "source_date": self.source_date or "",
            "decision_category": self.decision_category,
            "answer_relevancy_score": f"{self.answer_relevancy_score:.4f}" if self.answer_relevancy_score is not None else "N/A",
            "faithfulness_score": f"{self.faithfulness_score:.4f}" if self.faithfulness_score is not None else "N/A",
            "contextual_recall_score": f"{self.contextual_recall_score:.4f}" if self.contextual_recall_score is not None else "N/A",
            "answer_relevancy_reason": self.answer_relevancy_reason or "",
            "faithfulness_reason": self.faithfulness_reason or "",
            "contextual_recall_reason": self.contextual_recall_reason or "",
            "retrieval_latency_ms": f"{self.retrieval_latency_ms:.1f}" if self.retrieval_latency_ms is not None else "N/A",
            "generation_latency_ms": f"{self.generation_latency_ms:.1f}" if self.generation_latency_ms is not None else "N/A",
            "total_latency_ms": f"{self.total_latency_ms:.1f}",
            "prompt_tokens": self.prompt_tokens if self.prompt_tokens is not None else "N/A",
            "completion_tokens": self.completion_tokens if self.completion_tokens is not None else "N/A",
            "total_tokens": self.total_tokens if self.total_tokens is not None else "N/A",
            "cost_usd": f"${self.cost_usd:.6f}" if self.cost_usd is not None else "N/A",
            "status": self.status,
            "error_message": self.error_message or "",
            "retrieved_contexts": " || ".join(self.retrieval_context),
        }


def _is_refusal_or_unanswerable(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in REFUSAL_KEYWORDS)


class RAGEvaluator:
    """Evaluates pipeline outputs using DeepEval metrics."""

    def __init__(self, judge_model: Any | None = None) -> None:
        self.judge = judge_model
        self._init_deepeval()

    def _init_deepeval(self) -> None:
        if self.judge is None:
            self.has_deepeval = False
            return
        try:
            from deepeval.metrics import (
                AnswerRelevancyMetric,
                ContextualRecallMetric,
                FaithfulnessMetric,
            )
            from deepeval.test_case import LLMTestCase

            self.has_deepeval = True
            self.LLMTestCase = LLMTestCase
            # Instantiate metrics with judge
            self.relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self.judge)
            self.faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=self.judge)
            self.recall_metric = ContextualRecallMetric(threshold=0.5, model=self.judge)
        except Exception:
            self.has_deepeval = False

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        execution: PipelineExecutionResult,
    ) -> EvaluatedSampleResult:
        if execution.status != "SUCCESS":
            return EvaluatedSampleResult(
                sample_id=sample.id,
                question=sample.question,
                category=sample.category,
                ground_truth=sample.ground_truth,
                actual_output=execution.actual_output,
                retrieval_context=execution.retrieval_context,
                source_url=execution.source_url,
                source_date=execution.source_date,
                decision_allowed=execution.decision_allowed,
                decision_category=execution.decision_category,
                answer_relevancy_score=None,
                answer_relevancy_reason=f"Pipeline execution {execution.status}: {execution.error_message}",
                faithfulness_score=None,
                faithfulness_reason=f"Pipeline execution {execution.status}",
                contextual_recall_score=None,
                contextual_recall_reason=f"Pipeline execution {execution.status}",
                retrieval_latency_ms=execution.retrieval_latency_ms,
                generation_latency_ms=execution.generation_latency_ms,
                total_latency_ms=execution.total_latency_ms,
                prompt_tokens=execution.prompt_tokens,
                completion_tokens=execution.completion_tokens,
                total_tokens=execution.total_tokens,
                cost_usd=execution.cost_usd,
                status=execution.status,
                error_message=execution.error_message,
                retries=execution.retries,
            )

        is_unanswerable = (
            sample.category.lower() in {
                "unanswerable",
                "cannot_verify",
                "investment_advice",
                "performance_returns",
                "privacy",
                "account_specific",
                "unsupported",
            }
            or len(sample.reference_contexts) == 0
            or not execution.decision_allowed
        )

        relevancy_score: float | None = None
        relevancy_reason: str | None = None
        faithfulness_score: float | None = None
        faithfulness_reason: str | None = None
        recall_score: float | None = None
        recall_reason: str | None = None

        if self.has_deepeval and self.judge is not None:
            # 1. Answer Relevancy
            try:
                test_case = self.LLMTestCase(
                    input=sample.question,
                    actual_output=execution.actual_output,
                    expected_output=sample.ground_truth,
                )
                self.relevancy_metric.measure(test_case)
                relevancy_score = float(self.relevancy_metric.score)
                relevancy_reason = str(self.relevancy_metric.reason or "Evaluated by DeepEval")
            except Exception as exc:
                relevancy_score = None
                relevancy_reason = f"DeepEval Relevancy evaluation error: {exc}"

            # 2. Faithfulness
            if is_unanswerable and not execution.retrieval_context:
                # Proper N/A handling: No retrieval context exists because question was unanswerable/refused
                faithfulness_score = None
                faithfulness_reason = "N/A: Question is unanswerable or blocked before retrieval; no retrieval context to ground."
            elif not execution.retrieval_context:
                faithfulness_score = 0.0
                faithfulness_reason = "No chunks retrieved; response cannot be faithful to evidence."
            else:
                try:
                    test_case = self.LLMTestCase(
                        input=sample.question,
                        actual_output=execution.actual_output,
                        retrieval_context=execution.retrieval_context,
                    )
                    self.faithfulness_metric.measure(test_case)
                    faithfulness_score = float(self.faithfulness_metric.score)
                    faithfulness_reason = str(self.faithfulness_metric.reason or "Evaluated by DeepEval")
                except Exception as exc:
                    faithfulness_score = None
                    faithfulness_reason = f"DeepEval Faithfulness evaluation error: {exc}"

            # 3. Contextual Recall
            if is_unanswerable or len(sample.reference_contexts) == 0:
                recall_score = None
                recall_reason = "N/A: Unanswerable question has no reference context in ground truth."
            elif not execution.retrieval_context:
                recall_score = 0.0
                recall_reason = "No context was retrieved for an answerable question."
            else:
                try:
                    test_case = self.LLMTestCase(
                        input=sample.question,
                        actual_output=execution.actual_output,
                        expected_output=sample.ground_truth,
                        retrieval_context=execution.retrieval_context,
                    )
                    self.recall_metric.measure(test_case)
                    recall_score = float(self.recall_metric.score)
                    recall_reason = str(self.recall_metric.reason or "Evaluated by DeepEval")
                except Exception as exc:
                    recall_score = None
                    recall_reason = f"DeepEval Contextual Recall evaluation error: {exc}"

        else:
            # Fallback evaluation logic (token overlap & refusal adherence)
            relevancy_score, relevancy_reason = self._heuristic_relevancy(sample, execution, is_unanswerable)
            faithfulness_score, faithfulness_reason = self._heuristic_faithfulness(sample, execution, is_unanswerable)
            recall_score, recall_reason = self._heuristic_recall(sample, execution, is_unanswerable)

        return EvaluatedSampleResult(
            sample_id=sample.id,
            question=sample.question,
            category=sample.category,
            ground_truth=sample.ground_truth,
            actual_output=execution.actual_output,
            retrieval_context=execution.retrieval_context,
            source_url=execution.source_url,
            source_date=execution.source_date,
            decision_allowed=execution.decision_allowed,
            decision_category=execution.decision_category,
            answer_relevancy_score=relevancy_score,
            answer_relevancy_reason=relevancy_reason,
            faithfulness_score=faithfulness_score,
            faithfulness_reason=faithfulness_reason,
            contextual_recall_score=recall_score,
            contextual_recall_reason=recall_reason,
            retrieval_latency_ms=execution.retrieval_latency_ms,
            generation_latency_ms=execution.generation_latency_ms,
            total_latency_ms=execution.total_latency_ms,
            prompt_tokens=execution.prompt_tokens,
            completion_tokens=execution.completion_tokens,
            total_tokens=execution.total_tokens,
            cost_usd=execution.cost_usd,
            status=execution.status,
            error_message=execution.error_message,
            retries=execution.retries,
        )

    def _heuristic_relevancy(
        self,
        sample: EvaluationSample,
        execution: PipelineExecutionResult,
        is_unanswerable: bool,
    ) -> tuple[float, str]:
        actual = execution.actual_output.strip()
        gt = sample.ground_truth.strip()
        if not actual:
            return 0.0, "Empty output generated"

        if is_unanswerable:
            actual_refuses = _is_refusal_or_unanswerable(actual)
            gt_refuses = _is_refusal_or_unanswerable(gt)
            if actual_refuses and gt_refuses:
                return 1.0, "Appropriately recognized as unanswerable/refusal matching ground truth"
            if actual_refuses != gt_refuses:
                return 0.2, "Refusal mismatch between model and ground truth"

        # Word overlap Jaccard similarity
        actual_words = set(re.findall(r"\w+", actual.lower()))
        gt_words = set(re.findall(r"\w+", gt.lower()))
        if not gt_words:
            return 1.0, "Empty ground truth"
        overlap = len(actual_words & gt_words) / len(gt_words)
        score = min(1.0, max(0.0, overlap * 1.2))
        return round(score, 4), f"Lexical alignment score: {score:.2f}"

    def _heuristic_faithfulness(
        self,
        sample: EvaluationSample,
        execution: PipelineExecutionResult,
        is_unanswerable: bool,
    ) -> tuple[float | None, str]:
        if is_unanswerable and not execution.retrieval_context:
            return None, "N/A: Question is unanswerable or blocked before retrieval; no retrieval context to ground."
        if not execution.retrieval_context:
            return 0.0, "No context retrieved to support factual claims"

        combined_ctx = " ".join(execution.retrieval_context).lower()
        actual_words = set(re.findall(r"\w{4,}", execution.actual_output.lower()))
        if not actual_words:
            return 1.0, "No substantive claim terms"

        supported = sum(1 for w in actual_words if w in combined_ctx)
        score = supported / len(actual_words)
        return round(score, 4), f"Heuristic grounding: {supported}/{len(actual_words)} key terms found in context"

    def _heuristic_recall(
        self,
        sample: EvaluationSample,
        execution: PipelineExecutionResult,
        is_unanswerable: bool,
    ) -> tuple[float | None, str]:
        if is_unanswerable or len(sample.reference_contexts) == 0:
            return None, "N/A: Ground truth has no reference contexts for unanswerable question."
        if not execution.retrieval_context:
            return 0.0, "No context retrieved for answerable question"

        # Check if reference context concepts appear in retrieved context
        combined_retrieved = " ".join(execution.retrieval_context).lower()
        ref_matched = 0
        for ref in sample.reference_contexts:
            ref_words = set(re.findall(r"\w{4,}", ref.lower()))
            if not ref_words:
                ref_matched += 1
                continue
            matched = sum(1 for w in ref_words if w in combined_retrieved)
            if (matched / len(ref_words)) > 0.4:
                ref_matched += 1

        score = ref_matched / len(sample.reference_contexts)
        return round(score, 4), f"Contextual recall heuristic: {ref_matched}/{len(sample.reference_contexts)} reference sources covered"
