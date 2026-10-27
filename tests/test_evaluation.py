from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.dataset_validator import (
    EvaluationSample,
    load_and_validate_dataset,
    validate_entry,
)
from evaluation.deep_eval_judge import extract_clean_json
from evaluation.evaluator import EvaluatedSampleResult, RAGEvaluator
from evaluation.pipeline_runner import (
    PipelineExecutionResult,
    calculate_cost,
)
from evaluation.reporting import _percentile, _stats_for_list


class DatasetValidatorTests(unittest.TestCase):
    def test_valid_entry(self) -> None:
        raw = {
            "id": "q001",
            "question": "What is the exit load?",
            "ground_truth": "1% within 1 year.",
            "reference_contexts": ["Exit load is 1%."],
            "category": "factual",
        }
        sample, errors = validate_entry(0, raw)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.id, "q001")
        self.assertEqual(sample.category, "factual")

    def test_malformed_missing_fields(self) -> None:
        raw = {"id": "q002"}
        sample, errors = validate_entry(1, raw)
        self.assertIsNone(sample)
        self.assertIn("Missing required field: 'question'", errors)
        self.assertIn("Missing required field: 'ground_truth'", errors)
        self.assertIn("Missing required field: 'reference_contexts'", errors)

    def test_unanswerable_empty_reference_contexts_is_valid(self) -> None:
        raw = {
            "id": "q026",
            "question": "What is the future NAV?",
            "ground_truth": "The sources cannot verify future NAV.",
            "reference_contexts": [],
            "category": "unanswerable",
        }
        sample, errors = validate_entry(2, raw)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample.reference_contexts), 0)


class CostCalculationTests(unittest.TestCase):
    def test_known_model_cost(self) -> None:
        # qwen/qwen3.8-27b: 0.20 per million input, 0.20 per million output
        cost = calculate_cost("qwen/qwen3.8-27b", 1000, 500)
        self.assertIsNotNone(cost)
        # (1000 * 0.20 + 500 * 0.20) / 1_000_000 = (200 + 100) / 1_000_000 = 0.0003
        self.assertAlmostEqual(cost, 0.0003, places=6)

    def test_unconfigured_model_returns_none(self) -> None:
        cost = calculate_cost("unregistered-unknown-model", 1000, 500)
        self.assertIsNone(cost)

    def test_none_tokens_returns_none(self) -> None:
        cost = calculate_cost("qwen/qwen3.8-27b", None, None)
        self.assertIsNone(cost)


class JsonExtractionTests(unittest.TestCase):
    def test_strips_thinking_tags(self) -> None:
        raw = "<think>Let me reason about this step by step.</think>{\"score\": 1.0, \"reason\": \"Great answer\"}"
        clean = extract_clean_json(raw)
        self.assertEqual(clean, "{\"score\": 1.0, \"reason\": \"Great answer\"}")

    def test_extracts_from_markdown_code_fence(self) -> None:
        raw = "Here is the JSON verdict:\n```json\n{\"score\": 0.85}\n```"
        clean = extract_clean_json(raw)
        self.assertEqual(clean, "{\"score\": 0.85}")


class UnanswerableEvaluatorTests(unittest.TestCase):
    def test_unanswerable_marks_faithfulness_and_recall_as_none(self) -> None:
        evaluator = RAGEvaluator(judge_model=None)
        sample = EvaluationSample(
            id="q026",
            question="What is the NAV on 23 October 2026?",
            ground_truth="The documents do not contain enough info.",
            reference_contexts=[],
            category="unanswerable",
        )
        execution = PipelineExecutionResult(
            sample_id="q026",
            question=sample.question,
            category="unanswerable",
            ground_truth=sample.ground_truth,
            actual_output="The available approved sources cannot verify this answer.",
            retrieval_context=[],
            status="SUCCESS",
        )
        result = evaluator.evaluate_sample(sample, execution)

        # Faithfulness and contextual recall must be None (N/A), NOT 0.0!
        self.assertIsNone(result.faithfulness_score)
        self.assertIn("N/A", result.faithfulness_reason)
        self.assertIsNone(result.contextual_recall_score)
        self.assertIn("N/A", result.contextual_recall_reason)

        # Relevancy score should recognize proper unanswerable handling
        self.assertIsNotNone(result.answer_relevancy_score)
        self.assertGreaterEqual(result.answer_relevancy_score, 0.8)


class StatisticsTests(unittest.TestCase):
    def test_stats_computation(self) -> None:
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = _stats_for_list(vals)
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["mean"], 30.0)
        self.assertEqual(stats["median"], 30.0)
        self.assertEqual(stats["min"], 10.0)
        self.assertEqual(stats["max"], 50.0)


class ReportGenerationTests(unittest.TestCase):
    def test_generate_reports_creates_html_csv_json_and_plot(self) -> None:
        from evaluation.reporting import generate_reports

        sample_res = EvaluatedSampleResult(
            sample_id="q001",
            question="What is the exit load?",
            category="factual",
            ground_truth="1% exit load within 1 year.",
            actual_output="Exit load is 1% if redeemed within 1 year.",
            retrieval_context=["HDFC Flexi Cap: Exit load is 1% within 1 year."],
            source_url="https://files.hdfcfund.com/doc.pdf",
            source_date="2026-08",
            decision_allowed=True,
            decision_category="factual",
            answer_relevancy_score=1.0,
            answer_relevancy_reason="Direct answer",
            faithfulness_score=1.0,
            faithfulness_reason="Grounded in evidence",
            contextual_recall_score=1.0,
            contextual_recall_reason="Complete coverage",
            retrieval_latency_ms=120.0,
            generation_latency_ms=850.0,
            total_latency_ms=970.0,
            prompt_tokens=320,
            completion_tokens=45,
            total_tokens=365,
            cost_usd=0.000073,
            status="SUCCESS",
            error_message=None,
            retries=0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path, json_path, plot_path, html_path = generate_reports(
                results=[sample_res],
                output_dir=tmp_dir,
                model_name="qwen/qwen3.8-27b",
                judge_name="Groq-qwen/qwen3.8-27b",
                dataset_name="test_dataset.json",
            )
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertTrue(html_path.exists())

            # Verify report.html has essential elements
            html_content = html_path.read_text(encoding="utf-8")
            self.assertIn("Mutual Fund Facts RAG", html_content)
            self.assertIn("q001", html_content)
            self.assertIn("What is the exit load?", html_content)
            self.assertIn("evaluation_visualizations.png", html_content)
