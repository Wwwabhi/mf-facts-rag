from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.llm.answer_generator import generate_answer
from src.llm.response_formatter import format_answer
from src.retrieval.query_pipeline import answer_question
from src.retrieval.rerank import rerank_chunks
from src.retrieval import retrieve as retrieval_module
from src.retrieval.retrieve import retrieve_chunks
from src.data.source_registry import load_registry


OFFICIAL_URL = "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct"


class FakeModel:
    def encode(self, texts, **kwargs):
        self.texts = texts
        self.kwargs = kwargs
        return [[0.1, 0.2, 0.3]]


class FakeCollection:
    def count(self):
        return 2

    def query(self, **kwargs):
        self.kwargs = kwargs
        return {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["HDFC Flexi Cap exit load details.", "Unrelated older paragraph."]],
            "metadatas": [
                [
                    {"source_id": "1", "source_url": OFFICIAL_URL, "scheme": "HDFC Flexi Cap Fund", "document_type": "Scheme Page"},
                    {"source_id": "5", "source_url": "https://www.hdfcfund.com/large-cap", "scheme": "HDFC Large Cap Fund", "document_type": "Scheme Page"},
                ]
            ],
            "distances": [[0.2, 0.3]],
        }


class RetrievalTests(unittest.TestCase):
    def test_embedding_model_is_loaded_once_per_process(self) -> None:
        retrieval_module._MODEL_CACHE.clear()
        model = FakeModel()
        try:
            with patch.object(retrieval_module, "load_embedding_model", return_value=model) as loader:
                first = retrieval_module._shared_embedding_model()
                second = retrieval_module._shared_embedding_model()
            self.assertIs(first, second)
            loader.assert_called_once()
        finally:
            retrieval_module._MODEL_CACHE.clear()

    def test_uses_query_embedding_and_returns_ranked_chunk_metadata(self) -> None:
        model = FakeModel()
        collection = FakeCollection()

        chunks = retrieve_chunks("HDFC Flexi Cap exit load", top_k=2, model=model, collection=collection)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk_id"], "chunk-1")
        self.assertEqual(chunks[0]["metadata"]["source_id"], "1")
        self.assertTrue(model.kwargs["normalize_embeddings"])
        self.assertEqual(collection.kwargs["n_results"], 2)

    def test_explicit_scheme_match_ranks_above_closer_wrong_scheme(self) -> None:
        candidates = [
            {
                "chunk_id": "large-cap",
                "content": "HDFC Large Cap Fund exit load details.",
                "metadata": {"source_id": "5", "scheme": "HDFC Large Cap Fund"},
                "distance": 0.12,
            },
            {
                "chunk_id": "flexi-cap",
                "content": "HDFC Flexi Cap Fund exit load details.",
                "metadata": {"source_id": "1", "scheme": "HDFC Flexi Cap Fund"},
                "distance": 0.25,
            },
        ]

        reranked = rerank_chunks("What is the exit load for HDFC Flexi Cap Fund?", candidates, 2)

        self.assertEqual(reranked[0]["chunk_id"], "flexi-cap")

    def test_riskometer_ranks_matching_scheme_page_then_factsheet(self) -> None:
        candidates = [
            {
                "chunk_id": "wrong-scheme-page",
                "content": "Large Cap riskometer.",
                "metadata": {"source_id": "5", "scheme": "HDFC Large Cap Fund", "document_type": "Scheme Page"},
                "distance": 0.01,
            },
            {
                "chunk_id": "matching-factsheet",
                "content": "Flexi Cap riskometer.",
                "metadata": {"source_id": "4", "scheme": "HDFC Flexi Cap Fund", "document_type": "Factsheet"},
                "distance": 0.12,
            },
            {
                "chunk_id": "matching-kim",
                "content": "Flexi Cap riskometer.",
                "metadata": {"source_id": "3", "scheme": "HDFC Flexi Cap Fund", "document_type": "KIM"},
                "distance": 0.01,
            },
            {
                "chunk_id": "matching-scheme-page",
                "content": "Flexi Cap riskometer.",
                "metadata": {"source_id": "1", "scheme": "HDFC Flexi Cap Fund", "document_type": "Scheme Page"},
                "distance": 0.3,
            },
            {
                "chunk_id": "wrong-scheme-factsheet",
                "content": "Large Cap riskometer.",
                "metadata": {"source_id": "8", "scheme": "HDFC Large Cap Fund", "document_type": "Factsheet"},
                "distance": 0.01,
            },
        ]

        reranked = rerank_chunks("What is the riskometer for HDFC Flexi Cap Fund?", candidates, 5)

        self.assertEqual(
            [chunk["chunk_id"] for chunk in reranked[:3]],
            ["matching-scheme-page", "matching-factsheet", "matching-kim"],
        )


class GroqAnswerTests(unittest.TestCase):
    def test_sends_question_and_evidence_and_appends_provenance(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="The scheme has an exit load."))]
        )
        evidence = [
            {
                "content": "Exit load details for the scheme.",
                "metadata": {
                    "source_url": OFFICIAL_URL,
                    "source_date": "Live/current page",
                    "verified_on": "2026-09-24",
                    "scheme": "HDFC Flexi Cap Fund",
                    "document_type": "Scheme Page",
                    "section_breadcrumb": "HDFC Flexi Cap Fund > Exit Load",
                },
            }
        ]

        answer = generate_answer(
            "What is the exit load?",
            evidence,
            client=client,
            model_name="test-model",
        )

        self.assertIn("The scheme has an exit load.", answer)
        self.assertIn(f"Source URL: {OFFICIAL_URL}", answer)
        self.assertIn("Last updated from sources: 2026-09-24", answer)
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertIn("What is the exit load?", request["messages"][1]["content"])
        self.assertIn("Exit load details", request["messages"][1]["content"])


class QueryPipelineTests(unittest.TestCase):
    def test_each_guardrail_refusal_includes_official_metadata_source(self) -> None:
        blocked_questions = (
            ("Should I invest in HDFC Flexi Cap Fund?", "investment_advice"),
            ("What are the 5-year returns of HDFC Flexi Cap Fund?", "performance_returns"),
            ("My PAN is ABCDE1234F. What is my HDFC Flexi Cap Fund balance?", "privacy"),
            ("Show my HDFC folio balance.", "account_specific"),
            ("What will the weather be tomorrow?", "unsupported"),
        )
        faq = next(record for record in load_registry(retrieval_module.ROOT / "data" / "mf_rag_sources.csv") if record.document_type == "Investor FAQs")
        factsheet = next(
            record
            for record in load_registry(retrieval_module.ROOT / "data" / "mf_rag_sources.csv")
            if record.scheme == "HDFC Flexi Cap Fund" and record.document_type == "Factsheet"
        )

        for question, category in blocked_questions:
            with self.subTest(category=category):
                retriever = Mock()
                generator = Mock()
                response = answer_question(question, retriever=retriever, generator=generator)

                self.assertEqual(response.decision.category, category)
                self.assertFalse(response.decision.allowed)
                self.assertTrue(response.source_url)
                self.assertTrue(response.source_date)
                self.assertIn("hdfcfund.com", response.source_url or "")
                retriever.assert_not_called()
                generator.assert_not_called()
                if category == "performance_returns":
                    self.assertEqual(response.source_url, factsheet.source_url)
                    self.assertEqual(response.source_date, factsheet.source_date)
                else:
                    self.assertEqual(response.source_url, faq.source_url)
                    self.assertEqual(response.source_date, faq.verified_on)

    def test_cannot_verify_refusal_uses_official_source_metadata(self) -> None:
        retriever = Mock(return_value=[])
        generator = Mock()

        response = answer_question(
            "What is the exit load for HDFC Flexi Cap Fund?",
            retriever=retriever,
            generator=generator,
        )

        self.assertEqual(response.decision.category, "cannot_verify")
        self.assertEqual(response.source_url, "https://www.hdfcfund.com/services/faqs")
        self.assertEqual(response.source_date, "2026-09-24")
        generator.assert_not_called()

    def test_generated_source_date_must_match_retrieved_metadata(self) -> None:
        evidence = [
            {
                "chunk_id": "chunk-1",
                "content": "Exit load details.",
                "metadata": {
                    "source_url": OFFICIAL_URL,
                    "source_date": "Live/current page",
                    "verified_on": "2026-09-24",
                },
                "distance": 0.1,
            }
        ]
        generator = lambda question, chunks: format_answer(
            "The scheme has an exit load.", OFFICIAL_URL, "2025-01-01"
        )

        response = answer_question(
            "What is the exit load for HDFC Flexi Cap Fund?",
            retriever=lambda question, top_k: evidence,
            generator=generator,
        )

        self.assertEqual(response.decision.category, "cannot_verify")
        self.assertEqual(response.answer, "The available approved sources do not verify an answer to this question.")
        self.assertEqual(response.source_url, OFFICIAL_URL)
        self.assertEqual(response.source_date, "2026-09-24")

    def test_pipeline_returns_answer_source_and_date(self) -> None:
        evidence = [
            {
                "chunk_id": "chunk-1",
                "content": "Exit load details.",
                "metadata": {
                    "source_url": OFFICIAL_URL,
                    "source_id": "1",
                    "source_date": "Live/current page",
                    "verified_on": "2026-09-24",
                },
                "distance": 0.1,
            }
        ]
        generator = lambda question, chunks: format_answer("The scheme has an exit load.", OFFICIAL_URL, "2026-09-24")

        response = answer_question(
            "What is the exit load for HDFC Flexi Cap Fund?",
            retriever=lambda question, top_k: evidence,
            generator=generator,
        )

        self.assertTrue(response.decision.allowed)
        self.assertEqual(response.answer, "The scheme has an exit load.")
        self.assertEqual(response.source_url, OFFICIAL_URL)
        self.assertEqual(response.source_date, "2026-09-24")
        self.assertEqual(len(response.retrieved_chunks), 1)

    def test_guardrails_block_before_retrieval_and_generation(self) -> None:
        retriever = Mock()
        generator = Mock()

        response = answer_question(
            "Should I buy HDFC Flexi Cap Fund?",
            retriever=retriever,
            generator=generator,
        )

        self.assertFalse(response.decision.allowed)
        self.assertEqual(response.decision.category, "investment_advice")
        retriever.assert_not_called()
        generator.assert_not_called()

    def test_retrieve_only_skips_generation(self) -> None:
        retriever = Mock(return_value=[{"content": "Exit load.", "metadata": {"source_url": OFFICIAL_URL}}])
        generator = Mock()

        response = answer_question(
            "What is the exit load for HDFC Flexi Cap Fund?",
            retrieve_only=True,
            retriever=retriever,
            generator=generator,
        )

        self.assertTrue(response.decision.allowed)
        self.assertEqual(len(response.retrieved_chunks), 1)
        generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()