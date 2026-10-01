from __future__ import annotations

import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from streamlit.testing.v1 import AppTest

from app.main import _prepare_index
from src.llm.answer_generator import generate_answer
from src.llm.groq_client import GroqRequestError
from src.vectorstore.index_manager import ensure_index
from src.vectorstore.index_documents import index_chunks
from src.vectorstore.schema import deduplicate_chunks, validate_registered_sources
from src.data.source_registry import load_registry
from src.guardrails.refusal_templates import FACTS_ONLY_DISCLAIMER
from src.guardrails.answer_policy import GuardrailDecision
from src.retrieval.query_pipeline import QueryResponse
from scripts.generate_sample_qa import QUESTIONS, generate_sample_qa
from src.data import pipeline as ingestion_pipeline


OFFICIAL_URL = "https://www.hdfcfund.com/example"


class FakeGroqAPIError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class IndexStartupTests(unittest.TestCase):
    def test_builds_index_when_collection_is_empty(self) -> None:
        empty_collection = Mock()
        empty_collection.count.return_value = 0
        build_result = {
            "chunks_processed": 12,
            "vectors_stored": 12,
            "embedding_dimension": 384,
            "persistence_path": "/tmp/chroma",
        }
        with tempfile.TemporaryDirectory() as directory:
            persist_path = Path(directory) / "chroma"
            chunks_path = Path(directory) / "chunks.jsonl"
            preview_path = Path(directory) / "preview.txt"
            with (
                patch("src.vectorstore.index_manager.get_persistent_collection", return_value=empty_collection),
                patch("src.vectorstore.index_manager.index_chunks", return_value=build_result) as build,
            ):
                result = ensure_index(chunks_path, persist_path, preview_path)

        self.assertTrue(result["built"])
        self.assertEqual(result["vectors_stored"], 12)
        build.assert_called_once_with(
            chunks_path=chunks_path,
            persist_directory=persist_path,
            preview_path=preview_path,
        )

    def test_skips_build_when_collection_has_vectors(self) -> None:
        populated_collection = Mock()
        populated_collection.count.return_value = 12
        with tempfile.TemporaryDirectory() as directory:
            persist_path = Path(directory) / "chroma"
            with (
                patch("src.vectorstore.index_manager.get_persistent_collection", return_value=populated_collection),
                patch("src.vectorstore.index_manager.index_chunks") as build,
            ):
                result = ensure_index(persist_directory=persist_path)

        self.assertFalse(result["built"])
        self.assertEqual(result["vectors_stored"], 12)
        build.assert_not_called()

    def test_streamlit_caches_index_preparation_for_path(self) -> None:
        _prepare_index.clear()
        try:
            with patch("app.main.ensure_index", return_value={"built": True}) as build:
                first = _prepare_index("/tmp/chroma-cached-test")
                second = _prepare_index("/tmp/chroma-cached-test")
            self.assertEqual(first, second)
            build.assert_called_once_with(persist_directory="/tmp/chroma-cached-test")
        finally:
            _prepare_index.clear()


class ChunkIndexingTests(unittest.TestCase):
    def test_deduplicates_same_source_chunks_before_embedding(self) -> None:
        records = load_registry(Path(__file__).resolve().parents[1] / "data" / "mf_rag_sources.csv")
        first_url = next(record.source_url for record in records if record.id == "1")
        second_url = next(record.source_url for record in records if record.id == "5")
        repeated_content = "Section: Exit Load\nOne percent within a year."
        chunks = [
            {"chunk_id": "first", "content": repeated_content, "source_url": first_url},
            {"chunk_id": "duplicate", "content": "Section: Exit Load  One percent within a year.", "source_url": first_url},
            {"chunk_id": "different-source", "content": repeated_content, "source_url": second_url},
        ]
        collection = Mock()
        collection.get.return_value = {"ids": []}
        collection.count.return_value = 2
        model = Mock()
        model.get_embedding_dimension.return_value = 2

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            with (
                patch("src.vectorstore.index_documents.load_chunks", return_value=chunks),
                patch("src.vectorstore.index_documents.load_embedding_model", return_value=model),
                patch("src.vectorstore.index_documents.embed_texts", return_value=np.array([[0.1, 0.2], [0.3, 0.4]])) as embed,
                patch("src.vectorstore.index_documents.write_embeddings_preview"),
                patch("src.vectorstore.index_documents.get_persistent_collection", return_value=collection),
            ):
                result = index_chunks(
                    chunks_path=directory_path / "chunks.jsonl",
                    persist_directory=directory_path / "chroma",
                    preview_path=directory_path / "preview.txt",
                )

        self.assertEqual(result["duplicates_removed"], 1)
        self.assertEqual(result["chunks_processed"], 2)
        self.assertEqual(embed.call_args.args[1], [repeated_content, repeated_content])
        self.assertEqual(len(collection.upsert.call_args.kwargs["ids"]), 2)


class RegistryScopeTests(unittest.TestCase):
    def test_ingestion_processes_only_csv_registry_rows(self) -> None:
        fields = [
            "id", "product", "amc", "scheme", "source_type", "title", "url",
            "source_date", "verified_on", "use_in_rag",
        ]
        row = {
            "id": "1",
            "product": "Groww",
            "amc": "HDFC Asset Management Company Limited",
            "scheme": "HDFC Flexi Cap Fund",
            "source_type": "SID",
            "title": "Registered SID",
            "url": "https://files.hdfcfund.com/registered.pdf",
            "source_date": "2025-01-01",
            "verified_on": "2025-01-02",
            "use_in_rag": "true",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "sources.csv"
            with registry.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            (root / "raw").mkdir()
            (root / "processed").mkdir()
            (root / "raw" / "unregistered.pdf").write_bytes(b"not in the registry")
            with (
                patch.object(ingestion_pipeline, "CSV_PATH", registry),
                patch.object(ingestion_pipeline, "RAW_DIR", root / "raw"),
                patch.object(ingestion_pipeline, "PROCESSED_DIR", root / "processed"),
                patch.object(ingestion_pipeline, "process_pdf_source", return_value=True) as process_pdf,
                patch.object(ingestion_pipeline, "process_html_source") as process_html,
            ):
                successes, total, results = ingestion_pipeline.run_pipeline()

        self.assertEqual((successes, total), (1, 1))
        self.assertEqual([result["id"] for result in results], ["1"])
        process_pdf.assert_called_once()
        process_html.assert_not_called()

    def test_rejects_chunks_from_unregistered_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not listed in the registry"):
                validate_registered_sources(
                    [{"source_url": "https://example.com/unapproved.pdf", "content": "text"}],
                    Path(__file__).resolve().parents[1] / "data" / "mf_rag_sources.csv",
                )


class GroqFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = [
            {
                "content": "Exit load details.",
                "metadata": {
                    "source_url": OFFICIAL_URL,
                    "source_date": "2026-09-24",
                    "verified_on": "2026-09-24",
                },
            }
        ]

    def test_authentication_failure_is_useful_and_redacts_key(self) -> None:
        secret = "gsk_abcdefghijklmnopqrstuvwxyz0123456789"
        client = Mock()
        client.chat.completions.create.side_effect = FakeGroqAPIError(
            f"Authentication failed for {secret}", 401
        )

        with self.assertRaises(GroqRequestError) as raised:
            generate_answer("What is the exit load?", self.evidence, client=client, model_name="test-model")

        self.assertIn("HTTP 401", str(raised.exception))
        self.assertIn("Check GROQ_API_KEY", str(raised.exception))
        self.assertNotIn(secret, str(raised.exception))

    def test_invalid_model_failure_names_model_without_key(self) -> None:
        secret = "gsk_abcdefghijklmnopqrstuvwxyz0123456789"
        client = Mock()
        client.chat.completions.create.side_effect = FakeGroqAPIError(
            f"Model error while using {secret}", 404
        )

        with self.assertRaises(GroqRequestError) as raised:
            generate_answer("What is the exit load?", self.evidence, client=client, model_name="missing-model")

        self.assertIn("missing-model", str(raised.exception))
        self.assertIn("Check GROQ_MODEL", str(raised.exception))
        self.assertNotIn(secret, str(raised.exception))

    def test_streamlit_displays_safe_groq_failure(self) -> None:
        _prepare_index.clear()
        message = "Groq rejected the configured API key (HTTP 401). Check GROQ_API_KEY."
        try:
            with (
                patch("src.vectorstore.index_manager.ensure_index", return_value={"built": False, "vectors_stored": 12}),
                patch("src.retrieval.query_pipeline.answer_question", side_effect=GroqRequestError(message)),
            ):
                app_path = Path(__file__).resolve().parents[1] / "app" / "main.py"
                app = AppTest.from_file(str(app_path)).run()
                app.text_input(key="question_input").set_value("What is the exit load?")
                app.button[-1].click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(message in element.value for element in app.error))
        finally:
            _prepare_index.clear()


class DisclaimerTests(unittest.TestCase):
    def test_readme_uses_shared_ui_disclaimer(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertIn(FACTS_ONLY_DISCLAIMER, (root / "README.md").read_text(encoding="utf-8"))
        self.assertIn("FACTS_ONLY_DISCLAIMER", (root / "app" / "main.py").read_text(encoding="utf-8"))


class SampleQAGeneratorTests(unittest.TestCase):
    def test_writes_only_real_query_response_fields(self) -> None:
        calls: list[str] = []

        def query_runner(question: str) -> QueryResponse:
            calls.append(question)
            index = len(calls)
            return QueryResponse(
                answer=f"Pipeline answer {index}.",
                source_url=f"https://www.hdfcfund.com/test-source/{index}",
                source_date=f"2026-0{index}-01",
                retrieved_chunks=[],
                decision=GuardrailDecision(True, "factual", "test response"),
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample_qa.md"
            result = generate_sample_qa(output, query_runner)
            content = result.read_text(encoding="utf-8")

        self.assertEqual(len(calls), 10)
        for index in range(1, len(QUESTIONS) + 1):
            self.assertIn(f"Pipeline answer {index}.", content)
            self.assertIn(f"https://www.hdfcfund.com/test-source/{index}", content)
            self.assertIn(f"2026-0{index}-01", content)

    def test_incomplete_response_does_not_overwrite_existing_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample_qa.md"
            output.write_text("previous real sample\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source URL and source date"):
                generate_sample_qa(
                    output,
                    lambda question: QueryResponse(
                        answer="Incomplete pipeline response.",
                        source_url=None,
                        source_date=None,
                        retrieved_chunks=[],
                        decision=GuardrailDecision(False, "unsupported", "blocked"),
                    ),
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "previous real sample\n")


if __name__ == "__main__":
    unittest.main()