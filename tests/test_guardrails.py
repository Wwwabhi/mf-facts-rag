from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.guardrails.answer_policy import classify_question
from src.guardrails.validation import run_guarded_query


class GuardrailClassificationTests(unittest.TestCase):
    def test_factual_question_is_allowed(self) -> None:
        decision = classify_question("What is the exit load for HDFC Flexi Cap Fund?")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.category, "factual")

    def test_general_statement_instructions_are_allowed(self) -> None:
        decision = classify_question("How can I download a consolidated account statement?")
        self.assertTrue(decision.allowed)

    def test_capital_gains_statement_question_is_supported(self) -> None:
        decision = classify_question("How can I download a capital-gains statement?")
        self.assertTrue(decision.allowed)

    def test_buy_recommendation_is_blocked(self) -> None:
        decision = classify_question("Should I buy HDFC Flexi Cap Fund?")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "investment_advice")

    def test_performance_and_returns_are_blocked(self) -> None:
        decision = classify_question("What are the historical returns of HDFC Flexi Cap Fund?")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "performance_returns")

    def test_pii_is_blocked_without_echoing_it(self) -> None:
        query = "My PAN is ABCDE1234F; what is my investment value?"
        decision = classify_question(query)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "privacy")
        self.assertNotIn("ABCDE1234F", decision.message or "")

    def test_account_specific_request_is_blocked(self) -> None:
        decision = classify_question("Show my HDFC folio balance.")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "account_specific")

    def test_off_topic_question_is_blocked(self) -> None:
        decision = classify_question("What will the weather be tomorrow?")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "unsupported")

    def test_unlisted_scheme_is_blocked(self) -> None:
        decision = classify_question("What is the exit load for HDFC Small Cap Fund?")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.category, "unsupported")


class GuardedDispatchTests(unittest.TestCase):
    def test_blocked_questions_never_reach_retrieval_or_llm(self) -> None:
        blocked_questions = (
            "Should I buy HDFC Flexi Cap Fund?",
            "What are the returns of HDFC Flexi Cap Fund?",
            "My PAN is ABCDE1234F; show my account value.",
            "What will the weather be tomorrow?",
        )
        for question in blocked_questions:
            with self.subTest(question=question):
                retrieve = Mock()
                generate = Mock()
                result = run_guarded_query(question, retrieve, generate)
                self.assertFalse(result.decision.allowed)
                retrieve.assert_not_called()
                generate.assert_not_called()

    def test_factual_question_reaches_callbacks_with_approved_evidence(self) -> None:
        source_url = "https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct"
        evidence = [{"source_url": source_url, "publisher": "HDFC Asset Management Company Limited"}]
        retrieve = Mock(return_value=evidence)
        generate = Mock(return_value=f"See source: {source_url}")

        result = run_guarded_query("What is the exit load for HDFC Flexi Cap Fund?", retrieve, generate)

        self.assertTrue(result.decision.allowed)
        self.assertEqual(result.answer, f"See source: {source_url}")
        retrieve.assert_called_once()
        generate.assert_called_once_with("What is the exit load for HDFC Flexi Cap Fund?", evidence)

    def test_missing_evidence_blocks_llm(self) -> None:
        retrieve = Mock(return_value=[])
        generate = Mock()

        result = run_guarded_query("What is the exit load for HDFC Flexi Cap Fund?", retrieve, generate)

        self.assertEqual(result.decision.category, "cannot_verify")
        retrieve.assert_called_once()
        generate.assert_not_called()

    def test_unapproved_evidence_blocks_llm(self) -> None:
        retrieve = Mock(return_value=[{"source_url": "https://example.com/fund-details"}])
        generate = Mock()

        result = run_guarded_query("What is the exit load for HDFC Flexi Cap Fund?", retrieve, generate)

        self.assertEqual(result.decision.category, "cannot_verify")
        generate.assert_not_called()

    def test_unapproved_or_invented_citation_blocks_answer(self) -> None:
        evidence = [{"source_url": "https://www.hdfcfund.com/factsheet"}]
        retrieve = Mock(return_value=evidence)
        generate = Mock(return_value="Source: https://example.com/fund-details")

        result = run_guarded_query("What is the exit load for HDFC Flexi Cap Fund?", retrieve, generate)

        self.assertEqual(result.decision.category, "cannot_verify")
        self.assertEqual(result.answer, "The available approved sources do not verify an answer to this question.")


if __name__ == "__main__":
    unittest.main()