"""Dataset validator for RAG evaluation.

Validates test_dataset.json ensuring every record conforms to the required schema:
  - id: non-empty string
  - question: non-empty string
  - ground_truth: non-empty string
  - reference_contexts: list of strings (may be empty for unanswerable questions)
  - category: string (factual, multi_hop, procedural, unanswerable, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvaluationSample:
    id: str
    question: str
    ground_truth: str
    reference_contexts: list[str]
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "reference_contexts": list(self.reference_contexts),
            "category": self.category,
        }


@dataclass(frozen=True)
class ValidationIssue:
    index: int
    raw_item: Any
    errors: list[str]


@dataclass(frozen=True)
class ValidationResult:
    valid_samples: list[EvaluationSample]
    malformed_entries: list[ValidationIssue]
    total_loaded: int

    @property
    def is_valid(self) -> bool:
        return len(self.malformed_entries) == 0 and len(self.valid_samples) > 0


def validate_entry(index: int, item: Any) -> tuple[EvaluationSample | None, list[str]]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return None, [f"Entry #{index} is not a JSON object (got {type(item).__name__})."]

    # Check id
    sample_id = item.get("id")
    if sample_id is None:
        errors.append("Missing required field: 'id'")
    elif not isinstance(sample_id, str) or not sample_id.strip():
        errors.append("'id' must be a non-empty string")

    # Check question
    question = item.get("question")
    if question is None:
        errors.append("Missing required field: 'question'")
    elif not isinstance(question, str) or not question.strip():
        errors.append("'question' must be a non-empty string")

    # Check ground_truth
    ground_truth = item.get("ground_truth")
    if ground_truth is None:
        errors.append("Missing required field: 'ground_truth'")
    elif not isinstance(ground_truth, str) or not ground_truth.strip():
        errors.append("'ground_truth' must be a non-empty string")

    # Check reference_contexts
    reference_contexts = item.get("reference_contexts")
    cleaned_contexts: list[str] = []
    if reference_contexts is None:
        errors.append("Missing required field: 'reference_contexts'")
    elif not isinstance(reference_contexts, list):
        errors.append("'reference_contexts' must be a list of strings")
    else:
        for ctx_idx, ctx in enumerate(reference_contexts):
            if not isinstance(ctx, str):
                errors.append(f"'reference_contexts[{ctx_idx}]' must be a string (got {type(ctx).__name__})")
            else:
                cleaned_contexts.append(ctx.strip())

    # Check category
    category = item.get("category")
    if category is None:
        errors.append("Missing required field: 'category'")
    elif not isinstance(category, str) or not category.strip():
        errors.append("'category' must be a non-empty string")

    if errors:
        return None, errors

    return (
        EvaluationSample(
            id=str(sample_id).strip(),
            question=str(question).strip(),
            ground_truth=str(ground_truth).strip(),
            reference_contexts=cleaned_contexts,
            category=str(category).strip(),
        ),
        [],
    )


def load_and_validate_dataset(dataset_path: str | Path) -> ValidationResult:
    path = Path(dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Test dataset file not found at: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file {path}: {exc}") from exc

    if not isinstance(raw_data, list):
        raise ValueError(f"Root of dataset must be a JSON array, got {type(raw_data).__name__}")

    valid_samples: list[EvaluationSample] = []
    malformed_entries: list[ValidationIssue] = []

    for index, item in enumerate(raw_data):
        sample, errors = validate_entry(index, item)
        if errors:
            malformed_entries.append(ValidationIssue(index=index, raw_item=item, errors=errors))
        elif sample is not None:
            valid_samples.append(sample)

    return ValidationResult(
        valid_samples=valid_samples,
        malformed_entries=malformed_entries,
        total_loaded=len(raw_data),
    )
