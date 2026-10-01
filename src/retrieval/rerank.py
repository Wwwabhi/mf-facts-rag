from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "do", "does", "for", "from", "how", "i", "in", "is",
    "it", "of", "on", "the", "to", "what", "when", "where", "which", "who", "with",
}
SCHEME_ALIASES = (
    {"flexi", "cap"},
    {"large", "cap"},
    {"mid", "cap"},
    {"elss", "tax", "saver"},
    {"tax", "saver"},
)


def _terms(text: str) -> set[str]:
    return {word.casefold() for word in WORD_PATTERN.findall(text) if word.casefold() not in STOP_WORDS}


def _scheme_matches(query_terms: set[str], scheme: str) -> bool:
    scheme_terms = _terms(scheme)
    return any(alias.issubset(query_terms) and alias.issubset(scheme_terms) for alias in SCHEME_ALIASES)


def rerank_chunks(question: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_terms = _terms(question)
    ranked: list[dict[str, Any]] = []
    seen_content: set[tuple[str, str]] = set()

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        content = str(chunk.get("content", ""))
        source_id = str(metadata.get("source_id", ""))
        dedupe_key = (source_id, " ".join(content.casefold().split()))
        if dedupe_key in seen_content:
            continue
        seen_content.add(dedupe_key)

        distance = float(chunk.get("distance", 2.0))
        vector_score = 1.0 - min(max(distance, 0.0), 2.0) / 2.0
        content_terms = _terms(content)
        lexical_score = len(query_terms & content_terms) / len(query_terms) if query_terms else 0.0
        scheme_match = _scheme_matches(query_terms, str(metadata.get("scheme", "")))
        ranked.append(
            {
                **chunk,
                "relevance_score": 0.60 * vector_score + 0.15 * lexical_score + 0.25 * scheme_match,
            }
        )

    ranked.sort(key=lambda chunk: (-chunk["relevance_score"], chunk.get("distance", 2.0)))
    source_limit = max(2, (top_k + 1) // 2)
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)

    for chunk in ranked:
        source_id = str(chunk.get("metadata", {}).get("source_id", ""))
        if source_counts[source_id] < source_limit:
            selected.append(chunk)
            source_counts[source_id] += 1
        else:
            deferred.append(chunk)

    return (selected + deferred)[:top_k]