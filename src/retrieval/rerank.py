from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


WORD_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
RISKOMETER_QUERY = re.compile(r"\b(?:risk-o-meter|riskometer|risk level|risk rating|how risky)\b", re.IGNORECASE)
SCHEME_PAGE_QUERY = re.compile(r"\b(?:scheme\s*pages?|min(?:imum)?\s*sip)\b", re.IGNORECASE)
FOUR_FUNDS_PATTERN = re.compile(r"\b(?:four|4|all|each)\s+(?:of\s+the\s+)?(?:hdfc\s+)?funds?\b", re.IGNORECASE)

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

ALL_SCHEMES = [
    "HDFC Flexi Cap Fund",
    "HDFC Large Cap Fund",
    "HDFC Mid Cap Fund",
    "HDFC ELSS - Tax Saver Fund",
]

SCHEME_PATTERNS = [
    ("HDFC Flexi Cap Fund", re.compile(r"\b(?:flexi\s*cap)\b", re.IGNORECASE)),
    ("HDFC Large Cap Fund", re.compile(r"\b(?:large\s*cap)\b", re.IGNORECASE)),
    ("HDFC Mid Cap Fund", re.compile(r"\b(?:mid\s*cap)\b", re.IGNORECASE)),
    ("HDFC ELSS - Tax Saver Fund", re.compile(r"\b(?:elss|tax\s*saver)\b", re.IGNORECASE)),
]

SYNONYM_GROUPS = [
    {"previously", "earlier", "formerly", "old", "former", "prior", "preceding", "called", "known", "named"},
    {"aum", "assets", "management", "crore", "crores", "size"},
    {"benchmark", "index", "nifty", "tri", "returns"},
    {"allocation", "allocations", "allocating", "exposure", "market-cap", "cap", "stocks", "equity", "predominantly", "minimum"},
    {"idcw", "dividend", "payout", "reinvestment", "facility", "facilities"},
    {"lock-in", "lockin", "lock", "statutory", "three", "3", "years"},
    {"sip", "systematic", "investment", "monthly", "minimum", "100", "500"},
    {"turnover", "portfolio", "ptr", "ratio"},
    {"ter", "expense", "ratio", "expenses", "fees", "recurring"},
    {"nav", "net", "asset", "value", "per", "unit"},
]


def extract_mentioned_schemes(question: str) -> list[str]:
    if FOUR_FUNDS_PATTERN.search(question):
        return list(ALL_SCHEMES)
    found = []
    for scheme_name, pattern in SCHEME_PATTERNS:
        if pattern.search(question):
            found.append(scheme_name)
    return found


def _terms(text: str) -> set[str]:
    return {word.casefold() for word in WORD_PATTERN.findall(text) if word.casefold() not in STOP_WORDS}


def _scheme_matches(query_terms: set[str], scheme: str) -> bool:
    scheme_terms = _terms(scheme)
    return any(alias.issubset(query_terms) and alias.issubset(scheme_terms) for alias in SCHEME_ALIASES)


def _synonym_lexical_score(query_terms: set[str], content_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    matched = 0
    for q_term in query_terms:
        if q_term in content_terms:
            matched += 1
            continue
        found_synonym = False
        for group in SYNONYM_GROUPS:
            if q_term in group and (group & content_terms):
                found_synonym = True
                break
        if found_synonym:
            matched += 1
    return matched / len(query_terms)


def _riskometer_document_priority(document_type: str) -> int:
    normalized_type = document_type.strip().casefold()
    if normalized_type == "scheme page":
        return 0
    if normalized_type == "factsheet":
        return 1
    return 2


def rerank_chunks(question: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_terms = _terms(question)
    mentioned_schemes = extract_mentioned_schemes(question)
    is_scheme_page_q = bool(SCHEME_PAGE_QUERY.search(question))
    is_riskometer_q = bool(RISKOMETER_QUERY.search(question))
    
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
        lexical_score = _synonym_lexical_score(query_terms, content_terms)
        scheme_match = _scheme_matches(query_terms, str(metadata.get("scheme", "")))
        
        doc_boost = 0.0
        if is_scheme_page_q and str(metadata.get("document_type")) == "Scheme Page":
            doc_boost += 0.20
            if "min sip" in content.casefold() or "key facts" in content.casefold():
                doc_boost += 0.15

        ranked.append(
            {
                **chunk,
                "relevance_score": 0.50 * vector_score + 0.25 * lexical_score + 0.25 * scheme_match + doc_boost,
            }
        )

    if is_riskometer_q:
        ranked.sort(
            key=lambda chunk: (
                not _scheme_matches(query_terms, str(chunk.get("metadata", {}).get("scheme", ""))),
                _riskometer_document_priority(str(chunk.get("metadata", {}).get("document_type", ""))),
                "riskometer" not in str(chunk.get("content", "")).casefold(),
                -chunk["relevance_score"],
                chunk.get("distance", 2.0),
            )
        )
    else:
        ranked.sort(key=lambda chunk: (-chunk["relevance_score"], chunk.get("distance", 2.0)))

    source_limit = max(2, (top_k + 1) // 2)

    # Balanced per-scheme allocation when multiple schemes are queried
    if len(mentioned_schemes) >= 2:
        per_scheme_target = max(2, top_k // len(mentioned_schemes))
        selected: list[dict[str, Any]] = []
        scheme_counts: dict[str, int] = defaultdict(int)
        remaining: list[dict[str, Any]] = []

        for chunk in ranked:
            scheme = str(chunk.get("metadata", {}).get("scheme", ""))
            if scheme in mentioned_schemes and scheme_counts[scheme] < per_scheme_target:
                selected.append(chunk)
                scheme_counts[scheme] += 1
            else:
                remaining.append(chunk)

        source_counts: dict[str, int] = defaultdict(int)
        for chunk in selected:
            source_id = str(chunk.get("metadata", {}).get("source_id", ""))
            source_counts[source_id] += 1

        for chunk in remaining:
            if len(selected) >= top_k:
                break
            source_id = str(chunk.get("metadata", {}).get("source_id", ""))
            if source_counts[source_id] < source_limit:
                selected.append(chunk)
                source_counts[source_id] += 1

        return selected[:top_k]

    # Standard allocation with source limit
    selected = []
    deferred: list[dict[str, Any]] = []
    source_counts = defaultdict(int)

    for chunk in ranked:
        source_id = str(chunk.get("metadata", {}).get("source_id", ""))
        if source_counts[source_id] < source_limit:
            selected.append(chunk)
            source_counts[source_id] += 1
        else:
            deferred.append(chunk)

    return (selected + deferred)[:top_k]
