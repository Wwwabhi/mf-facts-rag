from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from src.guardrails.pii_guard import detect_sensitive_data, is_account_specific_request
from src.guardrails.refusal_templates import REFUSAL_MESSAGES
from src.guardrails.scope_checker import unsupported_reason


ADVICE_PATTERNS = (
    re.compile(r"\bshould\s+i\b.{0,60}\b(?:buy|sell|invest|switch|redeem|choose|select|hold|exit)\b", re.IGNORECASE),
    re.compile(r"\b(?:recommend(?:ation)?|suggest(?:ion)?)\b", re.IGNORECASE),
    re.compile(r"\bwhich\b.{0,60}\b(?:fund|scheme)\b.{0,40}\b(?:should|buy|invest|choose|pick)\b", re.IGNORECASE),
    re.compile(r"\b(?:best|top)\s+(?:mutual\s+)?fund\b", re.IGNORECASE),
    re.compile(r"\b(?:investment|portfolio)\s+advice\b", re.IGNORECASE),
    re.compile(r"\b(?:good|bad|right|suitable)\b.{0,35}\b(?:investment|for me)\b", re.IGNORECASE),
)

PERFORMANCE_PATTERNS = (
    re.compile(
        r"\b(?:performance|returns?|CAGR|XIRR|annuali[sz]ed returns?|absolute returns?|"
        r"historical returns?|past returns?|rate of return|return on investment|outperform|underperform)\b",
        re.IGNORECASE,
    ),
)

URL_PATTERN = re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)
OFFICIAL_HOSTS = ("hdfcfund.com", "sebi.gov.in", "amfiindia.com")


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    category: str
    reason: str
    message: str | None = None


def _blocked(category: str, reason: str) -> GuardrailDecision:
    return GuardrailDecision(False, category, reason, REFUSAL_MESSAGES[category])


def classify_question(question: str) -> GuardrailDecision:
    if not isinstance(question, str) or not question.strip():
        return _blocked("unsupported", "The question is empty or invalid.")

    sensitive_category = detect_sensitive_data(question)
    if sensitive_category:
        return _blocked("privacy", f"Sensitive data detected: {sensitive_category}.")
    if is_account_specific_request(question):
        return _blocked("account_specific", "The question requests personal account or folio information.")
    if any(pattern.search(question) for pattern in ADVICE_PATTERNS):
        return _blocked("investment_advice", "The question requests investment advice or a buy/sell recommendation.")
    if any(pattern.search(question) for pattern in PERFORMANCE_PATTERNS):
        return _blocked("performance_returns", "Performance and returns questions are outside the supported answer policy.")

    reason = unsupported_reason(question)
    if reason:
        return _blocked("unsupported", reason)
    return GuardrailDecision(True, "factual", "The question is within the supported factual scope.")


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip().rstrip(".,;"))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


def _is_official_url(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in OFFICIAL_HOSTS)


def has_approved_evidence(evidence_urls: list[str]) -> bool:
    return any(url and _is_official_url(url) for url in evidence_urls)


def answer_has_supported_citations(answer: str, evidence_urls: list[str]) -> bool:
    allowed_urls = {
        _canonical_url(url)
        for url in evidence_urls
        if url and _is_official_url(url)
    }
    citations = {_canonical_url(url) for url in URL_PATTERN.findall(answer or "")}
    return bool(allowed_urls and citations) and citations.issubset(allowed_urls)