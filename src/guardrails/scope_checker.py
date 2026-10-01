from __future__ import annotations

import re


SUPPORTED_SCHEMES = (
    "hdfc flexi cap fund",
    "hdfc large cap fund",
    "hdfc mid cap fund",
    "hdfc elss tax saver fund",
    "hdfc tax saver fund",
    "flexi cap fund",
    "large cap fund",
    "mid cap fund",
    "elss tax saver fund",
    "tax saver fund",
)

SUPPORTED_TOPIC = re.compile(
    r"\b(?:mutual funds?|HDFC\s+(?:AMC|Mutual Fund)|SEBI|AMFI|NAV|net asset value|"
    r"expense ratio|total expense ratio|\bTER\b|exit load|benchmark|riskometer|"
    r"minimum (?:SIP|investment|application)|SIP|systematic investment plan|"
    r"lock[- ]?in|scheme|fund manager|factsheets?|\bSID\b|\bKIM\b|investment objective|"
    r"portfolio|holdings?|asset allocation|IDCW|dividend|redemption|subscription|"
    r"switch(?:ing)?|CKYC|KYC|FATCA|CRS|account statement|capital[- ]gains?|nominee|"
    r"transaction charges|folio|repurchase|cut[- ]off|AUM|risk level)\b",
    re.IGNORECASE,
)

FUND_MENTION = re.compile(r"\b((?:[a-z0-9&'.-]+\s+){0,4}funds?)\b", re.IGNORECASE)
FUND_FILLER_WORDS = {
    "a", "an", "and", "about", "are", "buy", "cap", "does", "expense", "for", "hdfc",
    "how", "in", "is", "of", "on", "please", "ratio", "sell", "should", "the", "what",
    "which", "with",
}
GENERIC_FUND_NAMES = {"fund", "mutual fund", "mutual funds"}


def _normalize_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _has_unsupported_fund_mention(query: str) -> bool:
    for match in FUND_MENTION.finditer(query):
        phrase = _normalize_words(match.group(1))
        if phrase in GENERIC_FUND_NAMES:
            continue
        if any(phrase == scheme or phrase.endswith(f" {scheme}") for scheme in SUPPORTED_SCHEMES):
            continue
        meaningful_words = [word for word in phrase.split() if word not in FUND_FILLER_WORDS and word != "fund"]
        if meaningful_words:
            return True
    return False


def unsupported_reason(query: str) -> str | None:
    if _has_unsupported_fund_mention(query):
        return "The named fund is outside the supported scheme set."
    normalized = _normalize_words(query)
    if any(_normalize_words(scheme) in normalized for scheme in SUPPORTED_SCHEMES):
        return None
    if not SUPPORTED_TOPIC.search(query):
        return "The question is outside the supported mutual fund fact scope."
    return None