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
    "a", "an", "and", "about", "all", "any", "are", "both", "buy", "cap", "does", "each",
    "ended", "equity", "expense", "for", "four", "growth", "hdfc", "how", "in", "is",
    "mutual", "of", "on", "open", "other", "our", "please", "ratio", "sell", "should",
    "the", "these", "this", "what", "which", "with",
}
GENERIC_FUND_NAMES = {"fund", "mutual fund", "mutual funds"}

GENERIC_CATEGORIES = {
    "multi cap fund",
    "multi cap funds",
    "multicap fund",
    "multicap funds",
    "large cap fund",
    "large cap funds",
    "mid cap fund",
    "mid cap funds",
    "small cap fund",
    "small cap funds",
    "flexi cap fund",
    "flexi cap funds",
    "elss fund",
    "elss funds",
    "elss tax saver fund",
    "elss tax saver funds",
    "tax saver fund",
    "tax saver funds",
    "index fund",
    "index funds",
    "equity fund",
    "equity funds",
    "debt fund",
    "debt funds",
    "hybrid fund",
    "hybrid funds",
    "liquid fund",
    "liquid funds",
    "open ended fund",
    "open ended funds",
}

KNOWN_AMCS = {
    "hdfc", "sbi", "icici", "axis", "kotak", "nippon", "uti", "aditya birla",
    "tata", "mirae", "dsp", "franklin", "motilal", "parag parikh", "quant",
    "bandhan", "canara robeco", "hsbc", "invesco", "sundaram",
}


def _normalize_words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _has_unsupported_fund_mention(query: str) -> bool:
    for match in FUND_MENTION.finditer(query):
        phrase = _normalize_words(match.group(1))
        if phrase in GENERIC_FUND_NAMES:
            continue
        if any(phrase == scheme or phrase.endswith(f" {scheme}") for scheme in SUPPORTED_SCHEMES):
            continue
        # If it's a generic SEBI category without an AMC prefix, it's not an unsupported scheme
        words = phrase.split()
        has_amc = any(word in KNOWN_AMCS for word in words)
        if not has_amc and any(phrase == cat or phrase.endswith(f" {cat}") for cat in GENERIC_CATEGORIES):
            continue
        scheme_specific_words = [
            word for word in words
            if word not in FUND_FILLER_WORDS and word not in KNOWN_AMCS and word not in ("fund", "funds")
        ]
        if scheme_specific_words:
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