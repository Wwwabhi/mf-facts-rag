from __future__ import annotations

import re


SENSITIVE_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    ("aadhaar", re.compile(r"(?<!\d)(?:\d[ -]?){11}\d(?!\d)")),
    ("phone", re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)")),
    ("otp", re.compile(r"\b(?:otp|one[- ]time password)\s*(?:is|:|-)?\s*\d{4,8}\b", re.IGNORECASE)),
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)),
    ("card", re.compile(r"\b(?:card\s*)?(?:number|no\.?|cvv)\s*[:#-]?\s*(?:\d[ -]?){3,19}\b", re.IGNORECASE)),
    (
        "account_number",
        re.compile(r"\b(?:account|acct|folio)\s*(?:number|no\.?|#)\s*[:#=-]?\s*\d[\d -]{4,20}\b", re.IGNORECASE),
    ),
    ("personal_identifier", re.compile(r"\bmy\s+(?:PAN|Aadhaar|OTP|phone number|email|bank details?)\b", re.IGNORECASE)),
)

PERSONAL_ACCOUNT_PATTERNS = (
    re.compile(
        r"\b(?:my|our|mine)\s+(?:mutual fund\s+)?(?:account|folio|balance|units?|holdings?|transactions?|portfolio|investments?|investment value|redemption|SIPs?|NAV|statement)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:check|show|retrieve|look up|track|find|tell me)\b.{0,50}\b(?:my|our)\b.{0,50}\b(?:account|folio|balance|units?|holdings?|transactions?|portfolio|redemption|SIPs?|NAV)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:account|folio)\s+(?:balance|holdings?|transactions?|status|details)\b", re.IGNORECASE),
    re.compile(r"\b(?:change|update|link|register)\b.{0,50}\b(?:my\s+)?(?:bank|account|folio)\s+(?:details|number|information)\b", re.IGNORECASE),
)

GENERAL_STATEMENT_HELP = re.compile(
    r"^\s*how\s+(?:can|do|would)\s+i\s+(?:download|access|get|obtain)\b.{0,100}\b(?:account|consolidated|capital gains)\s+statement\b",
    re.IGNORECASE,
)


def detect_sensitive_data(query: str) -> str | None:
    for category, pattern in SENSITIVE_VALUE_PATTERNS:
        if pattern.search(query):
            return category
    return None


def is_account_specific_request(query: str) -> bool:
    if GENERAL_STATEMENT_HELP.search(query):
        return False
    return any(pattern.search(query) for pattern in PERSONAL_ACCOUNT_PATTERNS)