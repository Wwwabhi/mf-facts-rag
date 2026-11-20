# 02 - Guardrail & Scope Checker False Positive Fixes

This document details the exact flaws identified in the guardrail and policy modules (`scope_checker.py`, `pii_guard.py`, `answer_policy.py`, and `answer_generator.py`), the failing test cases, and the code modifications required to resolve them.

---

## 1. Issue: Flawed Scope Checking in `scope_checker.py`

### Mechanism of Failure
In `src/guardrails/scope_checker.py`:
```python
FUND_MENTION = re.compile(r"\b((?:[a-z0-9&'.-]+\s+){0,4}funds?)\b", re.IGNORECASE)
FUND_FILLER_WORDS = {
    "a", "an", "and", "about", "are", "buy", "cap", "does", "expense", "for", "hdfc",
    "how", "in", "is", "of", "on", "please", "ratio", "sell", "should", "the", "what",
    "which", "with",
}
GENERIC_FUND_NAMES = {"fund", "mutual fund", "mutual funds"}

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
```

### Affected Test Cases & Exact Triggers

1. **`q002`: *"How is a flexi cap fund different from a multi cap fund?"***
   - Regex matches: `"multi cap fund"`.
   - `meaningful_words`: `["multi"]` (since `"multi"` is not in `FUND_FILLER_WORDS`).
   - `phrase in GENERIC_FUND_NAMES`: False (`"multi cap fund"` is not in the set).
   - Result: Flagged as an unsupported competitor fund, returning refusal: *"The named fund is outside the supported scheme set."*
   - Reality: Multi-cap is an official SEBI mutual fund category, essential for explaining what a flexi-cap fund is.

2. **`q005`: *"What does a mutual fund's AUM represent?"***
   - Regex matches: `"what does a mutual fund"`.
   - `FUND_FILLER_WORDS` **does not contain the word `"mutual"`**!
   - `meaningful_words`: `["mutual"]`.
   - Result: Returns True, falsely flagging a standard industry definition query as an unsupported fund.

3. **`q010`: *"According to the supplied scheme pages, can I start a ₹100 SIP in each of the four HDFC funds?"***
   - Regex matches: `"the four hdfc funds"`.
   - `meaningful_words`: `["four"]` (`"four"` is missing from filler words).
   - Result: Returns True, blocking a direct question about the 4 supported schemes.

### Targeted Solution

1. Add generic categories, numbers, and mutual fund vocabulary to allowed sets:
   - Expand `FUND_FILLER_WORDS` with `{"mutual", "four", "all", "each", "these", "any", "other", "open", "ended", "equity"}`.
   - Add SEBI fund categories to a recognized categories list rather than treating them as competitor brand names: `{"multi cap", "large cap", "mid cap", "small cap", "flexi cap", "elss", "index fund", "equity fund"}`.
2. Only flag an unsupported fund if it explicitly names a competitor entity or brand (e.g., `sbi`, `icici`, `axis`, `nippon`, `mirae`, `tata`, `uti`, `kotak`, `parag parikh`).

---

## 2. Issue: False PII & Account-Specific Triggers

### Affected Test Cases & Exact Triggers

1. **`q023`: *"I need one account statement covering all my folios linked to my PAN. Which statement should I choose, and is there a fee?"***
   - In `pii_guard.py`:
     ```python
     if re.search(r"\bPAN\b", text): return "PAN"
     ```
   - The user asked about the **process** of requesting a PAN-level Consolidated Account Statement (CAS). They did not provide an actual 10-character alphanumeric PAN number (`[A-Z]{5}[0-9]{4}[A-Z]`).
   - Triggered: *"Please do not share PAN, Aadhaar, OTP... personal information."*
   - Fix: Only block when an **actual PAN credential pattern** is present:
     ```python
     PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
     ```
     Allow procedural terms like `"linked to my PAN"` or `"PAN statement"`.

2. **`q015`: *"HDFC ELSS - Tax Saver Fund has no exit load. Does that mean I can normally redeem my investment after one year?"***
   - In `answer_policy.py`:
     Triggered `is_account_specific_request()` because it matched `"redeem my investment"`.
   - Result: Returned canned disclaimer *"I cannot access individual accounts, folios, balances, or transactions"*, scoring 0.20 Relevancy.
   - Fix: Questions asking about general statutory rules (e.g. 3-year lock-in, minimum SIP, exit loads) using hypothetical first-person phrasing (*"can I normally redeem after one year"*) must be evaluated for scheme rules before defaulting to account-specific refusals.

---

## 3. Issue: Hard Failure on Sentence Limit in `answer_generator.py`

### Mechanism of Failure (`q013`)
```python
# src/llm/answer_generator.py
if len(SENTENCE_BOUNDARY.split(answer_text)) > 3:
    raise ValueError("Groq answer exceeded the three-sentence limit")
```
When answering `q013` (*"For eligible HDFC ELSS units whose lock-in has ended, which NAV applies if my valid redemption request is received at 2:30 p.m. versus 3:30 p.m. on a business day?"*), the LLM generated a correct 4-sentence explanation. 

Instead of formatting or trimming the answer, the code raised an unhandled `ValueError`, causing the entire pipeline run to mark `q013` as `FAILED`.

### Targeted Solution
Replace the exception with an automated soft trimmer and response compressor:
```python
sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(answer_text) if s.strip()]
if len(sentences) > 3:
    # Take first 3 sentences to guarantee compliance without crashing
    answer_text = " ".join(sentences[:3])
```
Also add explicit few-shot instruction in the system prompt:
```
"Constraint: You MUST strictly write between 1 and 3 sentences. Never write a fourth sentence."
```

---

## 4. Expected Impact on Re-Evaluation

| Metric | Baseline Score | Projected Post-Fix Score | Why |
| :--- | :--- | :--- | :--- |
| **Pipeline Failures** | 7 failed | Reduced by 1 (`q013` passes) | Sentence exception removed |
| **`q002` Relevancy** | 0.00 | **1.00** | Factual definition of flexi vs multi cap answered |
| **`q005` Relevancy** | 0.00 | **1.00** | AUM definition answered |
| **`q010` Relevancy** | 0.00 | **1.00** | Four HDFC funds SIP rules answered |
| **`q015` Relevancy** | 0.20 | **1.00** | 3-year lock-in vs exit load explained |
| **`q023` Procedural Pass**| Refused | **1.00** | CAS download guidance provided |
| **Overall Relevancy (Mean)** | 0.819 | **~0.950 (+13.1%)** | Resolves all 5 false refusals |
