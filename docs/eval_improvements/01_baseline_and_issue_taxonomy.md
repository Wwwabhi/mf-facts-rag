# 01 - Baseline Performance & Issue Taxonomy

This document records the exact baseline performance of the HDFC Mutual Fund Facts-Only RAG system, comparing the initial smoke test with the full baseline evaluation run, and categorizes every observed failure.

---

## 1. Run Metadata & Configuration

| Parameter | Run 1 (Smoke Test) | Run 2 (Full Baseline Run) |
| :--- | :--- | :--- |
| **Directory** | `evaluation/outputs/run_20261027_135608/` | `evaluation/outputs/run_20261027_140012/` |
| **Execution Timestamp** | 2026-10-27T08:27:01 UTC | 2026-10-27T09:00:07 UTC |
| **Target Generation Model** | `qwen/qwen3.8-27b` (via Groq API) | `qwen/qwen3.8-27b` (via Groq API) |
| **Evaluation Judge Model** | DeepEval with Groq `qwen/qwen3.8-27b` | DeepEval with Groq `qwen/qwen3.8-27b` |
| **Test Dataset** | `test_dataset.json` (first 2 samples) | `test_dataset.json` (all 30 samples) |
| **Total Prompt Tokens** | 794 | 24,309 |
| **Total Completion Tokens** | 49 | 780 |
| **Total Estimated Cost** | \$0.000169 USD | \$0.005018 USD |

---

## 2. Quantitative Metric Comparison

### Overall Quality Metrics

| Metric | Run 1 Mean (Median) | Run 2 Mean (Median) | Run 2 Min / Max | Run 2 P90 / P95 |
| :--- | :--- | :--- | :--- | :--- |
| **Answer Relevancy** | 0.5000 (0.5000) | **0.8190 (1.0000)** | 0.0000 / 1.0000 | 1.0000 / 1.0000 |
| **Faithfulness** | 0.5000 (0.5000) | **0.9792 (1.0000)** | 0.6667 / 1.0000 | 1.0000 / 1.0000 |
| **Contextual Recall** | 0.5000 (0.5000) | **0.6875 (1.0000)** | 0.0000 / 1.0000 | 1.0000 / 1.0000 |

*Note: Unanswerable/out-of-scope questions that appropriately returned refusal templates without retrieving evidence are marked as N/A for Faithfulness/Recall, matching evaluation standards.*

### Latency Profiles (ms)

| Latency Stage | Run 1 Mean (Median) | Run 2 Mean (Median) | Run 2 P90 / P95 |
| :--- | :--- | :--- | :--- |
| **Retrieval Latency** | 18,896.9 ms (18,896.9 ms) | **1,094.9 ms (466.7 ms)** | 851.5 ms / 1,083.6 ms |
| **Generation Latency** | 476.9 ms (476.9 ms) | **7,003.5 ms (6,680.4 ms)** | 14,665.6 ms / 18,174.2 ms |
| **Total Turnaround** | 19,393.9 ms (19,393.9 ms) | **9,741.5 ms (8,132.9 ms)** | 18,930.2 ms / 20,724.1 ms |

---

## 3. Breakdown by Query Category (Run 2)

| Category | Total Qs | Executed | Failed | Mean Relevancy | Mean Faithfulness | Mean Recall | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Factual** | 12 | 11 | 1 (`q024`) | 0.8182 | 0.9630 | 0.8750 | 9,872 ms |
| **Procedural** | 6 | 4 | 2 (`q013`, `q025`) | 1.0000 | 1.0000 | 1.0000 | 17,151 ms |
| **Multi-Hop** | 7 | 7 | 0 | 0.7429 | 1.0000 | **0.2000** | 5,586 ms |
| **Unanswerable**| 5 | 1 | 4 (`q026`, `q027`, `q029`, `q030`) | N/A | N/A | N/A | 6,353 ms |

---

## 4. Question-Level Baseline Result Matrix

| ID | Cat | Question Summary | Run 2 Status | Relevancy | Faithfulness | Recall | Bottleneck / Issue |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `q001` | Factual | Flexi Cap investment universe | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q002` | Factual | Flexi cap vs multi cap difference | SUCCESS | **0.0000** | N/A | N/A | **Scope checker blocked "multi cap"** |
| `q003` | Factual | Flexi Cap top 3 sectors | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q004` | Factual | TER definition & NAV accounting | SUCCESS | 1.0000 | 1.0000 | **0.0000** | Document chunks missed daily NAV accounting |
| `q005` | Factual | What does AUM represent | SUCCESS | **0.0000** | N/A | N/A | **Scope checker blocked "mutual"** |
| `q006` | Multi-hop | Former names of Large & Mid Cap | SUCCESS | 1.0000 | 1.0000 | **0.0000** | **Retrieval starvation for 2 schemes** |
| `q007` | Multi-hop | Large vs Mid Cap allocation limits | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q008` | Factual | Mid Cap fund manager & tenure | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q009` | Factual | Large Cap exit load structure | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q010` | Multi-hop | ₹100 SIP in all 4 HDFC funds | SUCCESS | **0.0000** | N/A | N/A | **Scope checker blocked "four"** |
| `q011` | Procedural | How to download capital gains statement | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q012` | Procedural | CKYC requirements for new investor | SUCCESS | 1.0000 | N/A | 1.0000 | Refusal correctly applied |
| `q013` | Procedural | 2:30pm vs 3:30pm redemption cut-off | **FAILED** | N/A | N/A | N/A | **Crashed on >3 sentence limit** |
| `q014` | Procedural | How to change bank mandate | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q015` | Multi-hop | ELSS nil exit load vs 1-yr redemption | SUCCESS | **0.2000** | N/A | N/A | **Misclassified as account-specific** |
| `q016` | Factual | ELSS statutory lock-in period | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q017` | Factual | HDFC Mid Cap riskometer | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q018` | Factual | Direct vs Regular plan difference | SUCCESS | 1.0000 | 1.0000 | 1.0000 | Perfect |
| `q019` | Multi-hop | Flexi Cap vs ELSS IDCW facilities | SUCCESS | 1.0000 | 1.0000 | **0.0000** | **Retrieval starvation for 2 schemes** |
| `q020` | Multi-hop | Flexi Cap vs Large Cap benchmarks | SUCCESS | 1.0000 | 1.0000 | **0.0000** | **Retrieval starvation for 2 schemes** |
| `q021` | Multi-hop | Flexi Cap vs Mid Cap July 2026 AUM | SUCCESS | 1.0000 | 1.0000 | **0.0000** | **Retrieval starvation for 2 schemes** |
| `q022` | Factual | Large Cap low risk verification | SUCCESS | 1.0000 | **0.6667** | N/A | Partial hallucination check discrepancy |
| `q023` | Procedural | Statement covering folios linked to PAN | SUCCESS | N/A | N/A | N/A | **Falsely triggered PII check** |
| `q024` | Factual | Account statement availability & details | **FAILED** | N/A | N/A | N/A | **Groq 429 Token Rate Limit** |
| `q025` | Procedural | AMFI historical NAV download steps | **FAILED** | N/A | N/A | N/A | **Groq 429 Token Rate Limit** |
| `q026` | Unanswer | Flexi Cap NAV on future date | **FAILED** | N/A | N/A | N/A | **Groq 429 Token Rate Limit** |
| `q027` | Unanswer | Mid Cap value 5 years from now | **FAILED** | N/A | N/A | N/A | **Groq 429 Token Rate Limit** |
| `q028` | Unanswer | How many units do I personally own | SUCCESS | N/A | N/A | N/A | Properly refused account info |
| `q029` | Unanswer | Yesterday redemption bank credit amount | **FAILED** | N/A | N/A | N/A | **Groq 429 Token Rate Limit** |
| `q030` | Unanswer | Exact IDCW payout next month | **FAILED** | N/A | N/A | N/A | **Groq 429 Token Rate Limit** |

---

## 5. Primary Root Cause Taxonomy

1. **Guardrail False Positives (Affects 5 questions: `q002`, `q005`, `q010`, `q015`, `q023`)**
   - Naive regex tokenization in `scope_checker.py` and `pii_guard.py` triggers hard-coded refusal templates for valid questions mentioning generic terminology.
2. **Multi-Hop Retrieval Starvation (Affects 4 questions: `q006`, `q019`, `q020`, `q021`)**
   - Single vector search with $top\_k = 5$ cannot represent multiple schemes simultaneously. The dominant scheme starves other schemes, yielding 0.00 contextual recall on comparative queries.
3. **Strict Validation Crash (Affects 1 question: `q013`)**
   - `answer_generator.py` raises `ValueError` if the LLM produces 4 sentences instead of gracefully formatting or truncating.
4. **Judge & Model Rate Limits (Affects 6 questions: `q024`–`q030`)**
   - Simultaneous generation and multi-metric DeepEval evaluation against a single free-tier Groq daily quota (200k tokens) triggers HTTP 429 errors.
