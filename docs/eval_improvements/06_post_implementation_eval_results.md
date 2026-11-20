# 06. Post-Implementation Evaluation Results & Delta Analysis

This document provides a comparative analysis of the evaluation results between the initial baseline (`run_20261027_140012`) and the post-improvement run (`run_phase4_improved`), validating the architectural and algorithmic fixes implemented across Phases 1 through 4.

---

## 1. High-Level Quality Metric Comparison

| Evaluation Metric | Baseline (`run_20261027_140012`) | Improved (`run_phase4_improved`) | Absolute Delta | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Answer Relevancy (Mean)** | **0.8190** (81.9%) | **1.0000** (100.0%) | **+0.1810** | **+22.1%** |
| **Answer Relevancy (Median)** | 1.0000 | 1.0000 | +0.0000 | 0.0% |
| **Contextual Recall (Mean)** | **0.6875** (68.8%) | **0.8667** (86.7%) | **+0.1792** | **+26.1%** |
| **Contextual Recall (Median)** | 1.0000 | 1.0000 | +0.0000 | 0.0% |
| **Faithfulness (Mean)** | 0.9792 (97.9%) | 0.9722 (97.2%) | -0.0070 | -0.7% |
| **Multi-Hop Recall (Mean)** | **0.2000** (20.0%) | **0.7500** (75.0%) | **+0.5500** | **+275.0%** |
| **Factual Relevancy (Mean)** | 0.8200 (82.0%) | 1.0000 (100.0%) | +0.1800 | +22.0% |
| **Factual Faithfulness (Mean)** | 0.9600 (96.0%) | 1.0000 (100.0%) | +0.0400 | +4.2% |

---

## 2. Category-by-Category Breakdown

### A. Multi-Hop Queries
The multi-hop queries experienced the most dramatic improvement due to balanced per-scheme chunk partitioning, scheme entity extraction, and synonym lexical expansion:
- **`q006`** (*"What were HDFC Large Cap Fund and HDFC Mid Cap Fund previously called?"*):
  - **Baseline**: Contextual Recall = **0.00**, Answer Relevancy = 0.00.
  - **Improved**: Contextual Recall = **1.00**, Faithfulness = **1.00**, Answer Relevancy = **1.00**.
  - *Fix*: Synonyms (`previously called` $\leftrightarrow$ `formerly known as`) and dual-scheme retrieval ensured both legacy names were captured in the top-5 context.
- **`q007`** (*"How do the minimum market-cap allocations of HDFC Large Cap Fund and HDFC Mid Cap Fund compare?"*):
  - **Baseline**: Contextual Recall = **0.00**, Answer Relevancy = 0.70.
  - **Improved**: Contextual Recall = **1.00**, Answer Relevancy = **1.00**.
  - *Fix*: Dynamic top-k scaling and balanced partitioning prevented HDFC Large Cap chunks from crowding out HDFC Mid Cap chunks.
- **`q010`** (*"According to the supplied scheme pages, can I start a SIP of ₹100 in each of the four HDFC funds?"*):
  - **Baseline**: Blocked as `unsupported` (guardrail false positive), Contextual Recall = N/A.
  - **Improved**: Contextual Recall = **1.00**, Answer Relevancy = **1.00**.
  - *Fix*: Corpus-wide query detection ("each of the four HDFC funds") allowed proper retrieval across all 4 scheme documents.

### B. Guardrail False Positive Resolutions
In the baseline run, legitimate financial queries were erroneously blocked before retrieval or generation:
- **`q002`** (*"How is a flexi cap fund different from a multi cap fund?"*):
  - **Baseline**: Blocked by `_has_unsupported_fund_mention` because `"multi cap"` was not an exact fund name in the supported registry.
  - **Improved**: Allowed via `GENERIC_CATEGORIES` whitelist. Evaluated successfully with **1.00 Relevancy, 1.00 Faithfulness, 1.00 Recall**.
- **`q005`** (*"What does a mutual fund's AUM represent?"*):
  - **Baseline**: Blocked by `_has_unsupported_fund_mention` due to the word `"fund's"`.
  - **Improved**: Allowed via `FUND_FILLER_WORDS` token stripping and generic financial term allowance. Evaluated with **1.00 Relevancy, 1.00 Faithfulness, 1.00 Recall**.

---

## 3. Harness Hardening & Checkpointing Validation

1. **State Persistence**: Disk checkpointing (`eval_checkpoint.json`) saved progress after each query. When the run was interrupted by external API daily token quotas, resuming via `--resume` cleanly reloaded 18 successful evaluations in under 1 second without wasting judge tokens or pipeline compute.
2. **Exponential Backoff**: Dynamic retry-after regex extraction successfully parsed Groq's wait intervals, avoiding aggressive hammer-and-fail loops.
3. **Decoupled Architecture**: Supported distinct evaluation judge keys and models (`--judge-api-key`, `--judge-model`) to isolate target RAG generation quotas from DeepEval scoring quotas.

---

## 4. Summary of Improvements by Issue Taxonomy

| Issue ID | Problem Area | Resolution | Status |
| :--- | :--- | :--- | :--- |
| **ISS-01** | Multi-hop single-scheme chunk starvation | Balanced scheme chunk partitioning + dynamic top-k scaling | **VERIFIED (Recall +275%)** |
| **ISS-02** | Synonym & terminology mismatch | Added `SYNONYM_GROUPS` with lexical scoring | **VERIFIED (`q006` Recall 0 $\to$ 1.0)** |
| **ISS-03** | Guardrail false positives on generic categories | Whitelisted SEBI categories (`multi cap`, `index`, etc.) | **VERIFIED (`q002` passed)** |
| **ISS-04** | Guardrail false positives on generic AUM | Token stripping on filler words (`mutual`, `four`, `each`) | **VERIFIED (`q005` passed)** |
| **ISS-05** | Fragile sentence length enforcement | Replaced hard `ValueError` with soft 3-sentence trimmer | **VERIFIED (0 crashes)** |
| **ISS-06** | Monolithic evaluation harness crashes | Disk checkpointing + resume + decoupled judge API | **VERIFIED (Fast resume)** |
