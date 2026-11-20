# RAG Evaluation Improvements & Resolution Roadmap

This directory contains comprehensive documentation on the baseline evaluation results, root cause analyses for all observed failures, targeted technical solutions, and a step-by-step execution plan to improve retrieval quality, guardrail precision, and evaluation stability.

---

## Document Index

1. **[01_baseline_and_issue_taxonomy.md](file:///Users/abhimanyusingh/Developer/mf-facts-rag/docs/eval_improvements/01_baseline_and_issue_taxonomy.md)**
   - Detailed quantitative baseline comparison: Run 1 (`run_20261027_135608`) vs. Latest Run (`run_20261027_140012`).
   - Granular breakdown across all 30 test questions in `test_dataset.json`.
   - Comprehensive issue taxonomy classifying guardrail false positives, retrieval collapse, and API rate limits.

2. **[02_guardrail_false_positive_fixes.md](file:///Users/abhimanyusingh/Developer/mf-facts-rag/docs/eval_improvements/02_guardrail_false_positive_fixes.md)**
   - Root causes and solutions for premature refusals in `src/guardrails/scope_checker.py`, `pii_guard.py`, and `answer_policy.py`.
   - Analysis of test cases `q002` (multi-cap vs flexi-cap), `q005` (AUM definition), `q010` (4 HDFC funds SIP), `q015` (ELSS lock-in vs exit load), and `q023` (PAN statement procedure).
   - Solution for pipeline crashes caused by strict sentence count validation in `src/llm/answer_generator.py` (`q013`).

3. **[03_multi_hop_retrieval_enhancements.md](file:///Users/abhimanyusingh/Developer/mf-facts-rag/docs/eval_improvements/03_multi_hop_retrieval_enhancements.md)**
   - Analysis of the multi-hop retrieval bottleneck (contextual recall = 0.20) in `src/retrieval/retrieve.py` and `rerank.py`.
   - Entity-aware sub-query decomposition architecture.
   - Dynamic top-$k$ chunk allocation per scheme.
   - Hybrid dense (vector) + lexical (BM25) search design to retrieve exact historical names and table metrics.

4. **[04_eval_harness_and_rate_limits.md](file:///Users/abhimanyusingh/Developer/mf-facts-rag/docs/eval_improvements/04_eval_harness_and_rate_limits.md)**
   - Analysis of Groq API daily token exhaustion (`429` Rate Limit) on `q024`–`q030`.
   - Strategy to decouple the evaluation judge (DeepEval) from the target generation model.
   - Disk-backed evaluation checkpointing, caching, and exponential backoff implementation.

5. **[05_step_by_step_implementation_plan.md](file:///Users/abhimanyusingh/Developer/mf-facts-rag/docs/eval_improvements/05_step_by_step_implementation_plan.md)**
   - Sequenced 4-phase implementation plan.
   - Verification procedures, test commands, and target metric thresholds for comparing post-fix runs against the baseline.

6. **[06_post_implementation_eval_results.md](file:///Users/abhimanyusingh/Developer/mf-facts-rag/docs/eval_improvements/06_post_implementation_eval_results.md)**
   - Side-by-side comparative analysis of the post-implementation benchmark run (`run_phase4_improved`) against the baseline.
   - Verification of the +275% relative gain in multi-hop recall and 100% answer relevancy.

---

## Baseline vs. Post-Implementation Results Summary

| Metric | Pilot Run (`135608`) | Baseline Run (`140012`) | Post-Implementation (`run_phase4_improved`) | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Answer Relevancy (Mean)** | 0.50 | 0.819 | **1.000 (100.0%)** | **+22.1%** |
| **Contextual Recall (Mean)** | 0.50 | 0.688 | **0.867 (86.7%)** | **+26.1%** |
| **Multi-Hop Recall (Mean)** | N/A | 0.200 | **0.750 (75.0%)** | **+275.0%** |
| **Faithfulness (Mean)** | 0.50 | 0.979 | **0.972 (97.2%)** | -0.7% |
| **Factual Relevancy (Mean)** | N/A | 0.820 | **1.000 (100.0%)** | **+22.0%** |
| **Factual Faithfulness (Mean)** | N/A | 0.960 | **1.000 (100.0%)** | **+4.2%** |
| **Guardrail False Positives** | N/A | 5 blocked (`q002`, `q005`, `q010`, `q015`, `q023`) | **0 blocked (All passed)** | **Resolved** |
