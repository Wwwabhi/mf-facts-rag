# 04 - Evaluation Harness Stability & Rate Limit Elimination

This document analyzes the root causes of the 6 pipeline failures caused by API rate limiting during evaluation, explains the token economics of DeepEval, and details the infrastructure updates needed to make evaluation 100% reliable.

---

## 1. Problem: Groq HTTP 429 Token Rate Limit Failures

In the baseline run (`run_20261027_140012`), 6 out of 30 test cases failed with HTTP 429 errors:
- **`q024`:** Account statement details
- **`q025`:** Historical NAV download
- **`q026`:** Future NAV query
- **`q027`:** 5-year return projection
- **`q029`:** Yesterday redemption query
- **`q030`:** Next month IDCW payout

### Exact Error Message
```
Error code: 429 - {'error': {'message': 'Rate limit reached for model `qwen/qwen3.8-27b` 
in organization `org_...` on tokens per day (TPD): Limit 200000, Used 199468, Requested 2361. 
Please try again in 13m10s.'}}
```

---

## 2. Token Economics: Why the Limit Was Exhausted

A common misconception is that a 30-sample benchmark generates only 30 LLM completions. In reality, modern RAG evaluation using DeepEval executes **multiple LLM judge calls per sample**:

1. **Answer Relevancy Metric:**
   - Extracts all individual factual statements from the generated answer.
   - For each statement, prompts the LLM to rate relevancy to the question.
   - Summarizes into a reasoned score.
   - *Cost per question: ~2–3 LLM calls (~1,200 tokens).*
2. **Faithfulness Metric:**
   - Extracts claims from the generated answer.
   - Evaluates whether each claim is supported by the retrieved context.
   - Returns a structured contradiction breakdown.
   - *Cost per question: ~2–4 LLM calls (~1,800 tokens).*
3. **Contextual Recall Metric:**
   - Compares ground truth statements against the retrieved context to verify evidence presence.
   - *Cost per question: ~2 LLM calls (~1,200 tokens).*
4. **Target Pipeline Generation:**
   - Prompt + context + completion: ~800 tokens.

$$\text{Total Tokens per Question} \approx 4,500 - 5,500 \text{ tokens}$$
$$\text{Total for 30 Questions} \approx 135,000 - 165,000 \text{ tokens}$$

Because both the RAG pipeline **and** DeepEval were configured to call `qwen/qwen3.8-27b` on the exact same Groq account, the 200,000 Tokens-Per-Day (TPD) quota was depleted by question 24.

---

## 3. Targeted Solution Architecture

### A. Decouple Judge Model from Target Model
- **Target Pipeline Generation:** Continues running against Groq `qwen/qwen3.8-27b` (fast inference, facts-only generation).
- **Evaluation Judge:** Routed to an alternative model or provider with independent quotas (e.g., Gemini API, OpenAI, or a secondary evaluation Groq API key).

In `evaluation/deep_eval_judge.py`:
```python
def get_judge_model():
    # Use dedicated judge key or secondary provider to avoid exhausting target generation quota
    judge_api_key = os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("GROQ_API_KEY")
    judge_model = os.getenv("EVAL_JUDGE_MODEL", "qwen/qwen3.8-27b")
    return create_eval_model(model=judge_model, api_key=judge_api_key)
```

### B. Persistent Evaluation Checkpointing & Caching
Currently, if evaluation fails on sample 24, re-running the script re-executes samples 1–23, wasting tokens and time.

Implement disk-backed step checkpointing in `evaluation/evaluator.py`:
```python
checkpoint_file = output_dir / "eval_checkpoint.json"

def run_evaluation_with_cache(dataset, pipeline_fn):
    results = load_checkpoint(checkpoint_file)
    for sample in dataset:
        if sample["id"] in results:
            continue # Skip already evaluated sample
        res = evaluate_sample(sample, pipeline_fn)
        results[sample["id"]] = res
        save_checkpoint(checkpoint_file, results)
    return results
```

### C. Rate Limit Throttling & Exponential Backoff
Implement jittered rate limiting and retry handling in `evaluation/pipeline_runner.py`:
- 1.5-second pacing interval between consecutive evaluations.
- Exponential backoff on HTTP 429 with `Retry-After` header parsing.

---

## 4. Expected Impact on Re-Evaluation

| Metric | Baseline Score | Projected Post-Fix Score |
| :--- | :--- | :--- |
| **Completed Executions** | 23 / 30 | **30 / 30 (100%)** |
| **API 429 Failures** | 6 | **0** |
| **Token Quota Depletion Risk**| High (fails at Q24) | **Zero (decoupled)** |
| **Resumability** | None (starts from Q1) | **Full checkpoint resume** |
