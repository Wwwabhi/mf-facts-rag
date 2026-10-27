# 🧪 Mutual Fund Facts RAG — Evaluation Pipeline

Repeatable, production-grade evaluation pipeline for the **Mutual Fund Facts** RAG chatbot, built for execution in **Google Colab** and local environments.

---

## 🏗️ Architecture & Codebase Inspection Findings

Before creating the pipeline, we inspected the codebase and identified:
1. **Question Ingestion**: Questions enter through `app/main.py` via `st.text_input` in Streamlit or programmatic calls to `src.retrieval.query_pipeline.answer_question(question, top_k=5)`.
2. **Pre-Retrieval Guardrails**: Evaluated deterministically in `src/guardrails/answer_policy.py`:
   - Checks empty input (`unsupported`)
   - Scans for PII via `detect_sensitive_data` (`privacy`)
   - Blocks account/folio queries via `is_account_specific_request` (`account_specific`)
   - Blocks advice / recommendations via `ADVICE_PATTERNS` (`investment_advice`)
   - Blocks returns / performance questions via `PERFORMANCE_PATTERNS` (`performance_returns`)
   - Enforces mutual fund topic scope via `unsupported_reason` (`unsupported`).
   - If blocked, canned refusal templates are returned immediately without querying ChromaDB or Groq.
3. **Retrieval & Vector Similarity Search**:
   - Embedding Model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, normalized).
   - Vector Store: Persistent ChromaDB collection `"mf_facts_rag"` at `vectorstore/chroma_db`.
   - Top-K Search: Cosine distance search returning `min(count, max(50, top_k * 3))` initial candidates.
   - Scheme-Aware Reranking (`src/retrieval/rerank.py`):
     - Weighted composite score: `0.60 * vector_score + 0.15 * lexical_score + 0.25 * scheme_match`.
     - Riskometer-specific boosting for scheme pages when queried.
     - Per-source diversity cap (`source_limit = max(2, (top_k + 1) // 2)`).
4. **Context Formatting & LLM Prompting**:
   - `src/llm/answer_generator.py` checks that retrieved evidence has official URLs (`hdfcfund.com`, `sebi.gov.in`, `amfiindia.com`).
   - Context is injected into the prompt:
     ```
     [Evidence 1]
     Scheme: ...
     Document type: ...
     Section: ...
     Source date: ...
     Content: ...
     ```
   - System prompt instructs Groq to answer using only evidence in max 3 sentences.
   - Answer provenance is formatted and checked against chunk metadata (`format_answer` and `parse_formatted_answer`).
5. **Groq Model & API**:
   - Default configured model: `qwen/qwen3.8-27b` (configured via `GROQ_MODEL` in `.env`).
   - API client: Official `groq.Groq` SDK with zero temperature and `max_tokens=350`.
6. **Live Deployment & Endpoint Reality**:
   - Deployment URL: `https://mf-facts-rag-ls8ctphagmcvyenzujm8mv.streamlit.app/`
   - **Crucial Finding**: Streamlit Community Cloud deployments communicate over internal WebSockets and **do not expose an HTTP/REST API endpoint**.
   - The evaluation pipeline supports:
     - **Mode 1 (`direct_pipeline`)**: Executes the exact production Python RAG pipeline in-process, capturing granular latencies (retrieval vs generation vs total), tokens, and chunk provenance.
     - **Mode 2 (`http_api`)**: Calls an external REST endpoint (e.g. `evaluation/api_server.py` or a deployed FastAPI microservice) with configurable request and response mappings.

---

## 📊 Evaluation Metrics (DeepEval)

- **Answer Relevancy (`AnswerRelevancyMetric`)**: Measures whether the answer directly addresses the user question without irrelevant filler.
- **Faithfulness (`FaithfulnessMetric`)**: Measures whether the generated answer is faithful to and grounded in the retrieved chunks.
- **Contextual Recall (`ContextualRecallMetric`)**: Measures whether the retrieved context contains all factual statements present in the ground truth.

### Groq as Judge Compatibility & Limitations
1. **Reasoning Tokens**: `qwen/qwen3.8-27b` emits reasoning thoughts. Our custom `GroqJudge` sanitizes `<think>...</think>` tags and extracts clean JSON matching the Pydantic schema requested by DeepEval.
2. **Rate Limits (HTTP 429)**: Groq free/on-demand tier has strict RPM/TPM limits. Our `GroqJudge` includes exponential backoff retry and configurable delay (`--delay 0.5`).
3. **Alternative Providers**: You can easily switch judge provider to OpenAI (`--judge-provider openai --judge-model gpt-4o-mini`) simply by configuring `OPENAI_API_KEY`.
4. **Unanswerable / Refusal Handling**:
   - For unanswerable questions or guardrail refusals with empty retrieval context or empty reference contexts, Faithfulness and Contextual Recall are recorded as **`N/A`** (not `0.0`), preventing artificial penalization of the RAG system.

---

## 💰 Performance & Cost Tracking

- Latencies: Wall-clock timing for `retrieval_latency_ms`, `generation_latency_ms`, and `total_latency_ms`.
- Token Usage: Captured directly from `completion.usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`).
- Pricing: Calculated using explicit published pricing (never invented).
  - `qwen/qwen3.8-27b`: $0.20 / 1M input tokens, $0.20 / 1M output tokens.
  - `llama-3.3-70b-versatile`: $0.59 / 1M input, $0.79 / 1M output.

---

---

## 🚀 Running Locally (Single Entry Point)

The recommended single entry point for running the evaluation suite is `evaluation.run_all`:

```bash
# 1. Install evaluation dependencies
pip install -r evaluation/requirements-eval.txt

# 2. Run the complete evaluation workflow
python -m evaluation.run_all
```

This single command automatically:
1. Validates the test dataset (`test_dataset.json`).
2. Executes all test cases through the RAG pipeline.
3. Scores Answer Relevancy, Faithfulness, Contextual Recall with DeepEval.
4. Generates a standalone, human-readable `report.html` interactive dashboard.
5. Saves CSV, JSON summary, and PNG visualizations into a timestamped directory (e.g., `evaluation/outputs/run_YYYYMMDD_HHMMSS/`).
6. Prints a concise terminal summary and the exact report path.

### CLI Flags & Customization

```bash
# Run a quick smoke test on 5 samples
python -m evaluation.run_all --limit 5

# Use OpenAI as DeepEval judge
python -m evaluation.run_all --judge-provider openai --judge-model gpt-4o-mini

# Evaluate an external live HTTP API
python -m evaluation.run_all --mode http_api --api-url http://localhost:8000/api/chat

# Customize delay to respect strict rate limits
python -m evaluation.run_all --delay 1.0
```

> [!TIP]
> The legacy evaluation script `python -m evaluation.run_eval` is also preserved and fully functional. Both commands generate the full artifact suite including `report.html`.

---

## ☁️ Running in Google Colab

Open [`evaluation/colab_evaluation_notebook.ipynb`](colab_evaluation_notebook.ipynb) in Google Colab:

1. Add your `GROQ_API_KEY` to **Colab Secrets** (Left sidebar 🔑 key icon).
2. Run Step 1 (`!pip install -r requirements-eval.txt`).
3. Run Step 2 (mount Drive or clone repo).
4. Run Step 5 to execute the evaluation.
5. Step 6 displays the interactive results table and visualizations.
6. Step 7 downloads `rag_evaluation_outputs.zip` containing `evaluation_results.csv`, `evaluation_summary.json`, and `evaluation_visualizations.png`.
