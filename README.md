# Mutual Fund Facts

A facts-only mutual-fund FAQ assistant in a Groww product context. This repository currently covers selected HDFC Asset Management Company schemes; it is not an integration with Groww accounts or services.

Live Demo - https://mf-facts-rag-ls8ctphagmcvyenzujm8mv.streamlit.app/

## Coverage

The corpus contains **22 approved source records**: 20 HDFC-hosted pages/documents, one SEBI SID registry entry, and one AMFI NAV source. Sources are limited to official HDFC AMC, SEBI, and AMFI URLs. The registry is [data/mf_rag_sources.csv](data/mf_rag_sources.csv).

The four supported schemes are HDFC Flexi Cap Fund, HDFC Large Cap Fund, HDFC Mid Cap Fund, and HDFC ELSS - Tax Saver Fund. The registry also includes general investor-service and regulator/industry sources.

## Architecture

```text
Official source registry
  -> raw HTML/PDF and processed text (requests, BeautifulSoup, Playwright fallback, Docling)
  -> section-aware chunks (500-token target, 650-token maximum, 75-token prose overlap)
  -> normalized sentence-transformers/all-MiniLM-L6-v2 embeddings (384 dimensions)
  -> persistent ChromaDB at vectorstore/chroma_db
  -> query guardrails -> top-k Chroma retrieval and scheme-aware reranking
  -> evidence supplied to Groq -> citation/source-date validation -> Streamlit answer
```

The current chunk snapshot contains 1,602 chunks. Chunks retain source, scheme, document, section, and date metadata. Factual responses include one official source URL and a “Last updated from sources” date.

## Tech Stack

- Python 3.12+
- Docling for PDF parsing; Requests and BeautifulSoup for HTML; Playwright as an HTML browser fallback
- `tiktoken` (`cl100k_base`) for chunk sizing
- Sentence Transformers with `sentence-transformers/all-MiniLM-L6-v2` for document and query embeddings
- ChromaDB persistent vector store with cosine distance
- Groq chat completions using the configured `GROQ_MODEL` (the example defaults to `qwen/qwen3.8-27b`)
- Streamlit UI; `python-dotenv` for project-root environment loading

## Setup

From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Create `.env` if it does not already exist, then set `GROQ_API_KEY` and `GROQ_MODEL`:

```bash
if [ ! -f .env ]; then cp .env.example .env; fi
```

Keep the API key in `.env`; `.gitignore` excludes it. Do not commit credentials.

`CHROMA_DB_PATH` is optional and defaults to `vectorstore/chroma_db`; set it in `.env` to use another persistent location. On first Streamlit startup, the app checks the configured collection and builds it from `data/chunks/chunks.jsonl` when missing or empty. This check/build is cached for the Streamlit process.

To rebuild chunks and embeddings from processed documents manually:

```bash
python -m src.chunking.chunker
python -m src.vectorstore.index_documents
python -m src.vectorstore.validate_index
```

To refetch and reprocess the registry sources before rebuilding, run `python -m src.data.pipeline`. That optional ingestion step needs the source-extraction dependencies (`docling`, `beautifulsoup4`, and, for the browser fallback, Playwright plus its Chromium runtime).

## Run

Start the UI:

```bash
streamlit run app/main.py --server.port 8501
```

Ask a factual question in the app. For a retrieval-only terminal check that does not call Groq:

```bash
python -m src.retrieval.cli --retrieve-only --top-k 5 "What is the exit load for HDFC Flexi Cap Fund?"
```

Run the automated tests with:

```bash
python -m unittest discover -v
```

Regenerate [data/sample_qa.md](data/sample_qa.md) by running the ten required questions through the real query pipeline:

```bash
python -m scripts.generate_sample_qa
```

This makes live Groq requests for questions allowed by the guardrails and only replaces the sample file if all ten pipeline responses include verified source metadata. The unit tests mock the Groq client; they do not make live API requests.

## Evaluation Workflow

This repository includes a production-grade, repeatable evaluation workflow designed to evaluate the RAG pipeline end-to-end on a representative 30-question benchmark dataset ([test_dataset.json](test_dataset.json)) covering factual questions, edge cases, guardrail refusals, and multi-scheme comparisons.

### 1. Setup Instructions

The evaluation pipeline uses [DeepEval](https://github.com/confident-ai/deepeval) and `matplotlib`. Install the evaluation dependencies into your virtual environment:

```bash
# Core package (if not already installed)
pip install -e .

# Evaluation suite dependencies
pip install -r evaluation/requirements-eval.txt
```

### 2. Environment Configuration (`.env`)

Ensure your `.env` file contains your credentials and configurations:

```bash
# Required for RAG generation and default Groq judge
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL="qwen/qwen3.8-27b"

# Optional: Path to ChromaDB vectorstore (defaults to vectorstore/chroma_db)
CHROMA_DB_PATH="vectorstore/chroma_db"

# Optional: OpenAI credentials (only needed if using --judge-provider openai)
OPENAI_API_KEY="your-openai-api-key"
```

> [!NOTE]
> If `GROQ_API_KEY` is absent, the evaluator gracefully falls back to deterministic heuristic scoring, allowing end-to-end execution without live API access.

### 3. Usage & Single Entry Point

Run the entire evaluation suite with a single command:

```bash
python -m evaluation.run_all
```

#### CLI Options & Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--dataset` | `test_dataset.json` | Path to evaluation test dataset JSON file |
| `--output-base-dir` | `evaluation/outputs` | Base directory for saving evaluation runs |
| `--run-name` | `run_YYYYMMDD_HHMMSS` | Custom run directory name |
| `--limit` | `None` (all) | Limit number of questions to evaluate (great for smoke tests) |
| `--judge-provider` | `groq` | Evaluation judge LLM provider: `groq`, `openai`, or `none` |
| `--judge-model` | `qwen/qwen3.8-27b` | Model identifier for evaluation judge |
| `--delay` | `0.5` | Inter-request delay in seconds to respect API rate limits |
| `--mode` | `direct_pipeline` | `direct_pipeline` (in-process Python) or `http_api` (HTTP endpoint) |
| `--api-url` | `http://localhost:8000/api/chat` | HTTP API endpoint URL (when `mode=http_api`) |
| `--top-k` | `5` | Context chunks retrieved per question in direct pipeline mode |

#### Common Examples

```bash
# Quick smoke test on 5 questions
python -m evaluation.run_all --limit 5

# Run using OpenAI gpt-4o-mini as judge
python -m evaluation.run_all --judge-provider openai --judge-model gpt-4o-mini

# Increase rate-limit delay between requests
python -m evaluation.run_all --delay 1.0

# Evaluate against an external live HTTP API
python -m evaluation.run_all --mode http_api --api-url http://localhost:8000/api/chat
```

### 4. Output Artifacts

Each evaluation run saves a complete set of audit artifacts into a timestamped directory (e.g. `evaluation/outputs/run_YYYYMMDD_HHMMSS/`):

1. **`report.html`**: A standalone, rich interactive dashboard featuring:
   - Overall KPI metric cards (Answer Relevancy, Faithfulness, Contextual Recall, Success Rate, Average Latency, and Estimated Cost).
   - Category performance breakdown table.
   - Flagged test cases panel for immediate debugging of failing or low-scoring queries.
   - Embedded multi-panel visualization preview.
   - Searchable, filterable interactive table with expandable drawer rows detailing questions, actual responses, ground truth, retrieved evidence, and judge reasoning.
2. **`evaluation_results.csv`**: Granular per-sample spreadsheet detailing scores, latencies, tokens, source URLs, and status.
3. **`evaluation_summary.json`**: Aggregate statistics, percentiles (p50, p90, p99), category breakdowns, and pass/fail counts.
4. **`evaluation_visualizations.png`**: High-resolution 4-panel publication-ready visualization charts.

The path to the latest run is recorded in `evaluation/outputs/latest_run.txt`.

#### Terminal Summary Example

Upon completion, `run_all.py` prints a concise terminal summary and the exact report path:

```text
================================================================================
📊 RAG EVALUATION BENCHMARK SUMMARY
================================================================================
Target Model:       qwen/qwen3.8-27b
Evaluation Judge:   groq (qwen/qwen3.8-27b)
Total Samples:      30  (Evaluated: 30, Successful: 30, Failed: 0)
Overall Success:    100.0%

--------------------------------------------------------------------------------
🎯 QUALITY METRICS (DeepEval)
--------------------------------------------------------------------------------
• Answer Relevancy:     0.94 / 1.00  (Passed: 28/30, 93.3%)
• Faithfulness:         0.98 / 1.00  (Passed: 26/26, 100.0%)
• Contextual Recall:    0.91 / 1.00  (Passed: 24/26, 92.3%)

--------------------------------------------------------------------------------
⚡ LATENCY & EFFICIENCY
--------------------------------------------------------------------------------
• Avg Retrieval Latency:  185 ms
• Avg Generation Latency: 1,420 ms
• Avg Total Latency:      1,605 ms  (p50: 1,510 ms, p90: 2,150 ms)
• Total Tokens Used:      18,420 tokens
• Total Cost:             $0.003684 USD

📁 GENERATED ARTIFACTS IN TIMESTAMPED RUN DIRECTORY:
  • 📊 HTML Report:    file:///path/to/evaluation/outputs/run_20261027_140012/report.html
                       (evaluation/outputs/run_20261027_140012/report.html)
  • 📁 Results CSV:    file:///path/to/evaluation/outputs/run_20261027_140012/evaluation_results.csv
  • 📄 Summary JSON:   file:///path/to/evaluation/outputs/run_20261027_140012/evaluation_summary.json
  • 📈 Visualizations: file:///path/to/evaluation/outputs/run_20261027_140012/evaluation_visualizations.png

💡 Open the HTML report directly in your browser:
   open evaluation/outputs/run_20261027_140012/report.html
```

### 5. Troubleshooting & FAQ

- **Groq Rate Limits (HTTP 429)**:
  - If you encounter 429 rate limit errors on Groq on-demand tiers, the custom `GroqJudge` automatically parses retry-after hints and executes exponential backoff.
  - You can also increase inter-request pacing using `--delay 1.0` or `--delay 2.0`.
- **Reasoning Tokens (`<think>` tags)**:
  - Models such as `qwen/qwen3.8-27b` emit reasoning thoughts. `GroqJudge` automatically strips `<think>...</think>` tags to ensure proper JSON extraction for DeepEval schemas.
- **Unanswerable / Refusal Cases**:
  - Out-of-scope questions (e.g. multi-cap funds or advice) are stopped before retrieval by policy guardrails (`decision_allowed=False`). In these cases, Faithfulness and Contextual Recall are logged as `N/A` rather than `0.0` so that legitimate refusals do not penalize retrieval quality metrics. Answer Relevancy measures whether the refusal matches the ground truth.
- **Offline / CI Testing Without API Keys**:
  - Run with `--judge-provider none` or without `GROQ_API_KEY` to use deterministic keyword-overlap heuristics.

## Guardrails

- Blocks investment advice and buy/sell recommendations.
- Blocks returns/performance questions, PII, and account-specific requests before retrieval.
- Restricts questions to the supported mutual-fund topic and scheme set.
- Requires retrieved evidence from an approved official host before answer generation and checks that the answer citation matches retrieved evidence.
- Returns a refusal or cannot-verify message when policy or evidence checks fail.

## Disclaimer

**Facts-only. No investment advice.** This is the same disclaimer constant rendered in the Streamlit UI. The assistant provides general information from selected public sources; it does not provide investment, financial, tax, legal, or portfolio advice, and cannot access individual accounts.

## Known Limitations

- Coverage is limited to the four named HDFC schemes and the 22 records in the registry; sources are not refreshed automatically for each query.
- PDF table cells and image-based details may not survive text extraction. The sample QA records an example where the riskometer value could not be verified from extracted evidence.
- The capital-gains statement source covers a general account statement, not specific capital-gains download steps; the assistant reports that limitation rather than inferring a process.
- Scope and PII checks are deterministic pattern rules, so novel phrasings may be missed or conservatively refused.
- Citation validation confirms that a URL belongs to retrieved approved evidence; it does not formally prove every generated claim is entailed by that evidence.
- Groq is required for generated answers. Retrieval-only CLI mode works without a Groq request.
