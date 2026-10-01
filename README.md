# Mutual Fund Facts

A facts-only mutual-fund FAQ assistant in a Groww product context. This repository currently covers selected HDFC Asset Management Company schemes; it is not an integration with Groww accounts or services.

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