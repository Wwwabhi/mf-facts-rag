# Implementation Plan

This plan breaks the project into six implementation phases. Each phase is intentionally small and testable, aligned with the facts-only product scope and the architecture described in the PRD and architecture documents.

---

## Phase 1: Source loading + document processing

### Files to create
- data/source_manifest.csv
- src/data/source_registry.py
- src/data/load_sources.py
- src/data/normalize_documents.py
- src/data/docling_parser.py
- src/data/metadata_extractor.py

### What it does
- Defines the approved source list: HDFC AMC, SEBI, and AMFI only
- Loads source URLs, document names, scheme mappings, and metadata into a structured manifest
- Normalizes raw source files into a consistent format for downstream processing
- Uses Docling to parse PDF and document content into clean text
- Extracts source metadata such as publisher, URL, scheme, document type, and publication date
- Validates that no unofficial or third-party content enters the corpus

### How to verify it works
- Confirm that all documents in the corpus are from approved publishers only
- Verify that each source file is parsed successfully into text without critical extraction errors
- Check that metadata fields are populated for every processed document
- Review a sample parsed document to ensure text structure is readable and usable for later chunking
- Confirm the pipeline rejects or flags unsupported source types before ingestion continues

---

## Phase 2: Chunking

### Files to create
- src/chunking/chunker.py
- src/chunking/strategy.py
- src/chunking/metadata_enrichment.py
- src/chunking/validate_chunks.py

### What it does
- Splits large documents into manageable text chunks while preserving semantic context
- Maintains scheme, source, and document metadata on each chunk
- Keeps chunks small enough to support targeted retrieval without losing meaning
- Avoids splitting across critical sections such as policy or fee details when possible
- Prepares chunk-level records that can be indexed in the vector database

### How to verify it works
- Inspect a sample document and confirm chunk boundaries make sense semantically
- Check that each chunk retains the correct metadata and source link
- Verify chunk size remains within the expected retrieval-friendly range
- Ensure no chunk contains broken or incomplete source text
- Run spot checks on fee, scheme, and FAQ content to confirm chunking preserves key facts

---

## Phase 3: Embeddings + ChromaDB

### Files to create
- src/vectorstore/chroma_client.py
- src/vectorstore/index_documents.py
- src/vectorstore/embed_chunks.py
- src/vectorstore/schema.py
- src/vectorstore/validate_index.py

### What it does
- Uses sentence-transformers to generate embeddings for each text chunk
- Uses the same embedding model for both ingestion and user queries
- Stores chunk embeddings and metadata in ChromaDB
- Preserves source traceability so every retrieved result can be linked back to the original document
- Enables semantic retrieval for factual questions related to scheme details, benchmarks, fees, and investor service information

### How to verify it works
- Confirm embeddings are created for all valid chunks without gaps
- Verify ChromaDB contains the expected number of stored records
- Run a sample factual query and ensure the retrieval returns relevant scheme-related chunks
- Validate that each retrieved result contains source metadata and document traceability
- Confirm that the embedding model and query model are the same implementation/version

---

## Phase 4: Guardrails

### Files to create
- src/guardrails/scope_checker.py
- src/guardrails/pii_guard.py
- src/guardrails/answer_policy.py
- src/guardrails/refusal_templates.py
- src/guardrails/validation.py

### What it does
- Checks whether a user question is in scope for the facts-only assistant
- Rejects investment advice, recommendation, buy/sell, performance prediction, and portfolio questions
- Blocks PII and account-specific information requests
- Ensures responses do not claim unsupported facts or invent citations
- Returns clear fallback responses such as “cannot verify” or a policy-based refusal

### How to verify it works
- Test in-scope factual questions and confirm they proceed normally
- Test disallowed questions such as “Should I buy this fund?” and confirm they are refused politely
- Test PII requests and confirm they are blocked before further processing
- Verify unsupported questions return a verified “not found / cannot verify” response instead of fabricated content
- Check that no answer can be produced without a valid approved source trace

---

## Phase 5: Retrieval + Groq

### Files to create
- src/retrieval/retrieve.py
- src/retrieval/query_pipeline.py
- src/retrieval/rerank.py
- src/llm/groq_client.py
- src/llm/answer_generator.py
- src/llm/response_formatter.py

### What it does
- Converts the user question into an embedding using the same model used at ingestion time
- Retrieves the top relevant chunks from ChromaDB
- Passes the relevant evidence and source metadata to the Groq-hosted LLM
- Generates a concise answer grounded in the retrieved source material only
- Appends one relevant official source link and the required “Last updated from sources: <date>” line
- Handles “not found” and unsupported-fact responses when evidence is insufficient

### How to verify it works
- Run representative factual questions and validate the retrieved context matches the expected source material
- Check that answer content stays within the required 3-sentence limit
- Confirm exactly one official source link is included in each factual answer
- Verify that the final answer includes the required “Last updated from sources: <date>” text
- Test missing-evidence scenarios and confirm the assistant responds with a clear verification failure instead of guessing

---

## Phase 6: Streamlit UI

### Files to create
- app/main.py
- app/ui/layout.py
- app/ui/examples.py
- app/ui/state.py
- app/ui/messages.py
- app/config/settings.py

### What it does
- Provides a simple user-facing interface with a welcome message, short description, facts-only notice, examples, input box, answer area, source link, and last-updated date
- Connects the frontend to the RAG flow and retrieval pipeline
- Displays concise answers in a lightweight, low-friction format
- Keeps the interface intentionally minimal and product-focused
- Makes the app easy to test and demo without adding unnecessary complexity

### How to verify it works
- Open the app and confirm the welcome text, facts-only note, and example questions are shown
- Submit a factual question and confirm the answer is returned with one source link and the last-updated line
- Test refusal scenarios and ensure they display friendly, policy-based messages
- Verify the app continues to work with no unsupported third-party facts or citations
- Check that the user experience stays simple, readable, and aligned with the product goal

---

## Suggested implementation order
1. Source loading + document processing
2. Chunking
3. Embeddings + ChromaDB
4. Guardrails
5. Retrieval + Groq
6. Streamlit UI

This sequencing keeps the system grounded in reliable data first, then safety, then retrieval and UI, so the product remains fact-based and compliant throughout development.
