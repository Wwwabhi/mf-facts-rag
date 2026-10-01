# Architecture Overview: Facts-Only Mutual Fund Assistant

## Components

### 1. Data ingestion layer
- Official public source documents from HDFC AMC, SEBI, and AMFI
- Document loading and normalization for PDFs, web pages, and text files
- Conversion and parsing using Docling to extract document structure and clean text
- Chunking and metadata extraction for scheme, source type, and publication date

### 2. Vector retrieval layer
- Sentence-transformers embedding model for document and query embeddings
- ChromaDB vector store for semantic retrieval of document chunks
- Metadata-aware indexing to preserve source, URL, date, and scheme information

### 3. Query and orchestration layer
- User question intake from the frontend
- Scope validation for in-scope factual mutual fund questions
- Query embedding generation using the same model as ingestion
- Retrieval of relevant chunks from ChromaDB
- Grounded answer generation using a Groq-hosted LLM
- Source citation mapping from retrieved metadata to a single approved source link

### 4. Application layer
- Streamlit-based frontend for a simple chat / FAQ interface
- Minimal UI with welcome text, examples, question input, answer area, source link, and last-updated notice
- Facts-only response formatting and safety enforcement

### 5. Guardrail and policy layer
- Investment-advice refusal logic
- PII and account-specific information blocking
- “Not verified / not found” fallback when evidence is insufficient
- Source policy enforcement to allow only approved publishers

## Ingestion flow

1. Collect approved official sources from HDFC AMC, SEBI, and AMFI.
2. Load and normalize source documents into a common format.
3. Parse and structure raw content with Docling.
4. Extract metadata such as source title, URL, scheme name, and last-updated date.
5. Split long documents into smaller chunks while preserving context.
6. Generate embeddings for each chunk using sentence-transformers.
7. Store the embeddings and metadata in ChromaDB.
8. Maintain a clean source registry so every chunk can be traced back to an approved document.

## Query flow

1. User asks a factual question in the Streamlit interface.
2. The app checks whether the question is in scope and safe.
3. If in scope, the question is embedded using the same sentence-transformers model used during ingestion.
4. Relevant chunks are retrieved from ChromaDB using semantic similarity.
5. The retrieved context, source metadata, and question are passed to the Groq LLM.
6. The model generates a concise, grounded answer based only on the retrieved evidence.
7. The app attaches one relevant official source link and the “Last updated from sources: <date>” line.
8. If evidence is insufficient or the question is out of scope, the app returns a safe refusal or “cannot verify” response.

## Tech stack
- Python: core application and orchestration
- Docling: document parsing and text extraction
- sentence-transformers: embeddings for documents and queries
- ChromaDB: vector storage and retrieval
- Groq: LLM-based answer generation
- Streamlit: lightweight frontend and demo UI

## Source / citation flow

- Every ingested document carries source metadata: publisher, URL, scheme, document type, and date.
- Each chunk is linked to its parent source record.
- During retrieval, the top matching chunk returns its metadata.
- The answer generation step uses that metadata to attach the correct source link.
- The system never invents a citation; it relies only on retrieved source metadata.
- The “Last updated from sources” value is pulled from the same source metadata and displayed in the final answer.

## Guardrails

### Content safety
- No investment advice or recommendation
- No buy/sell guidance
- No scheme ranking or return prediction
- No personalized portfolio or financial planning advice

### Privacy safety
- Block and ignore PAN, Aadhaar, OTP, bank details, phone numbers, email addresses, and mutual fund account numbers
- Do not store account-specific or personally identifiable information
- Reply with a safe message if sensitive data appears in a user query

### Source policy
- Only HDFC AMC, SEBI, and AMFI sources are allowed
- Third-party blogs, forums, and unofficial websites are excluded from the corpus
- Every factual answer must be traceable to an approved source

### Evidence quality
- If evidence is missing or weak, respond with “information could not be verified from the available sources”
- Do not fabricate facts or claim certainty beyond the retrieved source material

## Folder structure

- data/
  - source documents and raw corpus files
- docs/
  - PRD and architecture documentation
- src/
  - ingestion pipeline
  - retrieval logic
  - LLM orchestration
  - guardrail logic
  - UI code
- app/
  - Streamlit entry point and configuration
- models/
  - embedding model and runtime artifacts
- vectorstore/
  - ChromaDB persistent storage
- notebooks/
  - exploratory analysis and validation

## Simple text architecture diagram

```text
Official sources
   │
   ▼
Docling parsing / normalization
   │
   ▼
Chunking + metadata extraction
   │
   ▼
sentence-transformers embeddings
   │
   ▼
ChromaDB vector store
   │
   ├──────────────► Query embedding (same model)
   │
   ▼
Retrieval of relevant chunks
   │
   ▼
Groq LLM answer generation
   │
   ▼
Streamlit UI
   │
   ├────────► answer + one citation link
   └────────► “Last updated from sources: <date>”
```

## Summary
This architecture keeps the assistant intentionally small, compliant, and transparent. It uses a trusted official-source corpus, semantic retrieval, and tightly scoped answer generation to provide concise factual responses without crossing into investment advice or personal financial guidance.
