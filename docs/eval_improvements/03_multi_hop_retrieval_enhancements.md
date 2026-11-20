# 03 - Multi-Hop Retrieval & Contextual Recall Enhancements

This document explains why multi-hop queries experienced severe contextual recall collapse in the baseline evaluation and outlines the architectural changes needed to achieve comprehensive multi-document retrieval.

---

## 1. Problem: Retrieval Starvation on Multi-Entity Queries

In the baseline evaluation, multi-hop queries scored an average **Contextual Recall of 0.2000** (median 0.0000). The model demonstrated 100% faithfulness (meaning it refused to hallucinate), but answered with:
> *"The available approved sources cannot verify the... as the provided evidence does not contain..."*

### The Failing Questions
- **`q006`:** *"What were HDFC Large Cap Fund and HDFC Mid Cap Fund previously called?"* $\to$ Contextual Recall: **0.0000**
- **`q019`:** *"Do HDFC Flexi Cap Fund and HDFC ELSS Tax Saver offer the same IDCW facilities?"* $\to$ Contextual Recall: **0.0000**
- **`q020`:** *"What benchmarks do HDFC Flexi Cap Fund and HDFC Large Cap Fund use?"* $\to$ Contextual Recall: **0.0000**
- **`q021`:** *"According to the supplied July 2026 factsheets, which had the larger AUM: HDFC Flexi Cap Fund or HDFC Mid Cap Fund?"* $\to$ Contextual Recall: **0.0000**

---

## 2. Root Cause Analysis

### A. Vector Blending & Candidate Starvation
In `src/retrieval/retrieve.py`:
```python
vector = active_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
result = active_collection.query(query_embeddings=vector.tolist(), n_results=candidate_k)
```
When a question mentions two schemes (e.g. `HDFC Flexi Cap` and `HDFC Large Cap`), the single dense embedding is a mathematical average of both semantic spaces. Documents that have slightly higher term frequency or similarity for one scheme dominate the top-$k$ nearest neighbors.

### B. Rigid Candidate Cap ($top\_k = 5$) and Source Throttling
In `src/retrieval/rerank.py`:
```python
source_limit = max(2, (top_k + 1) // 2) # For top_k=5, source_limit is 3
```
If Scheme A consumes 3 chunks, only 2 slots remain for all other documents. Often, Scheme B is completely absent from the final top-5 chunks delivered to the generator.

### C. Absence of Lexical Matching (BM25)
Queries like `q006` look for exact historical names (*"formerly HDFC Top 100 Fund"*, *"formerly HDFC Mid-Cap Opportunities Fund"*). Semantic embeddings match broad descriptions of large/mid cap funds rather than the specific entity alias sentence.

---

## 3. Targeted Solution Architecture

```
                             [ User Multi-Hop Query ]
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ Query Classifier & Entity Detector    │
                   │ Detect schemes: [Large Cap, Mid Cap]  │
                   └───────────────────┬───────────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
       [Sub-Query 1: Large Cap]                [Sub-Query 2: Mid Cap]
       "HDFC Large Cap previous name"          "HDFC Mid Cap previous name"
                   │                                       │
       ┌───────────┴───────────┐               ┌───────────┴───────────┐
       ▼                       ▼               ▼                       ▼
  [Dense Embed]           [BM25 Lexical]  [Dense Embed]           [BM25 Lexical]
       │                       │               │                       │
       └───────────┬───────────┘               └───────────┬───────────┘
                   ▼                                       ▼
            [Fused Top 4]                           [Fused Top 4]
                   └───────────────────┬───────────────────┘
                                       ▼
                        [Balanced Context: 8 Chunks]
                               (4 from each scheme)
                                       │
                                       ▼
                         [Grounded LLM Generator]
```

### 1. Multi-Entity Sub-Query Decomposition
Implement a lightweight query splitter before vector query:
```python
def extract_mentioned_schemes(question: str) -> list[str]:
    found = []
    for scheme in ("hdfc flexi cap", "hdfc large cap", "hdfc mid cap", "hdfc elss tax saver"):
        if any(term in question.lower() for term in scheme.split()[1:]):
            found.append(scheme)
    return found
```
If multiple schemes are detected:
- Generate target sub-queries for each scheme.
- Query ChromaDB separately for each scheme with a guaranteed allocation (e.g. 3–4 chunks per scheme).

### 2. Dynamic Top-$k$ Scaling
- Single-scheme query: $top\_k = 5$
- Two-scheme query: $top\_k = 8$ (4 per scheme)
- Four-scheme query: $top\_k = 8$–$12$ (2–3 per scheme)

### 3. Hybrid Retrieval with BM25 Rank Fusion
Add BM25 indexing over chunk contents. For queries searching for specific named attributes (former names, benchmark titles, AUM numbers):
$$Score = 0.5 \times Score_{vector} + 0.5 \times Score_{BM25}$$
This ensures exact textual mentions (such as `"HDFC Top 100 Fund"`) rank directly into the top evidence set.

---

## 4. Expected Impact on Re-Evaluation

| Metric | Baseline Score | Projected Post-Fix Score | Why |
| :--- | :--- | :--- | :--- |
| **`q006` Contextual Recall** | 0.0000 | **1.0000** | Both former scheme names retrieved |
| **`q019` Contextual Recall** | 0.0000 | **1.0000** | IDCW rules for both Flexi Cap & ELSS retrieved |
| **`q020` Contextual Recall** | 0.0000 | **1.0000** | NIFTY 500 & NIFTY 100 benchmarks retrieved |
| **`q021` Contextual Recall** | 0.0000 | **1.0000** | Both July 2026 factsheet AUMs retrieved |
| **Overall Multi-Hop Recall** | 0.2000 | **$\ge$ 0.8500 (+65.0%)** | Eliminates retrieval starvation |
| **Overall Contextual Recall**| 0.6875 | **$\ge$ 0.8800 (+19.25%)** | Elevates entire test suite recall |
