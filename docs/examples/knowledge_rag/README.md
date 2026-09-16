# Knowledge Base QA (RAG Model Example)

This example demonstrates how to build and serve a Retrieval-Augmented Generation (RAG) knowledge base question-answering pipeline using ModelKit's first-class RAG abstractions.

## Architecture
- **Data Ingestion**: Chunks Markdown and text files into overlapping passages using `TextSplitter`.
- **Embeddings**: Vectorizes text using `sentence-transformers` (with deterministic hash fallback).
- **Vector Store**: `InMemoryVectorStore` with cosine similarity search.
- **Inference**: Returns grounded answers with source citations and confidence scores.

## Dependencies
Install the required packages using ModelKit or pip:
```bash
modelkit install sentence-transformers numpy
# or
pip install sentence-transformers numpy
```

## Quickstart

### 1. Initialize Project
```bash
# Option A: In a new folder
modelkit init support_rag
cd support_rag

# Option B: In the current directory
modelkit init .
```

### 2. Copy Knowledge Documents & Code
Place your `.txt` or `.md` files into the `data/` directory. Place `data.py`, `model.py`, `trainer.py`, and `inference.py` into your project package directory.

### 3. Build Vector Index
```bash
modelkit train
```
This chunks your knowledge documents, calculates semantic embeddings, and saves the index to `artifacts/rag_index.json`.

### 4. Serve Knowledge API
```bash
modelkit serve --port 8000
```

### 5. Query the RAG Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "How do session tokens expire?"}'
```
Response:
```json
{
  "query": "How do session tokens expire?",
  "answer": "Based on auth_policy.md: Authentication and Security Policy: ModelKit supports API key and Bearer token authentication. Session tokens expire after 24 hours of inactivity...",
  "sources": [
    {
      "source": "auth_policy.md",
      "snippet": "Authentication and Security Policy: ModelKit supports API key and Bearer token authentication. Session tokens expire after 24 hours...",
      "score": 0.92
    }
  ],
  "confidence": 0.92,
  "status": "success"
}
```
