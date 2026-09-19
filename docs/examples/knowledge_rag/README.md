# Knowledge Base QA (RAG Model Example)

This example demonstrates how to build, index, and serve a Retrieval-Augmented Generation (RAG) knowledge engine using AIMLite's enterprise RAG abstractions.

## Architecture
- **Data Ingestion & Smart Chunking**: Chunks Markdown and documentation files into structured passages using `SmartChunker`, preserving heading hierarchy and section metadata.
- **Query Intelligence**: Uses `QueryAnalyzer` with pre-defined system prompts (`PROMPT_QUERY_ANALYZER`) for intent analysis, multi-query expansion, and Hypothetical Document Embeddings (HyDE).
- **Embeddings & Vector Stores**: Dense vectors via `sentence-transformers`, `OpenAIChatProvider`, or built-in `TfidfEmbedding`, stored in `MemoryVectorStore` or `PostgresVectorStore` (`pgvector` / PostgreSQL ORM).
- **Chat Generation**: Modular `BaseChatProvider` (OpenAI, Anthropic, Gemini, Ollama, Local callable, or Mock baseline) with pre-defined task prompts (`PROMPT_RAG_QA`).
- **High-Level Model**: `KnowledgeModel` and `RAGTrainer` integrating the end-to-end indexing and inference workflow.

## Dependencies
Install optional packages for production vector indexing and neural embeddings:
```bash
aimlite install sentence-transformers psycopg2-binary
# or
pip install sentence-transformers psycopg2-binary
```

## Quickstart

### 1. Initialize Project
```bash
# Option A: In a new folder
aimlite init support_rag
cd support_rag

# Option B: In the current directory
aimlite init .
```

### 2. Copy Knowledge Documents & Code
Place your `.txt` or `.md` files into the `data/` directory. Place `data.py`, `model.py`, `trainer.py`, and `inference.py` into your project directory.

### 3. Build Vector Index
```bash
aimlite train
```
This chunks your knowledge documents via `SmartChunker`, generates semantic embeddings, and saves the index artifact to `artifacts/rag_index.json` (or syncs with PostgreSQL).

### 4. Serve Knowledge API
```bash
aimlite serve --port 8000
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
  "answer": "Based on auth_policy.md: Authentication and Security Policy: AIMLite supports API key and Bearer token authentication. Session tokens expire after 24 hours of inactivity...",
  "sources": [
    {
      "source": "auth_policy.md",
      "snippet": "Authentication and Security Policy: AIMLite supports API key and Bearer token authentication. Session tokens expire after 24 hours...",
      "score": 0.92
    }
  ],
  "confidence": 0.92,
  "status": "success"
}
```
