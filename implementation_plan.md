# RAG Architecture Overhaul: Chat Providers, PostgreSQL ORM, Smart Chunker, Query Analyzer & Knowledge Model

This plan upgrades AIMLite's Retrieval-Augmented Generation (RAG) architecture to provide enterprise-grade, modular RAG components while preserving zero-dependency runtime resilience.

## Overview of Enhancements

1. **Chat Provider Abstraction (`BaseChatProvider` & Implementations)**:
   - Modular interface supporting any LLM: API-based (OpenAI, Anthropic, Gemini, Groq, custom REST) and Local-based (Ollama, local HuggingFace / llama.cpp / callable, mock baseline).
   - Built-in embedding integration (`embed_text`, `embed_batch`).
   - Standardized pre-defined system prompts for all RAG tasks.

2. **Pre-defined System Prompts**:
   - `PROMPT_RAG_QA`: Grounded answer synthesis citing document IDs and metadata.
   - `PROMPT_QUERY_ANALYZER`: Query decomposition, intent extraction, sub-query rewriting, and Hypothetical Document Embeddings (HyDE).
   - `PROMPT_SMART_CHUNKER`: Contextual summary extraction and semantic boundary optimization.
   - `PROMPT_CONVERSATIONAL_RAG`: Multi-turn conversational RAG with chat history context.

3. **Query Analyzer (`QueryAnalyzer`)**:
   - Analyzes incoming user queries before retrieval.
   - Generates intent tags, expanded/sub-queries for multi-angle retrieval, and hypothetical answer passages (HyDE) for better embedding alignment.
   - Modular: works with any `BaseChatProvider` or built-in heuristic/regex fallback.

4. **Smart Chunker (`SmartChunker` / `SemanticChunker`)**:
   - Structure-aware text splitter (handles Markdown headers, code blocks, lists, paragraphs).
   - Semantic boundary preservation and contextual chunk enrichment (injects parent headers/document titles).

5. **PostgreSQL & ORM Database Integration**:
   - Declarative ORM models (`KnowledgeDocument`, `KnowledgeChunk`, `ChatSession`, `ChatMessage`).
   - `PostgresVectorStore`: Native PostgreSQL storage with `pgvector` support (`<=>` cosine / `<->` L2 distance) and SQL/ORM query fallback.
   - `SQLAlchemyKnowledgeStore`: Unified database manager for document lifecycle, chunks, and metadata.
   - SQLite / In-memory zero-dependency fallback for development and local testing.

6. **Basic Knowledge Model (`KnowledgeModel` & `RAGModel` Extensions)**:
   - High-level domain model extending `Model` for knowledge management, query routing, semantic retrieval, and conversational QA.
   - Integrates seamlessly with `QueryAnalyzer`, `SmartChunker`, `PostgresVectorStore`, and `ChatProvider`.

7. **Trainer Orchestration (`RAGTrainer` / `KnowledgeTrainer`)**:
   - Coordinates dataset loading -> smart chunking -> embedding generation -> database / vector index persistence.
   - Fully compatible with `aimlite train` and standard AIMLite lifecycle.

---

## Proposed Changes

### Core Library: `src/aimlite/rag.py` & `src/aimlite/`

#### [MODIFY] [rag.py](file:///home/atocodes/Projects/ModelKit/src/aimlite/rag.py)
- **Predefined System Prompts**:
  - `PROMPT_RAG_QA`, `PROMPT_QUERY_ANALYZER`, `PROMPT_SMART_CHUNKER`, `PROMPT_CONVERSATIONAL_RAG`.
- **Chat Provider Abstraction**:
  - `BaseChatProvider`, `OpenAIChatProvider`, `AnthropicChatProvider`, `GeminiChatProvider`, `OllamaChatProvider`, `LocalChatProvider`, `MockChatProvider`.
- **Query Analyzer**:
  - `QueryAnalysisResult` dataclass (`original_query`, `expanded_queries`, `hypothetical_document`, `keywords`, `intent`).
  - `QueryAnalyzer` class supporting LLM-based and heuristic-based query refinement.
- **Smart Chunker**:
  - `SmartChunker` (Markdown-aware, hierarchical splitting with heading metadata preservation).
- **Embedder Extensions**:
  - `BaseEmbedding`, `TfidfEmbedding`, `SentenceTransformerEmbedding`, `APIEmbedding`.
- **PostgreSQL & ORM Database Layer**:
  - `KnowledgeDocument`, `KnowledgeChunk` schemas (SQLAlchemy declarative models when available, standard dataclasses/records fallback).
  - `PostgresVectorStore`: PostgreSQL + pgvector / relational cosine similarity backend.
  - `DatabaseVectorStore`: Generic SQL/PostgreSQL/SQLite vector store.
- **Knowledge Model & RAG Model**:
  - `KnowledgeModel`: Full-featured knowledge management model combining analyzer, embedder, database store, and chat provider.
  - `RAGModel`: Updated to accept `chat_provider` and `query_analyzer`.
- **Lifecycle & Trainer**:
  - `RAGTrainer` / `KnowledgeTrainer` extending `BaseTrainer`.

#### [MODIFY] [__init__.py](file:///home/atocodes/Projects/ModelKit/src/aimlite/__init__.py)
- Re-export all new RAG abstractions, chat providers, prompts, query analyzers, chunkers, and database stores.

---

### Examples & Documentation: `docs/examples/knowledge_rag/`

#### [MODIFY] [data.py](file:///home/atocodes/Projects/ModelKit/docs/examples/knowledge_rag/data.py)
- Update dataset to showcase `SmartChunker` with markdown sections and metadata.

#### [MODIFY] [model.py](file:///home/atocodes/Projects/ModelKit/docs/examples/knowledge_rag/model.py)
- Showcase `KnowledgeModel` / `RAGModel` configured with `QueryAnalyzer`, `ChatProvider`, and vector store.

#### [MODIFY] [trainer.py](file:///home/atocodes/Projects/ModelKit/docs/examples/knowledge_rag/trainer.py)
- Demonstrate `RAGTrainer` indexing workflow.

---

### Testing Suite

#### [NEW] [test_rag.py](file:///home/atocodes/Projects/ModelKit/test/test_rag.py)
- Comprehensive test suite covering:
  - Chat providers (mock, custom, local, API configurations).
  - Predefined system prompts verification.
  - Query Analyzer (expansion, HyDE, keywords, fallback).
  - Smart Chunker (markdown splitting, context metadata, hierarchy).
  - Vector stores: MemoryVectorStore and PostgreSQL/Database vector store operations.
  - KnowledgeModel and RAGModel end-to-end question answering and chat.
  - RAGTrainer pipeline execution.

#### [MODIFY] [test_headers.py](file:///home/atocodes/Projects/ModelKit/test/test_headers.py) & [test_examples.py](file:///home/atocodes/Projects/ModelKit/test/test_examples.py)
- Ensure all existing tests continue passing without regression.

---

## Verification Plan

### Automated Tests
- Run all unit tests:
  ```bash
  PYTHONPATH=src:. python3 -m unittest discover -s test
  ```
- Run specialized RAG tests:
  ```bash
  PYTHONPATH=src:. python3 -m unittest test.test_rag
  PYTHONPATH=src:. python3 -m unittest test.test_examples
  PYTHONPATH=src:. python3 -m unittest test.test_headers
  ```

### Manual / Integration Verification
- Test local chat provider inference with mock payloads.
- Test query analysis expansion and HyDE document synthesis.
- Test Markdown smart chunking preserving header hierarchies.
- Test SQLite / PostgreSQL database vector store persistence and retrieval.
