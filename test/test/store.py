"""Vector Store & Database Storage: store.py

Starter code for semantic vector storage, embeddings, and PostgreSQL ORM records.
"""

import os
from typing import Any, Dict, Optional
from aimlite.rag import (
    APIEmbedding,
    BaseEmbedding,
    BaseVectorStore,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    MemoryVectorStore,
    OllamaEmbedding,
    PostgresVectorStore,
    SentenceTransformerEmbedding,
    TfidfEmbedding,
)


def get_embedding_model(
    engine: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseEmbedding:
    """Instantiates embedding model (Dense Neural, Ollama API, OpenAI API, or Pure-Python TF-IDF)."""
    resolved_engine = (engine or os.environ.get("EMBEDDING_ENGINE") or "ollama").lower()
    resolved_model = model_name or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    if resolved_engine == "ollama" or "nomic" in resolved_model.lower():
        return OllamaEmbedding(model=resolved_model, host=ollama_host)
    elif resolved_engine == "openai":
        return APIEmbedding(provider="openai", model=resolved_model)
    elif resolved_engine in ("sentence-transformers", "local"):
        return SentenceTransformerEmbedding(model_name_or_path=resolved_model)
    elif resolved_engine == "tfidf":
        return TfidfEmbedding()
    return SentenceTransformerEmbedding(model_name_or_path=resolved_model)


def get_vector_store(
    backend: Optional[str] = None,
    embedding_fn: Optional[BaseEmbedding] = None,
    db_url: Optional[str] = None,
    **kwargs: Any,
) -> BaseVectorStore:
    """Instantiates vector storage backend (PostgreSQL pgvector ORM or MemoryVectorStore)."""
    resolved_backend = (backend or os.environ.get("VECTOR_STORE") or "memory").lower()
    resolved_db = db_url or os.environ.get("DATABASE_URL")
    embed_fn = embedding_fn or get_embedding_model()

    if resolved_backend == "postgres" or (resolved_db and resolved_backend != "memory"):
        return PostgresVectorStore(
            db_url=resolved_db,
            embedding_fn=embed_fn,
            **kwargs,
        )
    return MemoryVectorStore(
        embedding_fn=embed_fn,
        **kwargs,
    )

