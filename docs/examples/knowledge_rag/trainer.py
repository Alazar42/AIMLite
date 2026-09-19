"""Document & Knowledge QA (RAG Paradigm): trainer.py

Vector index builder pipeline that reads knowledge base documents,
generates embeddings, and persists the vector index into artifacts.
"""

from __future__ import annotations

from aimlite.rag import RAGTrainer


class IndexBuilderTrainer(RAGTrainer):
    """Trainer orchestrator building and saving the semantic vector index."""
    pass
