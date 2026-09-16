"""Document & Knowledge QA (RAG Paradigm): trainer.py

Vector index builder pipeline that reads knowledge base documents,
generates embeddings, and persists the vector index into artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from modelkit import BaseTrainer, Dataset, Model


class IndexBuilderTrainer(BaseTrainer):
    """Trainer orchestrator building and saving the semantic vector index."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Builds semantic vector index from dataset documents."""
        if hasattr(dataset, "load_documents"):
            documents = dataset.load_documents()
        else:
            records = dataset.load()
            from modelkit.rag import Document

            documents = [
                Document(content=r.get("content", str(r)), metadata=r.get("metadata", {}))
                for r in records
            ]

        if not documents:
            return {"status": "failed", "error": "No documents found to index"}

        # Index documents in RAGModel
        if hasattr(model, "index_documents"):
            indexed_count = model.index_documents(documents)
        else:
            indexed_count = len(documents)

        # Persist index artifact
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        index_path = artifacts_dir / "rag_index.json"
        model.save(index_path)

        return {
            "status": "completed",
            "indexed_chunks": indexed_count,
            "index_path": str(index_path),
        }
