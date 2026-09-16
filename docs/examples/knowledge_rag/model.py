"""Document & Knowledge QA (RAG Paradigm): model.py

Specialized RAGModel subclass providing vector-indexed document retrieval
and context-grounded answer synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modelkit.rag import (
    BaseEmbedding,
    Document,
    MemoryVectorStore,
    RAGModel,
    TfidfEmbedding,
    VectorRetriever,
)


class SupportDocRAG(RAGModel):
    """Knowledge Base QA Model utilizing semantic vector search and passage synthesis."""

    def __init__(
        self,
        name: str = "support_doc_rag",
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self.top_k = top_k
        self.embedding_fn: BaseEmbedding = self._init_embedding()
        self.vector_store = MemoryVectorStore(embedding_fn=self.embedding_fn)
        self.retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedding_fn=self.embedding_fn,
        )

    def _init_embedding(self) -> BaseEmbedding:
        """Initializes embedding engine (SentenceTransformers if installed, else TfidfEmbedding)."""
        try:
            from sentence_transformers import SentenceTransformer

            class STEmbedding(BaseEmbedding):
                def __init__(self) -> None:
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")

                def embed_text(self, text: str) -> List[float]:
                    vec = self.model.encode(text, convert_to_numpy=True)
                    return [float(x) for x in vec.tolist()]

            return STEmbedding()
        except ImportError:
            # Built-in TF-IDF word frequency vector embedding
            return TfidfEmbedding()

    def index_documents(self, documents: List[Document]) -> int:
        """Indexes document chunks into vector store with computed embeddings."""
        # Compute embeddings
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_fn.embed_text(doc.content)
        self.vector_store.add_documents(documents)
        return len(documents)

    def predict(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Retrieves relevant passages and synthesizes an answer with citations."""
        query = inputs if isinstance(inputs, str) else str(inputs.get("query", ""))
        top_k = kwargs.get("top_k", self.top_k)

        # 1. Retrieve relevant passages
        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        # 2. Synthesize grounded answer
        if not retrieved_docs:
            return {
                "query": query,
                "answer": "No relevant documentation found for the provided query.",
                "sources": [],
                "confidence": 0.0,
            }

        # Build citations
        sources = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "snippet": doc.content[:160] + "...",
                "score": round(doc.score or 0.0, 4),
            }
            for doc in retrieved_docs
        ]

        top_passage = retrieved_docs[0].content
        answer = f"Based on {sources[0]['source']}: {top_passage}"

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "confidence": round(retrieved_docs[0].score or 0.85, 4),
        }

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes indexed vector documents to JSON."""
        self.vector_store.save(destination)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Loads and re-indexes vector documents from JSON."""
        self.vector_store.load(source)
