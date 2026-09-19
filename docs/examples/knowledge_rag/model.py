"""Document & Knowledge QA (RAG Paradigm): model.py

Specialized KnowledgeModel subclass providing vector-indexed document retrieval,
query intelligence analysis, and context-grounded answer synthesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite.rag import (
    BaseChatProvider,
    BaseEmbedding,
    BaseVectorStore,
    Document,
    KnowledgeModel,
    MemoryVectorStore,
    MockChatProvider,
    QueryAnalyzer,
    SmartChunker,
    TfidfEmbedding,
    VectorRetriever,
)


class SupportDocRAG(KnowledgeModel):
    """Knowledge Base QA Model utilizing semantic vector search, query analysis, and chat generation."""

    def __init__(
        self,
        name: str = "support_doc_rag",
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        chat_provider: Optional[BaseChatProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        **kwargs: Any,
    ) -> None:
        embedding_fn = self._init_embedding()
        provider = chat_provider or MockChatProvider()
        analyzer = QueryAnalyzer(chat_provider=provider)
        store = vector_store or MemoryVectorStore(embedding_fn=embedding_fn)
        chunker = SmartChunker()

        super().__init__(
            name=name,
            config=config,
            chat_provider=provider,
            embedding_fn=embedding_fn,
            vector_store=store,
            chunker=chunker,
            query_analyzer=analyzer,
            top_k=top_k,
            **kwargs,
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

    def predict(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Retrieves relevant passages and synthesizes an answer with citations."""
        query = inputs if isinstance(inputs, str) else str(inputs.get("query", ""))
        top_k = kwargs.get("top_k", self.top_k)

        # 1. Retrieve relevant passages (using QueryAnalyzer if enabled)
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
