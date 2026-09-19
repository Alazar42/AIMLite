"""Document & Knowledge QA (RAG Paradigm): model.py"""

from typing import Any, Dict, Optional
from aimlite.rag import (
    BaseChatProvider,
    BaseEmbedding,
    BaseVectorStore,
    KnowledgeModel,
    QueryAnalyzer,
    SmartChunker,
)
from test.chat_provider import get_chat_provider, get_query_analyzer
from test.store import get_embedding_model, get_vector_store


class SupportDocRAG(KnowledgeModel):
    """Enterprise Knowledge Base Model with Query Intelligence, Smart Chunking, and Grounded Chat."""

    def __init__(
        self,
        name: str = "test_rag",
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        chat_provider: Optional[BaseChatProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_fn: Optional[BaseEmbedding] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        provider = chat_provider or get_chat_provider()
        self.chat_provider = provider

        embed_model = embedding_fn or get_embedding_model()
        store = vector_store or get_vector_store(embedding_fn=embed_model)
        analyzer = query_analyzer or get_query_analyzer(chat_provider=provider)
        chunker = SmartChunker(max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        super().__init__(
            name=name,
            config=config,
            chat_provider=provider,
            embedding_fn=embed_model,
            vector_store=store,
            chunker=chunker,
            query_analyzer=analyzer,
            top_k=top_k,
            **kwargs,
        )
