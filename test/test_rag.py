"""Comprehensive tests for the AIMLite RAG subsystem.

Covers:
- Document creation, metadata, and chunking
- MemoryVectorStore add/search (real cosine similarity)
- TfidfEmbedding deterministic output
- KnowledgeModel explicit provider requirement
- RAGModel explicit provider requirement
- KnowledgeModel index_documents and retrieval
- RAGModel full predict pipeline (preprocess -> retrieve -> rerank -> synthesize)
- RAGTrainer.fit() indexes documents and persists index artifact
- SentenceTransformerEmbedding raises ImportError when package is missing
- DocumentLoader.load_directory raises on file errors
- QueryAnalyzer heuristic fallback path
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aimlite.rag import (
    Document,
    DocumentLoader,
    KnowledgeModel,
    MemoryVectorStore,
    MockChatProvider,
    QueryAnalyzer,
    RAGModel,
    RAGTrainer,
    SmartChunker,
    SentenceTransformerEmbedding,
    TfidfEmbedding,
    VectorRetriever,
)
from aimlite.data import Dataset


# =============================================================================
# Helpers
# =============================================================================


def make_knowledge_model(**kwargs) -> KnowledgeModel:
    """Creates a KnowledgeModel with MockChatProvider for tests."""
    kwargs.setdefault("chat_provider", MockChatProvider())
    kwargs.setdefault("name", "test_km")
    return KnowledgeModel(**kwargs)


def make_rag_model(**kwargs) -> RAGModel:
    """Creates a RAGModel with MockChatProvider for tests."""
    kwargs.setdefault("chat_provider", MockChatProvider())
    kwargs.setdefault("name", "test_rag")
    return RAGModel(**kwargs)


# =============================================================================
# Document tests
# =============================================================================


class TestDocument:
    def test_document_auto_id(self):
        """Documents must receive a unique auto-generated ID."""
        d1 = Document(content="First")
        d2 = Document(content="Second")
        assert d1.id != d2.id

    def test_document_to_dict(self):
        """to_dict() must include all standard fields."""
        doc = Document(content="Hello", metadata={"source": "test"}, score=0.9)
        d = doc.to_dict()
        assert d["content"] == "Hello"
        assert d["metadata"]["source"] == "test"
        assert d["score"] == 0.9


# =============================================================================
# TfidfEmbedding tests
# =============================================================================


class TestTfidfEmbedding:
    def test_embed_text_returns_list_of_floats(self):
        emb = TfidfEmbedding()
        vec = emb.embed_text("machine learning")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) > 0

    def test_embed_text_is_deterministic(self):
        emb = TfidfEmbedding()
        v1 = emb.embed_text("test query")
        v2 = emb.embed_text("test query")
        assert v1 == v2

    def test_embed_batch_length(self):
        emb = TfidfEmbedding()
        texts = ["alpha", "beta", "gamma"]
        vecs = emb.embed_batch(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert isinstance(v, list)


# =============================================================================
# MemoryVectorStore tests
# =============================================================================


class TestMemoryVectorStore:
    def test_add_and_search_returns_results(self):
        store = MemoryVectorStore(embedding_fn=TfidfEmbedding())
        docs = [
            Document(content="Python is a programming language"),
            Document(content="Machine learning uses data"),
            Document(content="Cats are domestic animals"),
        ]
        store.add_documents(docs)
        emb = TfidfEmbedding()
        q_vec = emb.embed_text("Python programming")
        results = store.similarity_search(q_vec, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, Document) for r in results)

    def test_similarity_search_score_range(self):
        """Cosine similarity scores must be in [-1, 1]."""
        store = MemoryVectorStore()
        docs = [Document(content="hello world"), Document(content="goodbye world")]
        store.add_documents(docs)
        emb = TfidfEmbedding()
        q_vec = emb.embed_text("hello")
        results = store.similarity_search(q_vec, top_k=2)
        for r in results:
            assert r.score is not None
            assert -1.0 <= r.score <= 1.0

    def test_similarity_search_top_k_respected(self):
        store = MemoryVectorStore()
        docs = [Document(content=f"doc {i}") for i in range(10)]
        store.add_documents(docs)
        emb = TfidfEmbedding()
        q_vec = emb.embed_text("doc 5")
        results = store.similarity_search(q_vec, top_k=3)
        assert len(results) <= 3

    def test_metadata_filter(self):
        """similarity_search() must respect metadata filters."""
        store = MemoryVectorStore()
        docs = [
            Document(content="apple pie recipe", metadata={"category": "food"}),
            Document(content="machine learning tutorial", metadata={"category": "tech"}),
        ]
        store.add_documents(docs)
        emb = TfidfEmbedding()
        q_vec = emb.embed_text("learning")
        results = store.similarity_search(q_vec, top_k=5, filters={"category": "tech"})
        assert all(r.metadata.get("category") == "tech" for r in results)

    def test_save_load_roundtrip(self):
        """save() and load() must restore all documents."""
        store = MemoryVectorStore()
        docs = [Document(content=f"content {i}", metadata={"idx": i}) for i in range(5)]
        store.add_documents(docs)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index.json"
            store.save(path)

            new_store = MemoryVectorStore()
            new_store.load(path)

            assert len(new_store.documents) == 5
            contents = {d.content for d in new_store.documents}
            for i in range(5):
                assert f"content {i}" in contents

    def test_search_empty_store_returns_empty(self):
        store = MemoryVectorStore()
        result = store.search("query", top_k=3)
        assert result == []


# =============================================================================
# SmartChunker tests
# =============================================================================


class TestSmartChunker:
    def test_split_long_document(self):
        """SmartChunker must split very long text into multiple chunks."""
        # Create content longer than max_chunk_size * 2 to guarantee splitting
        chunker = SmartChunker(max_chunk_size=50, min_chunk_size=10, chunk_overlap=5)
        long_text = " ".join(["longerword"] * 50)  # ~600 chars
        docs = [Document(content=long_text)]
        chunks = chunker.split_documents(docs)
        assert len(chunks) > 1, (
            f"SmartChunker should split text of length {len(long_text)} "
            f"into multiple chunks with max_chunk_size=50, got {len(chunks)} chunk(s)"
        )

    def test_chunk_preserves_metadata(self):
        chunker = SmartChunker(max_chunk_size=50)
        doc = Document(content="Short text.", metadata={"source": "myfile.txt"})
        chunks = chunker.split_documents([doc])
        for chunk in chunks:
            assert chunk.metadata.get("source") == "myfile.txt"

    def test_short_document_stays_as_one_chunk(self):
        chunker = SmartChunker(max_chunk_size=500)
        doc = Document(content="This is short.")
        chunks = chunker.split_documents([doc])
        assert len(chunks) == 1
        assert chunks[0].content == "This is short."


# =============================================================================
# KnowledgeModel and RAGModel provider requirement
# =============================================================================


class TestProviderRequirement:
    def test_knowledge_model_requires_chat_provider(self):
        """KnowledgeModel must raise ValueError when chat_provider is not provided."""
        with pytest.raises(ValueError, match="chat_provider"):
            KnowledgeModel(name="no_provider")

    def test_rag_model_requires_chat_provider(self):
        """RAGModel must raise ValueError when chat_provider is not provided."""
        with pytest.raises(ValueError, match="chat_provider"):
            RAGModel(name="no_provider")

    def test_knowledge_model_with_mock_provider(self):
        """KnowledgeModel must initialize successfully with MockChatProvider."""
        km = make_knowledge_model()
        assert km is not None
        assert isinstance(km.chat_provider, MockChatProvider)

    def test_rag_model_with_mock_provider(self):
        """RAGModel must initialize successfully with MockChatProvider."""
        rm = make_rag_model()
        assert rm is not None
        assert isinstance(rm.chat_provider, MockChatProvider)


# =============================================================================
# KnowledgeModel end-to-end (index + retrieve + predict)
# =============================================================================


class TestKnowledgeModelEndToEnd:
    def test_index_documents_returns_chunk_count(self):
        km = make_knowledge_model()
        docs = [
            Document(content="Python is a programming language created by Guido van Rossum."),
            Document(content="Machine learning is a subset of artificial intelligence."),
        ]
        count = km.index_documents(docs)
        assert count > 0

    def test_search_returns_relevant_results(self):
        km = make_knowledge_model()
        km.index_documents([
            Document(content="The Eiffel Tower is located in Paris, France."),
            Document(content="The Great Wall of China is a historic fortification."),
            Document(content="Python lists support slicing and indexing."),
        ])
        results = km.search("Eiffel Tower France", top_k=2)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all("content" in r for r in results)

    def test_predict_returns_structured_response(self):
        km = make_knowledge_model()
        km.index_documents([
            Document(content="AIMLite is a production-ready ML framework for Python."),
        ])
        result = km.predict("What is AIMLite?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "sources" in result
        assert "query" in result

    def test_predict_with_no_documents_returns_no_context_message(self):
        """predict() with no indexed documents must return an informative answer."""
        km = make_knowledge_model()
        result = km.predict("anything")
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_add_document_convenience_method(self):
        km = make_knowledge_model()
        count = km.add_document("This is a single document.", title="Test Doc")
        assert count >= 1

    def test_index_strings_and_dicts(self):
        """index_documents must accept mixed raw strings and dicts."""
        km = make_knowledge_model()
        count = km.index_documents([
            "Plain string document.",
            {"content": "Dict document.", "metadata": {"source": "api"}},
        ])
        assert count >= 2

    def test_save_load_roundtrip(self):
        km = make_knowledge_model()
        km.index_documents([Document(content="Persisted document.")])

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "rag_index.json"
            km.save(path)

            new_km = make_knowledge_model()
            new_km.load(path)

            assert len(new_km.vector_store.documents) == len(km.vector_store.documents)


# =============================================================================
# RAGTrainer.fit() tests
# =============================================================================


class TestRAGTrainer:
    def make_document_dataset(self, n: int = 5) -> Dataset:
        """Creates a dataset whose records contain 'content' keys for RAGTrainer."""
        ds = Dataset(name="rag_test")
        # RAGTrainer reads via dataset.load() then processes records
        ds._data = [
            {"content": f"Document {i} about machine learning and data science.", "metadata": {"source": f"doc{i}.txt"}}
            for i in range(n)
        ]
        # Override load() to return pre-set data (avoid file resolution)
        ds.load = lambda **kw: ds._data  # type: ignore
        return ds

    def test_fit_indexes_documents(self):
        km = make_knowledge_model()
        dataset = self.make_document_dataset(n=5)
        trainer = RAGTrainer()
        result = trainer.fit(km, dataset)
        assert result["status"] == "completed"
        assert result["indexed_chunks"] > 0

    def test_fit_persists_index_artifact(self):
        km = make_knowledge_model()
        dataset = self.make_document_dataset(n=3)
        trainer = RAGTrainer()

        with tempfile.TemporaryDirectory() as td:
            result = trainer.fit(km, dataset)

        assert "index_path" in result

    def test_fit_empty_dataset_returns_failed(self):
        km = make_knowledge_model()
        empty_ds = Dataset(name="empty")
        empty_ds._data = []
        trainer = RAGTrainer()
        result = trainer.fit(km, empty_ds)
        assert result["status"] == "failed"

    def test_fit_enables_subsequent_predict(self):
        """After fit(), knowledge model must answer queries using the indexed content."""
        km = make_knowledge_model()
        dataset = self.make_document_dataset(n=5)
        trainer = RAGTrainer()
        trainer.fit(km, dataset)

        result = km.predict("machine learning data science")
        assert "answer" in result
        assert "sources" in result


# =============================================================================
# QueryAnalyzer tests
# =============================================================================


class TestQueryAnalyzer:
    def test_heuristic_analyze_extracts_keywords(self):
        """_heuristic_analyze() must extract content keywords and skip stopwords."""
        analyzer = QueryAnalyzer(chat_provider=MockChatProvider())
        result = analyzer._heuristic_analyze("How does machine learning work?")
        assert "machine" in result.keywords or "learning" in result.keywords
        assert "how" not in result.keywords
        assert "does" not in result.keywords

    def test_analyze_returns_valid_structure(self):
        analyzer = QueryAnalyzer(chat_provider=MockChatProvider())
        result = analyzer.analyze("What is deep learning?")
        assert result.original_query == "What is deep learning?"
        assert isinstance(result.keywords, list)
        assert isinstance(result.expanded_queries, list)

    def test_all_search_queries_includes_original(self):
        analyzer = QueryAnalyzer(chat_provider=MockChatProvider())
        result = analyzer.analyze("neural networks")
        queries = result.all_search_queries()
        assert "neural networks" in queries


# =============================================================================
# SentenceTransformerEmbedding — explicit failure when not installed
# =============================================================================


class TestSentenceTransformerEmbedding:
    def test_raises_import_error_when_package_missing(self):
        """SentenceTransformerEmbedding must raise ImportError when sentence-transformers is absent."""
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            emb = SentenceTransformerEmbedding(model_name_or_path="all-MiniLM-L6-v2")
            with pytest.raises(ImportError, match="sentence-transformers"):
                emb.embed_text("test")


# =============================================================================
# DocumentLoader.load_directory error propagation
# =============================================================================


class TestDocumentLoader:
    def test_load_directory_raises_on_corrupt_file(self):
        """load_directory() must raise RuntimeError when any file fails to load."""
        with tempfile.TemporaryDirectory() as td:
            bad_file = Path(td) / "corrupt.json"
            bad_file.write_text("{ THIS IS NOT VALID JSON!!! }", encoding="utf-8")

            with pytest.raises(RuntimeError, match="file error"):
                DocumentLoader.load_directory(td, extensions=[".json"])

    def test_load_directory_succeeds_with_valid_files(self):
        """load_directory() must succeed when all files are valid."""
        with tempfile.TemporaryDirectory() as td:
            txt_file = Path(td) / "sample.txt"
            txt_file.write_text("This is valid text content.", encoding="utf-8")

            docs = DocumentLoader.load_directory(td, extensions=[".txt"])
            assert len(docs) >= 1
            assert any("valid text" in d.content for d in docs)
