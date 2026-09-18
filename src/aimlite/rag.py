"""RAG Paradigm: aimlite/rag.py

First-class Retrieval-Augmented Generation abstractions:
Documents, Loaders, Text Splitters, Embeddings, Vector Stores, Retrievers,
and RAGModel (which extends Model for unified inference and serving).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from aimlite.models import Model


@dataclass
class Document:
    """Represents a discrete text passage and its associated metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    embedding: Optional[List[float]] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes document to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
        }


class TextSplitter:
    """Chunks documents into sliding text windows with configurable overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Splits raw text string into overlapping chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
        return chunks

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Splits a sequence of documents into chunked Document instances."""
        chunked: List[Document] = []
        for doc in documents:
            sub_texts = self.split_text(doc.content)
            for idx, text in enumerate(sub_texts):
                sub_meta = dict(doc.metadata)
                sub_meta["chunk_index"] = idx
                sub_meta["parent_id"] = doc.id
                chunked.append(
                    Document(
                        content=text,
                        metadata=sub_meta,
                        id=f"{doc.id}-c{idx}",
                    )
                )
        return chunked


class DocumentLoader:
    """Utility to load documents from disk (text, markdown, CSV, JSON)."""

    @classmethod
    def load_file(cls, file_path: Union[str, Path]) -> List[Document]:
        """Loads a single file into one or more Document instances."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext in [".txt", ".md", ".rst"]:
            text = path.read_text(encoding="utf-8")
            return [Document(content=text, metadata={"source": str(path), "filename": path.name})]

        elif ext == ".json":
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            docs = []
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    content = item.get("text") or item.get("content") or json.dumps(item)
                    docs.append(
                        Document(
                            content=str(content),
                            metadata={"source": str(path), "index": idx, **{k: v for k, v in item.items() if k not in ["text", "content"]}},
                        )
                    )
            elif isinstance(data, dict):
                content = data.get("text") or data.get("content") or json.dumps(data)
                docs.append(Document(content=str(content), metadata={"source": str(path), **data}))
            return docs

        elif ext == ".csv":
            import csv

            docs = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    row_content = ", ".join(f"{k}: {v}" for k, v in row.items())
                    docs.append(Document(content=row_content, metadata={"source": str(path), "row": idx, **row}))
            return docs

        # Fallback raw read
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [Document(content=text, metadata={"source": str(path), "filename": path.name})]

    @classmethod
    def load_directory(
        cls,
        directory_path: Union[str, Path],
        glob_pattern: str = "*",
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """Recursively loads all matching files from a directory."""
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        valid_exts = [e.lower() for e in extensions] if extensions else None
        all_docs: List[Document] = []

        for p in dir_path.rglob(glob_pattern):
            if p.is_file():
                if valid_exts is None or p.suffix.lower() in valid_exts:
                    try:
                        all_docs.extend(cls.load_file(p))
                    except Exception:
                        pass
        return all_docs


class BaseEmbedding(ABC):
    """Abstract contract for generating vector representations from text."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generates a dense vector representation for a single text query."""
        ...

    def embed_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Embeds a sequence of Document instances in-place."""
        for doc in documents:
            doc.embedding = self.embed_text(doc.content)
        return list(documents)


class TfidfEmbedding(BaseEmbedding):
    """Pure-Python hash-based TF-IDF embedding baseline with zero external dependencies."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def embed_text(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        for token in tokens:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec


class BaseVectorStore(ABC):
    """Abstract interface for storing document vectors and performing similarity retrieval."""

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Stores document vectors into the index."""
        ...

    @abstractmethod
    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[Document]:
        """Returns the top_k most similar Document instances ranked by cosine score."""
        ...

    def save(self, destination: Union[str, Path]) -> None:
        """Serializes vector store state to disk."""
        ...

    def load(self, source: Union[str, Path]) -> None:
        """Restores vector store state from disk."""
        ...


class MemoryVectorStore(BaseVectorStore):
    """In-memory cosine similarity vector store with local serialization."""

    def __init__(self, embedding_fn: Optional[BaseEmbedding] = None) -> None:
        self.embedding_fn = embedding_fn or TfidfEmbedding()
        self.documents: List[Document] = []

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_documents(self, documents: Sequence[Document]) -> None:
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_fn.embed_text(doc.content)
            self.documents.append(doc)

    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[Document]:
        scored: List[tuple[float, Document]] = []
        for doc in self.documents:
            if doc.embedding is not None:
                sim = self._cosine_similarity(query_embedding, doc.embedding)
                scored.append((sim, doc))

        # Rank descending
        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Document] = []
        for sim, doc in scored[:top_k]:
            doc_copy = Document(
                content=doc.content,
                metadata=dict(doc.metadata),
                id=doc.id,
                embedding=doc.embedding,
                score=round(float(sim), 4),
            )
            results.append(doc_copy)
        return results

    def save(self, destination: Union[str, Path]) -> None:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "id": doc.id,
                "embedding": doc.embedding,
            }
            for doc in self.documents
        ]
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, source: Union[str, Path]) -> None:
        src = Path(source)
        if src.is_file():
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = [
                Document(
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                    id=item.get("id", ""),
                    embedding=item.get("embedding"),
                )
                for item in data
            ]


class BaseRetriever(ABC):
    """Abstract retrieval hook mapping queries to ranked context documents."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        """Retrieves top_k context documents relevant to the query."""
        ...


class VectorRetriever(BaseRetriever):
    """Retriever bound to a BaseVectorStore and BaseEmbedding."""

    def __init__(self, vector_store: BaseVectorStore, embedding_fn: BaseEmbedding) -> None:
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn

    def retrieve(self, query: str, top_k: int = 4) -> List[Document]:
        query_vec = self.embedding_fn.embed_text(query)
        return self.vector_store.similarity_search(query_vec, top_k=top_k)


class RAGModel(Model):
    """Model specialization for Retrieval-Augmented Generation.

    Subclasses Model so it inherits the standardized AIMLite lifecycle:
    predict(), save(), load(), get_dataset(), and auto-registration under 'model' and 'rag'.
    """

    retriever: Optional[BaseRetriever] = None

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init_subclass__(name=name, **kwargs)
        from aimlite.registry import register_class

        register_class("rag", cls, name=name)

    def __init__(
        self,
        name: str = "rag_model",
        config: Optional[Dict[str, Any]] = None,
        retriever: Optional[BaseRetriever] = None,
        generator_fn: Optional[Callable[[str, List[Document]], str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self.retriever = retriever
        self.generator_fn = generator_fn
        self.system_prompt = system_prompt or "Answer the query based on the provided context passages."

    def build_context(self, documents: List[Document]) -> str:
        """Formats retrieved documents into a consolidated context string."""
        parts = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("filename") or f"Doc {idx}"
            parts.append(f"[{idx}] (Source: {source})\n{doc.content}")
        return "\n\n".join(parts)

    def default_generator(self, query: str, context_docs: List[Document]) -> str:
        """Default baseline generator synthesis when no external LLM is attached."""
        if not context_docs:
            return "No relevant context documents found to answer the query."
        top_doc = context_docs[0]
        return f"Based on {top_doc.metadata.get('source', 'indexed knowledge')}: {top_doc.content[:300]}..."

    def predict(self, inputs: Any, top_k: int = 4, **kwargs: Any) -> Dict[str, Any]:
        """Executes forward RAG inference: Retrieval -> Synthesis -> Grounded Output.

        Args:
            inputs: User query text string or prompt payload.
            top_k: Number of context chunks to retrieve.

        Returns:
            Dictionary with 'query', 'answer', and 'sources' list with citations.
        """
        query_str = str(inputs)
        retrieved_docs: List[Document] = []

        if self.retriever is not None:
            retrieved_docs = self.retriever.retrieve(query_str, top_k=top_k)

        if self.generator_fn is not None:
            answer = self.generator_fn(query_str, retrieved_docs)
        else:
            answer = self.default_generator(query_str, retrieved_docs)

        return {
            "query": query_str,
            "answer": answer,
            "sources": [doc.to_dict() for doc in retrieved_docs],
        }
