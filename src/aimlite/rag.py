"""RAG Paradigm: aimlite/rag.py

First-class Retrieval-Augmented Generation abstractions:
Documents, Loaders, Text Splitters, Smart Chunkers, Embeddings, Chat Providers,
System Prompts, Query Analyzers, PostgreSQL & ORM Database Vector Stores,
Knowledge Models, Retrievers, and RAG Trainer Orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from aimlite.lifecycle import BaseTrainer
from aimlite.models import Model

if TYPE_CHECKING:
    from aimlite.data import Dataset


# =====================================================================
# 1. Pre-defined Standard System Prompts
# =====================================================================

PROMPT_RAG_QA = """You are a knowledgeable, factual, and helpful AI assistant.
Answer the user's query strictly based on the provided context passages.
- Cite your sources by referring to the document ID or source name (e.g. [1], [Source: filename]).
- If the context does not contain sufficient information to answer truthfully, state clearly that the answer is not available in the provided documents.
- Do not fabricate facts or hallucinate citations.
"""

PROMPT_QUERY_ANALYZER = """You are an expert Query Intelligence & Search Optimizer.
Your job is to analyze the user's search query and produce an optimized search strategy.
Deconstruct the query into:
1. Intent: The underlying user information need.
2. Keywords: Crucial search terms and entities.
3. Expanded Queries: 2-3 alternate search phrasings to maximize semantic recall.
4. Hypothetical Passage (HyDE): A concise, ideal answer snippet that would satisfy the query (used for vector embedding similarity matching).
5. Metadata Filters: Any inferred filters (category, topic, date, author) if discernible.

Respond in strict JSON format:
{
  "intent": "<user intent>",
  "keywords": ["<k1>", "<k2>"],
  "expanded_queries": ["<query_variation_1>", "<query_variation_2>"],
  "hypothetical_document": "<ideal answer passage>",
  "metadata_filters": {}
}
"""

PROMPT_SMART_CHUNKER = """You are a Document Structure & Semantic Segmentation Specialist.
Analyze the provided document text and divide it into coherent, self-contained semantic passages.
- Maintain topical coherence for each chunk.
- Generate a concise 1-sentence contextual summary or heading for each chunk to preserve document context.
- Keep chunk sizes balanced (200-600 words) without breaking mid-concept or mid-table.
"""

PROMPT_CONVERSATIONAL_RAG = """You are a conversational AI assistant with access to a knowledge base.
Maintain conversation context across user turns while answering questions grounded in the retrieved documentation.
If the user asks follow-up questions referencing previous messages, resolve pronouns and context accurately.
Always ground your factual assertions in the retrieved context passages.
"""


# =====================================================================
# 2. Document & Chunk Primitives
# =====================================================================

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
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Document:
        """Constructs Document instance from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            score=data.get("score"),
            embedding=data.get("embedding"),
        )


# =====================================================================
# 3. Text Splitters & Smart Chunkers
# =====================================================================

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


class SmartChunker:
    """Structure-aware and semantic document chunker.

    Understands Markdown headings (#, ##, ###), paragraph breaks, code blocks,
    and lists to divide text along natural semantic boundaries, while attaching
    parent section context and document hierarchy to chunk metadata.
    """

    def __init__(
        self,
        max_chunk_size: int = 600,
        min_chunk_size: int = 100,
        chunk_overlap: int = 60,
        preserve_markdown_headers: bool = True,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_markdown_headers = preserve_markdown_headers
        self.system_prompt = system_prompt or PROMPT_SMART_CHUNKER

    def _split_markdown_sections(self, text: str) -> List[Tuple[str, str]]:
        """Splits text into (section_header, section_body) pairs based on markdown headings."""
        lines = text.splitlines()
        sections: List[Tuple[str, List[str]]] = []
        current_header = "General"
        current_lines: List[str] = []

        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            match = header_pattern.match(line.strip())
            if match:
                if current_lines:
                    sections.append((current_header, current_lines))
                    current_lines = []
                current_header = match.group(2).strip()
            current_lines.append(line)

        if current_lines:
            sections.append((current_header, current_lines))

        return [(header, "\n".join(body_lines).strip()) for header, body_lines in sections if "\n".join(body_lines).strip()]

    def split_text(self, text: str, title: Optional[str] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """Splits text into chunks accompanied by contextual metadata."""
        text = text.strip()
        if not text:
            return []

        sections = self._split_markdown_sections(text) if self.preserve_markdown_headers else [("General", text)]
        results: List[Tuple[str, Dict[str, Any]]] = []

        for section_title, section_body in sections:
            # Split by paragraphs / double newlines first
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_body) if p.strip()]
            
            current_chunk_parts: List[str] = []
            current_len = 0

            for p in paragraphs:
                p_len = len(p)
                if current_len + p_len > self.max_chunk_size and current_chunk_parts:
                    chunk_text = "\n\n".join(current_chunk_parts).strip()
                    meta = {
                        "section": section_title,
                        "document_title": title,
                    }
                    results.append((chunk_text, meta))
                    current_chunk_parts = []
                    current_len = 0

                # If a single paragraph exceeds max_chunk_size, split by sentence or sliding window
                if p_len > self.max_chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", p)
                    sent_buf: List[str] = []
                    sent_len = 0
                    for s in sentences:
                        if sent_len + len(s) > self.max_chunk_size and sent_buf:
                            results.append(
                                (
                                    " ".join(sent_buf).strip(),
                                    {"section": section_title, "document_title": title},
                                )
                            )
                            sent_buf = []
                            sent_len = 0
                        sent_buf.append(s)
                        sent_len += len(s) + 1
                    if sent_buf:
                        current_chunk_parts.append(" ".join(sent_buf).strip())
                        current_len += sent_len
                else:
                    current_chunk_parts.append(p)
                    current_len += p_len + 2

            if current_chunk_parts:
                chunk_text = "\n\n".join(current_chunk_parts).strip()
                meta = {
                    "section": section_title,
                    "document_title": title,
                }
                results.append((chunk_text, meta))

        return results

    def split_documents(self, documents: Sequence[Document]) -> List[Document]:
        """Splits documents into contextualized chunks with enriched metadata."""
        chunked: List[Document] = []
        for doc in documents:
            doc_title = doc.metadata.get("title") or doc.metadata.get("filename") or doc.metadata.get("source")
            chunks_with_meta = self.split_text(doc.content, title=str(doc_title) if doc_title else None)
            for idx, (text, meta) in enumerate(chunks_with_meta):
                combined_meta = dict(doc.metadata)
                combined_meta.update(meta)
                combined_meta["chunk_index"] = idx
                combined_meta["parent_id"] = doc.id
                chunked.append(
                    Document(
                        content=text,
                        metadata=combined_meta,
                        id=f"{doc.id}-c{idx}",
                    )
                )
        return chunked


# =====================================================================
# 4. Document Loaders
# =====================================================================

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
            return [
                Document(
                    content=text,
                    metadata={"source": str(path), "filename": path.name, "title": path.stem},
                )
            ]

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
                            metadata={
                                "source": str(path),
                                "index": idx,
                                "title": item.get("title", f"{path.stem}_{idx}"),
                                **{k: v for k, v in item.items() if k not in ["text", "content"]},
                            },
                        )
                    )
            elif isinstance(data, dict):
                content = data.get("text") or data.get("content") or json.dumps(data)
                docs.append(
                    Document(
                        content=str(content),
                        metadata={"source": str(path), "title": data.get("title", path.stem), **data},
                    )
                )
            return docs

        elif ext == ".csv":
            import csv

            docs = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    row_content = ", ".join(f"{k}: {v}" for k, v in row.items())
                    docs.append(
                        Document(
                            content=row_content,
                            metadata={"source": str(path), "row": idx, "title": f"{path.stem}_row{idx}", **row},
                        )
                    )
            return docs

        # Fallback raw read
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [
            Document(
                content=text,
                metadata={"source": str(path), "filename": path.name, "title": path.stem},
            )
        ]

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


# =====================================================================
# 5. Embedding Models & Providers
# =====================================================================

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

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of text strings."""
        return [self.embed_text(t) for t in texts]


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


class SentenceTransformerEmbedding(BaseEmbedding):
    """Dense semantic embedding powered by local SentenceTransformers models with graceful pure-Python fallback."""

    def __init__(
        self,
        model_name_or_path: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        fallback_to_tfidf: bool = True,
    ) -> None:
        self.model_name = model_name_or_path
        self.device = device
        self.fallback_to_tfidf = fallback_to_tfidf
        self._fallback_tfidf: Optional[TfidfEmbedding] = None
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device)
            except (ImportError, Exception):
                if self.fallback_to_tfidf:
                    if self._fallback_tfidf is None:
                        self._fallback_tfidf = TfidfEmbedding()
                    return None
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerEmbedding. "
                    "Install it via 'aimlite install sentence-transformers' or 'pip install sentence-transformers'."
                )
        return self._model

    def embed_text(self, text: str) -> List[float]:
        model = self._get_model()
        if model is None:
            return self._fallback_tfidf.embed_text(text)  # type: ignore
        vec = model.encode(text, convert_to_numpy=True)
        return [float(x) for x in vec.tolist()]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        if model is None:
            return self._fallback_tfidf.embed_batch(texts)  # type: ignore
        vecs = model.encode(texts, convert_to_numpy=True)
        return [[float(x) for x in v.tolist()] for v in vecs]


class APIEmbedding(BaseEmbedding):
    """API-driven embedding engine supporting OpenAI, Ollama, and generic REST endpoints."""

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        provider: str = "openai",
        dim: int = 1536,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.provider = provider.lower()
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        batch = self.embed_batch([text])
        return batch[0] if batch else [0.0] * self.dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.provider == "openai":
            url = self.endpoint_url or "https://api.openai.com/v1/embeddings"
            payload = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    return [item["embedding"] for item in result.get("data", [])]
            except Exception:
                # Fallback to local TF-IDF if API is unreachable or not configured
                fallback = TfidfEmbedding(dim=self.dim)
                return [fallback.embed_text(t) for t in texts]

        elif self.provider == "ollama":
            url = self.endpoint_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if not url.endswith("/api/embeddings"):
                url = f"{url.rstrip('/')}/api/embeddings"
            embeddings = []
            for t in texts:
                payload = json.dumps({"model": self.model, "prompt": t}).encode("utf-8")
                req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                        embeddings.append(result.get("embedding", [0.0] * self.dim))
                except Exception as e:
                    raise ConnectionError(
                        f"Ollama Embedding Error: Failed to generate embeddings at '{url}'. "
                        f"Please ensure Ollama is running ('ollama serve') and model '{self.model}' is pulled ('ollama pull {self.model}'). "
                        f"Original error: {e}"
                    ) from e
            return embeddings

        raise ValueError(f"Unsupported embedding provider '{self.provider}'. Supported: 'openai', 'ollama', 'sentence-transformers', 'tfidf'.")


class OllamaEmbedding(BaseEmbedding):
    """Ollama REST API embedding client with direct connection verification."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: Optional[str] = None,
        dim: int = 768,
    ) -> None:
        self.model = model or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.dim = dim
        self._api = APIEmbedding(
            endpoint_url=self.host,
            model=self.model,
            provider="ollama",
            dim=self.dim,
        )

    def embed_text(self, text: str) -> List[float]:
        return self._api.embed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._api.embed_batch(texts)


# =====================================================================
# 6. Chat Provider Abstraction (Local & API LLMs)
# =====================================================================

class BaseChatProvider(ABC):
    """Abstract interface for LLM chat completion, text generation, and embedder hooks."""

    def __init__(
        self,
        model: str = "default",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or PROMPT_RAG_QA
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_kwargs = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """Generates a text completion for a single prompt."""
        ...

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Executes a multi-turn chat completion given message history."""
        # Default translation to generate() if not overridden
        sys_p = system_prompt or self.system_prompt
        full_prompt = f"System: {sys_p}\n\n"
        for m in messages:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            full_prompt += f"{role}: {content}\n"
        full_prompt += "Assistant: "
        return self.generate(full_prompt, system_prompt=sys_p, **kwargs)

    def get_embedder(self) -> BaseEmbedding:
        """Returns an associated BaseEmbedding instance for this provider."""
        return TfidfEmbedding()


class MockChatProvider(BaseChatProvider):
    """Deterministic, zero-dependency mock chat provider for tests, offline mode, and baseline benchmarks."""

    def __init__(
        self,
        model: str = "mock-gpt",
        system_prompt: Optional[str] = None,
        custom_responder: Optional[Callable[[str, Optional[str]], str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, system_prompt=system_prompt, **kwargs)
        self.custom_responder = custom_responder
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        sys_p = system_prompt or self.system_prompt
        self.call_history.append({"prompt": prompt, "system_prompt": sys_p, "kwargs": kwargs})

        if self.custom_responder:
            return self.custom_responder(prompt, sys_p)

        # Smart mock answering for JSON analysis prompts
        if "strict JSON format" in sys_p or "Query Intelligence" in sys_p:
            clean_q = prompt.split("Query:")[-1].strip() if "Query:" in prompt else prompt.strip()
            return json.dumps({
                "intent": f"Search intent for '{clean_q[:30]}'",
                "keywords": [w for w in re.findall(r"\w+", clean_q.lower()) if len(w) > 3][:4],
                "expanded_queries": [
                    f"Details about {clean_q}",
                    f"How to handle {clean_q}",
                ],
                "hypothetical_document": f"Relevant documentation passage explaining {clean_q} with operational guidance.",
                "metadata_filters": {},
            })

        return f"Grounded response from {self.model}: Processed request for '{prompt[:60]}...'"


class OpenAIChatProvider(BaseChatProvider):
    """API provider for OpenAI (GPT-4o, GPT-4o-mini, o1, o3, etc.) and OpenAI-compatible REST APIs."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        resolved_model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        super().__init__(
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                f"OpenAI Authentication Error ({self.model}): Missing API key. "
                "Please set OPENAI_API_KEY in your .env file or environment."
            )

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        chat_messages = []
        if system_prompt or (not any(m.get("role") == "system" for m in messages)):
            chat_messages.append({"role": "system", "content": system_prompt or self.system_prompt})
        chat_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            **self.extra_kwargs,
            **kwargs,
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(
                f"OpenAI API Error ({self.model}) [HTTP {e.code}]: {err_body}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"OpenAI Connection Error ({self.model}) at '{url}': {e}"
            ) from e

    def get_embedder(self) -> BaseEmbedding:
        return APIEmbedding(
            endpoint_url=f"{self.base_url}/embeddings",
            api_key=self.api_key,
            model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            provider="openai",
        )


class AnthropicChatProvider(BaseChatProvider):
    """API provider for Anthropic Claude (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        resolved_model = model or os.environ.get("ANTHROPIC_MODEL") or "claude-3-5-sonnet-20241022"
        super().__init__(
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                f"Anthropic Authentication Error ({self.model}): Missing API key. "
                "Please set ANTHROPIC_API_KEY in your .env file or environment."
            )

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "system": system_prompt or self.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(
                f"Anthropic API Error ({self.model}) [HTTP {e.code}]: {err_body}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Anthropic Connection Error ({self.model}): {e}"
            ) from e


class GeminiChatProvider(BaseChatProvider):
    """API provider for Google Gemini models (gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash)."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        resolved_model = model or os.environ.get("GEMINI_MODEL") or "gemini-1.5-flash"
        super().__init__(
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            raise ValueError(
                f"Gemini Authentication Error ({self.model}): Missing API key. "
                "Please set GEMINI_API_KEY in your .env file or environment."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        sys_instruction = system_prompt or self.system_prompt
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": sys_instruction}]},
            "generationConfig": {
                "temperature": self.temperature if temperature is None else temperature,
                "maxOutputTokens": self.max_tokens if max_tokens is None else max_tokens,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(
                f"Gemini API Error ({self.model}) [HTTP {e.code}]: {err_body}"
            ) from e
        except Exception as e:
            raise ConnectionError(
                f"Gemini Connection Error ({self.model}): {e}"
            ) from e


class OllamaChatProvider(BaseChatProvider):
    """Local LLM chat provider for Ollama (Llama 3, Mistral, Qwen, DeepSeek, Gemma)."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        resolved_model = model or os.environ.get("OLLAMA_MODEL") or "llama3.2"
        super().__init__(model=resolved_model, system_prompt=system_prompt, temperature=temperature, **kwargs)
        resolved_url = base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        self.base_url = resolved_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or self.system_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
            },
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            raise RuntimeError(
                f"Ollama API Error ({self.model}) [HTTP {e.code}]: {err_body}"
            ) from e
        except Exception as e:
            err_msg = str(e)
            if "111" in err_msg or "Connection refused" in err_msg or "urlopen error" in err_msg:
                raise ConnectionError(
                    f"Ollama Connection Error: Could not connect to Ollama server at '{self.base_url}'. "
                    f"Please make sure Ollama is running ('ollama serve') and model '{self.model}' is downloaded ('ollama pull {self.model}')."
                ) from e
            raise ConnectionError(f"Ollama Provider Error ({self.model}): {e}") from e

    def get_embedder(self) -> BaseEmbedding:
        return OllamaEmbedding(
            host=self.base_url,
            model=os.environ.get("EMBEDDING_MODEL", self.model),
        )


class LocalChatProvider(BaseChatProvider):
    """Provider for running local Hugging Face pipelines or custom callable models."""

    def __init__(
        self,
        model_or_fn: Any = None,
        model_name: str = "local-model",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model_name, system_prompt=system_prompt, temperature=temperature, **kwargs)
        self.model_or_fn = model_or_fn

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        sys_p = system_prompt or self.system_prompt
        combined_prompt = f"System: {sys_p}\nUser: {prompt}\nAssistant:"

        if callable(self.model_or_fn):
            return str(self.model_or_fn(combined_prompt))

        # Check if it's a Hugging Face pipeline
        if hasattr(self.model_or_fn, "__call__"):
            res = self.model_or_fn(
                combined_prompt,
                max_new_tokens=max_tokens or self.max_tokens,
                temperature=temperature or self.temperature,
            )
            if isinstance(res, list) and res and "generated_text" in res[0]:
                return res[0]["generated_text"]
            return str(res)

        return f"LocalChatProvider ({self.model}) answered: Processed query based on provided context."


# =====================================================================
# 7. Query Analyzer (Intent, Decomposition, Expansion & HyDE)
# =====================================================================

@dataclass
class QueryAnalysisResult:
    """Structured query analysis capturing intent, expanded queries, and HyDE documents."""

    original_query: str
    intent: str
    keywords: List[str] = field(default_factory=list)
    expanded_queries: List[str] = field(default_factory=list)
    hypothetical_document: Optional[str] = None
    metadata_filters: Dict[str, Any] = field(default_factory=dict)

    def all_search_queries(self) -> List[str]:
        """Returns all generated search query strings (original + expansions + HyDE)."""
        queries = [self.original_query]
        for q in self.expanded_queries:
            if q and q not in queries:
                queries.append(q)
        if self.hypothetical_document:
            queries.append(self.hypothetical_document)
        return queries


class QueryAnalyzer:
    """Intelligent query analyzer that decomposes, expands, and augments user search queries.

    Can operate using any `BaseChatProvider` (LLM-driven) or fallback to rule-based
    heuristic parsing when operating in offline / zero-dependency mode.
    """

    def __init__(
        self,
        chat_provider: Optional[BaseChatProvider] = None,
        system_prompt: Optional[str] = None,
        enable_hyde: bool = True,
        num_expansions: int = 2,
    ) -> None:
        self.chat_provider = chat_provider or MockChatProvider()
        self.system_prompt = system_prompt or PROMPT_QUERY_ANALYZER
        self.enable_hyde = enable_hyde
        self.num_expansions = num_expansions

    def _heuristic_analyze(self, query: str) -> QueryAnalysisResult:
        """Pure-Python heuristic keyword extractor and query expander."""
        words = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", query.lower())
        stop_words = {"how", "what", "where", "when", "why", "who", "the", "and", "for", "with", "does", "can", "are"}
        keywords = [w for w in words if w not in stop_words]

        expansions = [
            f"{' '.join(keywords)} explanation guide",
            f"overview documentation for {' '.join(keywords)}",
        ] if keywords else [query]

        hyde_doc = f"This document provides detailed information regarding {query}. It outlines the rules, steps, and procedures for implementation."

        return QueryAnalysisResult(
            original_query=query,
            intent=f"Retrieve information about {' '.join(keywords) if keywords else query}",
            keywords=keywords,
            expanded_queries=expansions[: self.num_expansions],
            hypothetical_document=hyde_doc if self.enable_hyde else None,
            metadata_filters={},
        )

    def analyze(self, query: str) -> QueryAnalysisResult:
        """Analyzes and optimizes a user query for RAG retrieval."""
        prompt = f"Query: {query}\n\nDeconstruct this query according to the instructions."

        try:
            raw_response = self.chat_provider.generate(prompt, system_prompt=self.system_prompt)
            # Try parsing JSON from LLM response
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return QueryAnalysisResult(
                    original_query=query,
                    intent=parsed.get("intent", f"Information request for {query}"),
                    keywords=parsed.get("keywords", []),
                    expanded_queries=parsed.get("expanded_queries", [])[: self.num_expansions],
                    hypothetical_document=parsed.get("hypothetical_document") if self.enable_hyde else None,
                    metadata_filters=parsed.get("metadata_filters", {}),
                )
        except Exception:
            pass

        return self._heuristic_analyze(query)


# =====================================================================
# 8. Vector Stores & PostgreSQL Database ORM Layer
# =====================================================================

class BaseVectorStore(ABC):
    """Abstract interface for storing document vectors and performing similarity retrieval."""

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Stores document vectors into the index."""
        ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Returns the top_k most similar Document instances ranked by cosine score."""
        ...

    def save(self, destination: Union[str, Path]) -> None:
        """Serializes vector store state to disk or external database."""
        ...

    def load(self, source: Union[str, Path]) -> None:
        """Restores vector store state from disk or database."""
        ...


class MemoryVectorStore(BaseVectorStore):
    """In-memory cosine similarity vector store with local JSON serialization."""

    def __init__(self, embedding_fn: Optional[BaseEmbedding] = None, **kwargs: Any) -> None:
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

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        scored: List[Tuple[float, Document]] = []
        for doc in self.documents:
            if filters:
                match = True
                for k, v in filters.items():
                    if doc.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            if doc.embedding is not None:
                sim = self._cosine_similarity(query_embedding, doc.embedding)
                scored.append((sim, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[Document] = []
        for sim, doc in scored[:top_k]:
            doc_copy = Document(
                id=doc.id,
                content=doc.content,
                metadata=dict(doc.metadata),
                embedding=doc.embedding,
                score=round(float(sim), 4),
            )
            results.append(doc_copy)
        return results

    def search(
        self,
        query: str,
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if not self.documents:
            return []
        query_embedding = self.embedding_fn.embed_text(query)
        return self.similarity_search(query_embedding, top_k=top_k, filters=filters)

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
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {}),
                    id=item.get("id", ""),
                    embedding=item.get("embedding"),
                )
                for item in data
            ]


# =====================================================================
# 8.1 PostgreSQL / Database ORM Vector Store
# =====================================================================

@dataclass
class KnowledgeDocumentRecord:
    """Database ORM record representing a knowledge document."""

    id: str
    title: str
    source: str
    content: str
    metadata_json: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class KnowledgeChunkRecord:
    """Database ORM record representing a chunk with vector embeddings."""

    id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: List[float]
    metadata_json: str


class PostgresVectorStore(BaseVectorStore):
    """PostgreSQL-backed vector store with support for pgvector and SQL ORM schema.

    Supports direct PostgreSQL connections (`psycopg2`, `asyncpg`, or `SQLAlchemy`),
    with native `<=>` cosine distance operations, metadata filtering, and strict database verification.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        db_url: Optional[str] = None,
        url: Optional[str] = None,
        table_name: str = "aimlite_knowledge_chunks",
        embedding_fn: Optional[BaseEmbedding] = None,
        vector_dim: int = 256,
        use_pgvector: bool = True,
        strict: bool = True,
        **kwargs: Any,
    ) -> None:
        conn = connection_string or db_url or url or os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or ""
        self.connection_string = conn
        self.table_name = table_name
        self.embedding_fn = embedding_fn or TfidfEmbedding(dim=vector_dim)
        self.vector_dim = vector_dim
        self.use_pgvector = use_pgvector
        self.strict = strict
        self._fallback_memory_store = MemoryVectorStore(embedding_fn=self.embedding_fn)
        self._db_initialized = False

    def init_db(self) -> None:
        """Initializes database schema and pgvector extension if PostgreSQL is connected."""
        if not self.connection_string:
            if self.strict:
                raise ValueError("PostgresVectorStore requires a valid 'db_url' or DATABASE_URL environment variable.")
            return

        try:
            import psycopg2  # type: ignore
        except ImportError as e:
            raise ImportError(
                "PostgreSQL driver 'psycopg2' is required for PostgresVectorStore. "
                "Install it via 'aimlite install psycopg2-binary' or 'pip install psycopg2-binary'."
            ) from e

        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    if self.use_pgvector:
                        try:
                            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                        except Exception:
                            conn.rollback()

                    cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id VARCHAR(64) PRIMARY KEY,
                            document_id VARCHAR(64),
                            chunk_index INT,
                            content TEXT NOT NULL,
                            metadata JSONB,
                            embedding {'vector(' + str(self.vector_dim) + ')' if self.use_pgvector else 'TEXT'},
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)
                    conn.commit()
            self._db_initialized = True
        except Exception as e:
            self._db_initialized = False
            raise ConnectionError(
                f"PostgreSQL Database Connection Error: Failed to connect to database at '{self.connection_string}'. "
                f"Please ensure PostgreSQL service is running and credentials/network are valid. Original error: {e}"
            ) from e

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Adds documents to PostgreSQL with strict database verification."""
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self.embedding_fn.embed_text(doc.content)

        if self.connection_string and not self._db_initialized:
            self.init_db()

        if self.connection_string and self._db_initialized:
            import psycopg2  # type: ignore

            try:
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor() as cur:
                        for doc in documents:
                            parent_id = doc.metadata.get("parent_id", doc.id)
                            chunk_idx = doc.metadata.get("chunk_index", 0)
                            meta_json = json.dumps(doc.metadata)
                            emb_val = str(doc.embedding) if not self.use_pgvector else f"[{','.join(map(str, doc.embedding))}]"

                            cur.execute(
                                f"""
                                INSERT INTO {self.table_name} (id, document_id, chunk_index, content, metadata, embedding)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (id) DO UPDATE SET
                                    content = EXCLUDED.content,
                                    metadata = EXCLUDED.metadata,
                                    embedding = EXCLUDED.embedding;
                                """,
                                (doc.id, parent_id, chunk_idx, doc.content, meta_json, emb_val),
                            )
                        conn.commit()
                return
            except Exception as e:
                raise RuntimeError(
                    f"PostgreSQL Storage Error: Failed to insert knowledge records into table '{self.table_name}': {e}"
                ) from e

        if self.strict:
            raise ConnectionError("PostgresVectorStore is uninitialized or missing DATABASE_URL connection string.")
        self._fallback_memory_store.add_documents(documents)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Performs vector similarity search via PostgreSQL."""
        if not self._db_initialized:
            self.init_db()

        if self.connection_string and self._db_initialized:
            import psycopg2  # type: ignore

            try:
                with psycopg2.connect(self.connection_string) as conn:
                    with conn.cursor() as cur:
                        if self.use_pgvector:
                            emb_str = f"[{','.join(map(str, query_embedding))}]"
                            cur.execute(
                                f"""
                                SELECT id, content, metadata, (1 - (embedding <=> %s::vector)) AS score
                                FROM {self.table_name}
                                ORDER BY embedding <=> %s::vector
                                LIMIT %s;
                                """,
                                (emb_str, emb_str, top_k),
                            )
                            rows = cur.fetchall()
                            results = []
                            for row in rows:
                                meta = row[2] if isinstance(row[2], dict) else json.loads(row[2] or "{}")
                                results.append(
                                    Document(
                                        id=row[0],
                                        content=row[1],
                                        metadata=meta,
                                        score=round(float(row[3]), 4),
                                    )
                                )
                            return results
            except Exception as e:
                raise RuntimeError(
                    f"PostgreSQL Search Error: Failed to execute vector similarity query on table '{self.table_name}': {e}"
                ) from e

        if self.strict:
            raise ConnectionError("PostgresVectorStore is uninitialized or missing DATABASE_URL connection string.")
        return self._fallback_memory_store.similarity_search(query_embedding, top_k=top_k, filters=filters)

    def save(self, destination: Union[str, Path]) -> None:
        self._fallback_memory_store.save(destination)

    def load(self, source: Union[str, Path]) -> None:
        self._fallback_memory_store.load(source)


# =====================================================================
# 9. Retrievers
# =====================================================================

class BaseRetriever(ABC):
    """Abstract retrieval hook mapping queries to ranked context documents."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 4, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieves top_k context documents relevant to the query."""
        ...


class VectorRetriever(BaseRetriever):
    """Retriever bound to a BaseVectorStore, BaseEmbedding, and optional QueryAnalyzer."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_fn: BaseEmbedding,
        query_analyzer: Optional[QueryAnalyzer] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        self.query_analyzer = query_analyzer

    def retrieve(self, query: str, top_k: int = 4, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        # If QueryAnalyzer is attached, execute multi-query / HyDE retrieval & fusion
        if self.query_analyzer is not None:
            analysis = self.query_analyzer.analyze(query)
            all_queries = analysis.all_search_queries()
            doc_scores: Dict[str, Tuple[Document, float]] = {}

            for q in all_queries:
                q_vec = self.embedding_fn.embed_text(q)
                docs = self.vector_store.similarity_search(
                    q_vec,
                    top_k=top_k,
                    filters=filters or analysis.metadata_filters,
                )
                for d in docs:
                    score = d.score if d.score is not None else 0.5
                    if d.id not in doc_scores or score > doc_scores[d.id][1]:
                        doc_scores[d.id] = (d, score)

            # Rank unified documents descending by score
            sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in sorted_docs[:top_k]]

        query_vec = self.embedding_fn.embed_text(query)
        return self.vector_store.similarity_search(query_vec, top_k=top_k, filters=filters)


# =====================================================================
# 10. High-Level KnowledgeModel & RAGModel
# =====================================================================

class RAGModel(Model):
    """Unified Model specialization for Retrieval-Augmented Generation.

    Extends AIMLite's core Model to enable seamless integration with the
    AIMLite lifecycle: predict(), save(), load(), get_dataset(), and CLI serve/train.
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
        chat_provider: Optional[BaseChatProvider] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        generator_fn: Optional[Callable[[str, List[Document]], str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self.retriever = retriever
        self.chat_provider = chat_provider or MockChatProvider()
        self.query_analyzer = query_analyzer
        self.generator_fn = generator_fn
        self.system_prompt = system_prompt or PROMPT_RAG_QA

    def build_context(self, documents: List[Document]) -> str:
        """Formats retrieved documents into a consolidated context string with citations."""
        parts = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("filename") or f"Doc {idx}"
            section = f" [{doc.metadata['section']}]" if "section" in doc.metadata else ""
            parts.append(f"[{idx}] (Source: {source}{section})\n{doc.content}")
        return "\n\n".join(parts)

    def default_generator(self, query: str, context_docs: List[Document]) -> str:
        """Synthesizes grounded response via ChatProvider or baseline context snippet."""
        if not context_docs:
            return "No relevant context documents found to answer the query."

        context_str = self.build_context(context_docs)
        prompt = f"Context Passages:\n{context_str}\n\nUser Question:\n{query}\n\nAnswer:"
        return self.chat_provider.generate(prompt, system_prompt=self.system_prompt)

    def predict(self, inputs: Any, top_k: int = 4, **kwargs: Any) -> Dict[str, Any]:
        """Executes forward RAG inference: Retrieval -> Grounded Generation -> Citations.

        Args:
            inputs: User query text string or dictionary payload.
            top_k: Number of context chunks to retrieve.

        Returns:
            Dictionary containing 'query', 'answer', and 'sources' list.
        """
        query_str = inputs.get("query", str(inputs)) if isinstance(inputs, dict) else str(inputs)
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


class KnowledgeModel(RAGModel):
    """Enterprise Knowledge Base model integrating PostgreSQL ORM, SmartChunker,

    Embedders, QueryAnalyzer, and ChatProviders into an autonomous knowledge engine.
    """

    def __init__(
        self,
        name: str = "knowledge_model",
        config: Optional[Dict[str, Any]] = None,
        chat_provider: Optional[BaseChatProvider] = None,
        embedding_fn: Optional[BaseEmbedding] = None,
        vector_store: Optional[BaseVectorStore] = None,
        chunker: Optional[SmartChunker] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        top_k: int = 3,
        **kwargs: Any,
    ) -> None:
        self.embedding_fn = embedding_fn or TfidfEmbedding()
        self.vector_store = vector_store or MemoryVectorStore(embedding_fn=self.embedding_fn)
        self.chunker = chunker or SmartChunker()
        self.chat_provider = chat_provider or MockChatProvider()
        self.query_analyzer = query_analyzer or QueryAnalyzer(chat_provider=self.chat_provider)
        self.retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedding_fn=self.embedding_fn,
            query_analyzer=self.query_analyzer,
        )

        super().__init__(
            name=name,
            config=config,
            retriever=self.retriever,
            chat_provider=self.chat_provider,
            query_analyzer=self.query_analyzer,
            **kwargs,
        )
        self.top_k = top_k

    def index_documents(self, documents: Sequence[Union[Document, str, Dict[str, Any]]]) -> int:
        """Processes, chunks, embeds, and stores documents into the vector database."""
        raw_docs: List[Document] = []
        for item in documents:
            if isinstance(item, Document):
                raw_docs.append(item)
            elif isinstance(item, dict):
                raw_docs.append(Document(content=item.get("content", str(item)), metadata=item.get("metadata", {})))
            else:
                raw_docs.append(Document(content=str(item)))

        # 1. Apply SmartChunker
        chunked = self.chunker.split_documents(raw_docs)

        # 2. Add to VectorStore (Postgres or Memory)
        self.vector_store.add_documents(chunked)
        return len(chunked)

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Persists vector store state to file or database checkpoint."""
        dest = Path(destination)
        if dest.suffix == ".json" or not dest.exists():
            self.vector_store.save(dest)
        else:
            self.vector_store.save(dest / "rag_index.json")

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Loads vector store state from checkpoint file."""
        src = Path(source)
        if src.is_file():
            self.vector_store.load(src)
        elif (src / "rag_index.json").is_file():
            self.vector_store.load(src / "rag_index.json")


# =====================================================================
# 11. Lifecycle RAG Trainer
# =====================================================================

class RAGTrainer(BaseTrainer):
    """AIMLite lifecycle trainer orchestrating knowledge ingestion, smart chunking,

    embedding generation, and vector index persistence into experiments/artifacts.
    """

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Builds semantic vector index from dataset documents."""
        if hasattr(dataset, "load_documents"):
            documents = dataset.load_documents()
        elif hasattr(dataset, "load"):
            records = dataset.load()
            documents = [
                Document(content=r.get("content", str(r)), metadata=r.get("metadata", {}))
                for r in records
            ]
        else:
            documents = []

        if not documents:
            return {"status": "failed", "error": "No documents found to index in dataset."}

        indexed_count = 0
        if hasattr(model, "index_documents"):
            indexed_count = model.index_documents(documents)
        elif hasattr(model, "vector_store"):
            model.vector_store.add_documents(documents)
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
