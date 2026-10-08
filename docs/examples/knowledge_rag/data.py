"""Document & Knowledge QA (RAG Paradigm): data.py

Knowledge base document loader and smart chunker for RAG pipelines.
Ingests text or markdown files and splits them into semantically coherent,
retrievable document passages enriched with header and document metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from aimlite import Dataset
from aimlite.rag import Document, SmartChunker


class KnowledgeDocsDataset(Dataset):
    """Ingests documentation articles and splits them into retrievable passages using SmartChunker."""

    filename: str = "knowledge_base.txt"

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        max_chunk_size: int = 400,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        src = source or data_path or self.filename
        super().__init__(source=src, **kwargs)
        self.chunker = SmartChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Parses documents and returns chunk records compatible with AIMLite Dataset."""
        docs = self.load_documents()
        return [
            {
                "id": d.id,
                "content": d.content,
                "metadata": d.metadata,
                "source": d.metadata.get("source", "unknown"),
            }
            for d in docs
        ]

    def load_documents(self) -> List[Document]:
        """Loads and parses documents from data/ and splits with SmartChunker."""
        raw_documents: List[Document] = []
        data_dir = self._get_data_dir() if hasattr(self, "_get_data_dir") else Path("data")

        # 1. Check for single resolved file source
        resolved = self._resolve_file_path(self.source)
        if resolved and resolved.is_file():
            text = resolved.read_text(encoding="utf-8")
            raw_documents.append(
                Document(
                    content=text,
                    metadata={"source": resolved.name, "title": resolved.stem},
                )
            )
        elif data_dir.is_dir():
            # Ingest all text and markdown files in data/
            text_files = sorted(
                list(data_dir.glob("*.txt"))
                + list(data_dir.glob("*.md"))
                + list(data_dir.glob("*.markdown"))
            )
            for file_path in text_files:
                try:
                    text = file_path.read_text(encoding="utf-8")
                    raw_documents.append(
                        Document(
                            content=text,
                            metadata={"source": file_path.name, "title": file_path.stem},
                        )
                    )
                except Exception:
                    continue

        if not raw_documents:
            raise FileNotFoundError(
                "No knowledge documents found in data/. "
                "Please place text (.txt) or markdown (.md) documents into data/ to build the vector index."
            )

        return self.chunker.split_documents(raw_documents)
