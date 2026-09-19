"""Document & Knowledge QA (RAG Paradigm): data.py"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from aimlite import Dataset
from aimlite.rag import Document, DocumentLoader, SmartChunker


class KnowledgeDocsDataset(Dataset):
    """Ingests documentation articles, datasets, and custom user files into chunked passages."""

    filename: str = "knowledge_base.md"

    def __init__(
        self,
        source: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        max_chunk_size: int = 400,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        src = source or data_path or "data"
        super().__init__(source=src, **kwargs)
        self.chunker = SmartChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        self.additional_documents: List[Document] = []

    def add_document(self, content: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Dynamically appends a new document or user-populated text record to the dataset."""
        meta = metadata or {}
        if title:
            meta["title"] = title
        self.additional_documents.append(Document(content=content, metadata=meta))

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        docs = self.load_documents()
        return [doc.to_dict() for doc in docs]

    def load_documents(self) -> List[Document]:
        """Loads and chunks all prepared files from the data directory and user-added documents."""
        raw_documents: List[Document] = list(self.additional_documents)
        data_target = Path(self.source) if self.source else Path("data")

        valid_extensions = {".md", ".txt", ".markdown", ".rst", ".json", ".csv"}

        if data_target.is_file():
            raw_documents.extend(DocumentLoader.load_file(data_target))
        elif data_target.is_dir():
            for p in sorted(data_target.rglob("*")):
                if p.is_file() and p.suffix.lower() in valid_extensions and not p.name.startswith("."):
                    try:
                        raw_documents.extend(DocumentLoader.load_file(p))
                    except Exception:
                        continue

        if not raw_documents:
            raise ValueError(
                f"No knowledge documents found in target path '{data_target}'. "
                "Please place your .md, .txt, .json, or .csv files inside the 'data/' folder or pass a custom path."
            )

        return self.chunker.split_documents(raw_documents)
