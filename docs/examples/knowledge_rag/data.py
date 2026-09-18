"""Document & Knowledge QA (RAG Paradigm): data.py

Knowledge base document loader and text chunker for RAG pipelines.
Ingests text or markdown files from the data directory and splits them into
retrievable document passages with metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from aimlite import Dataset
from aimlite.rag import Document, TextSplitter

SAMPLE_KNOWLEDGE_DOCS = [
    {
        "filename": "auth_policy.md",
        "content": (
            "Authentication and Security Policy: AIMLite supports API key and Bearer token authentication. "
            "Session tokens expire after 24 hours of inactivity. Multi-factor authentication (MFA) is required "
            "for administrative access to production model endpoints."
        ),
    },
    {
        "filename": "deployment_guide.md",
        "content": (
            "Production Deployment Guide: AIMLite models can be served via 'aimlite serve --port 8000'. "
            "For production deployments, containerize using Docker with the provided Dockerfile. "
            "Horizontal scaling can be achieved with Kubernetes by configuring the replica count."
        ),
    },
    {
        "filename": "adapter_tuning.md",
        "content": (
            "Parameter-Efficient Fine-Tuning: Adapter models use Low-Rank Adaptation (LoRA) to train "
            "lightweight delta matrices. This reduces checkpoint size from 14GB down to under 50MB, "
            "enabling rapid model swapping in multi-tenant environments."
        ),
    },
]


class KnowledgeDocsDataset(Dataset):
    """Ingests documentation articles and splits them into retrievable passages."""

    filename: str = "knowledge_base.txt"

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(source=src, **kwargs)
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Loads documents from disk or default knowledge base, chunking each file."""
        docs: List[Document] = self.load_documents()
        return [doc.to_dict() for doc in docs]

    def load_documents(self) -> List[Document]:
        """Returns parsed Document instances ready for indexing."""
        documents: List[Document] = []
        data_dir = Path("data")

        # 1. Read files in data/ if available
        text_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))
        if text_files:
            for file_path in text_files:
                try:
                    text = file_path.read_text(encoding="utf-8")
                    chunks = self.splitter.split_text(text)
                    for idx, chunk in enumerate(chunks):
                        documents.append(
                            Document(
                                content=chunk,
                                metadata={"source": file_path.name, "chunk_index": idx},
                            )
                        )
                except Exception:
                    continue

        # 2. Fall back to sample knowledge base if no files exist
        if not documents:
            for item in SAMPLE_KNOWLEDGE_DOCS:
                chunks = self.splitter.split_text(item["content"])
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            content=chunk,
                            metadata={"source": item["filename"], "chunk_index": idx},
                        )
                    )

        return documents
