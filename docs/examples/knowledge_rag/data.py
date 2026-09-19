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

SAMPLE_KNOWLEDGE_DOCS = [
    {
        "filename": "auth_policy.md",
        "content": (
            "# Authentication and Security Policy\n\n"
            "AIMLite supports API key and Bearer token authentication. "
            "Session tokens expire after 24 hours of inactivity.\n\n"
            "## Multi-Factor Authentication\n\n"
            "Multi-factor authentication (MFA) is required for administrative access to production model endpoints."
        ),
    },
    {
        "filename": "deployment_guide.md",
        "content": (
            "# Production Deployment Guide\n\n"
            "AIMLite models can be served via 'aimlite serve --port 8000'. "
            "For production deployments, containerize using Docker with the provided Dockerfile.\n\n"
            "## Scaling\n\n"
            "Horizontal scaling can be achieved with Kubernetes by configuring the replica count."
        ),
    },
    {
        "filename": "adapter_tuning.md",
        "content": (
            "# Parameter-Efficient Fine-Tuning\n\n"
            "Adapter models use Low-Rank Adaptation (LoRA) to train lightweight delta matrices.\n\n"
            "## Efficiency Analytics\n\n"
            "This reduces checkpoint size from 14GB down to under 50MB, "
            "enabling rapid model swapping in multi-tenant environments."
        ),
    },
]


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
        src = source or data_path
        super().__init__(source=src, **kwargs)
        self.chunker = SmartChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Loads documents from disk or default knowledge base, chunking each file."""
        docs: List[Document] = self.load_documents()
        return [doc.to_dict() for doc in docs]

    def load_documents(self) -> List[Document]:
        """Returns parsed Document instances ready for indexing."""
        raw_documents: List[Document] = []
        data_dir = Path("data")

        # 1. Read files in data/ if available
        if data_dir.is_dir():
            text_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))
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

        # 2. Fall back to sample knowledge base if no files exist
        if not raw_documents:
            for item in SAMPLE_KNOWLEDGE_DOCS:
                raw_documents.append(
                    Document(
                        content=item["content"],
                        metadata={"source": item["filename"], "title": item["filename"].split(".")[0]},
                    )
                )

        return self.chunker.split_documents(raw_documents)
