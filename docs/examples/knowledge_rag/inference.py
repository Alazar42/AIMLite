"""Document & Knowledge QA (RAG Paradigm): inference.py

Production inference endpoint for semantic document query retrieval
and context-grounded response generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aimlite import BaseInference, Model


class RAGInference(BaseInference):
    """Production inference handler for semantic knowledge retrieval."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.index_path = Path("artifacts") / "rag_index.json"

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        """Runs question-answering over indexed knowledge base documents."""
        if self.index_path.is_file():
            model.load(self.index_path)

        query = raw_input.get("query", "") if isinstance(raw_input, dict) else str(raw_input)
        if not query:
            return {"error": "Missing 'query' field in request body", "status": "failed"}

        result = model.predict(query)
        return {
            **result,
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        """Declares HTTP route mappings for AIMLite server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
