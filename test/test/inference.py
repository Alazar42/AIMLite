"""Document & Knowledge QA (RAG Paradigm): inference.py"""

from pathlib import Path
from typing import Any, Dict
from aimlite import BaseInference, Model


class RAGInference(BaseInference):
    """Production inference endpoint for semantic knowledge retrieval and answer synthesis."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.index_path = Path("artifacts") / "rag_index.json"

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
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
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
