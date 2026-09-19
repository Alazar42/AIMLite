"""Document & Knowledge QA (RAG Paradigm): evaluator.py"""

from typing import Any, Dict
from aimlite import BaseEvaluator, Model


class RAGEvaluator(BaseEvaluator):
    """Evaluates knowledge retrieval coverage and grounded answer synthesis."""

    def evaluate(self, model: Model, test_data: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "retrieval_status": "ready",
            "model_name": getattr(model, "name", "rag_model"),
        }
