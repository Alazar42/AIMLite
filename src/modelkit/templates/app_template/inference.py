"""ModelKit App: inference.py

Normalizes raw payloads and packages structured responses for serving.
"""

from typing import Any
from modelkit import BaseInference, Model


class AppInference(BaseInference):
    """Application inference handler."""

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Any:
        """Executes forward prediction and formats the payload output."""
        return model.predict(raw_input, **kwargs)
