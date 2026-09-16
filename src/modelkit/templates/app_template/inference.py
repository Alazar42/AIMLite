"""ModelKit App: inference.py

Custom inference, endpoints, and serving configuration.
"""

from typing import Any, Dict
from modelkit import BaseInference, Model


class AppInference(BaseInference):
    """Application inference handler.

    You can customize input normalization, output formatting,
    and add custom HTTP endpoints.
    """

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Any:
        """Executes forward prediction and formats the payload output."""
        return model.predict(raw_input, **kwargs)

    def get_routes(self) -> Dict[str, Any]:
        """Optionally declare custom API routes (e.g. POST /predict, GET /status)."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
            # Custom developer endpoints:
            # "POST /api/v1/analyze": self.custom_analyze,
            # "GET /status": self.custom_status,
        }

    # Optional: If you want to serve your own custom frontend (React, Vue, Vite, etc.),
    # place build files in `dist/` or `frontend/dist/` (or run `modelkit serve --frontend <dir>`).
    # ModelKit will automatically serve your custom index.html at http://127.0.0.1:8000/!
