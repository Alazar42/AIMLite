"""AIMLite App: inference.py

Custom inference, endpoints, and serving configuration.
"""

from typing import Any, Dict
from aimlite import BaseInference, Model


class AppInference(BaseInference):
    """Application inference handler.

    You can customize input normalization, output formatting,
    and add custom HTTP endpoints.
    """

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Any:
        """Executes forward prediction and formats the payload output."""
        return model.predict(raw_input, **kwargs)

    def get_routes(self) -> Dict[str, Any]:
        """Optionally declare custom API routes (e.g. POST /predict, GET /health).
        
        You can expose additional custom endpoints here. When using Voxide AI
        (@voxide/react), you can register matching tools on the client:
          ai.register({
            customTool: {
              description: "Run custom analysis",
              params: { query: { type: "string" } },
              handler: async ({ query }) => {
                const res = await fetch("/api/v1/analyze", { ... });
                return res.json();
              }
            }
          });
        """
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
            # Custom developer endpoints & Voxide AI agent tools:
            # "POST /api/v1/analyze": self.custom_analyze,
            # "GET /status": self.custom_status,
        }

    # Optional: If you want to serve your own custom frontend (React, Vue, Vite, etc.),
    # place build files in `dist/` or `frontend/dist/` (or run `aimlite serve --frontend <dir>`).
    # AIMLite will automatically serve your custom index.html at http://127.0.0.1:8000/!
