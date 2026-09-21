"""AIMLite App: model.py

Define your model architecture or pipeline here.
Subclasses automatically register under the hood without decorators.
You can inherit from:
- Model (standard / training from scratch)
- RAGModel (retrieval-augmented generation)
- AdapterModel (LoRA / fine-tuning)
"""

from typing import Any
from aimlite import Model


class AppModel(Model):
    """Application model definition."""

    def __init__(self, name: str = "app_model", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """Executes forward prediction across batch or single instance."""
        return inputs
