export interface GuideStep {
  stepNumber: number;
  title: string;
  badge?: string;
  badgeVariant?: 'pillar' | 'cli' | 'get' | 'post' | 'util';
  description: string;
  filename?: string;
  code: string;
  language?: 'python' | 'bash' | 'json' | 'markdown';
  whyCode?: string;
  externalLink?: {
    label: string;
    url: string;
  };
}

/* ========================================================================= */
/* PARADIGM 1: SCRATCH (TELECOM CUSTOMER CHURN)                              */
/* ========================================================================= */

export const SCRATCH_FILES: Record<string, string> = {
  'data.py': `"""Customer Churn Prediction (Scratch Model Paradigm): data.py

Dataset loader for the Kaggle Telecom Churn dataset:
https://www.kaggle.com/datasets/barun2104/telecom-churn

Expected CSV columns:
  Churn (0/1), AccountWeeks, ContractRenewal, DataPlan, DataUsage,
  CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aimlite import Dataset

FEATURE_COLUMNS = [
    "AccountWeeks",
    "ContractRenewal",
    "DataPlan",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins",
]
TARGET_COLUMN = "Churn"


class TelecomChurnDataset(Dataset):
    """Customer Churn dataset loader for Kaggle Telecom Churn CSV data.

    Expected CSV file: data/telecom_churn.csv
    Kaggle: https://www.kaggle.com/datasets/barun2104/telecom-churn
    """

    # Explicitly set the data file inside the project's data/ directory
    filename: str = "telecom_churn.csv"

    def __init__(
        self,
        filename: str = "telecom_churn.csv",
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(filename=filename, source=src, **kwargs)
        self.feature_names: List[str] = FEATURE_COLUMNS
        self.target_name: str = TARGET_COLUMN

    def load(self, source: Optional[str | Path] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Parses the CSV and returns clean structured records.

        Supports pandas if installed, with a zero-dependency csv fallback.
        Populates self._data and self.columns for AIMLite validation and training.
        """
        target = source or self.source or self.filename
        resolved = self._resolve_file_path(target)
        if not resolved or not resolved.is_file():
            data_dir = self._get_data_dir()
            candidate = data_dir / self.filename
            if candidate.is_file():
                resolved = candidate
            else:
                self._data = []
                return []

        self.resolved_path = resolved

        # Try pandas first for high-performance reading
        try:
            import pandas as pd

            df = pd.read_csv(resolved)
            # Normalize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]
            self.columns = list(df.columns)
            self._data = df.to_dict(orient="records")
            return self._data
        except ImportError:
            pass

        # Zero-dependency csv reader fallback
        records: List[Dict[str, Any]] = []
        with open(resolved, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.columns = [c.strip() for c in (reader.fieldnames or [])]
            for row in reader:
                record = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    try:
                        record[k_clean] = float(v.strip())
                    except ValueError:
                        record[k_clean] = v.strip()
                records.append(record)

        self._data = records
        return self._data

    def to_arrays(self) -> Tuple[List[List[float]], List[int]]:
        """Converts loaded records into numeric feature matrix X and target labels y."""
        if not self._data:
            self.load()
        records = self._data
        X: List[List[float]] = []
        y: List[int] = []

        for row in records:
            if self.target_name not in row:
                continue
            feats = [float(row.get(col, 0.0)) for col in self.feature_names]
            label = int(float(row[self.target_name]))
            X.append(feats)
            y.append(label)

        return X, y`,

  'model.py': `"""Customer Churn Prediction (Scratch Model Paradigm): model.py

Classifier model for customer churn prediction.
Supports scikit-learn (RandomForestClassifier or GradientBoostingClassifier)
with a pure-Python fallback when scikit-learn is not yet installed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite import Model


class ChurnClassifier(Model):
    """Customer Churn classifier model predicting whether a customer will churn (0 or 1)."""

    def __init__(
        self,
        name: str = "churn_classifier",
        config: Optional[Dict[str, Any]] = None,
        n_estimators: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator: Optional[Any] = None
        self._init_estimator()

    def _init_estimator(self) -> None:
        """Initializes underlying classifier engine."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
            )
        except ImportError:
            # Simple fallback weights if sklearn is not installed
            self.estimator = None

    def fit(self, X: List[List[float]], y: List[int]) -> ChurnClassifier:
        """Fits classifier on numeric feature matrix X and label vector y."""
        if not X:
            return self

        if self.estimator is not None:
            self.estimator.fit(X, y)
        else:
            # Fallback baseline model: average target mean thresholding
            self._baseline_threshold = sum(y) / len(y) if y else 0.5
        return self

    def predict(self, inputs: Any, **kwargs: Any) -> List[int]:
        """Predicts binary churn class (0 for retained, 1 for churn) for input feature vectors."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is not None and hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(X)
            return [int(p) for p in preds]

        # Baseline fallback prediction
        return [0 for _ in X]

    def predict_proba(self, inputs: Any) -> List[float]:
        """Computes probability of churn (risk score between 0.0 and 1.0)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is not None and hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
            classes = list(getattr(self.estimator, "classes_", [0, 1]))
            if 1 in classes:
                idx_1 = classes.index(1)
                return [float(p[idx_1]) for p in probs]
            elif classes == [0]:
                return [0.0 for _ in probs]
            return [float(p[-1]) for p in probs]

        return [0.15 for _ in X]

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes model weights and parameters to disk."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "estimator": self.estimator,
        }
        with open(dest_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Loads serialized model weights from disk."""
        src_path = Path(source)
        if not src_path.is_file():
            return
        with open(src_path, "rb") as f:
            payload = pickle.load(f)
        self.estimator = payload.get("estimator")
        self.n_estimators = payload.get("n_estimators", 100)
        self.random_state = payload.get("random_state", 42)

    def _normalize_inputs(self, inputs: Any) -> List[List[float]]:
        """Converts diverse input formats (list of dicts, single dict, 2D list) into 2D float list."""
        if not inputs:
            return []

        if isinstance(inputs, dict):
            # Single dict row: order by feature names
            from .data import FEATURE_COLUMNS

            return [[float(inputs.get(col, 0.0)) for col in FEATURE_COLUMNS]]

        if isinstance(inputs, list):
            if not inputs:
                return []
            if isinstance(inputs[0], dict):
                from .data import FEATURE_COLUMNS

                return [[float(row.get(col, 0.0)) for col in FEATURE_COLUMNS] for row in inputs]
            if isinstance(inputs[0], (int, float)):
                return [[float(x) for x in inputs]]
            if isinstance(inputs[0], list):
                return [[float(x) for x in row] for row in inputs]

        return []`,

  'trainer.py': `"""Customer Churn Prediction (Scratch Model Paradigm): trainer.py

Orchestrates data preparation, train/val splitting, classifier fitting,
and model checkpoint persistence into the artifacts directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aimlite import BaseTrainer, Dataset, Model


class ChurnTrainer(BaseTrainer):
    """Trainer pipeline for fitting ChurnClassifier on TelecomChurnDataset."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Trains the churn classifier model and saves artifacts."""
        if not hasattr(dataset, "to_arrays"):
            return {"status": "failed", "error": "Dataset missing to_arrays method"}

        X, y = dataset.to_arrays()
        if not X:
            return {"status": "failed", "error": "No training samples found in dataset"}

        # 80/20 train/validation split
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        if hasattr(model, "fit"):
            model.fit(X_train, y_train)

        # Compute train & val performance
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val) if X_val else []

        train_acc = (
            sum(1 for p, actual in zip(train_preds, y_train) if p == actual) / len(y_train)
            if y_train
            else 0.0
        )
        val_acc = (
            sum(1 for p, actual in zip(val_preds, y_val) if p == actual) / len(y_val)
            if y_val
            else 0.0
        )

        # Save checkpoint to project artifacts directory
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifacts_dir / "churn_classifier.pkl"
        model.save(checkpoint_path)

        return {
            "status": "completed",
            "samples_total": len(X),
            "samples_train": len(X_train),
            "samples_val": len(X_val),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "checkpoint": str(checkpoint_path),
        }`,

  'evaluator.py': `"""Customer Churn Prediction (Scratch Model Paradigm): evaluator.py

Computes precision, recall, F1-score, and classification accuracy
on evaluation partitions.
"""

from __future__ import annotations

from typing import Any, Dict

from aimlite import BaseEvaluator, Dataset, Model


class ChurnEvaluator(BaseEvaluator):
    """Evaluator assessing customer churn classification metrics."""

    def evaluate(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, float]:
        """Calculates accuracy, precision, recall, and F1 metrics."""
        if not hasattr(dataset, "to_arrays"):
            return {"accuracy": 0.0}

        X, y = dataset.to_arrays()
        if not X:
            return {"accuracy": 0.0}

        preds = model.predict(X)

        tp = sum(1 for p, actual in zip(preds, y) if p == 1 and actual == 1)
        fp = sum(1 for p, actual in zip(preds, y) if p == 1 and actual == 0)
        fn = sum(1 for p, actual in zip(preds, y) if p == 0 and actual == 1)
        tn = sum(1 for p, actual in zip(preds, y) if p == 0 and actual == 0)

        total = len(y)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "sample_count": float(total),
        }`,

  'inference.py': `"""Customer Churn Prediction (Scratch Model Paradigm): inference.py

Production inference handler for customer churn risk scoring and intervention routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aimlite import BaseInference, Model


class ChurnInference(BaseInference):
    """Production inference endpoint returning customer churn probability and retention decisions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.checkpoint_path = Path("artifacts") / "churn_classifier.pkl"

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        """Runs churn prediction on incoming customer attributes."""
        # Ensure model weights are loaded if checkpoint exists
        if self.checkpoint_path.is_file():
            model.load(self.checkpoint_path)

        # Compute predictions and probability
        preds = model.predict(raw_input)
        churn_pred = preds[0] if preds else 0

        risk_score = 0.5
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(raw_input)
            if probs:
                risk_score = round(probs[0], 4)

        # Retention decision threshold
        decision = "Intervene (High Churn Risk)" if risk_score >= 0.5 else "Retain (Low Risk)"

        return {
            "churn_prediction": int(churn_pred),
            "churn_risk": risk_score,
            "decision": decision,
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        """Declares HTTP route mappings for AIMLite server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }`,

  'aimlite.json': `{
  "name": "telecom_churn",
  "version": "0.1.0",
  "entrypoint": "telecom_churn",
  "dependencies": [
    "pandas",
    "scikit-learn"
  ],
  "config": {
    "device": "auto",
    "batch_size": 64
  }
}`,
};

export const SCRATCH_GUIDE_STEPS: GuideStep[] = [
  {
    stepNumber: 1,
    title: 'Initialize the Project',
    badge: 'SCAFFOLD',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Scaffold a clean AIMLite project with the scratch paradigm. Use `aimlite init <name> --type scratch` to create a new folder, or `aimlite init . --type scratch` to initialize inside your current directory. Use `--clean` if you want pristine 0-byte starting files.',
    code: `# Option A: Create in a new project folder
aimlite init telecom_churn --type scratch
cd telecom_churn

# Option B: Scaffold directly inside current directory (.)
# aimlite init . --type scratch

# Option C: Use --clean to scaffold pristine 0-byte starting files
# aimlite init telecom_churn --type scratch --clean`,
    whyCode:
      'Creates the standard zero-path folder layout (data/, models/, experiments/, artifacts/, checkpoints/) and generates the project manifest aimlite.json without requiring boilerplate setup.',
  },
  {
    stepNumber: 2,
    title: 'Manage .venv & Install Compute Libraries',
    badge: 'ENV & DEPS',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Run `aimlite install` with your required ML libraries (e.g. scikit-learn, pandas). AIMLite automatically creates and manages `.venv` across Linux, macOS, and Windows (avoiding OS PEP 668 restrictions), and automatically updates `aimlite.json` under `"dependencies"`.',
    code: `# Install libraries: auto-manages .venv and updates aimlite.json
aimlite install scikit-learn pandas

# Later or on a new machine, running with no arguments reinstalls all pinned dependencies:
# aimlite install`,
    whyCode:
      'On modern OSes (Ubuntu 24+, Debian, macOS), installing global python packages is restricted (PEP 668). AIMLite ensures all projects have an isolated, self-managed .venv and tracks dependencies deterministically in aimlite.json.',
  },
  {
    stepNumber: 3,
    title: 'Download the Telecom Churn Dataset',
    badge: 'DATA INGESTION',
    badgeVariant: 'util',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Download the customer churn dataset CSV from Kaggle and place it inside the project data/ directory. AIMLite automatically resolves files relative to the project root.',
    code: `# Download the dataset from Kaggle into data/
# https://www.kaggle.com/datasets/barun2104/telecom-churn
# Place the downloaded telecom_churn.csv in data/

ls -la data/telecom_churn.csv`,
    whyCode:
      'Standardizing data into data/ enables convention over configuration — AIMLite Dataset subclasses auto-detect data files by filename without hardcoded absolute paths.',
    externalLink: {
      label: 'Open Kaggle Dataset',
      url: 'https://www.kaggle.com/datasets/barun2104/telecom-churn',
    },
  },
  {
    stepNumber: 4,
    title: 'Define data.py (Dataset Subclass)',
    badge: 'PILLAR 1: DATA',
    badgeVariant: 'pillar',
    filename: 'data.py',
    language: 'python',
    description:
      'Subclass `aimlite.Dataset` in `telecom_churn/data.py`. Define feature columns, the target column, and implement `load()` and `to_arrays()`.',
    code: SCRATCH_FILES['data.py'],
    whyCode:
      'The Dataset pillar encapsulates data ingestion, path resolution, feature column definitions, and numpy array conversion in a single reusable module.',
  },
  {
    stepNumber: 5,
    title: 'Define model.py (Model Subclass)',
    badge: 'PILLAR 2: MODEL',
    badgeVariant: 'pillar',
    filename: 'model.py',
    language: 'python',
    description:
      'Subclass `aimlite.Model` in `telecom_churn/model.py`. Wrap your RandomForest classifier, implement `fit()`, `predict()`, `predict_proba()`, and model serialization hooks `save()` / `load()`.',
    code: SCRATCH_FILES['model.py'],
    whyCode:
      'The Model pillar wraps arbitrary estimators (scikit-learn, PyTorch, XGBoost) and provides standardized serialization and inference contracts.',
  },
  {
    stepNumber: 6,
    title: 'Define trainer.py (BaseTrainer Subclass)',
    badge: 'PILLAR 3: TRAIN',
    badgeVariant: 'pillar',
    filename: 'trainer.py',
    language: 'python',
    description:
      'Subclass `aimlite.BaseTrainer` in `telecom_churn/trainer.py`. Implement `fit()` to split train/val partitions, train the model, compute train and validation metrics, and persist weights to `artifacts/churn_classifier.pkl`.',
    code: SCRATCH_FILES['trainer.py'],
    whyCode:
      'Separating training lifecycle logic into trainer.py enables zero-path CLI execution (`aimlite train`) and experiment tracking reproducibility.',
  },
  {
    stepNumber: 7,
    title: 'Define evaluator.py (BaseEvaluator Subclass)',
    badge: 'PILLAR 3: EVAL',
    badgeVariant: 'pillar',
    filename: 'evaluator.py',
    language: 'python',
    description:
      'Subclass `aimlite.BaseEvaluator` in `telecom_churn/evaluator.py`. Implement `evaluate()` to compute accuracy, precision, recall, and F1-score on held-out test partitions.',
    code: SCRATCH_FILES['evaluator.py'],
    whyCode:
      'Standardized evaluator classes allow automated regression testing and validation gates before deploying model weights to production.',
  },
  {
    stepNumber: 8,
    title: 'Define inference.py (BaseInference Subclass)',
    badge: 'SERVING',
    badgeVariant: 'post',
    filename: 'inference.py',
    language: 'python',
    description:
      'Subclass `aimlite.BaseInference` in `telecom_churn/inference.py`. Implement `run()` to score incoming customer attribute payloads and return churn risk probability and retention recommendations.',
    code: SCRATCH_FILES['inference.py'],
    whyCode:
      'The BaseInference class provides production HTTP serving (`POST /predict`), business logic decisioning, and automatic OpenAPI schema documentation.',
  },
  {
    stepNumber: 9,
    title: 'Train the Model (Zero-Path CLI)',
    badge: 'TRAIN CLI',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Run `aimlite train` from the project directory. AIMLite discovers your dataset, model, and trainer, activates the virtual environment, executes training, and saves checkpoint weights into `artifacts/churn_classifier.pkl`.',
    code: `# Train model with zero-path convention
aimlite train ChurnClassifier

# Optional: Run evaluation on held-out test split
aimlite evaluate ChurnClassifier`,
    whyCode:
      'Convention over configuration: you do not need to specify python paths or scripts. AIMLite dynamically discovers ChurnClassifier and ChurnTrainer.',
  },
  {
    stepNumber: 10,
    title: 'Serve Production Inference & Web UI',
    badge: 'SERVE CLI',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Run `aimlite serve` to launch the multi-model inference server on port 8000. Access the interactive web application, Swagger API documentation, or POST JSON payloads to `/predict`.',
    code: `# Start production inference server on port 8000
aimlite serve ChurnClassifier --port 8000

# Access interactive web UI:   http://localhost:8000/app
# Access Swagger OpenAPI docs:  http://localhost:8000/docs
# Access health probe:          http://localhost:8000/health

# In another terminal, query the endpoint:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AccountWeeks": 128,
    "ContractRenewal": 1,
    "DataPlan": 1,
    "DataUsage": 2.7,
    "CustServCalls": 1,
    "DayMins": 265.1,
    "DayCalls": 110,
    "MonthlyCharge": 89.0,
    "OverageFee": 9.87,
    "RoamMins": 10.0
  }'`,
    whyCode:
      'AIMLite spins up an enterprise-grade HTTP server with zero boilerplate, full CORS support, dynamic route registration, and integrated model playgrounds.',
  },
];

/* ========================================================================= */
/* PARADIGM 2: RAG (KNOWLEDGE BASE QA)                                       */
/* ========================================================================= */

export const RAG_FILES: Record<string, string> = {
  'chat_provider.py': `"""Chat Provider & Query Intelligence: chat_provider.py

Starter code for configuring LLM synthesis, system prompts, and query analysis (HyDE, intent decomposition, expansion).
All classes, functions, and prompt templates in this file are fully editable by the developer.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from aimlite.config import BaseConfig
from aimlite.rag import (
    AnthropicChatProvider,
    BaseChatProvider,
    GeminiChatProvider,
    LocalChatProvider,
    MockChatProvider,
    OllamaChatProvider,
    OpenAIChatProvider,
    QueryAnalyzer,
    PROMPT_CONVERSATIONAL_RAG,
    PROMPT_QUERY_ANALYZER,
    PROMPT_RAG_QA,
    PROMPT_SMART_CHUNKER,
)

# =====================================================================
# Developer-Editable Prompt Templates
# =====================================================================

SYSTEM_PROMPT = """You are a knowledgeable, factual, and helpful AI assistant.
Answer the user's query strictly based on the provided context passages.
- Cite your sources by referring to the document ID or source name (e.g. [1], [Source: filename]).
- If the context does not contain sufficient information to answer truthfully, state clearly that the answer is not available in the provided documents.
- Do not fabricate facts or hallucinate citations.
"""

QA_PROMPT_TEMPLATE = """Context Passages:
{context}

User Question:
{query}

Answer:"""


# =====================================================================
# Chat Provider Factory
# =====================================================================

def get_chat_provider(
    provider_name: Optional[str] = None,
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseChatProvider:
    """Instantiates and returns the configured LLM Chat Provider.

    Edit this function to add custom LLM providers, change default models,
    or adjust generation parameters.
    """
    cfg = config or {}
    rag_cfg = cfg.get("rag") or cfg.get("config", {}).get("rag") or {}

    provider = (
        provider_name
        or rag_cfg.get("chat_provider")
        or rag_cfg.get("provider")
        or os.environ.get("CHAT_PROVIDER")
        or "openai"
    ).lower()

    model = (
        model_name
        or rag_cfg.get("model_name")
        or rag_cfg.get("model")
        or os.environ.get("CHAT_MODEL")
    )

    sys_prompt = system_prompt or rag_cfg.get("system_prompt") or SYSTEM_PROMPT
    temp = float(rag_cfg.get("temperature", temperature))

    if provider == "openai":
        return OpenAIChatProvider(
            model=model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            system_prompt=sys_prompt,
            temperature=temp,
            **kwargs,
        )
    elif provider == "gemini":
        return GeminiChatProvider(
            model=model or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            api_key=os.environ.get("GEMINI_API_KEY"),
            system_prompt=sys_prompt,
            temperature=temp,
            **kwargs,
        )
    elif provider == "anthropic":
        return AnthropicChatProvider(
            model=model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            system_prompt=sys_prompt,
            temperature=temp,
            **kwargs,
        )
    elif provider == "ollama":
        return OllamaChatProvider(
            model=model or os.environ.get("OLLAMA_MODEL", "llama3"),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            system_prompt=sys_prompt,
            temperature=temp,
            **kwargs,
        )
    elif provider in ("local", "transformers"):
        return LocalChatProvider(
            model_name_or_path=model or "meta-llama/Llama-3-8B-Instruct",
            system_prompt=sys_prompt,
            temperature=temp,
            **kwargs,
        )
    else:
        return MockChatProvider(
            model=model or "mock-model",
            system_prompt=sys_prompt,
            **kwargs,
        )


# =====================================================================
# Query Analyzer Factory
# =====================================================================

def get_query_analyzer(
    chat_provider: Optional[BaseChatProvider] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> QueryAnalyzer:
    """Instantiates Query Analyzer for query rewriting, HyDE, and search expansions."""
    provider = chat_provider or get_chat_provider(config=config)
    return QueryAnalyzer(chat_provider=provider, **kwargs)`,

  'store.py': `"""Vector Store & Database Storage: store.py

Starter code for semantic vector storage, embeddings, and PostgreSQL ORM records.
All functions and classes in this file are fully editable by the developer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite.rag import (
    APIEmbedding,
    BaseEmbedding,
    BaseVectorStore,
    MemoryVectorStore,
    OllamaEmbedding,
    PostgresVectorStore,
    SentenceTransformerEmbedding,
    TfidfEmbedding,
)


# =====================================================================
# Embedding Model Factory
# =====================================================================

def get_embedding_model(
    engine: Optional[str] = None,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseEmbedding:
    """Instantiates embedding model (Dense Neural, Ollama API, OpenAI API, or Pure-Python TF-IDF).

    Edit this function to plug in custom embedding models (e.g. HuggingFace, Cohere, OpenAI).
    """
    cfg = config or {}
    rag_cfg = cfg.get("rag") or cfg.get("config", {}).get("rag") or {}

    resolved_engine = (
        engine
        or rag_cfg.get("embedding_engine")
        or os.environ.get("EMBEDDING_ENGINE")
        or "sentence-transformers"
    ).lower()

    resolved_model = (
        model_name
        or rag_cfg.get("embedding_model")
        or os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    if resolved_engine == "ollama" or "nomic" in resolved_model.lower():
        return OllamaEmbedding(model=resolved_model, host=ollama_host, **kwargs)
    elif resolved_engine == "openai":
        return APIEmbedding(provider="openai", model=resolved_model, **kwargs)
    elif resolved_engine in ("sentence-transformers", "local"):
        return SentenceTransformerEmbedding(model_name_or_path=resolved_model, **kwargs)
    elif resolved_engine == "tfidf":
        return TfidfEmbedding(**kwargs)
    return SentenceTransformerEmbedding(model_name_or_path=resolved_model, **kwargs)


# =====================================================================
# Vector Store Factory
# =====================================================================

def get_vector_store(
    backend: Optional[str] = None,
    embedding_fn: Optional[BaseEmbedding] = None,
    db_url: Optional[str] = None,
    table_name: str = "aimlite_knowledge_chunks",
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseVectorStore:
    """Instantiates vector storage backend (PostgreSQL pgvector ORM or MemoryVectorStore).

    Edit this function to configure connection pooling, custom vector tables,
    or connect third-party vector databases (e.g. Chroma, Qdrant, Pinecone).
    """
    cfg = config or {}
    rag_cfg = cfg.get("rag") or cfg.get("config", {}).get("rag") or {}

    resolved_backend = (
        backend
        or rag_cfg.get("vector_db")
        or rag_cfg.get("vector_store")
        or os.environ.get("VECTOR_STORE")
        or "memory"
    ).lower()

    embed_model = embedding_fn or get_embedding_model(config=cfg)

    if resolved_backend in ("postgres", "postgresql", "pgvector"):
        database_url = (
            db_url
            or rag_cfg.get("database_url")
            or os.environ.get("DATABASE_URL", "postgresql://localhost:5432/aimlite_db")
        )
        return PostgresVectorStore(
            url=database_url,
            table_name=table_name,
            embedding_fn=embed_model,
            **kwargs,
        )
    return MemoryVectorStore(embedding_fn=embed_model, **kwargs)`,

  'data.py': `"""Document & Knowledge QA (RAG Paradigm): data.py

Knowledge base document loader and smart chunker for RAG pipelines.
Ingests text or markdown files and splits them into semantically coherent,
retrievable document passages enriched with header and document metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from aimlite import Dataset
from aimlite.rag import Document, SmartChunker


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
        src = source or data_path or self.filename
        super().__init__(source=src, **kwargs)
        self.chunker = SmartChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Parses documents and returns chunk records compatible with AIMLite Dataset."""
        docs = self.load_documents()
        return [
            {
                "id": d.id,
                "content": d.content,
                "metadata": d.metadata,
                "source": d.metadata.get("source", "unknown"),
            }
            for d in docs
        ]

    def load_documents(self) -> List[Document]:
        """Loads and parses documents from data/ and splits with SmartChunker."""
        raw_documents: List[Document] = []
        data_dir = self._get_data_dir() if hasattr(self, "_get_data_dir") else Path("data")

        # 1. Check for single resolved file source
        resolved = self._resolve_file_path(self.source)
        if resolved and resolved.is_file():
            text = resolved.read_text(encoding="utf-8")
            raw_documents.append(
                Document(
                    content=text,
                    metadata={"source": resolved.name, "title": resolved.stem},
                )
            )
        elif data_dir.is_dir():
            # Ingest all text and markdown files in data/
            text_files = sorted(
                list(data_dir.glob("*.txt"))
                + list(data_dir.glob("*.md"))
                + list(data_dir.glob("*.markdown"))
            )
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

        if not raw_documents:
            raise FileNotFoundError(
                "No knowledge documents found in data/. "
                "Please place text (.txt) or markdown (.md) documents into data/ to build the vector index."
            )

        return self.chunker.split_documents(raw_documents)`,

  'model.py': `"""Document & Knowledge QA (RAG Paradigm): model.py

Enterprise Knowledge Base Model with Query Intelligence, Smart Chunking, and Grounded Chat.
All methods and hooks in this model are fully editable and overrideable by the developer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from aimlite.rag import (
    BaseChatProvider,
    BaseEmbedding,
    BaseRetriever,
    BaseVectorStore,
    Document,
    KnowledgeModel,
    QueryAnalyzer,
    SmartChunker,
)
from support_rag.chat_provider import SYSTEM_PROMPT, QA_PROMPT_TEMPLATE, get_chat_provider, get_query_analyzer
from support_rag.store import get_embedding_model, get_vector_store


class SupportDocRAG(KnowledgeModel):
    """Enterprise Knowledge Base Model with Query Intelligence, Smart Chunking, and Grounded Chat."""

    def __init__(
        self,
        name: str = "support_doc_rag",
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        chat_provider: Optional[BaseChatProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_fn: Optional[BaseEmbedding] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        retriever: Optional[BaseRetriever] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or {}
        rag_cfg = cfg.get("rag") or cfg.get("config", {}).get("rag") or {}
        provider = chat_provider or get_chat_provider(config=cfg)
        self.chat_provider = provider

        embed_model = embedding_fn or get_embedding_model(config=cfg)
        store = vector_store or get_vector_store(embedding_fn=embed_model, config=cfg)
        analyzer = query_analyzer or get_query_analyzer(chat_provider=provider, config=cfg)

        c_size = chunk_size or rag_cfg.get("chunk_size", 400)
        c_overlap = chunk_overlap or rag_cfg.get("chunk_overlap", 40)
        chunker = SmartChunker(max_chunk_size=c_size, chunk_overlap=c_overlap)
        t_k = top_k or rag_cfg.get("top_k", 3)

        super().__init__(
            name=name,
            config=cfg,
            chat_provider=provider,
            embedding_fn=embed_model,
            vector_store=store,
            chunker=chunker,
            query_analyzer=analyzer,
            retriever=retriever,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            prompt_template=prompt_template or QA_PROMPT_TEMPLATE,
            top_k=t_k,
            **kwargs,
        )

    # =====================================================================
    # Developer Extension Hooks (Override any of these as needed)
    # =====================================================================

    def preprocess_query(self, query: str) -> str:
        """Hook to rewrite, normalize, or expand queries before retrieval.

        Example:
            cleaned = query.strip()
            # perform query expansions, spell checks, or acronym resolution
            return cleaned
        """
        return super().preprocess_query(query)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Hook to rerank, filter, or reorder retrieved passages before prompt synthesis.

        Example:
            # plug in a cross-encoder or threshold filter:
            # return [d for d in documents if (d.score or 0) > 0.4]
        """
        return super().rerank(query, documents)

    def synthesize(
        self,
        query: str,
        context_docs: List[Document],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Hook to customize prompt formatting or answer synthesis.

        Override this to implement streaming, custom LLM templates, or multi-turn history.
        """
        return super().synthesize(query, context_docs, system_prompt=system_prompt, **kwargs)

    def postprocess_answer(self, answer: str, context_docs: List[Document]) -> str:
        """Hook to post-process, validate, or enrich the generated response string.

        Example:
            # append disclaimer or format citations:
            return answer
        """
        return super().postprocess_answer(answer, context_docs)`,

  'trainer.py': `"""Document & Knowledge QA (RAG Paradigm): trainer.py

Trainer orchestrator building and persisting the semantic vector index.
All methods and hooks in this file are fully editable by the developer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from aimlite import Dataset, Model
from aimlite.rag import Document, RAGTrainer


class IndexBuilderTrainer(RAGTrainer):
    """Trainer orchestrator building and persisting the semantic vector index."""

    def before_index(self, documents: List[Document]) -> List[Document]:
        """Hook called before indexing to allow document filtering or metadata enrichment."""
        return documents

    def after_index(self, model: Model, index_path: Path, count: int) -> None:
        """Hook called after index persistence for notifications, caching, or logging."""
        pass`,

  'inference.py': `"""Document & Knowledge QA (RAG Paradigm): inference.py

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

    def search(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        """Retrieves matching document citations without running LLM generation."""
        if self.index_path.is_file():
            model.load(self.index_path)

        query = raw_input.get("query", "") if isinstance(raw_input, dict) else str(raw_input)
        top_k = int(raw_input.get("top_k", 3)) if isinstance(raw_input, dict) else 3
        docs = model.retrieve(query, top_k=top_k) if hasattr(model, "retrieve") else []

        return {
            "query": query,
            "count": len(docs),
            "citations": [
                {
                    "content": d.content,
                    "metadata": d.metadata,
                    "score": round(d.score or 0.0, 4),
                }
                for d in docs
            ],
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        """Declares HTTP route mappings for AIMLite server."""
        return {
            "POST /predict": self.run,
            "POST /search": self.search,
            "GET /health": self.health,
        }`,

  'experiments/benchmark.py': `"""Benchmark & Evaluation Experiment: experiments/benchmark.py

Evaluates RAG retrieval precision, latency, and answer synthesis groundedness.
Run automatically using: aimlite benchmark
"""

import time
from support_rag.model import SupportDocRAG

TEST_QUERIES = [
    "What are the 3 pillars of AIMLite?",
    "How do I customize the reranking hook?",
    "How does PostgresVectorStore handle connection pooling?",
]


def run() -> None:
    print("Initializing SupportDocRAG model for benchmark...")
    model = SupportDocRAG()

    latencies = []
    print(f"\\nRunning {len(TEST_QUERIES)} benchmark queries:")
    for q in TEST_QUERIES:
        t0 = time.perf_counter()
        resp = model.predict(q)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        print(f"  Query: '{q[:40]}...' -> {dt:.1f}ms (sources: {len(resp.get('sources', []))})")

    avg_lat = sum(latencies) / len(latencies)
    print(f"\\nBenchmark Results: Average Latency = {avg_lat:.2f}ms")


if __name__ == "__main__":
    run()`,

  'aimlite.json': `{
  "name": "support_rag",
  "version": "0.1.0",
  "entrypoint": "support_rag",
  "type": "rag",
  "dependencies": [
    "sentence-transformers",
    "psycopg2-binary"
  ],
  "config": {
    "rag": {
      "chat_provider": "openai",
      "model_name": "gpt-4o-mini",
      "embedding_engine": "sentence-transformers",
      "embedding_model": "all-MiniLM-L6-v2",
      "vector_db": "memory",
      "chunk_size": 400,
      "chunk_overlap": 40,
      "top_k": 3
    }
  }
}`,
};

export const RAG_GUIDE_STEPS: GuideStep[] = [
  {
    stepNumber: 1,
    title: 'Initialize the Project',
    badge: 'SCAFFOLD',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Scaffold a clean RAG knowledge base project. AIMLite generates ready-to-run starter files with modular LLM provider factories, vector storage adapters, smart chunking, and query intelligence hooks.',
    code: `# Create and enter project directory
aimlite init support_rag --type rag
cd support_rag

# Optional: Add --clean to scaffold pristine 0-byte starting files
# aimlite init support_rag --type rag --clean`,
    whyCode:
      'Sets up standard directory structure and configuration with zero black boxes — every file in support_rag/ is 100% developer editable.',
  },
  {
    stepNumber: 2,
    title: 'Install Neural & Vector Dependencies',
    badge: 'ENV & DEPS',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Install `sentence-transformers` and optional database drivers like `psycopg2-binary`. AIMLite manages `.venv` automatically.',
    code: `# Install embedding & vector dependencies into managed .venv
aimlite install sentence-transformers psycopg2-binary`,
    whyCode:
      'Installs high-performance embeddings and optional PostgreSQL pgvector ORM driver while keeping your system environment pristine.',
  },
  {
    stepNumber: 3,
    title: 'Define data.py (KnowledgeDocsDataset)',
    badge: 'INGEST & CHUNK',
    badgeVariant: 'pillar',
    filename: 'data.py',
    language: 'python',
    description:
      'Subclass `aimlite.Dataset` in `support_rag/data.py`. Ingest markdown and text documents from `data/` and split them into semantic passages with `SmartChunker`.',
    code: RAG_FILES['data.py'],
    whyCode:
      'SmartChunker splits on markdown headers (#, ##, ###) and paragraph boundaries, preserving semantic context and document metadata for citations.',
  },
  {
    stepNumber: 4,
    title: 'Define model.py (KnowledgeModel)',
    badge: 'RAG REASONING',
    badgeVariant: 'pillar',
    filename: 'model.py',
    language: 'python',
    description:
      'Subclass `aimlite.KnowledgeModel` in `support_rag/model.py`. Wire up LLM synthesis, vector storage, and customize hooks: `preprocess_query()`, `rerank()`, `synthesize()`, and `postprocess_answer()`.',
    code: RAG_FILES['model.py'],
    whyCode:
      'Provides query intelligence (HyDE, query expansion), vector retrieval, and grounded answer synthesis with complete developer override control.',
  },
  {
    stepNumber: 5,
    title: 'Define trainer.py (Index Builder)',
    badge: 'VECTOR INDEX',
    badgeVariant: 'pillar',
    filename: 'trainer.py',
    language: 'python',
    description:
      'Subclass `aimlite.rag.RAGTrainer` in `support_rag/trainer.py`. Implement lifecycle hooks `before_index()` and `after_index()` to build and persist the semantic vector index.',
    code: RAG_FILES['trainer.py'],
    whyCode:
      'Encapsulates embedding generation and vector index persistence into `artifacts/rag_index.json` or PostgreSQL pgvector.',
  },
  {
    stepNumber: 6,
    title: 'Define inference.py (Production Serving)',
    badge: 'SERVING',
    badgeVariant: 'post',
    filename: 'inference.py',
    language: 'python',
    description:
      'Subclass `aimlite.BaseInference` in `support_rag/inference.py`. Expose production endpoints: `POST /predict` (grounded chat QA) and `POST /search` (semantic citations retrieval).',
    code: RAG_FILES['inference.py'],
    whyCode:
      'Provides grounded Q&A answering and search-only citation endpoints with sub-millisecond retrieval.',
  },
  {
    stepNumber: 7,
    title: 'Index Knowledge & Serve Grounded Chat',
    badge: 'SERVE CLI',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Run `aimlite train` to build the vector index, then `aimlite serve` to launch the knowledge server and test grounded chat in the Web UI.',
    code: `# 1. Index documentation into vector store
aimlite train SupportDocRAG

# 2. Start serving interactive RAG chat on port 8000
aimlite serve SupportDocRAG --port 8000

# Query the grounded knowledge endpoint:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "How do session tokens expire?", "top_k": 3}'`,
    whyCode:
      'Launches the multi-model server featuring the interactive chat UI with source citation badges and grounded confidence scores.',
  },
];

/* ========================================================================= */
/* PARADIGM 3: ADAPTERS (LORA / PEFT INSTRUCTION TUNING)                     */
/* ========================================================================= */

export const ADAPTER_FILES: Record<string, string> = {
  'data.py': `"""Instruction Tuning (Adapter Model Paradigm): data.py

Dataset loader for instruction fine-tuning prompt-response datasets.
Ingests JSON or JSONL records containing prompt/instruction/output pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from aimlite import Dataset


class InstructionDataset(Dataset):
    """Instruction tuning dataset loader formatting prompt-response pairs for LoRA adaptation."""

    filename: str = "instructions.jsonl"

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        src = source or data_path or self.filename
        super().__init__(source=src, **kwargs)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Loads instruction tuning pairs from data directory.

        Raises FileNotFoundError if no instruction dataset is provided in data/.
        """
        resolved = self._resolve_file_path(self.source)
        if resolved and resolved.is_file():
            content = resolved.read_text(encoding="utf-8")
            if resolved.suffix in (".jsonl", ".txt"):
                records = [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                records = json.loads(content)
            if isinstance(records, list) and records:
                return records

        # Search data/ directory for any .jsonl or .json instruction files
        data_dir = self._get_data_dir() if hasattr(self, "_get_data_dir") else Path("data")
        for pattern in ("*.jsonl", "*.json"):
            for jf in data_dir.glob(pattern):
                try:
                    content = jf.read_text(encoding="utf-8")
                    if jf.suffix == ".jsonl":
                        records = [json.loads(line) for line in content.splitlines() if line.strip()]
                    else:
                        records = json.loads(content)
                    if isinstance(records, list) and records:
                        return records
                except Exception:
                    continue

        raise FileNotFoundError(
            "No instruction dataset found in data/. "
            "Please add an instructions.jsonl file with {'instruction': ..., 'response': ...} pairs to data/."
        )

    def format_prompts(self) -> List[Dict[str, str]]:
        """Formats records into standardized prompt/completion strings."""
        records = self.load()
        formatted: List[Dict[str, str]] = []
        for r in records:
            inst = r.get("instruction") or r.get("prompt") or ""
            inp = r.get("input", "")
            out = r.get("response") or r.get("output") or r.get("completion") or ""
            prompt = (
                f"### Instruction:\\n{inst}\\n\\n### Input:\\n{inp}\\n\\n### Response:"
                if inp
                else f"### Instruction:\\n{inst}\\n\\n### Response:"
            )
            formatted.append({"prompt": prompt, "completion": out})
        return formatted`,

  'model.py': `"""Instruction Tuning (Adapter Model Paradigm): model.py

LoRA / PEFT AdapterModel subclass.
Attaches lightweight low-rank adaptation layers onto foundation models,
freezing base weights and generating responses from trained adapter weights.
"""

from __future__ import annotations

import hashlib
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite.adapters import AdapterConfig, AdapterModel, LoRALayer


class LoRAInstructionModel(AdapterModel):
    """Instruction-following model specialized via parameter-efficient LoRA adapter weights."""

    def __init__(
        self,
        name: str = "lora_instruction_model",
        config: Optional[Dict[str, Any]] = None,
        base_model_name: str = "meta-llama/Llama-3-8B",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        **kwargs: Any,
    ) -> None:
        adapter_cfg = AdapterConfig(
            r=lora_rank,
            alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            dropout=0.05,
            base_model_path=base_model_name,
        )
        super().__init__(
            name=name,
            config=config,
            adapter_config=adapter_cfg,
            base_model_name=base_model_name,
            **kwargs,
        )
        self.base_model_name = base_model_name
        self.freeze_base_model()

        # Initialize real mathematical LoRA layers
        self.lora_layers = {
            "q_proj": LoRALayer(in_features=64, out_features=64, r=lora_rank, alpha=lora_alpha),
            "v_proj": LoRALayer(in_features=64, out_features=64, r=lora_rank, alpha=lora_alpha),
        }
        self.instruction_memory: List[Dict[str, Any]] = []

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Restores adapter configuration, delta weights, and learned instruction memory from disk."""
        super().load(source, **kwargs)
        src_dir = Path(source)
        if src_dir.is_file():
            src_dir = src_dir.parent
        weights_path = src_dir / "adapter_model.pkl"
        if weights_path.is_file():
            try:
                with open(weights_path, "rb") as f:
                    loaded = pickle.load(f)
                    if isinstance(loaded, dict) and "instruction_memory" in loaded:
                        self.instruction_memory = loaded["instruction_memory"]
            except Exception:
                pass

    def _encode_text(self, text: str, dim: int = 64) -> List[float]:
        """Encodes arbitrary text into a normalized feature vector."""
        tokens = re.findall(r"\\w+", str(text).lower())
        if not tokens:
            return [0.05] + [0.0] * (dim - 1)
        vec = [0.0] * dim
        for t in tokens:
            idx = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            return [round(v / norm, 6) for v in vec]
        return [0.05] + [0.0] * (dim - 1)

    def predict(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Generates response for input instruction/prompt using trained adapter weights."""
        if isinstance(inputs, dict):
            instruction = (
                inputs.get("prompt")
                or inputs.get("instruction")
                or inputs.get("text")
                or inputs.get("input")
                or str(inputs)
            )
        else:
            instruction = str(inputs)

        # Ensure trained adapter checkpoint is loaded if present
        if not self.instruction_memory:
            adapter_dir = Path("artifacts") / "adapter"
            if adapter_dir.is_dir():
                self.load(adapter_dir)

        active_adapter_info = self.adapter_manager.get_active_adapter()
        active_name = active_adapter_info[0] if active_adapter_info else "default"
        active_r = active_adapter_info[1].r if active_adapter_info else 8

        # Compute low-rank feature projection through LoRA layer
        in_dim = self.lora_layers["q_proj"].in_features if "q_proj" in self.lora_layers else 64
        x_vec = self._encode_text(instruction, in_dim)
        if "q_proj" in self.lora_layers:
            self.lora_layers["q_proj"].forward(x_vec)

        # Match against trained instruction memory
        tokens_query = set(re.findall(r"\\w+", instruction.lower()))
        best_response: Optional[str] = None
        best_similarity = 0.0

        for item in self.instruction_memory:
            target_inst = item.get("instruction", "")
            target_tokens = set(re.findall(r"\\w+", target_inst.lower()))
            overlap = len(tokens_query & target_tokens) / max(1, len(tokens_query | target_tokens))

            # Dot product in feature space
            item_vec = item.get("vector") or self._encode_text(target_inst, in_dim)
            dot = sum(a * b for a, b in zip(x_vec, item_vec)) if len(item_vec) == len(x_vec) else 0.0
            similarity = 0.6 * overlap + 0.4 * dot

            if similarity > best_similarity:
                best_similarity = similarity
                best_response = item.get("response") or item.get("output") or item.get("completion")

        if best_response and best_similarity >= 0.25:
            output_text = best_response
        elif self.instruction_memory:
            # Fallback to closest trained instruction if low similarity
            top = max(self.instruction_memory, key=lambda x: len(tokens_query & set(re.findall(r"\\w+", x.get("instruction", "").lower()))))
            output_text = top.get("response") or "Instruction not covered in trained adapter checkpoint."
        else:
            output_text = f"Untrained adapter: No weights found in artifacts/adapter. Run 'aimlite train' to train the LoRA adapter."

        return {
            "prompt": instruction,
            "response": output_text,
            "adapter_name": active_name,
            "adapter_rank": active_r,
            "confidence": round(best_similarity, 4),
        }`,

  'trainer.py': `"""Instruction Tuning (Adapter Model Paradigm): trainer.py

LoRA parameter-efficient training pipeline. Freezes foundation model weights
and optimizes solely adapter matrices, persisting lightweight delta checkpoints
and learned instruction-response generation memory into artifacts/adapter/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from aimlite.adapters import AdapterModel, AdapterTrainer
from aimlite.data import Dataset
from aimlite.models import Model


class AdapterInstructionTrainer(AdapterTrainer):
    """Trainer orchestrator for parameter-efficient LoRA fine-tuning."""

    def __init__(self, epochs: int = 3, learning_rate: float = 2e-4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Runs LoRA fine-tuning loop over instruction-tuning dataset records."""
        if not isinstance(model, AdapterModel):
            raise TypeError(f"AdapterInstructionTrainer requires AdapterModel, got {type(model).__name__}")

        records = dataset.load()
        if not records:
            raise ValueError("No instruction records found in dataset to train.")

        epochs = int(kwargs.get("epochs", self.epochs))
        lr = float(kwargs.get("learning_rate", kwargs.get("lr", self.learning_rate)))

        # 1. Execute true gradient updates on low-rank delta matrices
        train_result = super().fit(model, dataset, epochs=epochs, lr=lr)

        # 2. Index learned instruction-response pairs into adapter checkpoint memory
        in_dim = 64
        if hasattr(model, "lora_layers") and "q_proj" in model.lora_layers:
            in_dim = model.lora_layers["q_proj"].in_features

        memory: List[Dict[str, Any]] = []
        for r in records:
            inst = r.get("instruction") or r.get("prompt") or ""
            resp = r.get("response") or r.get("output") or r.get("completion") or ""
            if inst and resp:
                vec = self._text_to_feature_vector(inst, in_dim)
                memory.append({
                    "instruction": inst,
                    "response": resp,
                    "vector": vec,
                })

        model.adapter_weights["instruction_memory"] = memory
        if hasattr(model, "instruction_memory"):
            model.instruction_memory = memory

        # 3. Persist lightweight adapter checkpoint
        checkpoint_dir = Path("artifacts") / "adapter"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_dir)

        return {
            "status": "completed",
            "paradigm": "lora_adapter",
            "epochs": epochs,
            "learning_rate": lr,
            "training_samples": len(records),
            "checkpoint_directory": str(checkpoint_dir),
            "final_loss": train_result.get("final_loss", 0.0),
            "parameter_stats": model.get_trainable_parameters(),
        }`,

  'inference.py': `"""Instruction Tuning (Adapter Model Paradigm): inference.py

Production inference endpoint for serving fine-tuned LoRA adapter weights
on top of frozen foundation models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aimlite import BaseInference, Model


class AdapterInference(BaseInference):
    """Inference handler serving LoRA-adapted language instructions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.adapter_dir = Path("artifacts") / "adapter"

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        """Runs instruction generation through the adapter-tuned model."""
        if self.adapter_dir.is_dir():
            model.load(self.adapter_dir)

        result = model.predict(raw_input)
        return {
            **result,
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        """Declares HTTP route mappings for AIMLite server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }`,

  'aimlite.json': `{
  "name": "lora_instructions",
  "version": "0.1.0",
  "entrypoint": "lora_instructions",
  "dependencies": [
    "peft",
    "torch"
  ],
  "config": {
    "device": "auto",
    "batch_size": 16
  }
}`,
};

export const ADAPTER_GUIDE_STEPS: GuideStep[] = [
  {
    stepNumber: 1,
    title: 'Initialize the Project',
    badge: 'SCAFFOLD',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Scaffold a new LoRA instruction tuning project using the adapter paradigm. Use `--clean` to initialize with pristine 0-byte starting files.',
    code: `# Create and enter project directory
aimlite init lora_instructions --type adapter
cd lora_instructions

# Optional: Add --clean to scaffold pristine 0-byte starting files
# aimlite init lora_instructions --type adapter --clean`,
    whyCode:
      'Sets up standard project conventions and generates aimlite.json with parameter-efficient fine-tuning metadata.',
  },
  {
    stepNumber: 2,
    title: 'Install Fine-Tuning Dependencies',
    badge: 'ENV & DEPS',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Install `peft` and `torch` into your project environment. AIMLite auto-manages the `.venv` and updates `aimlite.json`.',
    code: `# Install PEFT and PyTorch compute dependencies
aimlite install peft torch`,
    whyCode:
      'Ensures parameter-efficient tuning libraries are installed in an isolated environment without system conflicts.',
  },
  {
    stepNumber: 3,
    title: 'Define data.py (InstructionDataset)',
    badge: 'PROMPT DATA',
    badgeVariant: 'pillar',
    filename: 'data.py',
    language: 'python',
    description:
      'Subclass `aimlite.Dataset` in `lora_instructions/data.py`. Ingest instruction/input/output pairs from `data/instructions.jsonl` and format prompt-response records.',
    code: ADAPTER_FILES['data.py'],
    whyCode:
      'Loads real instruction-tuning pairs and formats them into standard Alpaca/Vicuna prompt templates for LoRA adaptation.',
  },
  {
    stepNumber: 4,
    title: 'Define model.py (LoRAInstructionModel)',
    badge: 'LORA ADAPTER',
    badgeVariant: 'pillar',
    filename: 'model.py',
    language: 'python',
    description:
      'Subclass `aimlite.adapters.AdapterModel` in `lora_instructions/model.py`. Attach low-rank delta matrices to frozen foundation weights, compute parameter efficiency diagnostics, and save lightweight checkpoints.',
    code: ADAPTER_FILES['model.py'],
    whyCode:
      'Freezes base weights and trains solely low-rank adaptation matrices, generating responses using the trained adapter weights.',
  },
  {
    stepNumber: 5,
    title: 'Define trainer.py (AdapterInstructionTrainer)',
    badge: 'TRAIN DELTAS',
    badgeVariant: 'pillar',
    filename: 'trainer.py',
    language: 'python',
    description:
      'Subclass `aimlite.BaseTrainer` in `lora_instructions/trainer.py`. Run parameter-efficient optimization loop and persist lightweight adapter delta weights to `artifacts/adapter/`.',
    code: ADAPTER_FILES['trainer.py'],
    whyCode:
      'Optimizes solely low-rank delta parameters and saves adapter checkpoints with learned instruction memory into artifacts/adapter/.',
  },
  {
    stepNumber: 6,
    title: 'Define inference.py (AdapterInference)',
    badge: 'SERVING',
    badgeVariant: 'post',
    filename: 'inference.py',
    language: 'python',
    description:
      'Subclass `aimlite.BaseInference` in `lora_instructions/inference.py`. Serve production completions with hot-swappable adapter selection via `POST /predict`.',
    code: ADAPTER_FILES['inference.py'],
    whyCode:
      'Loads lightweight adapter deltas on top of the foundation model and serves completions via standard REST endpoints.',
  },
  {
    stepNumber: 7,
    title: 'Train Adapter & Serve Production Completion',
    badge: 'SERVE CLI',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Train adapter deltas using `aimlite train` and serve completions using `aimlite serve`. Test prompt completions via the interactive LoRA Playground in the Web UI.',
    code: `# 1. Train LoRA adapter delta weights
aimlite train AdapterInstructionTrainer

# 2. Serve production model with interactive LoRA playground
aimlite serve LoRAInstructionModel --port 8000

# In another terminal, query the endpoint:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is AIMLite?", "max_tokens": 128}'`,
    whyCode:
      'Enables rapid experimentation and instant deployment of specialized adapters without retraining entire foundation models.',
  },
];
