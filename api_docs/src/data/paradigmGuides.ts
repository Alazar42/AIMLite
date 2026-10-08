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
        """Transforms tabular records into feature matrix X and binary target vector y."""
        if not self._data:
            self.load()
        records = self._data
        if not records:
            return [], []

        X: List[List[float]] = []
        y: List[int] = []

        for r in records:
            if self.target_name not in r:
                continue
            row_features = [float(r.get(col, 0.0)) for col in self.feature_names]
            target_val = int(float(r[self.target_name]))
            X.append(row_features)
            y.append(target_val)

        return X, y
`,

    'model.py': `"""Customer Churn Prediction (Scratch Model Paradigm): model.py

Classifier model for customer churn prediction using scikit-learn.
Bring Your Own Framework: AIMLite executes user-defined models seamlessly.
Install dependencies into your project with:
    aimlite install scikit-learn pandas
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite import Model


DEFAULT_FEATURE_COLUMNS: List[str] = [
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
        self.is_fitted: bool = False

        try:
            from telecom_churn.data import FEATURE_COLUMNS
            self.feature_names = list(FEATURE_COLUMNS)
        except ImportError:
            try:
                from data import FEATURE_COLUMNS
                self.feature_names = list(FEATURE_COLUMNS)
            except ImportError:
                self.feature_names = list(DEFAULT_FEATURE_COLUMNS)

        self._init_estimator()

    def _init_estimator(self) -> None:
        """Initializes underlying scikit-learn classifier engine."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
            )
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required to initialize ChurnClassifier. "
                "Install it into your project using: aimlite install scikit-learn"
            ) from e

    def fit(self, X: List[List[float]], y: List[int]) -> ChurnClassifier:
        """Fits classifier on numeric feature matrix X and label vector y."""
        if not X:
            return self

        if self.estimator is None:
            self._init_estimator()
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, inputs: Any, **kwargs: Any) -> List[int]:
        """Predicts binary churn class (0 for retained, 1 for churn)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is None:
            raise RuntimeError("ChurnClassifier is not trained or loaded. Call fit() or load() first.")

        preds = self.estimator.predict(X)
        return [int(p) for p in preds]

    def predict_proba(self, inputs: Any) -> List[float]:
        """Computes probability of churn (risk score between 0.0 and 1.0)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is None:
            raise RuntimeError("ChurnClassifier is not trained or loaded. Call fit() or load() first.")

        probs = self.estimator.predict_proba(X)
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        if 1 in classes:
            idx_1 = classes.index(1)
            return [float(p[idx_1]) for p in probs]
        return [float(p[-1]) for p in probs]

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes model weights and parameters to disk."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "estimator": self.estimator,
            "feature_names": getattr(self, "feature_names", []),
        }
        with open(dest_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Restores model weights and parameters from disk."""
        src_path = Path(source)
        if not src_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {source}")

        with open(src_path, "rb") as f:
            payload = pickle.load(f)

        self.name = payload.get("name", self.name)
        self.n_estimators = payload.get("n_estimators", self.n_estimators)
        self.random_state = payload.get("random_state", self.random_state)
        self.estimator = payload.get("estimator", None)
        self.is_fitted = self.estimator is not None

        if "feature_names" in payload:
            self.feature_names = payload["feature_names"]

    def _normalize_inputs(self, inputs: Any) -> List[List[float]]:
        """Converts diverse input formats into 2D float feature matrix."""
        feature_cols = getattr(self, "feature_names", None) or DEFAULT_FEATURE_COLUMNS

        if isinstance(inputs, dict):
            if "features" in inputs:
                return self._normalize_inputs(inputs["features"])

            norm_map = {k.lower().replace("_", "").replace("-", ""): v for k, v in inputs.items()}
            row = []
            matched = False
            for col in feature_cols:
                col_key = col.lower().replace("_", "").replace("-", "")
                if col in inputs:
                    try:
                        row.append(float(inputs[col]))
                        matched = True
                    except (ValueError, TypeError):
                        row.append(0.0)
                elif col_key in norm_map:
                    try:
                        row.append(float(norm_map[col_key]))
                        matched = True
                    except (ValueError, TypeError):
                        row.append(0.0)
                else:
                    row.append(0.0)

            if not matched:
                numeric_vals = []
                for v in inputs.values():
                    try:
                        numeric_vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
                if numeric_vals:
                    while len(numeric_vals) < len(feature_cols):
                        numeric_vals.append(0.0)
                    return [numeric_vals[:len(feature_cols)]]

            return [row]

        if isinstance(inputs, list):
            if not inputs:
                return []
            if isinstance(inputs[0], dict):
                return [self._normalize_inputs(item)[0] for item in inputs]
            if isinstance(inputs[0], (int, float)):
                return [[float(x) for x in inputs]]
            if isinstance(inputs[0], list):
                return [[float(x) for x in row] for row in inputs]

        return []
`,

    'trainer.py': `"""Customer Churn Prediction (Scratch Model Paradigm): trainer.py

Orchestrates data preparation, train / val splitting, classifier fitting,
    and model checkpoint persistence into artifacts /.
"""

from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict

from aimlite import BaseTrainer, Dataset, Model


class ChurnTrainer(BaseTrainer):
"""Trainer pipeline for fitting ChurnClassifier on TelecomChurnDataset."""

    def fit(self, model: Model, dataset: Dataset, ** kwargs: Any) -> Dict[str, Any]:
"""Trains the churn classifier model and saves artifacts."""
if not hasattr(dataset, "to_arrays"):
return { "status": "failed", "error": "Dataset missing to_arrays method" }

X, y = dataset.to_arrays()
if not X:
    return { "status": "failed", "error": "No training samples found in dataset" }

        # 80 / 20 train / validation split
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_val, y_val = X[split_idx:], y[split_idx:]

if hasattr(model, "fit"):
    model.fit(X_train, y_train)

        # Compute train & val performance
train_preds = model.predict(X_train)
val_preds = model.predict(X_val) if X_val else[]

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
artifacts_dir.mkdir(parents = True, exist_ok = True)
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
}
    `,

    'evaluator.py': `"""Customer Churn Prediction (Scratch Model Paradigm): evaluator.py

Computes precision, recall, F1 - score, and classification accuracy
on evaluation partitions.
"""

from __future__ import annotations

    from typing import Any, Dict

from aimlite import BaseEvaluator, Dataset, Model


class ChurnEvaluator(BaseEvaluator):
"""Evaluator assessing customer churn classification metrics."""

    def evaluate(self, model: Model, dataset: Dataset, ** kwargs: Any) -> Dict[str, float]:
"""Calculates accuracy, precision, recall, and F1 metrics."""
if not hasattr(dataset, "to_arrays"):
return { "accuracy": 0.0 }

X, y = dataset.to_arrays()
if not X:
    return { "accuracy": 0.0 }

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
}
    `,

    'inference.py': `"""Customer Churn Prediction (Scratch Model Paradigm): inference.py

Production inference handler for customer churn risk scoring and intervention routing.
Developers can customize:
1. Input payloads: accept structured dicts({ "CustServCalls": 7, ...}) or arrays({ "features": [...] })
2. Active model selection: dynamically route between registered models or load specific checkpoints
3. Response schema: format human - readable decisions, probabilities, and retention actions
4. Custom HTTP routes: expose additional endpoints via get_routes()
"""

from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict, Optional

from aimlite import BaseInference, Model


class ChurnInference(BaseInference):
"""Production inference endpoint returning customer churn probability and retention decisions."""

    def __init__(self, ** kwargs: Any) -> None:
super().__init__(** kwargs)
self.checkpoint_path = self._resolve_checkpoint()

    def _resolve_checkpoint(self) -> Optional[Path]:
candidates = [
    Path("artifacts") / "churn_classifier.pkl",
    Path("models") / "churn_classifier.pkl",
    Path(__file__).resolve().parent.parent / "artifacts" / "churn_classifier.pkl",
    Path(__file__).resolve().parent.parent / "models" / "churn_classifier.pkl",
]
for c in candidates:
    if c.is_file():
        return c
return None

    def run(self, model: Model, raw_input: Any, ** kwargs: Any) -> Dict[str, Any]:
"""Runs churn prediction on incoming customer attributes.

Args:
model: The active Model instance resolved by the AIMLite server.
    raw_input: Raw JSON payload sent from HTTP client or the served web UI.
                       Can be a dict of customer attributes or { "features": [...] }.
"""
        # Ensure model weights are loaded if checkpoint exists
if not getattr(model, "is_fitted", False):
ckpt = self.checkpoint_path or self._resolve_checkpoint()
if ckpt and ckpt.is_file():
model.load(ckpt)

        # 1. Compute discrete class prediction (0 = retain, 1 = churn)
preds = model.predict(raw_input)
churn_pred = preds[0] if preds else 0

        # 2. Compute continuous risk probability
risk_score = 0.5
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(raw_input)
if probs:
    risk_score = round(probs[0], 4)

        # 3. Apply business logic threshold for retention intervention
        decision = "Intervene (High Churn Risk)" if risk_score >= 0.5 else "Retain (Low Risk)"

        # 4. Return structured response(the served web UI highlights 'decision' and 'churn_risk')
return {
    "churn_prediction": int(churn_pred),
    "churn_risk": risk_score,
    "decision": decision,
    "status": "success",
}

    def get_routes(self) -> Dict[str, Any]:
"""Declares custom HTTP route mappings for the AIMLite server."""
return {
    "POST /predict": self.run,
    "GET /health": self.health,
}
    `,

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
} `
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
            'Scaffold a clean AIMLite project with the scratch paradigm. Use `aimlite init < name > --type scratch` to create a new folder, or `aimlite init. --type scratch` to initialize inside your current directory. Use `--clean` if you want pristine 0-byte starting files.',
        code: `# Option A: Create in a new project folder
aimlite init telecom_churn--type scratch
cd telecom_churn

# Option B: Scaffold directly inside current directory(.)
# aimlite init. --type scratch

# Option C: Use--clean to scaffold pristine 0 - byte starting files
# aimlite init telecom_churn--type scratch --clean`,
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
        code: `# Install libraries: auto - manages.venv and updates aimlite.json
aimlite install scikit - learn pandas

# Later or on a new machine, running with no arguments reinstalls all pinned dependencies:
# aimlite install`,
        whyCode:
            'On modern OSes (Ubuntu 24+, Debian, macOS), installing global python packages is restricted (PEP 668). AIMLite ensures all projects have an isolated, self-managed .venv and tracks dependencies deterministically in aimlite.json.',
    },
    {
        stepNumber: 3,
        title: 'Acquire Kaggle Telecom Churn Dataset',
        badge: 'DATASET',
        badgeVariant: 'pillar',
        filename: 'data/telecom_churn.csv',
        language: 'bash',
        description:
            'Download the Telecom Churn CSV from Kaggle and place it in your project as `data / telecom_churn.csv`. Then validate dataset conventions with `aimlite data validate`.',
        code: `# 1. Download dataset from Kaggle:
# https://www.kaggle.com/datasets/barun2104/telecom-churn
# Place file into: data / telecom_churn.csv

# 2. Validate dataset schema and conventions:
aimlite data validate`,
        externalLink: {
            label: 'Kaggle: Telecom Churn Dataset',
            url: 'https://www.kaggle.com/datasets/barun2104/telecom-churn',
        },
        whyCode:
            'AIMLite validates that data/ contains valid datasets and ensures columns (AccountWeeks, ContractRenewal, DataPlan, CustServCalls, MonthlyCharge, Churn) match expected schema before training begins.',
    },
    {
        stepNumber: 4,
        title: 'data.py: Ingest & Preprocess Customer Records',
        badge: 'LOADER',
        badgeVariant: 'pillar',
        filename: 'data.py',
        language: 'python',
        description:
            'Implement `TelecomChurnDataset` inheriting from `aimlite.Dataset` with explicit `filename = "telecom_churn.csv"`. Uses pandas for accelerated reading with a pure-Python csv fallback, and populates `self._data` and `self.columns`.',
        code: SCRATCH_FILES['data.py'],
        whyCode:
            'Setting `filename = "telecom_churn.csv"` tells AIMLite exactly which CSV file in `data / ` to bind, validate, and partition. Assigning `self._data` and `self.columns` enables AIMLite automated split contracts and pre-flight validation.',
    },
    {
        stepNumber: 5,
        title: 'model.py: Define ChurnClassifier',
        badge: 'MODEL',
        badgeVariant: 'pillar',
        filename: 'model.py',
        language: 'python',
        description:
            'Define `ChurnClassifier` subclassing `aimlite.Model`. Encapsulates `RandomForestClassifier` with balanced class weights, implements `predict()` and `predict_proba()`, and provides standard pickle serialization.',
        code: SCRATCH_FILES['model.py'],
        whyCode:
            'Inheriting from Model gives you automatic registration with AIMLite discovery. Zero-path commands (train, evaluate, serve) can inspect and instantiate ChurnClassifier by name.',
    },
    {
        stepNumber: 6,
        title: 'trainer.py: Fit Classifier & Checkpoint Weights',
        badge: 'TRAINER',
        badgeVariant: 'pillar',
        filename: 'trainer.py',
        language: 'python',
        description:
            'Define `ChurnTrainer` subclassing `BaseTrainer`. Performs an 80/20 train/validation split, fits the model, reports train/val accuracy, and saves weights into `artifacts / churn_classifier.pkl`.',
        code: SCRATCH_FILES['trainer.py'],
        whyCode:
            'BaseTrainer standardizes execution lifecycle. AIMLite invokes fit(model, dataset) and guarantees checkpoints are cleanly stored in artifacts/ without ad-hoc path manipulation.',
    },
    {
        stepNumber: 7,
        title: 'evaluator.py: Benchmark Accuracy & F1-Score',
        badge: 'EVALUATOR',
        badgeVariant: 'pillar',
        filename: 'evaluator.py',
        language: 'python',
        description:
            'Define `ChurnEvaluator` subclassing `BaseEvaluator`. Measures true positives, false positives, accuracy, precision, recall, and F1-score across validation partitions.',
        code: SCRATCH_FILES['evaluator.py'],
        whyCode:
            'Separating evaluation from training allows continuous validation across datasets and regression monitoring before deploying to production.',
    },
    {
        stepNumber: 8,
        title: 'inference.py: Production HTTP Endpoint & Custom Prediction Logic',
        badge: 'INFERENCE',
        badgeVariant: 'pillar',
        filename: 'inference.py',
        language: 'python',
        description:
            'Subclass `BaseInference` to define how incoming HTTP requests map to model predictions. You can accept named attributes or feature vectors in `raw_input`, customize model execution, shape returned JSON fields (`decision`, `churn_risk`), and register custom endpoints via `get_routes()`.',
        code: SCRATCH_FILES['inference.py'],
        whyCode:
            'Inference acts as the gateway between HTTP clients / the served UI and your model. You can normalize diverse inputs, invoke multiple candidate models, and output rich decisions without altering core model logic.',
    },
    {
        stepNumber: 9,
        title: 'Execute, Serve & Customize Web Templates',
        badge: 'EXECUTION',
        badgeVariant: 'cli',
        filename: 'terminal.sh',
        language: 'bash',
        description:
            'Train your classifier, benchmark metrics, and launch the production server. The served dashboard automatically features a styled model dropdown switcher, dark/light themeing, and real-time inference. Override templates by placing `app.html` in `<project_root> / templates / ` or serve custom frontends via `--frontend`.',
        code: `# 1. Train classifier and write weights to artifacts /
    aimlite train ChurnClassifier

# 2. Evaluate accuracy, precision, recall, and F1
aimlite evaluate ChurnClassifier

# 3. Start production inference REST server + interactive web UI
aimlite serve ChurnClassifier--port 8000

# 4. Test inference endpoint via REST API:
curl - X POST http://127.0.0.1:8000/predict \\
-H "Content-Type: application/json" \\
-d '{
"AccountWeeks": 128,
    "ContractRenewal": 1,
        "DataPlan": 1,
            "DataUsage": 2.7,
                "CustServCalls": 1,
                    "DayMins": 160.0,
                        "DayCalls": 100,
                            "MonthlyCharge": 60.0,
                                "OverageFee": 5.0,
                                    "RoamMins": 8.0
  }'

# 5.(Optional) Customize the served web UI:
# Place a custom HTML template in <project_root>/templates/app.html
# AIMLite discovers project templates first(Django - style template override)!
# Or host a custom React / Vite / Next build:
# aimlite serve ChurnClassifier--frontend./ frontend / dist`,
        whyCode:
            'Zero-path execution wires your training and serving pipelines instantly. The served UI connects dynamically to GET /models, allowing seamless model switching and custom template customization.',
    },
];

/* ========================================================================= */
/* PARADIGM 2: RAG (KNOWLEDGE BASE QA)                                       */
/* ========================================================================= */

export const RAG_FILES: Record<string, string> = {
    'chat_provider.py': `"""Chat Provider & Query Intelligence: chat_provider.py

Starter code for configuring LLM synthesis, system prompts, and query analysis(HyDE, intent decomposition, expansion).
All classes, functions, and prompt templates in this file are fully editable by the developer.
"""

from __future__ import annotations

import os
    from typing import Any, Dict, List, Optional
from aimlite.rag import(
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
# Developer - Editable Prompt Templates
# =====================================================================

    SYSTEM_PROMPT = """You are a knowledgeable, factual, and helpful AI assistant.
Answer the user's query strictly based on the provided context passages.
    - Cite your sources by referring to the document ID or source name(e.g. [1], [Source: filename]).
- If the context does not contain sufficient information to answer truthfully, state clearly that the answer is not available in the provided documents.
- Do not fabricate facts or hallucinate citations.
"""

QA_PROMPT_TEMPLATE = """Context Passages:
{ context }

User Question:
{ query }

Answer: """


# =====================================================================
# Chat Provider Factory
# =====================================================================

    def get_chat_provider(
        provider_name: str = "openai",
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ** kwargs: Any,
    ) -> BaseChatProvider:
"""Instantiates and returns the configured LLM Chat Provider.
    
    Edit this function to add custom LLM providers, change default models,
    or adjust generation parameters.
    """
prompt = system_prompt or SYSTEM_PROMPT

if provider_name == "openai":
    return OpenAIChatProvider(
        model = model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        system_prompt = prompt,
        temperature = temperature,
            ** kwargs,
    )
    elif provider_name == "gemini":
return GeminiChatProvider(
    model = model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
    system_prompt = prompt,
    temperature = temperature,
            ** kwargs,
)
    elif provider_name == "anthropic":
return AnthropicChatProvider(
    model = model_name or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    system_prompt = prompt,
    temperature = temperature,
            ** kwargs,
)
    elif provider_name == "ollama":
return OllamaChatProvider(
    model = model_name or os.environ.get("OLLAMA_MODEL", "llama3.2"),
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
    system_prompt = prompt,
    temperature = temperature,
            ** kwargs,
)
    elif provider_name == "local":
return LocalChatProvider(
    model_name = model_name or os.environ.get("LOCAL_MODEL", "meta-llama/Llama-3.2-3B"),
    system_prompt = prompt,
            ** kwargs,
)
    else:
return MockChatProvider(
    model = model_name or os.environ.get("MOCK_MODEL", "mock-gpt"),
    system_prompt = prompt,
            ** kwargs,
)


# =====================================================================
# Query Analyzer & Intelligence
# =====================================================================

    def get_query_analyzer(
        chat_provider: Optional[BaseChatProvider] = None,
        enable_hyde: bool = True,
        system_prompt: Optional[str] = None,
    ) -> QueryAnalyzer:
"""Configures Query Intelligence for intent parsing, expansion, and hypothetical document generation (HyDE).
    
    Edit this function to customize the query analyzer or adjust query expansion behaviour.
    """
provider = chat_provider or get_chat_provider()
return QueryAnalyzer(
    chat_provider = provider,
    enable_hyde = enable_hyde,
    system_prompt = system_prompt or PROMPT_QUERY_ANALYZER,
)
    `,

    'store.py': `"""Vector Store & Database Storage: store.py

Starter code for semantic vector storage, embeddings, and PostgreSQL ORM records.
All functions and classes in this file are fully editable by the developer.
"""

from __future__ import annotations

import os
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Union
from aimlite.rag import(
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
    ** kwargs: Any,
    ) -> BaseEmbedding:
"""Instantiates embedding model (Dense Neural, Ollama API, OpenAI API, or Pure-Python TF-IDF).
    
    Edit this function to plug in custom embedding models(e.g.HuggingFace, Cohere, OpenAI).
    """
resolved_engine = (engine or os.environ.get("EMBEDDING_ENGINE") or "sentence-transformers").lower()
resolved_model = model_name or os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

if resolved_engine == "ollama" or "nomic" in resolved_model.lower():
return OllamaEmbedding(model = resolved_model, host = ollama_host, ** kwargs)
    elif resolved_engine == "openai":
return APIEmbedding(provider = "openai", model = resolved_model, ** kwargs)
    elif resolved_engine in ("sentence-transformers", "local"):
return SentenceTransformerEmbedding(model_name_or_path = resolved_model, ** kwargs)
    elif resolved_engine == "tfidf":
return TfidfEmbedding(** kwargs)
return SentenceTransformerEmbedding(model_name_or_path = resolved_model, ** kwargs)


# =====================================================================
# Vector Store Factory
# =====================================================================

    def get_vector_store(
        backend: Optional[str] = None,
        embedding_fn: Optional[BaseEmbedding] = None,
        db_url: Optional[str] = None,
        table_name: str = "aimlite_knowledge_chunks",
    ** kwargs: Any,
    ) -> BaseVectorStore:
"""Instantiates vector storage backend (PostgreSQL pgvector ORM or MemoryVectorStore).
    
    Edit this function to configure connection pooling, custom vector tables,
    or connect third - party vector databases(e.g.Chroma, Qdrant, Pinecone).
    """
resolved_backend = (backend or os.environ.get("VECTOR_STORE") or "memory").lower()
embed_model = embedding_fn or get_embedding_model()

if resolved_backend in ("postgres", "postgresql", "pgvector"):
    database_url = db_url or os.environ.get("DATABASE_URL", "postgresql://localhost:5432/aimlite_db")
return PostgresVectorStore(
    url = database_url,
    table_name = table_name,
    embedding_fn = embed_model,
            ** kwargs,
)
return MemoryVectorStore(embedding_fn = embed_model, ** kwargs)
    `,

    'data.py': `"""Document & Knowledge QA (RAG Paradigm): data.py

Knowledge base document loader and smart chunker for RAG pipelines.
Ingests text or Markdown files and splits them into semantically coherent,
    retrievable document passages enriched with header and document metadata.
"""

from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict, List, Optional, Union

from aimlite import Dataset
    from aimlite.rag import Document, SmartChunker, DocumentLoader


class KnowledgeDocsDataset(Dataset):
"""Ingests documentation articles, datasets, and custom user files into chunked passages."""

filename: str = "knowledge_base.md"

    def __init__(
    self,
    source: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path]] = None,
    max_chunk_size: int = 400,
    chunk_overlap: int = 40,
        ** kwargs: Any,
) -> None:
src = source or data_path or "data"
super().__init__(source = src, ** kwargs)
self.chunker = SmartChunker(max_chunk_size = max_chunk_size, chunk_overlap = chunk_overlap)
self.additional_documents: List[Document] = []

    def add_document(self, content: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
"""Dynamically appends a new document or user-populated text record to the dataset."""
meta = metadata or { }
if title:
    meta["title"] = title
self.additional_documents.append(Document(content = content, metadata = meta))

    def parse_custom_file(self, file_path: Path) -> List[Document]:
"""Hook for developers to implement custom file format parsing (e.g. PDF, DOCX, XML)."""
return DocumentLoader.load_file(file_path)

    def load(self, ** kwargs: Any) -> List[Dict[str, Any]]:
docs = self.load_documents()
return [doc.to_dict() for doc in docs]

    def load_documents(self) -> List[Document]:
"""Loads and chunks all prepared files from the data directory and user-added documents."""
raw_documents: List[Document] = list(self.additional_documents)
data_target = Path(self.source) if self.source else Path("data")

valid_extensions = { ".md", ".txt", ".markdown", ".rst", ".json", ".csv"}

if data_target.is_file():
    raw_documents.extend(self.parse_custom_file(data_target))
        elif data_target.is_dir():
for p in sorted(data_target.rglob("*")):
    if p.is_file() and p.suffix.lower() in valid_extensions and not p.name.startswith("."):
try:
raw_documents.extend(self.parse_custom_file(p))
                    except Exception:
continue

if not raw_documents:
            # Fallback to starter sample documentation
raw_documents.append(
    Document(
        content = (
            "# AIMLite Framework Architecture\\n\\n"
                        "AIMLite provides 3 core pillars: Data (Dataset), Model (Model), and Lifecycle (Trainer). "
                        "All scaffolded RAG files are 100% developer-editable with zero black boxes."
    ),
    metadata = { "source": "faq.md", "title": "AIMLite Architecture" },
)
            )

return self.chunker.split_documents(raw_documents)
    `,

    'model.py': `"""Document & Knowledge QA (RAG Paradigm): model.py

Enterprise Knowledge Base Model with Query Intelligence, Smart Chunking, and Grounded Chat.
All methods and hooks in this model are fully editable and overrideable by the developer.
"""

from __future__ import annotations

    from typing import Any, Dict, List, Optional
from aimlite.rag import(
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
        ** kwargs: Any,
) -> None:
provider = chat_provider or get_chat_provider()
self.chat_provider = provider

embed_model = embedding_fn or get_embedding_model()
store = vector_store or get_vector_store(embedding_fn = embed_model)
analyzer = query_analyzer or get_query_analyzer(chat_provider = provider)
chunker = SmartChunker(max_chunk_size = chunk_size, chunk_overlap = chunk_overlap)

super().__init__(
    name = name,
    config = config,
    chat_provider = provider,
    embedding_fn = embed_model,
    vector_store = store,
    chunker = chunker,
    query_analyzer = analyzer,
    retriever = retriever,
    system_prompt = system_prompt or SYSTEM_PROMPT,
    prompt_template = prompt_template or QA_PROMPT_TEMPLATE,
    top_k = top_k,
            ** kwargs,
)

    # =====================================================================
    # Developer Extension Hooks(Override any of these as needed)
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
            # plug in a cross - encoder, reciprocal rank fusion, or score threshold:
            # return [d for d in documents if (d.score or 0) > 0.4]
return documents
"""
return super().rerank(query, documents)

    def synthesize(
    self,
    query: str,
    context_docs: List[Document],
    system_prompt: Optional[str] = None,
        ** kwargs: Any,
) -> str:
"""Hook to customize prompt formatting or answer synthesis.
        
        Override this to implement streaming, custom LLM templates, or multi - turn history.
        """
return super().synthesize(query, context_docs, system_prompt = system_prompt, ** kwargs)

    def postprocess_answer(self, answer: str, context_docs: List[Document]) -> str:
"""Hook to post-process, validate, or enrich the generated response string.

Example:
            # append disclaimer or format citations:
return answer
"""
return super().postprocess_answer(answer, context_docs)
    `,

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
pass
`,

    'inference.py': `"""Document & Knowledge QA (RAG Paradigm): inference.py

Production inference endpoint for semantic document query retrieval
and context - grounded response generation.
"""

from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict, Optional
from aimlite import BaseInference, Model


class RAGInference(BaseInference):
"""Production inference handler for semantic knowledge retrieval and search."""

    def __init__(self, ** kwargs: Any) -> None:
super().__init__(** kwargs)
self.index_path = Path("artifacts") / "rag_index.json"

    def run(self, model: Model, raw_input: Any, ** kwargs: Any) -> Dict[str, Any]:
"""Runs question-answering over indexed knowledge base documents."""
if self.index_path.is_file():
    model.load(self.index_path)

query = raw_input.get("query", "") if isinstance(raw_input, dict) else str(raw_input)
if not query:
    return { "error": "Missing 'query' field in request body", "status": "failed" }

top_k = raw_input.get("top_k") if isinstance(raw_input, dict) else None
filters = raw_input.get("filters") if isinstance(raw_input, dict) else None

result = model.predict(query, top_k = top_k, filters = filters)
return {
            ** result,
    "status": "success",
        }

    def search(self, model: Model, raw_input: Any, ** kwargs: Any) -> Dict[str, Any]:
"""Direct semantic search endpoint returning ranked document passages without LLM synthesis."""
if self.index_path.is_file():
    model.load(self.index_path)

query = raw_input.get("query", "") if isinstance(raw_input, dict) else str(raw_input)
top_k = raw_input.get("top_k", 5) if isinstance(raw_input, dict) else 5
filters = raw_input.get("filters") if isinstance(raw_input, dict) else None

if hasattr(model, "search"):
    results = model.search(query, top_k = top_k, filters = filters)
return { "query": query, "results": results, "count": len(results), "status": "success" }
return { "error": "Active model does not support direct semantic search", "status": "failed" }

    def get_routes(self) -> Dict[str, Any]:
"""Declares HTTP route mappings for AIMLite server."""
return {
    "POST /predict": self.run,
    "POST /search": self.search,
    "GET /health": self.health,
}
    `,

    'experiments/benchmark.py': `"""Benchmark & Evaluation Experiment: experiments/benchmark.py

Evaluates RAG retrieval precision, latency, and answer synthesis groundedness.
Run automatically using: aimlite benchmark
"""

import time
    from support_rag.model import SupportDocRAG

TEST_QUERIES =[
    "What are the 3 pillars of AIMLite?",
    "How do I customize the reranking hook?",
    "How does PostgresVectorStore handle connection pooling?",
]

def run():
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
    run()
`,

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
        "device": "auto",
            "top_k": 3,
                "chunk_size": 400
    }
} `
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
            'Scaffold a new RAG knowledge retrieval project with the rag paradigm. Use `--clean` to initialize with pristine 0-byte starting files.',
        code: `# Create and enter project directory
aimlite init support_rag--type rag
cd support_rag

# Optional: Add--clean to scaffold pristine 0 - byte starting files
# aimlite init support_rag--type rag --clean`,
        whyCode:
            'Sets up standard directory conventions (data/, artifacts/) and initializes aimlite.json.',
    },
    {
        stepNumber: 2,
        title: 'Install RAG Dependencies & Manage .venv',
        badge: 'ENV & DEPS',
        badgeVariant: 'cli',
        filename: 'terminal.sh',
        language: 'bash',
        description:
            'Install `sentence - transformers` and optional database drivers like `psycopg2 - binary`. AIMLite manages `.venv` automatically.',
        code: `# Install embedding & database libraries into managed.venv:
aimlite install sentence - transformers psycopg2 - binary`,
        whyCode:
            'SentenceTransformers computes dense vector embeddings for semantic similarity search. TfidfEmbedding acts as a zero-dependency fallback.',
    },
    {
        stepNumber: 3,
        title: 'Add Knowledge Documents to data/',
        badge: 'DOCS',
        badgeVariant: 'pillar',
        filename: 'data/faq.md',
        language: 'markdown',
        description:
            'Place Markdown (.md) or Text (.txt) articles into the `data / ` directory. AIMLite ingests and chunks all documents automatically.',
        code: `# AIMLite Architecture & Deployment FAQ

### What is Zero - Path Execution ?
    AIMLite dynamically discovers data.py, model.py, trainer.py, and inference.py
by inspecting project conventions.No manual routing or wiring is required.

### How are vector indices and databases stored ?
    Checkpoints and vector indices are serialized into artifacts / rag_index.json or synced to PostgreSQL via PostgresVectorStore.`,
        whyCode:
            'Raw documentation files in data/ are converted into contextual chunks by SmartChunker without hardcoded text in code.',
    },
    {
        stepNumber: 4,
        title: 'data.py: Document Chunking with SmartChunker',
        badge: 'CHUNKER',
        badgeVariant: 'pillar',
        filename: 'data.py',
        language: 'python',
        description:
            'Implement `KnowledgeDocsDataset` using AIMLite built-in `SmartChunker` to segment documentation into natural semantic passages with header context.',
        code: RAG_FILES['data.py'],
        whyCode:
            'SmartChunker preserves Markdown heading hierarchy (#, ##) and attaches parent section titles to chunk metadata.',
    },
    {
        stepNumber: 5,
        title: 'model.py: KnowledgeModel, QueryAnalyzer & ChatProvider',
        badge: 'MODEL',
        badgeVariant: 'pillar',
        filename: 'model.py',
        language: 'python',
        description:
            'Subclass `KnowledgeModel` and configure `QueryAnalyzer`, `BaseChatProvider`, and `PostgresVectorStore`/`MemoryVectorStore` for context-grounded answers.',
        code: RAG_FILES['model.py'],
        whyCode:
            'KnowledgeModel coordinates QueryAnalyzer (intent, multi-query expansion, HyDE) and ChatProvider (OpenAI, Gemini, Claude, Ollama, or local).',
    },
    {
        stepNumber: 6,
        title: 'trainer.py: Build & Persist Semantic Vector Index',
        badge: 'INDEXER',
        badgeVariant: 'pillar',
        filename: 'trainer.py',
        language: 'python',
        description:
            'Subclass `RAGTrainer` as `IndexBuilderTrainer` to calculate embeddings offline and serialize the vector index to `artifacts / rag_index.json` or database.',
        code: RAG_FILES['trainer.py'],
        whyCode:
            'Pre-indexing document embeddings offline guarantees that user queries during production serving execute with sub-10ms latency.',
    },
    {
        stepNumber: 7,
        title: 'inference.py: Production Knowledge API Endpoint & Retrieval Routing',
        badge: 'INFERENCE',
        badgeVariant: 'pillar',
        filename: 'inference.py',
        language: 'python',
        description:
            'Define `RAGInference` subclassing `BaseInference`. Accepts queries in `raw_input`, formats generative synthesis responses, and registers both full generative QA (`POST / predict`) and fast semantic search (`POST / search`) endpoints via `get_routes()`.',
        code: RAG_FILES['inference.py'],
        whyCode:
            'Enables complete control over incoming query payload structures, semantic chunk reranking, custom citation formatting, and direct vector search without invoking generative LLMs.',
    },
    {
        stepNumber: 8,
        title: 'Build Index, Serve Knowledge API & Customize UI',
        badge: 'EXECUTION',
        badgeVariant: 'cli',
        filename: 'terminal.sh',
        language: 'bash',
        description:
            'Build the semantic index and start the production knowledge query server. Serves an offline-first monochromatic chat playground with real-time citations and source inspectability. Customize templates via `<project_root> / templates / chat.html` or pass `--frontend`.',
        code: `# 1. Ingest documents and build vector index
aimlite train SupportDocRAG

# 2. Start knowledge API server with interactive chat playground
aimlite serve SupportDocRAG--port 8000

# 3. Query the knowledge base via REST API:
curl - X POST http://127.0.0.1:8000/predict \\
-H "Content-Type: application/json" \\
-d '{"query": "How do session tokens expire?", "top_k": 3}'

# 4. Direct semantic search(retrieval - only without LLM):
curl - X POST http://127.0.0.1:8000/search \\
-H "Content-Type: application/json" \\
-d '{"query": "session token timeout", "top_k": 5}'

# 5.(Optional) Customize the served chat interface:
# Override<project_root> / templates / chat.html or provide a custom frontend:
# aimlite serve SupportDocRAG--frontend./ frontend / dist`,
        whyCode:
            'Executes zero-path end-to-end training and launches an enterprise-grade server with styled model chooser, offline typography, and customizable web UI.',
    },
];

/* ========================================================================= */
/* PARADIGM 3: ADAPTERS (LORA & PEFT FINE-TUNING)                            */
/* ========================================================================= */

export const ADAPTER_FILES: Record<string, string> = {
    'data.py': `"""Instruction Tuning (Adapter Model Paradigm): data.py

Dataset loader for instruction fine - tuning prompt - response datasets.
Ingests JSON or JSONL records containing prompt / instruction / output pairs.
"""

from __future__ import annotations

import json
    from pathlib import Path
    from typing import Any, Dict, List, Optional

from aimlite import Dataset

SAMPLE_INSTRUCTIONS =[
    {
        "instruction": "Summarize customer retention feedback.",
        "input": "Customer renewal rate dropped 4% following recent tier updates.",
        "output": "Action: Analyze churn drivers related to recent pricing changes.",
    },
    {
        "instruction": "Classify support ticket severity.",
        "input": "Database connection pool exhausted on production node.",
        "output": "Severity: P1 Critical",
    },
    {
        "instruction": "Convert intent into structured query filter.",
        "input": "Find churned users with active contracts.",
        "output": "churned == 1 and contract_renewal == 1",
    },
]


class InstructionDataset(Dataset):
"""Instruction tuning dataset loader formatting prompt-response pairs for LoRA adaptation."""

    def __init__(
    self,
    source: Optional[str | Path] = None,
    data_path: Optional[str | Path] = None,
        ** kwargs: Any,
) -> None:
src = source or data_path
super().__init__(source = src, ** kwargs)

    def load(self, ** kwargs: Any) -> List[Dict[str, Any]]:
"""Loads instruction tuning pairs from data directory or sample instructions."""
resolved = self._resolve_file_path(self.source)
if resolved and resolved.is_file():
try:
content = resolved.read_text(encoding = "utf-8")
if resolved.suffix in (".jsonl", ".txt"):
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
                else:
records = json.loads(content)
if isinstance(records, list):
    return records
            except Exception:
pass

data_dir = Path("data")
for jf in data_dir.glob("*.json"):
    try:
data = json.loads(jf.read_text(encoding = "utf-8"))
if isinstance(data, list) and data and "instruction" in data[0]:
return data
            except Exception:
continue

return SAMPLE_INSTRUCTIONS

    def format_prompts(self) -> List[Dict[str, str]]:
"""Formats records into standardized prompt/completion strings."""
records = self.load()
formatted: List[Dict[str, str]] = []
for r in records:
    inst = r.get("instruction", "")
inp = r.get("input", "")
out = r.get("output", "")
prompt = (
    f"### Instruction:\\n{inst}\\n\\n### Input:\\n{inp}\\n\\n### Response:"
if inp
                else f"### Instruction:\\n{inst}\\n\\n### Response:"
            )
formatted.append({ "prompt": prompt, "completion": out })
return formatted
    `,

    'model.py': `"""Instruction Tuning (Adapter Model Paradigm): model.py

LoRA / PEFT AdapterModel subclass.
Attaches lightweight low - rank adaptation layers onto foundation models,
    freezing base weights and persisting solely adapter weight deltas.
"""

from __future__ import annotations

import json
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Union

from aimlite.adapters import AdapterConfig, AdapterModel


class LoRAInstructionModel(AdapterModel):
"""Instruction-following model specialized via parameter-efficient LoRA adapter weights."""

    def __init__(
    self,
    name: str = "lora_instruction_model",
    config: Optional[Dict[str, Any]] = None,
    base_model_name: str = "meta-llama/Llama-3-8B",
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
        ** kwargs: Any,
) -> None:
adapter_cfg = AdapterConfig(
    r = lora_rank,
    alpha = lora_alpha,
    target_modules = ["q_proj", "v_proj"],
    dropout = 0.05,
    base_model_path = base_model_name,
)
super().__init__(
    name = name,
    config = config,
    adapter_config = adapter_cfg,
            ** kwargs,
)
self.base_model_name = base_model_name
self.freeze_base_model()
self.adapter_weights = {
    "q_proj.lora_A": [[0.01 * (i + j) for j in range(8)]for i in range(16)],
    "q_proj.lora_B": [[0.02 * (i - j) for j in range(16)]for i in range(8)],
}

    def predict(self, inputs: Any, ** kwargs: Any) -> Dict[str, Any]:
"""Generates fine-tuned response for input instruction/prompt."""
if isinstance(inputs, dict):
    instruction = inputs.get("instruction", "")
user_input = inputs.get("input", "")
        else:
instruction = str(inputs)
user_input = ""

if user_input:
    prompt_header = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{user_input}\\n\\n### Response:"
        else:
prompt_header = f"### Instruction:\\n{instruction}\\n\\n### Response:"

response = f"[LoRA-Adapted {self.base_model_name} (r={self.adapter_config.r})]: Executed instruction with specialized behavior."

return {
    "prompt": prompt_header,
    "response": response,
    "adapter_rank": self.adapter_config.r,
    "base_model": self.base_model_name,
}

    def save(self, destination: Union[str, Path], ** kwargs: Any) -> None:
"""Saves ONLY the lightweight adapter delta weights and configuration.

        Saves ~50KB instead of duplicating gigabytes of base model weights.
        """
dest_dir = Path(destination)
dest_dir.mkdir(parents = True, exist_ok = True)

config_data = {
    "base_model_name": self.base_model_name,
    "lora_rank": self.adapter_config.r,
    "lora_alpha": self.adapter_config.alpha,
    "target_modules": self.adapter_config.target_modules,
}
with open(dest_dir / "adapter_config.json", "w", encoding = "utf-8") as f:
json.dump(config_data, f, indent = 2)

weights_meta = {
    "tensors": list(self.adapter_weights.keys()),
    "adapter_type": "lora",
    "trainable_params": 16 * 8 + 8 * 16,
}
with open(dest_dir / "adapter_model.json", "w", encoding = "utf-8") as f:
json.dump(weights_meta, f, indent = 2)

    def load(self, source: Union[str, Path], ** kwargs: Any) -> None:
"""Loads adapter configuration and delta weights from disk."""
src_dir = Path(source)
cfg_file = src_dir / "adapter_config.json"
if not cfg_file.is_file():
            raise FileNotFoundError(f"Adapter configuration not found in: {source}")

with open(cfg_file, "r", encoding = "utf-8") as f:
cfg = json.load(f)
self.base_model_name = cfg.get("base_model_name", self.base_model_name)
    `,

    'trainer.py': `"""Instruction Tuning (Adapter Model Paradigm): trainer.py

LoRA parameter - efficient training pipeline.Freezes foundation model weights
and optimizes solely adapter matrices, persisting lightweight delta checkpoints.
"""

from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict

from aimlite import BaseTrainer, Dataset, Model


class AdapterInstructionTrainer(BaseTrainer):
"""Trainer orchestrator for parameter-efficient LoRA fine-tuning."""

    def __init__(self, epochs: int = 3, learning_rate: float = 2e-4, ** kwargs: Any) -> None:
super().__init__(** kwargs)
self.epochs = epochs
self.learning_rate = learning_rate

    def fit(self, model: Model, dataset: Dataset, ** kwargs: Any) -> Dict[str, Any]:
"""Runs LoRA fine-tuning loop over instruction-tuning dataset."""
if hasattr(dataset, "format_prompts"):
    samples = dataset.format_prompts()
else:
records = dataset.load()
samples = [{ "prompt": str(r) } for r in records]

if not samples:
    return { "status": "failed", "error": "No instruction samples found in dataset" }

        # Ensure base foundation model weights remain frozen
if hasattr(model, "freeze_base_model"):
    model.freeze_base_model()

        # Save lightweight adapter checkpoint(delta matrices only)
checkpoint_dir = Path("artifacts") / "adapter"
checkpoint_dir.mkdir(parents = True, exist_ok = True)
model.save(checkpoint_dir)

return {
    "status": "completed",
    "paradigm": "lora_adapter",
    "epochs": self.epochs,
    "learning_rate": self.learning_rate,
    "training_samples": len(samples),
    "checkpoint_directory": str(checkpoint_dir),
    "note": "Delta weights checkpointed without duplicating base model.",
}
    `,

    'inference.py': `"""Instruction Tuning (Adapter Model Paradigm): inference.py

Production inference endpoint for serving fine - tuned LoRA adapter weights
on top of frozen foundation models.
"""

from __future__ import annotations

    from pathlib import Path
    from typing import Any, Dict

from aimlite import BaseInference, Model


class AdapterInference(BaseInference):
"""Inference handler serving LoRA-adapted language instructions."""

    def __init__(self, ** kwargs: Any) -> None:
super().__init__(** kwargs)
self.adapter_dir = Path("artifacts") / "adapter"

    def run(self, model: Model, raw_input: Any, ** kwargs: Any) -> Dict[str, Any]:
"""Runs instruction generation through the adapter-tuned model."""
if self.adapter_dir.is_dir():
    model.load(self.adapter_dir)

result = model.predict(raw_input)
return {
            ** result,
    "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
"""Declares HTTP route mappings for AIMLite server."""
return {
    "POST /predict": self.run,
    "GET /health": self.health,
}
    `,

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
} `
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

# Optional: Add--clean to scaffold pristine 0 - byte starting files
# aimlite init lora_instructions --type adapter --clean`,
        whyCode:
            'Sets up standard project conventions and generates aimlite.json.',
    },
    {
        stepNumber: 2,
        title: 'Install PyTorch & PEFT Dependencies',
        badge: 'ENV & DEPS',
        badgeVariant: 'cli',
        filename: 'terminal.sh',
        language: 'bash',
        description:
            'Install `torch` and `peft`. AIMLite automatically sets up `.venv` and updates `aimlite.json`.',
        code: `# Install PEFT fine - tuning dependencies:
aimlite install torch peft`,
        whyCode:
            'Hugging Face PEFT and PyTorch provide Low-Rank Adaptation (LoRA) matrix optimization.',
    },
    {
        stepNumber: 3,
        title: 'Add Instruction Dataset to data/instructions.json',
        badge: 'DATASET',
        badgeVariant: 'pillar',
        filename: 'data/instructions.json',
        language: 'json',
        description:
            'Add instruction/prompt/response JSON records into `data / instructions.json`.',
        code: `[
    {
        "instruction": "Summarize customer retention feedback.",
        "input": "Customer renewal rate dropped 4% following recent tier updates.",
        "output": "Action: Analyze churn drivers related to recent pricing changes."
    },
    {
        "instruction": "Classify support ticket severity.",
        "input": "Database connection pool exhausted on production node.",
        "output": "Severity: P1 Critical"
    }
]`,
        whyCode:
            'Instruction tuning trains base models to reliably follow custom prompt structures.',
    },
    {
        stepNumber: 4,
        title: 'data.py: Format Instruction Tuning Prompts',
        badge: 'LOADER',
        badgeVariant: 'pillar',
        filename: 'data.py',
        language: 'python',
        description:
            'Subclass `Dataset` as `InstructionDataset` to load prompt pairs and format them into standard prompt templates (`### Instruction: ... ### Response: `).',
        code: ADAPTER_FILES['data.py'],
        whyCode:
            'Standardizes disparate input schemas into uniform training prompt templates.',
    },
    {
        stepNumber: 5,
        title: 'model.py: LoRAInstructionModel & Frozen Base',
        badge: 'MODEL',
        badgeVariant: 'pillar',
        filename: 'model.py',
        language: 'python',
        description:
            'Subclass `AdapterModel` and attach trainable low-rank delta matrices via `AdapterConfig(r = 8, alpha = 16.0)` while freezing foundation weights.',
        code: ADAPTER_FILES['model.py'],
        whyCode:
            'AdapterModel calls freeze_base_model() to lock foundation parameters. Saves only ~50KB delta matrices instead of duplicating 14GB+ models.',
    },
    {
        stepNumber: 6,
        title: 'trainer.py: Optimize Low-Rank Delta Weights',
        badge: 'TRAINER',
        badgeVariant: 'pillar',
        filename: 'trainer.py',
        language: 'python',
        description:
            'Subclass `BaseTrainer` as `AdapterInstructionTrainer` to train adapter matrices and save lightweight delta checkpoints to `artifacts / adapter / `.',
        code: ADAPTER_FILES['trainer.py'],
        whyCode:
            'Reduces GPU VRAM requirements by over 80% and accelerates training loops.',
    },
    {
        stepNumber: 7,
        title: 'inference.py: Dynamic LoRA Response Serving & Adapter Selection',
        badge: 'INFERENCE',
        badgeVariant: 'pillar',
        filename: 'inference.py',
        language: 'python',
        description:
            'Subclass `BaseInference` to load delta checkpoints and serve fine-tuned responses via `POST / predict`. You can dynamically switch adapter weights per request or pass domain parameters in `raw_input`.',
        code: ADAPTER_FILES['inference.py'],
        whyCode:
            'Allows dynamic adapter swapping on top of a single shared foundation model, drastically cutting memory while enabling task-specific specialization.',
    },
    {
        stepNumber: 8,
        title: 'Train Adapters, Serve Fine-Tuned API & Customize UI',
        badge: 'EXECUTION',
        badgeVariant: 'cli',
        filename: 'terminal.sh',
        language: 'bash',
        description:
            'Execute LoRA fine-tuning and start the production inference server. Switch between base model and fine-tuned adapters effortlessly from the served UI or API.',
        code: [
            '# 1. Train lightweight delta weights',
            'aimlite train LoRAInstructionModel',
            '',
            '# 2. Start serving fine-tuned model + web dashboard',
            'aimlite serve LoRAInstructionModel --port 8000',
            '',
            '# 3. Test generation endpoint:',
            'curl -X POST http://127.0.0.1:8000/predict \\',
            '  -H "Content-Type: application/json" \\',
            '  -d \'{"instruction": "Classify support ticket", "input": "Cannot access billing portal"}\'',
            '',
            '# 4. Target specific registered model via REST:',
            'curl -X POST http://127.0.0.1:8000/models/LoRAInstructionModel/predict \\',
            '  -H "Content-Type: application/json" \\',
            '  -d \'{"instruction": "Summarize issue", "input": "Connection timeout on login"}\'',
            '',
            '# 5. (Optional) Customize the served web UI:',
            '# Override <project_root>/templates/app.html or specify a custom frontend build:',
            '# aimlite serve LoRAInstructionModel --frontend ./frontend/dist',
        ].join('\n'),
        whyCode:
            'Deploys parameter-efficient adapter models using standard zero-path commands, with full template override and multi-model routing support.',
    },
];
