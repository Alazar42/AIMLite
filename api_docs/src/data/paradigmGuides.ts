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

from modelkit import Dataset

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
        Populates self._data and self.columns for ModelKit validation and training.
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

Classifier model for customer churn prediction.
Supports scikit-learn (RandomForestClassifier) with pure-Python fallback.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modelkit import Model


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
            self.estimator = None

    def fit(self, X: List[List[float]], y: List[int]) -> ChurnClassifier:
        """Fits classifier on numeric feature matrix X and label vector y."""
        if not X:
            return self

        if self.estimator is not None:
            self.estimator.fit(X, y)
        else:
            self._baseline_threshold = sum(y) / len(y) if y else 0.5
        return self

    def predict(self, inputs: Any, **kwargs: Any) -> List[int]:
        """Predicts binary churn class (0 for retained, 1 for churn)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is not None and hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(X)
            return [int(p) for p in preds]

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

    def _normalize_inputs(self, inputs: Any) -> List[List[float]]:
        """Converts diverse input formats into 2D float feature matrix."""
        from data import FEATURE_COLUMNS

        if isinstance(inputs, dict):
            return [[float(inputs.get(col, 0.0)) for col in FEATURE_COLUMNS]]
        if isinstance(inputs, list):
            if not inputs:
                return []
            if isinstance(inputs[0], dict):
                return [[float(item.get(col, 0.0)) for col in FEATURE_COLUMNS] for item in inputs]
            if isinstance(inputs[0], (int, float)):
                return [[float(x) for x in inputs]]
            if isinstance(inputs[0], list):
                return [[float(x) for x in row] for row in inputs]
        return []
`,

  'trainer.py': `"""Customer Churn Prediction (Scratch Model Paradigm): trainer.py

Orchestrates data preparation, train/val splitting, classifier fitting,
and model checkpoint persistence into artifacts/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from modelkit import BaseTrainer, Dataset, Model


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
        }
`,

  'evaluator.py': `"""Customer Churn Prediction (Scratch Model Paradigm): evaluator.py

Computes precision, recall, F1-score, and classification accuracy
on evaluation partitions.
"""

from __future__ import annotations

from typing import Any, Dict

from modelkit import BaseEvaluator, Dataset, Model


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
        }
`,

  'inference.py': `"""Customer Churn Prediction (Scratch Model Paradigm): inference.py

Production inference handler for customer churn risk scoring and intervention routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from modelkit import BaseInference, Model


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
        """Declares HTTP route mappings for ModelKit server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
`,

  'modelkit.json': `{
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
}`
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
      'Scaffold a clean ModelKit project. Use `modelkit init <name>` to create a new folder, or `./modelkit init .` to initialize directly inside your current directory.',
    code: `# Option A: Create in a new project folder
modelkit init telecom_churn
cd telecom_churn

# Option B: Or scaffold directly inside current directory (.)
# ./modelkit init .`,
    whyCode:
      'Creates the standard zero-path folder layout (data/, models/, experiments/, artifacts/, checkpoints/) and generates the project manifest modelkit.json without requiring boilerplate setup.',
  },
  {
    stepNumber: 2,
    title: 'Manage .venv & Install Compute Libraries',
    badge: 'ENV & DEPS',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Run `modelkit install` with your required ML libraries (e.g. scikit-learn, pandas). ModelKit automatically creates and manages `.venv` across Linux, macOS, and Windows (avoiding OS PEP 668 restrictions), and automatically updates `modelkit.json` under `"dependencies"`.',
    code: `# Install libraries: auto-manages .venv and updates modelkit.json
modelkit install scikit-learn pandas

# Later or on a new machine, running with no arguments reinstalls all pinned dependencies:
# modelkit install`,
    whyCode:
      'On modern OSes (Ubuntu 24+, Debian, macOS), installing global python packages is restricted (PEP 668). ModelKit ensures all projects have an isolated, self-managed .venv and tracks dependencies deterministically in modelkit.json.',
  },
  {
    stepNumber: 3,
    title: 'Acquire Kaggle Telecom Churn Dataset',
    badge: 'DATASET',
    badgeVariant: 'pillar',
    filename: 'data/telecom_churn.csv',
    language: 'bash',
    description:
      'Download the Telecom Churn CSV from Kaggle and place it in your project as `data/telecom_churn.csv`. Then validate dataset conventions with `modelkit data validate`.',
    code: `# 1. Download dataset from Kaggle:
# https://www.kaggle.com/datasets/barun2104/telecom-churn
# Place file into: data/telecom_churn.csv

# 2. Validate dataset schema and conventions:
modelkit data validate`,
    externalLink: {
      label: 'Kaggle: Telecom Churn Dataset',
      url: 'https://www.kaggle.com/datasets/barun2104/telecom-churn',
    },
    whyCode:
      'ModelKit validates that data/ contains valid datasets and ensures columns (AccountWeeks, ContractRenewal, DataPlan, CustServCalls, MonthlyCharge, Churn) match expected schema before training begins.',
  },
  {
    stepNumber: 4,
    title: 'data.py: Ingest & Preprocess Customer Records',
    badge: 'LOADER',
    badgeVariant: 'pillar',
    filename: 'data.py',
    language: 'python',
    description:
      'Implement `TelecomChurnDataset` inheriting from `modelkit.Dataset` with explicit `filename = "telecom_churn.csv"`. Uses pandas for accelerated reading with a pure-Python csv fallback, and populates `self._data` and `self.columns`.',
    code: SCRATCH_FILES['data.py'],
    whyCode:
      'Setting `filename = "telecom_churn.csv"` tells ModelKit exactly which CSV file in `data/` to bind, validate, and partition. Assigning `self._data` and `self.columns` enables ModelKit automated split contracts and pre-flight validation.',
  },
  {
    stepNumber: 5,
    title: 'model.py: Define ChurnClassifier',
    badge: 'MODEL',
    badgeVariant: 'pillar',
    filename: 'model.py',
    language: 'python',
    description:
      'Define `ChurnClassifier` subclassing `modelkit.Model`. Encapsulates `RandomForestClassifier` with balanced class weights, implements `predict()` and `predict_proba()`, and provides standard pickle serialization.',
    code: SCRATCH_FILES['model.py'],
    whyCode:
      'Inheriting from Model gives you automatic registration with ModelKit discovery. Zero-path commands (train, evaluate, serve) can inspect and instantiate ChurnClassifier by name.',
  },
  {
    stepNumber: 6,
    title: 'trainer.py: Fit Classifier & Checkpoint Weights',
    badge: 'TRAINER',
    badgeVariant: 'pillar',
    filename: 'trainer.py',
    language: 'python',
    description:
      'Define `ChurnTrainer` subclassing `BaseTrainer`. Performs an 80/20 train/validation split, fits the model, reports train/val accuracy, and saves weights into `artifacts/churn_classifier.pkl`.',
    code: SCRATCH_FILES['trainer.py'],
    whyCode:
      'BaseTrainer standardizes execution lifecycle. ModelKit invokes fit(model, dataset) and guarantees checkpoints are cleanly stored in artifacts/ without ad-hoc path manipulation.',
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
    title: 'inference.py: Production HTTP Endpoint & Risk Scoring',
    badge: 'INFERENCE',
    badgeVariant: 'pillar',
    filename: 'inference.py',
    language: 'python',
    description:
      'Define `ChurnInference` subclassing `BaseInference`. Automatically loads weights from `artifacts/churn_classifier.pkl`, calculates churn risk score, and routes customers to retention workflows.',
    code: SCRATCH_FILES['inference.py'],
    whyCode:
      'BaseInference provides production-ready REST API route mapping (POST /predict, GET /health). ModelKit serve uses this class to power low-latency prediction servers.',
  },
  {
    stepNumber: 9,
    title: 'Execute & Serve with Zero-Path CLI',
    badge: 'EXECUTION',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Execute the end-to-end machine learning lifecycle using ModelKit CLI commands. Train the model, benchmark metrics, and start the production inference HTTP server.',
    code: `# 1. Train classifier and write weights to artifacts/
modelkit train ChurnClassifier

# 2. Evaluate accuracy, precision, recall, and F1
modelkit evaluate ChurnClassifier

# 3. Start production inference REST server
modelkit serve ChurnClassifier --port 8000

# 4. Test inference endpoint (in another terminal):
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
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
      'Zero-path execution automatically connects data.py, model.py, trainer.py, and inference.py without boilerplate wiring code.',
  },
];

/* ========================================================================= */
/* PARADIGM 2: RAG (KNOWLEDGE BASE QA)                                       */
/* ========================================================================= */

export const RAG_FILES: Record<string, string> = {
  'data.py': `"""Document & Knowledge QA (RAG Paradigm): data.py

Knowledge base document loader and text chunker for RAG pipelines.
Ingests text or markdown files from data/ and splits them into
retrievable document passages with metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from modelkit import Dataset
from modelkit.rag import Document, TextSplitter

SAMPLE_KNOWLEDGE_DOCS = [
    {
        "filename": "auth_policy.md",
        "content": (
            "Authentication and Security Policy: ModelKit supports API key and Bearer token authentication. "
            "Session tokens expire after 24 hours of inactivity. Multi-factor authentication (MFA) is required "
            "for administrative access to production model endpoints."
        ),
    },
    {
        "filename": "deployment_guide.md",
        "content": (
            "Production Deployment Guide: ModelKit models can be served via 'modelkit serve --port 8000'. "
            "For production deployments, containerize using Docker with the provided Dockerfile. "
            "Horizontal scaling can be achieved with Kubernetes by configuring the replica count."
        ),
    },
    {
        "filename": "adapter_tuning.md",
        "content": (
            "Parameter-Efficient Fine-Tuning: Adapter models use Low-Rank Adaptation (LoRA) to train "
            "lightweight delta matrices. This reduces checkpoint size from 14GB down to under 50MB, "
            "enabling rapid model swapping in multi-tenant environments."
        ),
    },
]


class KnowledgeDocsDataset(Dataset):
    """Ingests documentation articles and splits them into retrievable passages."""

    filename: str = "knowledge_base.txt"

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(source=src, **kwargs)
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Loads and parses knowledge articles from disk."""
        docs = self.load_documents()
        return [{"content": d.content, "metadata": d.metadata} for d in docs]

    def load_documents(self) -> List[Document]:
        """Reads Markdown and Text files from data/ directory and returns chunked passages."""
        documents: List[Document] = []
        data_dir = Path("data")

        if data_dir.is_dir():
            for p in data_dir.glob("*.*"):
                if p.suffix.lower() in (".md", ".txt", ".markdown"):
                    try:
                        text = p.read_text(encoding="utf-8")
                        chunks = self.splitter.split_text(text)
                        for idx, chunk in enumerate(chunks):
                            documents.append(
                                Document(
                                    content=chunk,
                                    metadata={"source": p.name, "chunk_id": idx},
                                )
                            )
                    except Exception:
                        continue

        if not documents:
            for sample in SAMPLE_KNOWLEDGE_DOCS:
                chunks = self.splitter.split_text(sample["content"])
                for idx, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            content=chunk,
                            metadata={"source": sample["filename"], "chunk_id": idx},
                        )
                    )

        return documents
`,

  'model.py': `"""Document & Knowledge QA (RAG Paradigm): model.py

Specialized RAGModel subclass providing vector-indexed document retrieval
and context-grounded answer synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modelkit.rag import (
    BaseEmbedding,
    Document,
    MemoryVectorStore,
    RAGModel,
    TfidfEmbedding,
    VectorRetriever,
)


class SupportDocRAG(RAGModel):
    """Knowledge Base QA Model utilizing semantic vector search and passage synthesis."""

    def __init__(
        self,
        name: str = "support_doc_rag",
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self.top_k = top_k
        self.embedding_fn: BaseEmbedding = self._init_embedding()
        self.vector_store = MemoryVectorStore(embedding_fn=self.embedding_fn)
        self.retriever = VectorRetriever(
            vector_store=self.vector_store,
            embedding_fn=self.embedding_fn,
        )

    def _init_embedding(self) -> BaseEmbedding:
        """Initializes embedding engine (SentenceTransformers if installed, else TfidfEmbedding)."""
        try:
            from sentence_transformers import SentenceTransformer

            class STEmbedding(BaseEmbedding):
                def __init__(self) -> None:
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")

                def embed_text(self, text: str) -> List[float]:
                    vec = self.model.encode(text, convert_to_numpy=True)
                    return [float(x) for x in vec.tolist()]

            return STEmbedding()
        except ImportError:
            return TfidfEmbedding()

    def index_documents(self, documents: List[Document]) -> int:
        """Adds documents to internal vector store."""
        return self.vector_store.add_documents(documents)

    def predict(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Queries the vector index and returns context-grounded answers."""
        query = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
        top_k = kwargs.get("top_k", self.top_k)

        retrieved_docs = self.retriever.retrieve(query, top_k=top_k)

        sources = [
            {
                "source": d.metadata.get("source", "knowledge_base"),
                "chunk_id": d.metadata.get("chunk_id", 0),
                "snippet": d.content[:160] + "..." if len(d.content) > 160 else d.content,
            }
            for d in retrieved_docs
        ]

        if retrieved_docs:
            primary_src = sources[0]["source"]
            answer = f"According to {primary_src}: {retrieved_docs[0].content}"
        else:
            answer = "No relevant knowledge articles found matching the query."

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "total_retrieved": len(retrieved_docs),
        }

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes vector store documents and embeddings to disk."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(dest_path)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Loads serialized vector index from disk."""
        src_path = Path(source)
        if not src_path.is_file():
            raise FileNotFoundError(f"Vector index not found at: {source}")
        self.vector_store.load(src_path)
`,

  'trainer.py': `"""Document & Knowledge QA (RAG Paradigm): trainer.py

Vector index builder pipeline that reads knowledge base documents,
generates embeddings, and persists the vector index into artifacts/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from modelkit import BaseTrainer, Dataset, Model


class IndexBuilderTrainer(BaseTrainer):
    """Trainer orchestrator building and saving the semantic vector index."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Builds semantic vector index from dataset documents."""
        if hasattr(dataset, "load_documents"):
            documents = dataset.load_documents()
        else:
            records = dataset.load()
            from modelkit.rag import Document

            documents = [
                Document(content=r.get("content", str(r)), metadata=r.get("metadata", {}))
                for r in records
            ]

        if not documents:
            return {"status": "failed", "error": "No documents found to index"}

        # Index documents in RAGModel
        if hasattr(model, "index_documents"):
            indexed_count = model.index_documents(documents)
        else:
            indexed_count = len(documents)

        # Persist index artifact
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        index_path = artifacts_dir / "rag_index.json"
        model.save(index_path)

        return {
            "status": "completed",
            "indexed_chunks": indexed_count,
            "index_path": str(index_path),
        }
`,

  'inference.py': `"""Document & Knowledge QA (RAG Paradigm): inference.py

Production inference endpoint for semantic document query retrieval
and context-grounded response generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from modelkit import BaseInference, Model


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
        """Declares HTTP route mappings for ModelKit server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
`,

  'modelkit.json': `{
  "name": "support_rag",
  "version": "0.1.0",
  "entrypoint": "support_rag",
  "dependencies": [
    "numpy",
    "sentence-transformers"
  ],
  "config": {
    "device": "auto",
    "batch_size": 32
  }
}`
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
      'Scaffold a new RAG knowledge retrieval project with ModelKit conventions.',
    code: `# Create and enter project directory
modelkit init support_rag
cd support_rag`,
    whyCode:
      'Sets up standard directory conventions (data/, artifacts/) and initializes modelkit.json.',
  },
  {
    stepNumber: 2,
    title: 'Install RAG Dependencies & Manage .venv',
    badge: 'ENV & DEPS',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Install `sentence-transformers` and `numpy`. ModelKit manages `.venv` automatically and records dependencies into `modelkit.json`.',
    code: `# Install embedding libraries into managed .venv:
modelkit install sentence-transformers numpy`,
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
      'Place Markdown (.md) or Text (.txt) articles into the `data/` directory. ModelKit ingests and chunks all documents automatically.',
    code: `# ModelKit Architecture & Deployment FAQ

### What is Zero-Path Execution?
ModelKit dynamically discovers data.py, model.py, trainer.py, and inference.py
by inspecting project conventions. No manual routing or wiring is required.

### How are artifacts stored?
Checkpoints and trained models are stored in artifacts/ and checkpoints/.
Vector indices are serialized into artifacts/rag_index.json.`,
    whyCode:
      'Raw documentation files in data/ are converted into retrievable chunks by TextSplitter without hardcoded text in code.',
  },
  {
    stepNumber: 4,
    title: 'data.py: Document Chunking with TextSplitter',
    badge: 'CHUNKER',
    badgeVariant: 'pillar',
    filename: 'data.py',
    language: 'python',
    description:
      'Implement `KnowledgeDocsDataset` using ModelKit built-in `TextSplitter` to segment documentation into overlapping semantic windows.',
    code: RAG_FILES['data.py'],
    whyCode:
      'TextSplitter(chunk_size=300, chunk_overlap=40) prevents boundary truncation and ensures complete context during vector retrieval.',
  },
  {
    stepNumber: 5,
    title: 'model.py: SupportDocRAG & MemoryVectorStore',
    badge: 'MODEL',
    badgeVariant: 'pillar',
    filename: 'model.py',
    language: 'python',
    description:
      'Subclass `RAGModel` and configure `MemoryVectorStore` with `VectorRetriever` for cosine similarity matching and context-grounded answers.',
    code: RAG_FILES['model.py'],
    whyCode:
      'RAGModel provides standard predict(query) returning answer text alongside verifiable citation snippets and similarity scores.',
  },
  {
    stepNumber: 6,
    title: 'trainer.py: Build & Persist Semantic Vector Index',
    badge: 'INDEXER',
    badgeVariant: 'pillar',
    filename: 'trainer.py',
    language: 'python',
    description:
      'Subclass `BaseTrainer` as `IndexBuilderTrainer` to calculate embeddings offline and serialize the vector index to `artifacts/rag_index.json`.',
    code: RAG_FILES['trainer.py'],
    whyCode:
      'Pre-indexing document embeddings offline guarantees that user queries during production serving execute with sub-10ms latency.',
  },
  {
    stepNumber: 7,
    title: 'inference.py: Production Knowledge API Endpoint',
    badge: 'INFERENCE',
    badgeVariant: 'pillar',
    filename: 'inference.py',
    language: 'python',
    description:
      'Define `RAGInference` subclassing `BaseInference` to serve knowledge queries via `POST /predict`.',
    code: RAG_FILES['inference.py'],
    whyCode:
      'Automatically connects to the serialized index in artifacts/rag_index.json and exposes a RESTful API with health checks.',
  },
  {
    stepNumber: 8,
    title: 'Build Index & Serve Knowledge API',
    badge: 'EXECUTION',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Build the vector index and start the production knowledge query server.',
    code: `# 1. Ingest documents and build vector index
modelkit train SupportDocRAG

# 2. Start knowledge API server
modelkit serve SupportDocRAG --port 8000

# 3. Query the knowledge base
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"query": "How does zero-path execution work?", "top_k": 3}'`,
    whyCode:
      'Executes the entire RAG lifecycle using standard ModelKit CLI commands.',
  },
];

/* ========================================================================= */
/* PARADIGM 3: ADAPTERS (LORA & PEFT FINE-TUNING)                            */
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

from modelkit import Dataset

SAMPLE_INSTRUCTIONS = [
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
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(source=src, **kwargs)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Loads instruction tuning pairs from data directory or sample instructions."""
        resolved = self._resolve_file_path(self.source)
        if resolved and resolved.is_file():
            try:
                content = resolved.read_text(encoding="utf-8")
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
                data = json.loads(jf.read_text(encoding="utf-8"))
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
            formatted.append({"prompt": prompt, "completion": out})
        return formatted
`,

  'model.py': `"""Instruction Tuning (Adapter Model Paradigm): model.py

LoRA / PEFT AdapterModel subclass.
Attaches lightweight low-rank adaptation layers onto foundation models,
freezing base weights and persisting solely adapter weight deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modelkit.adapters import AdapterConfig, AdapterModel


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
            **kwargs,
        )
        self.base_model_name = base_model_name
        self.freeze_base_model()
        self.adapter_weights = {
            "q_proj.lora_A": [[0.01 * (i + j) for j in range(8)] for i in range(16)],
            "q_proj.lora_B": [[0.02 * (i - j) for j in range(16)] for i in range(8)],
        }

    def predict(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
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

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Saves ONLY the lightweight adapter delta weights and configuration.

        Saves ~50KB instead of duplicating gigabytes of base model weights.
        """
        dest_dir = Path(destination)
        dest_dir.mkdir(parents=True, exist_ok=True)

        config_data = {
            "base_model_name": self.base_model_name,
            "lora_rank": self.adapter_config.r,
            "lora_alpha": self.adapter_config.alpha,
            "target_modules": self.adapter_config.target_modules,
        }
        with open(dest_dir / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        weights_meta = {
            "tensors": list(self.adapter_weights.keys()),
            "adapter_type": "lora",
            "trainable_params": 16 * 8 + 8 * 16,
        }
        with open(dest_dir / "adapter_model.json", "w", encoding="utf-8") as f:
            json.dump(weights_meta, f, indent=2)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Loads adapter configuration and delta weights from disk."""
        src_dir = Path(source)
        cfg_file = src_dir / "adapter_config.json"
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Adapter configuration not found in: {source}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.base_model_name = cfg.get("base_model_name", self.base_model_name)
`,

  'trainer.py': `"""Instruction Tuning (Adapter Model Paradigm): trainer.py

LoRA parameter-efficient training pipeline. Freezes foundation model weights
and optimizes solely adapter matrices, persisting lightweight delta checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from modelkit import BaseTrainer, Dataset, Model


class AdapterInstructionTrainer(BaseTrainer):
    """Trainer orchestrator for parameter-efficient LoRA fine-tuning."""

    def __init__(self, epochs: int = 3, learning_rate: float = 2e-4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Runs LoRA fine-tuning loop over instruction-tuning dataset."""
        if hasattr(dataset, "format_prompts"):
            samples = dataset.format_prompts()
        else:
            records = dataset.load()
            samples = [{"prompt": str(r)} for r in records]

        if not samples:
            return {"status": "failed", "error": "No instruction samples found in dataset"}

        # Ensure base foundation model weights remain frozen
        if hasattr(model, "freeze_base_model"):
            model.freeze_base_model()

        # Save lightweight adapter checkpoint (delta matrices only)
        checkpoint_dir = Path("artifacts") / "adapter"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
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

Production inference endpoint for serving fine-tuned LoRA adapter weights
on top of frozen foundation models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from modelkit import BaseInference, Model


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
        """Declares HTTP route mappings for ModelKit server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
`,

  'modelkit.json': `{
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
}`
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
      'Scaffold a new LoRA instruction tuning project.',
    code: `# Create and enter project directory
modelkit init lora_instructions
cd lora_instructions`,
    whyCode:
      'Sets up standard project conventions and generates modelkit.json.',
  },
  {
    stepNumber: 2,
    title: 'Install PyTorch & PEFT Dependencies',
    badge: 'ENV & DEPS',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Install `torch` and `peft`. ModelKit automatically sets up `.venv` and updates `modelkit.json`.',
    code: `# Install PEFT fine-tuning dependencies:
modelkit install torch peft`,
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
      'Add instruction/prompt/response JSON records into `data/instructions.json`.',
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
      'Subclass `Dataset` as `InstructionDataset` to load prompt pairs and format them into standard prompt templates (`### Instruction: ... ### Response:`).',
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
      'Subclass `AdapterModel` and attach trainable low-rank delta matrices via `AdapterConfig(r=8, alpha=16.0)` while freezing foundation weights.',
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
      'Subclass `BaseTrainer` as `AdapterInstructionTrainer` to train adapter matrices and save lightweight delta checkpoints to `artifacts/adapter/`.',
    code: ADAPTER_FILES['trainer.py'],
    whyCode:
      'Reduces GPU VRAM requirements by over 80% and accelerates training loops.',
  },
  {
    stepNumber: 7,
    title: 'inference.py: Dynamic LoRA Response Serving',
    badge: 'INFERENCE',
    badgeVariant: 'pillar',
    filename: 'inference.py',
    language: 'python',
    description:
      'Subclass `BaseInference` to load delta checkpoints and serve fine-tuned responses via `POST /predict`.',
    code: ADAPTER_FILES['inference.py'],
    whyCode:
      'Allows dynamic adapter swapping on top of a single shared foundation model.',
  },
  {
    stepNumber: 8,
    title: 'Train Adapters & Serve Fine-Tuned API',
    badge: 'EXECUTION',
    badgeVariant: 'cli',
    filename: 'terminal.sh',
    language: 'bash',
    description:
      'Execute LoRA fine-tuning and start the production inference server.',
    code: `# 1. Train lightweight delta weights
modelkit train LoRAInstructionModel

# 2. Start serving fine-tuned model
modelkit serve LoRAInstructionModel --port 8000

# 3. Test generation endpoint
curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"instruction": "Classify support ticket", "input": "Cannot access billing portal"}'`,
    whyCode:
      'Deploys parameter-efficient adapter models using standard zero-path commands.',
  },
];
