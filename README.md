# ModelKit

**The Django for Machine Learning & AI** — an opinionated, convention-over-configuration Python ML framework with zero-path CLI execution.

---

## Authors & Collaborators

- **Mickyas Tesfaye** ([alazartesfaye42@gmail.com](mailto:alazartesfaye42@gmail.com))
- **Beamlak Tadesse** ([atocodes@gmail.com](mailto:atocodes@gmail.com))

---

## Why ModelKit?

Traditional machine learning projects suffer from repetitive boilerplate: scattered scripts, brittle path configurations, unstandardized train/test splits, hardcoded checkpoint paths, and ad-hoc serving code.

ModelKit provides a standardized, convention-based structure inspired by modern web frameworks like Django and Vite:
- **Zero-Path Execution**: Run `modelkit` or `./modelkit` anywhere inside your project directory. ModelKit resolves modules, sets up paths, and locates your data and models automatically.
- **Automated Virtual Environment Management**: Automatically detects and uses project `.venv` (powered by `uv` or `pip`).
- **First-Class IDE Typing Headers**: Generates `.modelkit/modelkit.pyi` on project initialization, giving VS Code, Cursor, and PyCharm immediate auto-completion, parameter hints, and docstrings with zero global installs.
- **Per-Model Artifacts & Lifecycle**: Supports multiple model classes per project with standardized naming (`models/<model_name>.pkl` and `experiments/<model_name>_snapshot.json`).
- **Built-in Inference Serving**: Production-ready HTTP server with `POST /predict`, `GET /health`, `GET /docs`, custom user-defined endpoints, and optional static frontend hosting.

---

## Quickstart

Requires Python **>= 3.10** (tested on 3.14) and [`uv`](https://docs.astral.sh/uv/) (or `pip`).

```bash
git clone https://github.com/Alazar42/ModelKit.git
cd ModelKit
uv sync
```

### Essential Commands

```bash
uv run cli --help             # Run CLI entrypoint via uv
./build/modelkit --help       # Run standalone compiled CLI binary directly
uv run test                   # Run core test suite (41 tests)
uv run pytest                 # Full pytest runner
```

---

## Project Structure Conventions

When you run `modelkit init <project_name>` (or `modelkit init .`), ModelKit scaffolds a standardized convention layout:

```text
my_project/
├── .modelkit/
│   └── modelkit.pyi          # IDE typing stubs for full autocomplete & IntelliSense
├── .venv/                    # Project-isolated virtual environment
├── data/                     # Raw datasets (.csv, .json, .txt, .md)
├── models/                   # Model architectures and saved weights (*.pkl)
├── experiments/              # Training run logs and metric snapshots (*.json)
├── data.py                   # Dataset definitions extending Dataset
├── model.py                  # Model classes extending Model / RAGModel / AdapterModel
├── trainer.py                # Training lifecycle hooks extending BaseTrainer
├── evaluator.py              # Metric benchmarks extending BaseEvaluator
├── inference.py              # Inference pipeline extending BaseInference
├── modelkit.json             # Project config, dependencies manifest, and paradigms
└── modelkit                  # Project-local CLI binary (self-contained executable)
```

---

## The 3 AI Paradigms

ModelKit is architected around the 3 primary modern machine learning paradigms:

### Paradigm 1: Training from Scratch (Customer Churn Classifier)

Bespoke tabular architectures, full optimization loops, and custom weights.

- **Dataset**: Kaggle Telecom Churn Dataset ([Kaggle Source](https://www.kaggle.com/datasets/barun2104/telecom-churn))
- **Required Dependencies**:
  ```bash
  modelkit install scikit-learn pandas
  # or
  pip install scikit-learn pandas
  ```
- **Example Files**: [`docs/examples/churn_scratch/`](docs/examples/churn_scratch/)
  - `data.py`: `TelecomChurnDataset(Dataset)` ingesting `data/telecom_churn.csv` with automatic 80/10/10 train/val/test splits.
  - `model.py`: `ChurnClassifier(Model)` wrapping scikit-learn's `RandomForestClassifier`.
  - `trainer.py`: `ChurnTrainer(BaseTrainer)` fitting and saving weights to `models/churn_classifier.pkl`.
  - `evaluator.py`: `ChurnEvaluator(BaseEvaluator)` calculating accuracy, precision, recall, and F1 score.
  - `inference.py`: `ChurnInference(BaseInference)` scoring churn risk probability and returning retention decisions.
- **Workflow**:
  ```bash
  # Initialize (new directory or in-place with .)
  modelkit init churn_model && cd churn_model
  # Place telecom_churn.csv in data/
  modelkit data validate
  modelkit train ChurnClassifier
  modelkit evaluate ChurnClassifier
  modelkit serve ChurnClassifier --port 8000
  ```

---

### Paradigm 2: RAG (Knowledge Base Question Answering)

Ground foundation models in enterprise documents with semantic vector search and zero hallucination.

- **Data Formats**: Markdown (`.md`) or text (`.txt`) documents placed in `data/`.
- **Required Dependencies**:
  ```bash
  modelkit install sentence-transformers numpy
  # or
  pip install sentence-transformers numpy
  ```
- **Example Files**: [`docs/examples/knowledge_rag/`](docs/examples/knowledge_rag/)
  - `data.py`: `KnowledgeDocsDataset(Dataset)` chunking documents with `TextSplitter`.
  - `model.py`: `SupportDocRAG(RAGModel)` binding `MemoryVectorStore` and `VectorRetriever`.
  - `trainer.py`: `IndexBuilderTrainer(BaseTrainer)` building and persisting the vector index to `models/rag_index.json`.
  - `inference.py`: `RAGInference(BaseInference)` querying the retriever and synthesizing grounded answers with citations.
- **Workflow**:
  ```bash
  modelkit init support_rag && cd support_rag
  # Place knowledge documents in data/
  modelkit train SupportDocRAG
  modelkit serve SupportDocRAG --port 8000
  ```

---

### Paradigm 3: Fine-Tuning (LoRA & PEFT Adapters)

Parameter-efficient adaptation with lightweight delta checkpoints (~50KB to 50MB instead of 14GB+).

- **Data Formats**: Prompt-response instruction pairs in `data/instructions.json` or `data/instructions.jsonl`.
- **Required Dependencies**:
  ```bash
  modelkit install torch peft
  # or
  pip install torch peft
  ```
- **Example Files**: [`docs/examples/instruction_adapter/`](docs/examples/instruction_adapter/)
  - `data.py`: `InstructionDataset(Dataset)` formatting instruction prompt templates.
  - `model.py`: `LoRAInstructionModel(AdapterModel)` configuring `AdapterConfig(r=8, alpha=16.0)` and freezing foundation weights.
  - `trainer.py`: `AdapterInstructionTrainer(BaseTrainer)` optimizing low-rank delta matrices.
  - `inference.py`: `AdapterInference(BaseInference)` executing fine-tuned generations via `POST /predict`.
- **Workflow**:
  ```bash
  modelkit init lora_app && cd lora_app
  # Place instructions.json in data/
  modelkit train LoRAInstructionModel
  modelkit serve LoRAInstructionModel --port 8000
  ```

---

## Zero-Path CLI Reference

ModelKit provides zero-path convention-over-configuration commands:

| Command | Description |
|---|---|
| `modelkit init [project_name \| .]` | Scaffolds a project in a new folder or directly in the current directory (`.`), setting up `.venv`, `.modelkit/modelkit.pyi` IDE stubs, starter files, and the local binary. |
| `modelkit install [packages...]` | Without arguments, installs all dependencies listed in `modelkit.json`. With packages, installs them into `.venv` using `uv` (or `pip`) and adds them to `modelkit.json`. |
| `modelkit data validate` | Validates dataset schema, record count, column names, and partition readiness across all datasets in `data/`. |
| `modelkit train [ModelName]` | Automatically discovers registered models and executes training. Saves checkpoint to `models/<model_name>.pkl` and snapshots to `experiments/`. |
| `modelkit evaluate [ModelName]` | Loads the model's checkpoint and calculates benchmark metrics on test partitions. |
| `modelkit serve [ModelName] [--port 8000] [--frontend <dir>]` | Starts an HTTP inference server exposing `POST /predict`, `GET /health`, `GET /docs`, custom routes, and optional static frontend hosting. |
| `modelkit doctor` | Diagnoses runtime health, virtual environment, hardware accelerator availability (`cuda`, `mps`, `cpu`), and directory permissions. |

---

## Standalone Executable & Build System (`build/`)

The standalone binary is packaged using Python's native `zipapp` format into a single self-contained executable with zero runtime dependencies:

```bash
# Compile / rebuild the standalone binary:
python3 build/build_cli.py

# Run directly without python invocation:
./build/modelkit doctor
```

---

## Interactive API Documentation Portal (`api_docs/`)

The full documentation portal is built with React, Vite, and modern styling, featuring interactive paradigm step-by-step guides, code explanation walkthroughs, copy-to-clipboard code blocks, and an API test playground:

```bash
cd api_docs
npm install
npm run dev     # Launch local docs dev server at http://localhost:5173
npm run build   # Build production bundle into api_docs/dist/
```

---

## Testing

ModelKit includes a comprehensive 41-test suite validating CLI commands, header generation, data validation, model lifecycle, inference serving, and end-to-end paradigm workflows:

```bash
uv run test        # Core test runner
uv run pytest      # Full pytest runner (41/41 passing)
```
