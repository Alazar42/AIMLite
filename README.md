# AIMLite
# AIMLite

**The Django for AI & Machine Learning** — an opinionated, convention-over-configuration Python framework with zero-path CLI execution.

---

## PyPI Installation

```bash
pip install aimlite
```

Or with [`uv`](https://docs.astral.sh/uv/):
```bash
uv add aimlite
```

```python
from aimlite import Model, Dataset, BaseTrainer, BaseConfig, BaseEvaluator, BaseInference
```
**The Django for AI & Machine Learning** — an opinionated, convention-over-configuration Python framework with zero-path CLI execution.

---

## PyPI Installation

```bash
pip install aimlite
```

Or with [`uv`](https://docs.astral.sh/uv/):
```bash
uv add aimlite
```

```python
from aimlite import Model, Dataset, BaseTrainer, BaseConfig, BaseEvaluator, BaseInference
```

---

## Authors & Collaborators

- **Mickyas Tesfaye** ([alazartesfaye42@gmail.com](mailto:alazartesfaye42@gmail.com))
- **Beamlak Tadesse** ([atocodes@gmail.com](mailto:atocodes@gmail.com))

---

## Why AIMLite?
## Why AIMLite?

Traditional AI and machine learning projects suffer from repetitive boilerplate: scattered scripts, brittle path configurations, unstandardized train/test splits, hardcoded checkpoint paths, and ad-hoc serving code.
Traditional AI and machine learning projects suffer from repetitive boilerplate: scattered scripts, brittle path configurations, unstandardized train/test splits, hardcoded checkpoint paths, and ad-hoc serving code.

AIMLite provides a standardized, convention-based structure inspired by modern web frameworks like Django and Vite:
- **Zero-Path Execution**: Run `aimlite` anywhere inside your project directory. AIMLite resolves modules, sets up paths, and locates your data and models automatically.
- **Automated Virtual Environment Management**: Automatically creates `.venv` and pre-installs `aimlite` on `init`.
- **Instant IDE Type Safety**: Because `aimlite` is installed in the project `.venv` with PEP 561 headers (`py.typed`), editors like VS Code, Cursor, and PyCharm deliver immediate auto-completion, parameter hints, and docstrings with zero configuration.
- **Per-Model Artifacts & Lifecycle**: Supports multiple model classes per project with standardized naming (`models/<model_name>.pkl` and `experiments/<model_name>_snapshot.json`).
- **Interactive Chat Playground & Inference Serving**: Production-ready HTTP server with built-in Interactive Chat UI (`/chat`), Swagger API documentation (`/docs`), OpenAPI schema (`/openapi.json`), `POST /predict`, `GET /health`, custom user-defined endpoints, and optional static frontend hosting.
- **Strict Connection Enforcement**: Direct error diagnostics for offline local daemons (Ollama), missing/invalid API keys, and database connections with zero silent fallbacks or mock data masks.

---

## Quickstart

Requires Python **>= 3.10** (tested on 3.14) and [`uv`](https://docs.astral.sh/uv/) (or `pip`).

```bash
git clone https://github.com/Alazar42/aimlite.git
cd aimlite
git clone https://github.com/Alazar42/aimlite.git
cd aimlite
uv sync
```

### Essential Commands

```bash
uv run aimlite --help         # Run AIMLite CLI
uv run test                   # Run core test suite (41 tests)
uv run pytest                 # Full pytest runner
```

---

## Project Structure Conventions

When you run `aimlite init <project_name>` (or `aimlite init .`), AIMLite scaffolds a standardized convention layout:

```text
my_project/
├── .aimlite/
│   └── workspace.json        # IDE workspace configuration
├── .venv/                    # Project-isolated virtual environment
├── data/                     # Raw datasets (.csv, .json, .txt, .md)
├── models/                   # Model architectures and saved weights (*.pkl)
├── experiments/              # Training run logs and metric snapshots (*.json)
├── data.py                   # Dataset definitions extending Dataset
├── model.py                  # Model classes extending Model / RAGModel / AdapterModel
├── trainer.py                # Training lifecycle hooks extending BaseTrainer
├── evaluator.py              # Metric benchmarks extending BaseEvaluator
├── inference.py              # Inference pipeline extending BaseInference
└── aimlite.json              # Project config, dependencies manifest, and paradigms
```

---

## The 3 AI Paradigms

AIMLite is architected around the 3 primary modern machine learning paradigms:

### Paradigm 1: Training from Scratch (Customer Churn Classifier)

Bespoke tabular architectures, full optimization loops, and custom weights.

- **Dataset**: Kaggle Telecom Churn Dataset ([Kaggle Source](https://www.kaggle.com/datasets/barun2104/telecom-churn))
- **Required Dependencies**:
  ```bash
  aimlite install scikit-learn pandas
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
  aimlite init churn_model && cd churn_model
  # Place telecom_churn.csv in data/
  aimlite data validate
  aimlite train ChurnClassifier
  aimlite evaluate ChurnClassifier
  aimlite serve ChurnClassifier --port 8000
  ```

---

### Reference Template 2: Knowledge Base & RAG (Enterprise Knowledge Engine)

Demonstrates semantic vector retrieval, query intelligence, smart document chunking, grounded LLM synthesis, and interactive chat serving. Ingest text, Markdown, CSV, JSON, or any document source with any filename you choose.

- **Data Formats**: Markdown (`.md`) or text (`.txt`) documents placed in `data/`.
- **Required Dependencies**:
  ```bash
  aimlite install sentence-transformers numpy
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
  aimlite init support_rag && cd support_rag
  # Place knowledge documents in data/
  aimlite train SupportDocRAG
  aimlite serve SupportDocRAG --port 8000
  ```

---

### Paradigm 3: Fine-Tuning (LoRA & PEFT Adapters)

Parameter-efficient adaptation with lightweight delta checkpoints (~50KB to 50MB instead of 14GB+).

- **Data Formats**: Prompt-response instruction pairs in `data/instructions.json` or `data/instructions.jsonl`.
- **Required Dependencies**:
  ```bash
  aimlite install torch peft
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
  aimlite init lora_app && cd lora_app
  # Place instructions.json in data/
  aimlite train LoRAInstructionModel
  aimlite serve LoRAInstructionModel --port 8000
  ```

---

## Zero-Path CLI Reference

AIMLite provides zero-path convention-over-configuration commands:
AIMLite provides zero-path convention-over-configuration commands:

| Command | Description |
|---|---|
| `aimlite init [project_name \| .]` | Scaffolds a project in a new folder or directly in the current directory (`.`), setting up `.venv`, pre-installing `aimlite`, starter files, and conventions. |
| `aimlite install [packages...]` | Without arguments, installs all dependencies listed in `aimlite.json`. With packages, installs them into `.venv` using `uv` (or `pip`) and adds them to `aimlite.json`. |
| `aimlite data validate` | Validates dataset schema, record count, column names, and partition readiness across all datasets in `data/`. |
| `aimlite train [ModelName]` | Automatically discovers registered models and executes training. Saves checkpoint to `models/<model_name>.pkl` and snapshots to `experiments/`. |
| `aimlite evaluate [ModelName]` | Loads the model's checkpoint and calculates benchmark metrics on test partitions. |
| `aimlite serve [ModelName] [--port 8000] [--frontend <dir>]` | Starts an HTTP inference server exposing `POST /predict`, `GET /health`, `GET /docs`, custom routes, and optional static frontend hosting. |
| `aimlite doctor` | Diagnoses runtime health, virtual environment, hardware accelerator availability (`cuda`, `mps`, `cpu`), and directory permissions. |

---

## How to Build & Package

AIMLite supports multiple build and packaging targets:

### 1. Build Standalone CLI Executable (`build/`)
Bundle the entire AIMLite framework and CLI into a single, portable executable using Python's native `zipapp` format (compressed bytecode, zero external binary dependencies):
```bash
# Compile standalone binary:
python3 build/build_cli.py

# Run directly without python invocation:
./build/aimlite doctor
```

### 2. Build Python Package (Wheel & Source Distribution)
Build standard distribution packages using Hatchling / uv:
```bash
# Using uv:
uv build

# Or using standard python build:
python3 -m pip install --upgrade build
python3 -m build
```

### 3. Build Interactive Documentation Portal (`api_docs/`)
Build the React/TypeScript/Vite API documentation portal:
```bash
cd api_docs
npm install
npm run build   # Produces optimized production bundle in api_docs/dist/
```

---

## Testing

AIMLite includes a comprehensive 41-test suite validating CLI commands, header generation, data validation, model lifecycle, inference serving, and end-to-end paradigm workflows:

```bash
# RAG Subsystem (Chat providers, SmartChunker, QueryAnalyzer, PostgresVectorStore, KnowledgeModel):
PYTHONPATH=src:. python3 -m unittest test.test_rag

# Adapters & LoRA Fine-Tuning:
PYTHONPATH=src:. python3 -m unittest test.test_adapters

# Zero-Path CLI & Discovery:
PYTHONPATH=src:. python3 -m unittest test.test_cli

# End-to-End Reference Paradigms:
PYTHONPATH=src:. python3 -m unittest test.test_examples

# Pillar Headers & Core Interfaces:
PYTHONPATH=src:. python3 -m unittest test.test_headers
```

---

## Business & Enterprise

AIMLite is **100% free and open-source under the Apache 2.0 license** for individual developers, students, researchers, and startups.

For companies and teams seeking enterprise extensions or dedicated collaboration:
- **Custom Data Connectors**: Direct integrations with enterprise data warehouses (Snowflake, BigQuery, PostgreSQL, Databricks).
- **Turnkey Production Starter Kits**: Pre-built commercial templates for customer retention/churn, enterprise RAG knowledge bases, and multi-adapter LoRA specialists.
- **Consulting & Implementation**: Architecture design, model optimization, and white-glove deployment directly with the core framework creators.

For commercial inquiries, partnerships, or custom integrations:
- **Mickyas Tesfaye**: [alazartesfaye42@gmail.com](mailto:alazartesfaye42@gmail.com)
- **Beamlak Tadesse**: [atocodes@gmail.com](mailto:atocodes@gmail.com)

---

## License

AIMLite is released under the [Apache 2.0 License](LICENSE). Free for commercial and non-commercial use.

