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

---

## Authors & Collaborators

- **Mickyas Tesfaye** ([alazartesfaye42@gmail.com](mailto:alazartesfaye42@gmail.com))
- **Beamlak Tadesse** ([atocodes@gmail.com](mailto:atocodes@gmail.com))

---

## Why AIMLite?

Traditional AI and machine learning projects suffer from repetitive boilerplate: scattered scripts, brittle path configurations, unstandardized train/test splits, hardcoded checkpoint paths, and ad-hoc serving code.

AIMLite provides a standardized, convention-based structure inspired by modern web frameworks like Django and Vite:
- **Zero-Path Execution**: Run `aimlite` anywhere inside your project directory. AIMLite resolves modules, sets up paths, and locates your data and models automatically.
- **Automated Virtual Environment Management**: Automatically creates `.venv` and pre-installs `aimlite` on `init`.
- **Instant IDE Type Safety**: Because `aimlite` is installed in the project `.venv` with PEP 561 headers (`py.typed`), editors like VS Code, Cursor, and PyCharm deliver immediate auto-completion, parameter hints, and docstrings with zero configuration.
- **Per-Model Artifacts & Lifecycle**: Supports multiple model classes per project with standardized naming (`models/<model_name>.pkl` and `experiments/<model_name>_snapshot.json`).
- **Built-in Inference Serving**: Production-ready HTTP server with `POST /predict`, `GET /health`, `GET /docs`, custom user-defined endpoints, and optional static frontend hosting.

---

## Quickstart

Requires Python **>= 3.10** (tested on 3.14) and [`uv`](https://docs.astral.sh/uv/) (or `pip`).

```bash
git clone https://github.com/Alazar42/aimlite.git
cd aimlite
uv sync
```

### Essential Commands

```bash
uv run aimlite --help         # Run AIMLite CLI
uv run test                   # Run core test suite (49 tests)
uv run pytest                 # Full pytest runner (49/49 passing)
```

---

## Project Structure Conventions

When you run `aimlite init <project_name>` (or `aimlite init .`), AIMLite scaffolds a clean, convention-based layout:

```text
my_project/
├── .venv/                    # Project-isolated virtual environment with aimlite pre-installed
├── data/                     # Raw datasets (.csv, .json, .parquet, .txt, etc.)
├── models/                   # Serialized production models and saved weights (*.pkl)
├── experiments/              # Training run logs and metric snapshots (*.json)
├── artifacts/                # Generated artifacts, vector embeddings, and outputs
├── checkpoints/              # Model weights and checkpoint deltas
├── my_project/               # Python application package
│   ├── __init__.py
│   ├── config.py             # Project configuration
│   ├── data.py               # Dataset definitions extending Dataset
│   ├── model.py              # Model classes extending Model
│   ├── trainer.py            # Training lifecycle hooks extending BaseTrainer
│   ├── evaluator.py          # Metric benchmarks extending BaseEvaluator
│   └── inference.py          # Inference pipeline extending BaseInference
└── aimlite.json              # Project manifest and config
```

---

## Flexible Dataset Naming (Any File, Any Format)

AIMLite gives developers total freedom over data file names and formats. You are never forced to use specific filenames or conventions.

Simply declare `filename = "<your_filename>"` on your `Dataset` subclass to match any file placed inside `data/`:

```python
from aimlite import Dataset

class UserChurnDataset(Dataset):
    filename = "telecom_churn.csv"  # Name it whatever you want!

class MonthlySalesDataset(Dataset):
    filename = "sales_q4_2026.parquet"

class LogAuditDataset(Dataset):
    filename = "system_logs.jsonl"
```

You can also pass `source` directly when instantiating or calling `load()`:

```python
dataset = UserChurnDataset(source="data/custom_export.csv")
```

### Supported Data Formats Out-of-the-Box
- **Tabular**: CSV (`.csv`), TSV (`.tsv`, `.tab`)
- **JSON**: Structured JSON arrays (`.json`), newline-delimited JSON (`.jsonl`)
- **Columnar**: Apache Parquet (`.parquet`, `.pq`)
- **Text & Docs**: Plain text (`.txt`), Markdown (`.md`)

---

## Unconstrained Modeling & Reference Templates

AIMLite is completely unopinionated about your modeling choices and does not force you into fixed paradigms. You can build **any** AI or machine learning system you want:
- **Classical & Tabular ML** (scikit-learn, XGBoost, LightGBM, CatBoost)
- **Deep Neural Networks** (PyTorch, TensorFlow, JAX)
- **LLMs & Prompt Pipelines** (LangChain, LlamaIndex, vLLM, Ollama, Hugging Face)
- **Computer Vision & Multimodal** (torchvision, timm, albumentations)
- **Reinforcement Learning & Custom Heuristics**

You can use **any file name** and **any data format** across all workflows. Simply declare `filename = "<your_filename>"` on your `Dataset` class or pass `source` directly.

The following reference implementations in `docs/examples/` are simply starting templates to demonstrate project structure — you are free to adapt, replace, or build completely custom workflows:

### Reference Template 1: Tabular ML (e.g. Churn Classifier)

Demonstrates a standard supervised learning workflow with feature preprocessing, scikit-learn models, and metric evaluation. You can use any tabular format (CSV, Parquet, TSV, etc.) with any filename you choose.

- **Example Implementation**: [`docs/examples/churn_scratch/`](docs/examples/churn_scratch/)
  - `data.py`: `TelecomChurnDataset(Dataset)` declaring `filename = "telecom_churn.csv"` with automatic 80/10/10 train/val/test splits.
  - `model.py`: `ChurnClassifier(Model)` wrapping scikit-learn's `RandomForestClassifier`.
  - `trainer.py`: `ChurnTrainer(BaseTrainer)` fitting and saving weights to `models/churn_classifier.pkl`.
  - `evaluator.py`: `ChurnEvaluator(BaseEvaluator)` calculating accuracy, precision, recall, and F1 score.
  - `inference.py`: `ChurnInference(BaseInference)` scoring churn risk probability and returning retention decisions.
- **Workflow**:
  ```bash
  # Initialize (new directory or in-place with .)
  aimlite init churn_model && cd churn_model
  # Place your data file in data/
  aimlite data validate
  aimlite train ChurnClassifier
  aimlite evaluate ChurnClassifier
  aimlite serve ChurnClassifier --port 8000
  ```

---

### Reference Template 2: Knowledge Base & RAG

Demonstrates semantic vector retrieval and context augmentation. Ingest text, Markdown, CSV, JSON, or any document source with any filename you choose.

- **Example Implementation**: [`docs/examples/knowledge_rag/`](docs/examples/knowledge_rag/)
  - `data.py`: `KnowledgeDocsDataset(Dataset)` chunking text passages.
  - `model.py`: `SupportDocRAG(RAGModel)` binding `MemoryVectorStore` and `VectorRetriever`.
  - `trainer.py`: `IndexBuilderTrainer(BaseTrainer)` building and persisting the vector index to `models/rag_index.json`.
  - `inference.py`: `RAGInference(BaseInference)` querying the retriever and synthesizing grounded answers with citations.
- **Workflow**:
  ```bash
  aimlite init support_rag && cd support_rag
  # Place your knowledge documents in data/
  aimlite train SupportDocRAG
  aimlite serve SupportDocRAG --port 8000
  ```

---

### Reference Template 3: Fine-Tuning & Adapters (LoRA & PEFT)

Demonstrates parameter-efficient fine-tuning with mathematical low-rank matrix decomposition, multi-adapter routing, zero-latency weight merging, and lightweight delta checkpoints (~50KB to 50MB instead of 14GB+). Ingest any prompt format, CSV, Parquet, JSON, JSONL, or custom schema with any filename you choose.

- **Key LoRA Capabilities**:
  - **Low-Rank Decomposition (`LoRALayer`)**: Freezes base foundation weights $W_0$ and trains low-rank matrices $A \in \mathbb{R}^{r \times d_{in}}$ and $B \in \mathbb{R}^{d_{out} \times r}$:
    $$h = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$
  - **Zero-Latency Serving**: In-place `model.merge_weights()` folds delta matrices directly into foundation weights with 0 inference overhead; `model.unmerge_weights()` restores base parameters.
  - **Multi-Adapter Hot-Swapping (`MultiAdapterManager`)**: Register multiple domain adapters on a single running model (`model.add_adapter("coder", cfg)`) and hot-swap at runtime per request (`model.set_active_adapter("coder")`).
  - **Parameter Diagnostics**: Real-time parameter accounting (`model.print_trainable_parameters()`) showing trainable vs total weights and memory reduction percentages.
  - **Hugging Face PEFT Format**: Serializes to standard `adapter_config.json`, `adapter_model.pkl`, and `adapter_metadata.json`.
  - **Zero Heavy Dependencies**: Runs out of the box in pure Python / NumPy, while fully supporting PyTorch tensors and PEFT.
- **Example Implementation**: [`docs/examples/instruction_adapter/`](docs/examples/instruction_adapter/)
  - `data.py`: `InstructionDataset(Dataset)` preparing fine-tuning data.
  - `model.py`: `LoRAInstructionModel(AdapterModel)` configuring adapter rank, alpha, and routing.
  - `trainer.py`: `AdapterInstructionTrainer(BaseTrainer)` optimizing low-rank delta matrices with parameter diagnostics.
  - `inference.py`: `AdapterInference(BaseInference)` executing fine-tuned generations via `POST /predict`.
- **Workflow**:
  ```bash
  aimlite init lora_app && cd lora_app
  # Place your dataset file in data/
  aimlite train LoRAInstructionModel
  aimlite serve LoRAInstructionModel --port 8000
  ```

---

## Zero-Path CLI Reference

AIMLite provides zero-path convention-over-configuration commands:

| Command | Description |
|---|---|
| `aimlite init [project_name \| .]` | Scaffolds a project in a new folder or directly in the current directory (`.`), setting up `.venv`, pre-installing `aimlite`, starter files, and conventions. |
| `aimlite install [packages...]` | Without arguments, installs all dependencies listed in `aimlite.json`. With packages, installs them into `.venv` using `uv` (or `pip`) and adds them to `aimlite.json`. |
| `aimlite data validate [target]` | Validates dataset schema, record count, column names, and partition readiness across all datasets in `data/`, or selectively validates by dataset class name, declared filename, path, or raw unbound data file. |
| `aimlite train [ModelName] [--resume [path]] [--checkpoint-dir <dir>]` | Automatically discovers registered models and executes training. Supports resuming from latest or specific past checkpoints with `--resume`, and saving to custom directories with `--checkpoint-dir`. Saves weights to `models/<model_name>.pkl` and snapshots to `experiments/`. |
| `aimlite evaluate [ModelName] [--checkpoint <path>]` | Loads the model's checkpoint and calculates benchmark metrics on test partitions. Discovers latest checkpoint automatically or loads an explicit past checkpoint with `--checkpoint`. |
| `aimlite serve [ModelName] [--checkpoint <path>] [--port 8000] [--frontend <dir>]` | Starts an HTTP inference server exposing `POST /predict`, `GET /health`, `GET /docs`, custom routes, and optional static frontend hosting. Supports loading a specific past checkpoint with `--checkpoint`. |
| `aimlite doctor` | Diagnoses runtime health, virtual environment, hardware accelerator availability (`cuda`, `mps`, `cpu`), and directory permissions. |

---

## Standalone Executable & Build System (`build/`)

The standalone binary is packaged using Python's native `zipapp` format into a single self-contained executable with zero runtime dependencies:

```bash
# Compile / rebuild the standalone binary:
python3 build/build_cli.py

# Run directly without python invocation:
./build/aimlite doctor
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

AIMLite includes a comprehensive 49-test suite validating CLI commands, header generation, data validation, model lifecycle, inference serving, LoRA low-rank adaptation, and end-to-end paradigm workflows:

```bash
uv run test        # Core test runner
uv run pytest      # Full pytest runner (49/49 passing)
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

