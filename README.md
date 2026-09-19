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
- **Automated Virtual Environment Management**: Automatically creates `.venv`, generates a tailored `requirements.txt`, and auto-installs dependencies on `init`.
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
uv sync
```

### Essential Commands

```bash
uv run aimlite --help         # Run AIMLite CLI
uv run test                   # Run core test suite
uv run pytest                 # Full pytest runner
```

---

## Project Structure Conventions

When you run `aimlite init <project_name>` (or `aimlite init .`), AIMLite scaffolds a clean, convention-based layout:

```text
my_project/
├── .venv/                    # Project-isolated virtual environment with dependencies pre-installed
├── .env                      # Active environment variables (model names, host URLs, API keys)
├── .env.example              # Template reference for credentials and database connections
├── requirements.txt          # Auto-generated tailored requirements for your paradigm & provider
├── data/                     # Raw datasets (.csv, .json, .parquet, .txt, .md, etc.)
├── models/                   # Serialized production models and saved weights (*.pkl)
├── experiments/              # Training run logs, benchmarks, and metric snapshots (*.json)
├── artifacts/                # Generated artifacts, vector embeddings (rag_index.json), and outputs
├── checkpoints/              # Model weights and checkpoint deltas
├── client.py                 # Ready-to-run interactive/batch client inference starter
├── my_project/               # Python application package
│   ├── __init__.py
│   ├── config.py             # Project configuration with zero-dependency load_dotenv
│   ├── chat_provider.py      # Chat Provider, system prompts, and Query Intelligence config
│   ├── store.py              # Semantic vector store (PostgreSQL pgvector or MemoryVectorStore)
│   ├── data.py               # Dataset definitions extending Dataset (e.g. KnowledgeDocsDataset)
│   ├── model.py              # Model classes extending Model (e.g. SupportDocRAG)
│   ├── trainer.py            # Training lifecycle hooks extending BaseTrainer
│   ├── evaluator.py          # Metric benchmarks extending BaseEvaluator
│   └── inference.py          # Inference pipeline extending BaseInference
└── aimlite.json              # Project manifest and config
```

---

## Environment Variables & Configuration (.env)

AIMLite includes built-in, zero-dependency environment configuration via `aimlite.load_dotenv()`:

```ini
# ==============================================================================
# AIMLite RAG Knowledge Engine: Environment Variables (.env)
# ==============================================================================

# --- Chat LLM Model Identifiers ---
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
GEMINI_MODEL=gemini-1.5-flash
OLLAMA_MODEL=llama3.2
LOCAL_MODEL=meta-llama/Llama-3.2-3B

# --- Embedding Engine & Model ---
EMBEDDING_ENGINE=ollama                 # 'ollama', 'sentence-transformers', 'openai', 'tfidf'
EMBEDDING_MODEL=nomic-embed-text        # e.g. nomic-embed-text, all-MiniLM-L6-v2, text-embedding-3-small

# --- Cloud API Credentials (when using cloud providers) ---
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=

# --- Local Ollama Endpoint ---
OLLAMA_HOST=http://localhost:11434

# --- Storage & Database Vector Backend ---
VECTOR_STORE=memory                     # 'memory' (artifacts/rag_index.json) or 'postgres' (PostgreSQL pgvector)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/knowledge_db

# --- Hardware Acceleration & Serving ---
AIMLITE_DEVICE=auto
AIMLITE_PORT=8000
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
- **Text & Docs**: Plain text (`.txt`), Markdown (`.md`, `.markdown`), reStructuredText (`.rst`)

---

## Unconstrained Modeling & Reference Templates

AIMLite is completely unopinionated about your modeling choices and does not force you into fixed paradigms. You can build **any** AI or machine learning system you want:
- **Classical & Tabular ML** (scikit-learn, XGBoost, LightGBM, CatBoost)
- **Deep Neural Networks** (PyTorch, TensorFlow, JAX)
- **LLMs & RAG Pipelines** (LangChain, LlamaIndex, vLLM, Ollama, OpenAI, Anthropic, Gemini, Hugging Face)
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
  aimlite install -r requirements.txt
  # Place your data file in data/
  aimlite data validate
  aimlite train ChurnClassifier
  aimlite evaluate ChurnClassifier
  aimlite serve ChurnClassifier --port 8000
  ```

---

### Reference Template 2: Knowledge Base & RAG (Enterprise Knowledge Engine)

Demonstrates semantic vector retrieval, query intelligence, smart document chunking, grounded LLM synthesis, and interactive chat serving. Ingest text, Markdown, CSV, JSON, or any document source with any filename you choose.

- **Key RAG Capabilities**:
  - **Interactive Chat Playground (`/chat`)**: Full-featured web interface served out of the box with conversation threading, expandable **📚 Retrieved Knowledge Sources** citation drawers, confidence scores, Top-K sliders, and latency metrics.
  - **Universal Chat Provider Abstraction (`BaseChatProvider`)**: First-class support for **Official SDKs** (`ollama`, `openai`, `anthropic`, `google-genai`) with automatic fallback to standard library REST HTTP calls and complete error response decoding.
  - **Strict Error Enforcement**: Halts immediately with clear, actionable diagnostics if an Ollama daemon is offline (`ollama serve`), an embedding model is missing (`ollama pull nomic-embed-text`), API keys are missing, or database connections fail.
  - **Structure-Aware Smart Chunking (`SmartChunker`)**: Parses Markdown headers (`#`, `##`, `###`), paragraph boundaries, and lists while preserving parent document title and section hierarchies in chunk metadata.
  - **Query Intelligence (`QueryAnalyzer` & HyDE)**: Deconstructs user queries into search intent, keyword tags, multi-query variations, and Hypothetical Document Embeddings (HyDE) for maximum semantic retrieval recall.
  - **PostgreSQL ORM & `pgvector` (`PostgresVectorStore`)**: Production vector persistence using PostgreSQL with native `pgvector` cosine similarity operators (`<=>`) and declarative ORM records (`KnowledgeDocumentRecord`, `KnowledgeChunkRecord`), alongside zero-dependency in-memory JSON storage (`MemoryVectorStore`).
  - **Dynamic Ingestion & Live Population (`KnowledgeDocsDataset`)**: Recursively scans all files in `data/` (`.md`, `.txt`, `.json`, `.csv`) and supports live dynamic document addition (`dataset.add_document(content, title, metadata)`).
  - **High-Level Model & Trainer (`KnowledgeModel` & `RAGTrainer`)**: Enterprise domain model and lifecycle trainer automating document loading $\to$ smart chunking $\to$ vector embedding $\to$ database/artifact persistence.
- **Example Implementation**: [`docs/examples/knowledge_rag/`](docs/examples/knowledge_rag/)
- **Workflow**:
  ```bash
  aimlite init support_rag --template rag --chat-provider ollama
  cd support_rag
  
  # Install dependencies into .venv
  aimlite install -r requirements.txt
  
  # Ingest documents, chunk, compute embeddings, and build vector store
  aimlite train
  
  # Test via CLI client
  python client.py
  
  # Launch HTTP server with Interactive Chat UI at http://localhost:8000/chat
  aimlite serve --port 8000
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
- **Workflow**:
  ```bash
  aimlite init lora_app --template fine-tuning
  cd lora_app
  aimlite install -r requirements.txt
  aimlite train LoRAInstructionModel
  aimlite serve LoRAInstructionModel --port 8000
  ```

---

## Zero-Path CLI Reference

AIMLite provides zero-path convention-over-configuration commands:

| Command | Description |
|---|---|
| `aimlite init [project_name \| .] [options]` | Scaffolds a project in a new folder or current directory (`.`), setting up `.venv`, auto-installing dependencies, `requirements.txt`, `.env`, starter files, and conventions. Supports `--template`, `--chat-provider`, `--model`, `--vector-db`, and `--embedding`. |
| `aimlite install [-r requirements.txt \| packages...]` | Without arguments, installs from `requirements.txt` / `aimlite.json`. With `-r <file>`, installs from the specified requirements file. With packages, installs them into `.venv` using `uv` (or `pip`) and tracks them in `aimlite.json`. |
| `aimlite data validate [target]` | Validates dataset schema, record count, column names, and partition readiness across all datasets in `data/`, or selectively validates by dataset class name, declared filename, path, or raw unbound data file. |
| `aimlite train [ModelName] [--resume [path]] [--checkpoint-dir <dir>]` | Automatically discovers registered models and executes training/indexing. Supports resuming from past checkpoints with `--resume`, and saving to custom directories with `--checkpoint-dir`. Saves weights to `models/<model_name>.pkl` and snapshots to `experiments/`. |
| `aimlite evaluate [ModelName] [--checkpoint <path>]` | Loads the model's checkpoint and calculates benchmark metrics on test partitions. Discovers latest checkpoint automatically or loads an explicit past checkpoint with `--checkpoint`. |
| `aimlite serve [ModelName] [--checkpoint <path>] [--port 8000] [--frontend <dir>]` | Starts an HTTP inference server exposing **Interactive Chat UI (`/chat`)**, Swagger API Docs (`/docs`), OpenAPI Schema (`/openapi.json`), `POST /predict`, `GET /health`, custom routes, and optional static frontend hosting. |
| `aimlite doctor` | Diagnoses runtime health, virtual environment, hardware accelerator availability (`cuda`, `mps`, `cpu`), and directory permissions. |

---

## How to Build & Package

AIMLite supports multiple build and packaging targets:

### 1. Build Standalone CLI Executable (`build/`)
Bundle the entire AIMLite framework and CLI into a single, portable executable using Python's native `zipapp` format (compressed bytecode, zero external binary dependencies):
```bash
# Compile standalone binary:
python3 build/build_cli.py

# Test the compiled binary directly:
./build/aimlite --help
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

## How to Test

AIMLite includes a comprehensive 57-test suite validating CLI commands, header generation, data validation, model lifecycle, inference serving, LoRA adaptation, and the RAG subsystem:

### 1. Run Complete Test Suite
```bash
# Run via unittest discovery:
PYTHONPATH=src:. python3 -m unittest discover -s test

# Or using uv / pytest:
uv run test
uv run pytest
```

### 2. Run Module-Specific Test Suites
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

