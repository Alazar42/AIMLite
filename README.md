<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo_white.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo.png">
    <img src="https://i.ibb.co/5W4Gns1v/logo-white.png" alt="AIMLite Logo" width="460" />
  </picture>
</p>

<h1 align="center">AIMLite</h1>

<p align="center">
  <strong>The Django for AI & Machine Learning</strong> — an opinionated, convention-over-configuration Python framework with zero-path CLI execution.
</p>

<p align="center">
  <a href="https://pypi.org/project/aimlite/"><img src="https://img.shields.io/pypi/v/aimlite.svg?color=black" alt="PyPI version"></a>
  <a href="https://pypi.org/project/aimlite/"><img src="https://img.shields.io/pypi/pyversions/aimlite.svg?color=black" alt="Python versions"></a>
  <a href="https://github.com/Alazar42/AIMLite/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-black.svg" alt="License"></a>
</p>

---

## Installation

### One-Line Install (Linux, macOS, Windows WSL / Git Bash)

```bash
curl -fsSL https://raw.githubusercontent.com/Alazar42/AIMLite/main/install.sh | bash
```

### PyPI Installation

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
uv run test                   # Run core test suite (49 tests)
uv run pytest                 # Full pytest runner (49/49 passing)
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

Demonstrates semantic vector retrieval, query intelligence, smart document chunking, and grounded LLM synthesis. Ingest text, Markdown, CSV, JSON, or any document source with any filename you choose.

- **Key RAG Capabilities**:
  - **Structure-Aware Smart Chunking (`SmartChunker`)**: Parses Markdown headers (`#`, `##`, `###`), paragraph boundaries, and lists while preserving parent document title and section hierarchies in chunk metadata.
  - **Query Intelligence (`QueryAnalyzer` & HyDE)**: Deconstructs user queries into search intent, keyword tags, multi-query variations, and Hypothetical Document Embeddings (HyDE) for maximum semantic retrieval recall.
  - **Universal Chat Provider Abstraction (`BaseChatProvider`)**: Native integrations for **API models** (`OpenAIChatProvider`, `AnthropicChatProvider`, `GeminiChatProvider`), **Local models** (`OllamaChatProvider`, `LocalChatProvider` with HuggingFace/custom callables), and zero-dependency `MockChatProvider`.
  - **PostgreSQL ORM & `pgvector` (`PostgresVectorStore`)**: Production vector persistence using PostgreSQL with native `pgvector` similarity operators (`<=>` cosine / `<->` L2 distance) and declarative ORM records (`KnowledgeDocumentRecord`, `KnowledgeChunkRecord`), with seamless in-memory fallback.
  - **Pre-defined Standard System Prompts**: Built-in prompts for grounded Q&A (`PROMPT_RAG_QA`), search analysis (`PROMPT_QUERY_ANALYZER`), semantic chunking (`PROMPT_SMART_CHUNKER`), and multi-turn conversations (`PROMPT_CONVERSATIONAL_RAG`).
  - **High-Level Model & Trainer (`KnowledgeModel` & `RAGTrainer`)**: Enterprise domain model and lifecycle trainer automating document loading $\to$ smart chunking $\to$ vector embedding $\to$ database/artifact persistence.
- **Example Implementation**: [`docs/examples/knowledge_rag/`](docs/examples/knowledge_rag/)
  - `data.py`: `KnowledgeDocsDataset(Dataset)` ingesting and chunking articles using `SmartChunker`.
  - `model.py`: `SupportDocRAG(KnowledgeModel)` coordinating `QueryAnalyzer`, `PostgresVectorStore`/`MemoryVectorStore`, and `BaseChatProvider`.
  - `trainer.py`: `IndexBuilderTrainer(RAGTrainer)` orchestrating semantic index building and artifact/database persistence.
  - `inference.py`: `RAGInference(BaseInference)` serving grounded Q&A with verifiable document citations and confidence scores via `POST /predict`.
- **Workflow**:
  ```bash
  aimlite init support_rag && cd support_rag
  # Place knowledge documents in data/
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
  # Place instructions.json in data/
  aimlite train LoRAInstructionModel
  aimlite serve LoRAInstructionModel --port 8000
  ```

---

## Zero-Path CLI Reference

AIMLite provides zero-path convention-over-configuration commands:

| Command | Description |
|---|---|
| `aimlite init [project_name \| .] [--clean]` | Scaffolds a project in a new folder or directly in the current directory (`.`). Supports `--clean` to generate a pure scratch project with zero sample code or dummy datasets (only class contracts and guiding comments). |
| `aimlite install [packages...]` | Without arguments, installs all dependencies listed in `aimlite.json`. With packages, installs them into `.venv` using `uv` (or `pip`) and adds them to `aimlite.json`. |
| `aimlite data validate [target]` | Validates dataset schema, record count, column names, and partition readiness across all datasets in `data/`, or selectively validates by dataset class name, declared filename, path, or raw unbound data file. |
| `aimlite train [ModelName] [--resume [path]] [--checkpoint-dir <dir>]` | Automatically discovers registered models and executes training. Supports resuming from latest or specific past checkpoints with `--resume`, and saving to custom directories with `--checkpoint-dir`. Saves weights to `models/<model_name>.pkl` and snapshots to `experiments/`. |
| `aimlite evaluate [ModelName] [--checkpoint <path>]` | Loads the model's checkpoint and calculates benchmark metrics on test partitions. Discovers latest checkpoint automatically or loads an explicit past checkpoint with `--checkpoint`. |
| `aimlite serve [ModelName] [--checkpoint <path>] [--port 8000] [--frontend <dir>] [--api]` | Starts an HTTP inference server with multi-model auto-discovery, paradigm-adaptive web UI, `GET /models`, `POST /models/{name}/predict`, optional headless API mode (`--api`), and custom SPA frontend hosting (`--frontend <dir>`). |
| `aimlite doctor` | Diagnoses runtime health, virtual environment, hardware accelerator availability (`cuda`, `mps`, `cpu`), and directory permissions. |
| `aimlite benchmark [script] [args...]` | Runs any Python script with the project root automatically injected into `PYTHONPATH`. Eliminates `ModuleNotFoundError` when running experiment or benchmark scripts. Uses the project `.venv` Python automatically. Auto-discovers scripts in `experiments/` when no path is given. |

---

## Containerization & Production Docker Deployment

Deploying an AIMLite application to production is zero-friction. Below is the production-ready `Dockerfile` engineered for ultra-fast builds, layer-cached dependencies, pre-baked model weights, active health monitoring, and dynamic port binding on cloud platforms (Render, Railway, Fly.io, Google Cloud Run, AWS ECS/Fargate, and Kubernetes).

### Production `Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies (curl for healthchecks & network tooling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install uv (blazing fast backend) and AIMLite framework
RUN pip install --no-cache-dir uv aimlite

# 2. Set project working directory
WORKDIR /app

# 3. Copy project manifest
COPY aimlite.json .

# 4. Install all project dependencies into managed .venv using aimlite CLI
RUN aimlite install

# 5. Copy the remaining application files, data, and frontend assets
COPY . .

# 6. Train and calibrate model weights for serving
RUN aimlite train

# 7. Expose serving port
EXPOSE 8000

# 8. Health check verifying inference server status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# 9. Host multi-model inference server with custom frontend (dynamically binds to $PORT on cloud hosts)
CMD ["sh", "-c", "aimlite serve --host 0.0.0.0 --port ${PORT:-8000} --frontend frontend"]
```

### Key Architectural Highlights

1. **Layer Caching Optimization**: By copying `aimlite.json` and executing `aimlite install` *before* copying application code, Docker caches your compiled virtual environment layer. Code edits will rebuild in seconds without re-downloading dependencies.
2. **Baked Checkpoints (Zero Cold-Start)**: `RUN aimlite train` runs at image build time. Weight matrices (`models/*.pkl`) and vector artifacts are pre-baked directly into the container image, guaranteeing instant startup and zero inference lag on container spin-up.
3. **Active Healthcheck Probe**: Built-in `HEALTHCHECK` periodically verifies `GET /health` with `curl`. Orchestrators (Kubernetes, ECS, Docker Swarm) automatically restart degraded instances.
4. **Dynamic Cloud Port Binding (`${PORT:-8000}`)**: Cloud platforms (Render, Railway, Heroku, Google Cloud Run) dynamically inject the `$PORT` environment variable. The shell wrapper dynamically binds to this port, avoiding binding conflicts.
5. **Unified SPA & REST API Serving (`--frontend frontend`)**: AIMLite serves your custom compiled frontend SPA (React, Vue, Vite, Next.js static) alongside backend inference endpoints on the exact same port, eliminating CORS configurations entirely.

### Build and Run Instructions

```bash
# 1. Build the production Docker image:
docker build -t my-aimlite-app .

# 2. Run container locally with port 8000 mapped:
docker run -d --name aimlite-srv -p 8000:8000 my-aimlite-app

# 3. Verify server health:
curl http://localhost:8000/health

# 4. Submit inference request:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [128.0, 1.0, 2.7, 1.0, 265.1, 110.0, 89.0, 9.8, 10.0]}'

# 5. View live streaming container logs:
docker logs -f aimlite-srv
```

### Docker Compose (`docker-compose.yml`)

For local multi-service orchestration (e.g. AIMLite + PostgreSQL with `pgvector` for enterprise RAG):

```yaml
services:
  aimlite:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${PORT:-8000}:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

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

