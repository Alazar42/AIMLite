# Changelog

All notable changes to the AIMLite framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.5] - 2026-09-23

### Added
- **`aimlite benchmark` command**:
  - New `benchmark` subcommand that runs any Python script with the project root automatically injected into `PYTHONPATH`, eliminating `ModuleNotFoundError: No module named '<project>'` when running experiment or benchmark scripts directly.
  - Auto-discovers the first script in `experiments/` when no path is supplied.
  - Uses the project `.venv` Python automatically — no manual `source .venv/bin/activate` required.
  - All extra arguments are forwarded to the target script unchanged.
- **Pyfiglet ASCII art banner on `aimlite init`**:
  - Replaced hand-drawn ASCII art with [`pyfiglet`](https://pypi.org/project/pyfiglet/) for a reliable, correctly-rendered large title banner.
  - Font selectable; ships with the `basic` font by default. Falls back gracefully to plain text if `pyfiglet` is not installed.
  - `pyfiglet>=1.0.4` added as an explicit project dependency.
- **Interactive package search on `aimlite init`**:
  - New `+` prompt after template/config selection lets users type extra packages (space-separated) to add to the project during initialization.
  - Extra packages are merged into `aimlite.json` dependencies and installed if the user chooses immediate installation.

### Changed
- **`aimlite.json` as single dependency source of truth**:
  - Removed `requirements.txt` generation from `aimlite init`. All project dependencies are now stored exclusively in `aimlite.json`.
  - `aimlite install` reads from `aimlite.json` when no packages are specified (no `-r` flag needed).
  - Next-steps hint updated from `aimlite install -r requirements.txt` → `aimlite install`.

---

## [1.0.4] - 2026-09-23


### Added
- **Vite-Inspired Modern Terminal UI (`questionary` & `rich`)**:
  - Re-architected CLI project initialization and prompts after Vite (`create-vite`) and Clack.
  - Interactive arrow-key navigation (`↑`/`↓`), cyan pointer indicators (`❯`), styled question marks (`?`), answered checkmarks (`✔`), and dimmed contextual guidance for templates and chat providers.
  - Rich terminal status spinners during project filesystem generation.
  - Added `--install` CLI flag to optionally install dependencies into `.venv` immediately upon project initialization.
- **Root `.gitignore`**:
  - Added comprehensive repository root `.gitignore` covering Python bytecode, virtual environments (`.venv`), distribution artifacts, model checkpoints (`*.pkl`), IDE configs, test caches, and temporary project workspaces.

### Fixed
- **Project Creation Freeze/Hang on ML Templates**:
  - Fixed an issue where `aimlite init` would hang or appear frozen after selecting `rag` or `fine-tuning` templates due to silent background `pip` downloads of multi-gigabyte machine learning dependencies (`torch`, CUDA packages, `sentence-transformers`) in `capture_output=True`.
  - Project creation is now instantaneous (< 50ms). Heavy dependency installation is deferred by default to `aimlite install -r requirements.txt`, or prompted interactively with live progress.
- **Template Selection on `aimlite init <project_name>`**:
  - Fixed an argument routing bug where passing a project name positional argument inadvertently flagged the command as non-interactive, skipping template selection and defaulting to `scratch`.
  - Supplying a project name now confirms the name and opens the interactive template selection menu.
- **Scratch Template Dependencies**:
  - Corrected default scratch template dependencies to `[]` so that scratch projects start with zero external dependencies.

---

## [1.0.3] - 2026-09-20

### Added
- **Production Modular RAG & Enterprise Knowledge Base Engine (`aimlite.rag`)**:
  - **Comprehensive Multi-Provider Chat Integrations**:
    - `OpenAIChatProvider`: Support for GPT-4o, GPT-4o-mini, o1, and o3 with official `openai` SDK support and zero-dependency HTTP fallback.
    - `GeminiChatProvider`: Support for Google Gemini 1.5 Flash/Pro and Gemini 2.0 with official `google-genai` / `google-generativeai` SDKs and REST fallback.
    - `AnthropicChatProvider`: Support for Claude 3.5 Sonnet and Claude 3 Opus with official `anthropic` SDK and REST API fallback.
    - `OllamaChatProvider`: Support for local open-weights LLMs (Llama 3.2, DeepSeek-R1, Mistral, Gemma) with official `ollama` SDK and HTTP API.
    - `LocalChatProvider`: Direct execution of local Hugging Face pipelines, PyTorch models, or custom Python callables.
    - `MockChatProvider`: Deterministic zero-dependency provider for fast offline unit tests and baseline verification.
  - **Strict Pre-Flight Connection & Credential Validation**:
    - Added `validate_connection()` across all chat providers, embeddings, and vector databases to catch network issues and authentication errors early.
    - Automatic HTTP error body decoding with actionable terminal diagnostics (unreachable host, invalid API key, missing endpoint, offline model).
  - **Query Intelligence & Decomposition (`QueryAnalyzer`)**:
    - Intent decomposition, sub-question extraction, search keyword expansion, and HyDE (Hypothetical Document Embeddings) answer synthesis.
    - Heuristic fallback analysis when external LLM synthesis is unavailable.
  - **Structure-Aware Smart Document Chunking (`SmartChunker`)**:
    - Heading hierarchy-aware document chunker for Markdown, Text, and structured documents to preserve context across headers and structural boundaries.
  - **Flexible Vector Storage & Embedding Backends**:
    - `SentenceTransformerEmbedding`: Dense neural vectors (`all-MiniLM-L6-v2`) via `sentence-transformers`.
    - `APIEmbedding` & `OllamaEmbedding`: Direct embedding generation from remote APIs or local Ollama endpoints.
    - `TfidfEmbedding`: Pure-Python hash-based TF-IDF vectorization with zero external dependencies.
    - `MemoryVectorStore`: In-memory JSON-persisted vector store with pure-Python cosine similarity.
    - `PostgresVectorStore`: PostgreSQL and `pgvector` SQL ORM integration with connection pooling and `url`/`db_url` alias support.
  - **High-Level Orchestration Models & Workflows**:
    - `KnowledgeModel`: High-level orchestrator connecting chat providers, embeddings, vector store, smart chunker, and query analyzer.
    - `RAGTrainer`: Dedicated trainer workflow for building, validating, and persisting semantic vector indexes.
    - Prompt Engineering Templates: `PROMPT_RAG_QA`, `PROMPT_QUERY_ANALYZER`, `PROMPT_SMART_CHUNKER`, and `PROMPT_CONVERSATIONAL_RAG`.

- **Interactive Web Chat Playground (`aimlite serve`)**:
  - Dedicated interactive Chat Playground served at `/chat` and `/playground` (with automatic browser detection on `/`).
  - Dark-mode interface featuring animated responses, source context citations inspector, configurable generation parameters (temperature, top_k, system prompt), and session management.
  - Live Swagger UI at `/docs` alongside standard JSON endpoints (`POST /predict`, `GET /health`).

- **Interactive Project Scaffolding & Virtual Environment Automation (`aimlite init` / `aimlite install`)**:
  - Interactive CLI wizard with arrow-key navigation for selecting system paradigms (RAG & Knowledge Base, LoRA Fine-Tuning, Custom/Scratch ML) and granular components (Chat Provider, Vector DB, Embedding Engine, Query Intelligence, Smart Chunking).
  - CLI flags for non-interactive / CI automation: `--type`, `--chat-provider`, `--vector-db`, `--model`, `--embedding`, `-y` / `--non-interactive`.
  - Automatic virtual environment (`.venv`) creation, package installation, and generation of `requirements.txt`, `.env.example`, `.gitignore`, `client.py`, and custom `README.md`.
  - Added `aimlite install -r requirements.txt` / `--requirement` with automated virtual environment target detection.

- **Zero-Dependency Dotenv Auto-Loading (`aimlite.config`)**:
  - Pure-Python `.env` parser and environment loader (`load_dotenv()`) discovering configuration up directory trees without external dependencies.
  - Integrated into `BaseConfig` for automatic environment resolution on initialization.

### Changed
- **Unified Root Package Exports**:
  - Exported all core RAG classes, chat providers, embeddings, vector stores, chunkers, and prompt templates from root `aimlite` and `aimlite.rag`.
- **Standalone Executable Builder**:
  - Updated CLI builder (`build/build_cli.sh`) with complete support for modular RAG scaffolding and template packaging.

---

## [0.1.2] - 2026-09-18

### Added
- **Production-Grade LoRA & PEFT Adaptation Engine (`aimlite.adapters`)**:
  - **Mathematical Low-Rank Decomposition**: Implemented `LoRALayer` ($h = W_0 x + \frac{\alpha}{r} (B \cdot A) x$) with frozen foundation weights $W_0$, Gaussian low-rank initialization for matrix $A$, and zero initialization for matrix $B$ ($\Delta W = 0$ at step 0).
  - **Zero-Latency Serving via Weight Merging**: In-place `merge_weights()` folds adapter deltas directly into foundation weights for zero inference latency overhead; `unmerge_weights()` restores base parameters for continuous fine-tuning or swapping.
  - **Multi-Adapter Hot-Swapping Registry**: Added `MultiAdapterManager` to host and route multiple named adapters (`add_adapter()`, `set_active_adapter()`) on a single running foundation model with sub-millisecond latency.
  - **Parameter Efficiency Diagnostics**: Integrated `get_trainable_parameters()` and `print_trainable_parameters()` into `AdapterModel` and `aimlite train` CLI, calculating trainable vs total weights and memory reduction percentages.
  - **Hugging Face PEFT Compatibility**: Generates standard `adapter_config.json`, lightweight delta matrix checkpointing `adapter_model.pkl` (~50KB to 50MB instead of 14GB+), and `adapter_metadata.json`.
  - **Zero-Heavy-Dependency Guarantee**: Runs seamlessly in pure Python and NumPy without requiring PyTorch/CUDA downloads, while fully supporting PyTorch tensor interop.
  - **Dedicated Adapter Test Suite**: Added 5 comprehensive unit tests covering PEFT serialization, low-rank mathematics, zero-step delta, weight merging/unmerging, and training loops (47/47 passing tests).

### Fixed
- **Clean Workspace & Test Isolation**:
  - Isolated test execution inside temporary directories, preventing test suites (`test_cli`, `test_examples`) from generating stray `artifacts/`, `checkpoints/`, `data/`, `experiments/`, and `models/` folders in the repository root.
  - Updated `aimlite doctor` directory diagnostics to clean up non-preexisting test probe directories, ensuring health checks never leave uninitialized folders behind.

---

## [0.1.1] - 2026-09-18

### Changed
- **Developer Freedom & Flexibility**:
  - Clarified that developers have complete freedom to use any data file name and any format (CSV, Parquet, JSON, JSONL, text, TSV).
  - Reframed paradigms as unconstrained reference templates, supporting any ML framework (scikit-learn, PyTorch, TensorFlow, Hugging Face, pure Python).
  - Cleaned up CLI and API documentation.

---

## [0.1.0] - 2026-09-18

### Changed
- **Project Renamed to AIMLite**:
  - Rebranded package and ecosystem from `ModelKit` to `AIMLite` (`aimlite` on PyPI).
  - Python package import path is now `import aimlite` / `from aimlite import Model, Dataset, BaseConfig, BaseTrainer`.
  - CLI binary and commands renamed to `aimlite` (e.g. `aimlite init`, `aimlite train`, `aimlite serve`).
  - Documentation, examples, and API docs migrated to reference `aimlite`.

- **Explicit Dataset Filename Requirement**:
  - Removed implicit file auto-loading and convention guessing from `Dataset`.
  - Developers explicitly declare `filename = "<filename>"` (or provide `source`) on `Dataset` subclasses.
  - `aimlite data validate` now validates explicitly declared files and alerts if a dataset lacks a `filename`.

- **Clean Project Scaffolding**:
  - `aimlite init` automatically creates `.venv` and installs `aimlite` directly into the environment.
  - Removed IDE helper directories and configuration files (`.aimlite/`, `pyrightconfig.json`, `.vscode/settings.json`).
  - Native IDE autocompletion and type-checking powered by bundled PEP 561 `py.typed`.

### Added
- Multi-app architecture with support for 3 modern AI paradigms:
  - **Scratch Training** (Classical ML / Custom Architectures).
  - **Adapters & Parameter-Efficient Fine-Tuning** (LoRA deltas).
  - **Retrieval-Augmented Generation (RAG)** (Document chunking, embedding index, and context generation).
- Fast standalone executable CLI generator (`build/build_cli.py`).
- Automatic environment health checks (`aimlite doctor`).
- Built-in local inference API server with Swagger documentation (`aimlite serve`).
