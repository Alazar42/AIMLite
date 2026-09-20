# Changelog

All notable changes to the AIMLite framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.1] - 2026-09-20

### Added
- **Major 1.0.0 Milestone Release**: Stabilized core architecture, inference server, and developer ergonomics across the ecosystem.
- **Built-in Interactive Web Interfaces (`aimlite serve`)**:
  - **Inference Playground (`/`, `/playground`)**: Real-time forward pass execution with latency telemetry and response inspector.
  - **Chat & RAG Studio (`/chat`)**: Conversational interface for querying retrieval-augmented pipelines with chat history and message streaming.
  - **Zero-Hallucination Dataset Inspection**: Automatically extracts actual records and schema from project datasets in `data/` (CSV, JSON), populating input payloads with real data.
  - **Transparent Error Handling**: Unconfigured or untrained starter models explicitly raise `NotImplementedError` and report HTTP 500, preventing silent echo or fake mock predictions.
- **Voxide AI Voice & Text Assistant Integration**:
  - Embedded floating Voxide AI helper widget into served templates and documentation (`@voxide/react`).
  - **Root `.env` API Key Enforcement**: Automatically searches for `VOXIDE_API_KEY` / `VOXIDE_PUBLIC_KEY` in the project root `.env` or process environment.
  - **Graceful Error Display**: Widget remains visible on all served pages; when no API key is configured, displays a clear in-panel authentication error guiding the developer to add their key.
  - Live voice recognition and hands-free capability execution (`/predict`, `/health`, model diagnostics).
- **Frontend Serving & Build Pipeline**:
  - Automatic discovery and hosting of SPA frontends (`api_docs/dist`, `frontend/dist`, `dist`, `static`).
  - Dynamic `window.VOXIDE_PUBLIC_KEY` injection into served static HTML pages.
  - Single standalone binary distribution via `build/aimlite`.

### Changed
- Bumped package version, CLI metadata, and documentation to `1.0.0`.
- All 50 test suites passing (`test_adapters`, `test_cli`, `test_examples`, `test_headers`).

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

- **Flexible Dataset Filenames**:
  - Developers have complete freedom over dataset file naming and formats.
  - Simply declare `filename = "<any_filename>"` (or provide `source`) on `Dataset` subclasses without any convention restrictions.
  - Supports CSV, TSV, JSON, JSONL, Parquet, and text files out-of-the-box.
  - `aimlite data validate` checks the declared file and reports partition readiness.

- **Clean Project Scaffolding**:
  - `aimlite init` automatically creates `.venv` and installs `aimlite` directly into the environment.
  - Removed IDE helper directories and configuration files (`.aimlite/`, `pyrightconfig.json`, `.vscode/settings.json`).
  - Native IDE autocompletion and type-checking powered by bundled PEP 561 `py.typed`.

### Added
- Flexible reference workflow templates without forcing developers into fixed paradigms:
  - **Classical & Tabular ML** (e.g. Scikit-learn, XGBoost, Custom Architectures).
  - **Adapters & Parameter-Efficient Fine-Tuning** (e.g. LoRA deltas).
  - **Retrieval-Augmented Generation (RAG)** (Document chunking, vector embeddings, and retrieval).
  - Support for any custom modeling framework (PyTorch, TensorFlow, JAX, Hugging Face, pure Python).
- Fast standalone executable CLI generator (`build/build_cli.py`).
- Automatic environment health checks (`aimlite doctor`).
- Built-in local inference API server with Swagger documentation (`aimlite serve`).
