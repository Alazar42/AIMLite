# Changelog

All notable changes to the AIMLite framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
