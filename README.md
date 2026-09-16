# ModelKit

**The Django for Machine Learning & AI** — an opinionated, convention-over-configuration Python ML framework with zero-path CLI execution.

---

## Authors & Collaborators

- **Mickyas Tesfaye** ([alazartesfaye42@gmail.com](mailto:alazartesfaye42@gmail.com))
- **Beamlak Tadesse** ([atocodes@gmail.com](mailto:atocodes@gmail.com))

---

## Quickstart

Requires Python **>= 3.14** and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Alazar42/ModelKit.git
cd ModelKit
uv sync
```

### Essential Commands

```bash
uv run cli --help             # Run CLI entrypoint via uv
./build/modelkit --help       # Run standalone compiled CLI binary directly
uv run test                   # Run core test suite (test/main.py)
uv run pytest                 # Full pytest runner
```

---

## Project Status

- [x] **Headers and core frameworks** (`src/modelkit/`)
- [x] **API Docs & Interactive Portal** (`api_docs/`)
- [x] **Zero-Path CLI Implementation** (`src/cli/`)
- [x] **Standalone Executable & Build System** (`build/modelkit`)

---

## Architecture & Guide

### 1. Core Framework (`src/modelkit/`)
- **`Dataset`**: Zero-config tabular loader, automatic `data/` discovery, and train/val/test splits.
- **`Model`**: Weights persistence (`save`/`load`), `predict()`, and `dataset` binding.
- **`BaseTrainer`, `BaseEvaluator`, `BaseInference`**: Standardized ML lifecycle hooks.
- **`BaseConfig`**: Automated hardware device detection (`cuda`, `mps`, `cpu`) and project path resolver.
- **Registry**: Subclasses auto-register under the hood via `__init_subclass__` (zero decorator boilerplate).
- **RAG & Adapters**: Native support for Document loaders, vector stores, retrievers, and LoRA/PEFT parameter adapters.

### 2. Zero-Path CLI (`src/cli/` & `build/modelkit`)

ModelKit CLI provides zero-path execution with Django-style conventions:

| Command | Description |
|---|---|
| `modelkit init [project_name]` | Scaffolds a new project with convention layout (`data/`, `models/`, `artifacts/`, etc.) and starter files. |
| `modelkit install <pkgs...>` | Safely installs packages into project `.venv` using `uv` (or `pip`). |
| `modelkit data validate` | Validates dataset schema, record count, columns, and split readiness. |
| `modelkit train [class_name]` | Executes training cycle for a specific model class or single project model. |
| `modelkit evaluate [class_name]` | Automatically loads checkpoint and calculates benchmark metrics for target model. |
| `modelkit serve [class_name] [--port 8000] [--frontend <dir>]` | Launches local HTTP inference API (`/predict`, `/health`, `/docs`, custom routes) and optional custom frontend hosting. |
| `modelkit doctor` | Inspects environment health, hardware accelerators (`cuda`, `mps`, `cpu`), and directory permissions. |

### 3. Standalone Executable & Build System (`build/`)

The standalone binary is packaged using Python's native `zipapp` format:

```bash
# Compile/rebuild the standalone executable:
python3 build/build_cli.py
# or
./build/build_cli.sh

# Run directly:
./build/modelkit doctor
```

---

## Testing

```bash
uv run test        # Fast runner
uv run pytest      # Full pytest runner
```
