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
uv run cli        # Run CLI entrypoint (src/cli/main.py)
uv run test       # Run core test suite (test/main.py)
```

---

## Project Status

- [x] **Headers and core frameworks** (`src/modelkit/`)
- [x] **API Docs** (`api_docs/`)
- [ ] **CLI implementation** (`src/cli/` — in progress by @atocodes)

---

## Architecture & Guide

### 1. Core Framework (`src/modelkit/`)
- **`Dataset`**: Zero-config tabular loader, automatic `data/` discovery, and train/val/test splits.
- **`Model`**: Weights persistence (`save`/`load`), `predict()`, and `dataset` binding.
- **`BaseTrainer`, `BaseEvaluator`, `BaseInference`**: Standardized ML lifecycle hooks.
- **`BaseConfig`**: Automated hardware device detection (`cuda`, `mps`, `cpu`) and project path resolver.
- **Registry**: Subclasses auto-register under the hood via `__init_subclass__` (zero decorator boilerplate).

### 2. CLI Implementation (`src/cli/`)
The CLI entrypoint scaffold is ready at `src/cli/main.py`.

> **Guide for @atocodes**: Launch the interactive documentation portal to view the specifications, expected inputs/outputs, and command designs:
> ```bash
> cd api_docs
> npm install && npm run dev
> ```
> Open **`http://localhost:5173/`** to follow the visual blueprint for `init`, `train`, `evaluate`, `serve`, `doctor`, `install`, and `data`.

---

## Testing

```bash
uv run test        # Fast runner
uv run pytest      # Full pytest runner
```
