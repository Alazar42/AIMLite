# Changelog

All notable changes to the AIMLite framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.2] - 2026-10-12

### Changed
- **Decoupled Standalone Visual Studio & Documentation Restructuring**:
  - Reverted the documentation website (`api_docs`) to purely documentation, removing embedded project creation components (`ProjectStudio.tsx`).
  - Added dedicated documentation in `api_docs` for the local visual Studio workflow (`aimlite` / `aimlite init` default browser experience) vs `--cli` terminal mode.
  - Redesigned the visual Web Studio (`src/aimlite/templates/studio.html`) to match the premium dark obsidian aesthetic of the AIMLite `api_docs` website:
    - 100% full-screen width responsive layout with sticky navbar and fluid split configuration & preview panels.
    - Added custom tokenized syntax highlighting for live `aimlite.json` manifest code preview with distinct color tokens (`hl-key`, `hl-string`, `hl-number`, `hl-bool`, `hl-punct`).
    - Added syntax-styled File Tree and CLI command previews.
- **Clean Zero-Bloat Dependencies by Default & Bidirectional Dependency Management**:
  - All projects (`scratch`, `rag`, `adapter`) now initialize with a completely clean, empty dependencies list (`"dependencies": []`) on start unless explicitly added by the developer.
  - In the visual Web Studio (`studio.html`), dependencies start empty (`[]`) with a clean-state indicator. Developers can dynamically add packages via text input or quick-add chips, and any added dependency can be removed on demand via individual `×` removal buttons or the "Clear all" button.
  - Added new zero-path CLI command `aimlite remove <pkgs...>` (and alias `aimlite uninstall`) to safely uninstall packages from the project `.venv` and remove them from `aimlite.json`.
- Bumped framework, server, CLI, skill guides, and documentation versions to `2.2.2`.

## [2.2.1] - 2026-10-12

### Changed
- **Documentation Suite & Changelog Synchronization**:
  - Synchronized documentation web application (`api_docs`), landing page terminal demos, release badges, and live interactive docs to reflect the new 2.2.0 Web Studio release and 2.2.1 updates.
  - Updated latest version references and dynamic version pills across `App.tsx`, `Navbar.tsx`, `LandingPage.tsx`, `EndpointDoc.tsx`, `index.html`, and `chat.html`.
  - Rebuilt static distribution bundle (`studio_dist`) and standalone CLI executable (`build/aimlite`).
- Bumped framework and CLI version to `2.2.1` across `pyproject.toml`, `src/aimlite/__init__.py`, `main.py`, `studio.py`, `ui.py`, `server.py`, `agent_assets.py`, and agent skill guides.

## [2.2.0] - 2026-10-12

### Added
- **AIMLite Studio Web-Based Project Creator (`aimlite` & `aimlite init`)**:
  - Default behavior overhaul: bare `aimlite` and `aimlite init` now automatically boot a locally hosted, multi-threaded interactive Web Studio (`AIMLite Studio`) in the developer's default browser on port 8000 (with automatic port collision detection fallback).
  - Terminal isolation: only the explicit flag `aimlite init --cli` triggers terminal prompts for CLI project creation steps.
  - Added CLI flags: `--port <int>`, `--host <str>`, and `--no-browser` to both root `aimlite` parser and `init` subcommand.
- **Full 6-Step Web Creation Flow Matching CLI Wizard**:
  - Ported every interactive prompt and choice from the CLI wizard into the Web Studio UI:
    - **Step 1: Project Identity**: Project Name and Target Directory inputs with real-time sanitized Python package identifier preview (`sanitize_package_name`).
    - **Step 2: Architecture Paradigm**: Visual selector for `scratch` (Scratch / Custom ML), `rag` (RAG Knowledge Base), and `adapter` (LoRA Adapter Tuning).
    - **Step 3: Paradigm Configurations**:
      - *RAG*: Chat Provider selector (`openai`, `gemini`, `anthropic`, `ollama`, `local`, `mock`), Model Identifier input, Vector Database backend (`memory` vs `postgres`), Embedding Engine (`sentence-transformers`, `api`, `tfidf`), plus feature toggles for HyDE (Hypothetical Document Embeddings) and Smart Semantic Chunker.
      - *LoRA*: Hyperparameter tuning for Rank $r$ (4, 8, 16, 32, 64), Alpha $\alpha$ (8, 16, 32, 64), Base Model Identifier, and dynamic Target Attention Modules chip manager (`q_proj`, `v_proj`, etc.).
    - **Step 4: Code Generation Mode (`--clean`)**: Mode switch between *Starter Boilerplate* (end-to-end working baseline code) and *Clean Skeleton* (0-byte contracts and interfaces with zero sample data).
    - **Step 5: Extra PyPI Packages Search**: Interactive PyPI package search with quick-add chips (`+pandas`, `+numpy`, `+scikit-learn`, `+torch`, `+fastapi`, `+rich`, `+matplotlib`) and tag management (`extra_packages`).
    - **Step 6: Environment & Setup**: Checkboxes for provisioning an isolated `.venv` and immediate dependency installation (`--install`).
- **Real-Time Live Reactive Preview Panel**:
  - Real-time reactive preview pane with 3 synchronized tabs:
    - `aimlite.json`: Live-generated manifest reflecting all selections, hyperparameter configurations, and resolved dependencies.
    - `File Tree`: Visual folder and code file tree of the scaffolded directory.
    - `CLI Equiv`: The exact equivalent command line string (e.g. `aimlite init my_ai --cli --type rag ...`) for terminal reproducibility.
- **Dual-Mode Serving & Zipapp Executable Compatibility**:
  - Multi-threaded HTTP server (`ThreadingStudioServer`) in `src/aimlite/cli/studio.py` with `/api/status` and `/api/init` endpoints.
  - Dual-mode frontend serving: serves pre-built React SPA static bundle (`api_docs/dist` or `studio_dist`), with automatic fallback to self-contained single-page studio HTML (`templates/studio.html`).
  - Packaged executable compatibility: integrated `pkgutil.get_data("aimlite.templates", "studio.html")` enabling flawless operation inside standalone zipapp binaries (`build/aimlite`).
- **ProjectStudio React Component in Documentation & Web App**:
  - Implemented `ProjectStudio.tsx` in `api_docs/src/components/ProjectStudio.tsx` using Tailwind CSS and Lucide icons following the dark obsidian aesthetic (`#07080a`).
  - Added Studio mode switcher in documentation navbar, sidebar quick navigation, and URL query parameter prefilling (`?name=...&type=...`).
- **Programmatic Package Injection in `run_init`**:
  - Extended `run_init()` with `extra_packages: Optional[List[str]] = None`, allowing programmatic and Web Studio injection of custom PyPI dependencies into `aimlite.json`.

### Changed
- Default entrypoint routing: running `aimlite` or `aimlite init` launches the local web studio rather than printing the CLI help screen or starting interactive terminal prompts.
- Bumped framework and CLI version to `2.2.0` across `pyproject.toml`, `src/aimlite/__init__.py`, `main.py`, `studio.py`, `ui.py`, `server.py`, `agent_assets.py`, and documentation.

### Fixed
- Ensured standalone executable packaging (`build/build_cli.py`) bundles `templates/studio.html` and `templates/studio_dist/` cleanly within the zipapp archive.
- Added full test coverage for Web Studio routing, argument parsing, API status endpoint, and project initialization in `test/test_web_studio_init.py` (all 138 test cases passing).

## [2.1.2] - 2026-10-08

### Fixed
- **LoRA Adapter Scaffolding & Prompt Resolution**:
  - Enhanced `predict()` in scaffolded adapter models (`LoRAInstructionModel`) to check `inputs.get("prompt")`, `inputs.get("instruction")`, `inputs.get("text")`, and `inputs.get("input")`.
  - Fixes dictionary stringification in the Web Application LoRA playground where requests send `{"prompt": "...", "max_tokens": ...}` without the `"instruction"` key.
  - Aligned prompt parsing across documentation examples and live API specifications.
- **Documentation Paradigm Guides Indentation Overhaul**:
  - Restored clean 4-space Python indentation across all 16 template snippets in `api_docs/src/data/paradigmGuides.ts` (`SCRATCH_FILES`, `RAG_FILES`, `ADAPTER_FILES`).
  - Fixed syntax and indentation errors when developers copy-paste starter code from the documentation or interactive code playground into local `.py` files.
  - Standardized JSON formatting to 2-space indentation in all scaffolded manifest files.
  - Corrected CLI command spacing in guide steps (e.g. `scikit-learn`, `sentence-transformers`, `--type`, `--port`).
- **Interactive Documentation App Update**:
  - Synchronized `api_docs` to `v2.1.2` release with verified TypeScript compilation and production bundle build.

## [2.1.1] - 2026-10-08

### Fixed (Minor Fixes & Improvements)
- **Dynamic Model Configuration Synchronization & Hot-Reloading**:
  - Added dynamic config synchronization in `KnowledgeModel.sync_config()` and `aimlite serve` inference loop.
  - Active chat model and `top_k` settings in `aimlite.json` are hot-reloaded without requiring a server restart.
  - Added `BaseConfig.load_active()`, `BaseConfig.reload()`, and dedicated accessors (`get_rag_config()`, `get_adapter_config()`, `get_model_config()`).
- **Ollama Provider Robustness & Timeouts**:
  - Added configurable timeout parameter and `OLLAMA_TIMEOUT` environment variable support (default 300s, replacing hardcoded 90s REST timeout).
  - Explicit `TimeoutError` exception raised with clear troubleshooting instructions when Ollama generation times out.
  - Added `max_tokens` (`num_predict`) support in `OllamaChatProvider` and capped query analyzer generation to 256 tokens.
  - Graceful fallback from Ollama SDK to native REST API on connection or import errors.
- **Package Name Sanitization & Auto-Healing**:
  - Implemented `sanitize_package_name()` to convert project directory names with leading dots (e.g. `.custom_rag`), hyphens, numbers, or reserved keywords into valid Python identifiers.
  - Enhanced `resolve_project_context()` with auto-healing: automatically renames invalid package directories on disk, updates `aimlite.json` entrypoints, and rewrites local imports.
  - Expanded project class discovery and bare module imports to register `store`, `chat_provider`, and `adapter` modules.
- **Scaffolding & Agent Customizations**:
  - Updated RAG and LoRA starter templates to dynamically resolve configuration defaults from `aimlite.json` via `BaseConfig.load_active()`.
  - Updated `AGENTS.md` and `.agents/skills/aimlite/SKILL.md` to reference sanitized package names.
- **Web UI Inference Error Handling**:
  - Enhanced `app.html` interactive chat playground to trap and display server errors and inference failure messages inline instead of failing silently.
- **`aimlite update` Self-Updater & Package Synchronizer**:
  - Added zero-friction `aimlite update` CLI command to check and upgrade globally, user-level (`--user`), or virtualenv installed AIMLite packages directly from PyPI.
  - Multi-tier cross-platform engine supporting Windows, macOS, and Linux across virtual environments and non-virtualenv system environments.
  - Automatically identifies whether AIMLite is already up to date, installs when missing, cleanly bypasses Linux PEP 668 externally-managed environment locks (`--break-system-packages`), and supports `--force`, `--user`, `--global`, and specific `--version` targets.
- **Repository Hygiene & `.gitignore` Hardening**:
  - Untracked all tracked CPython `.pyc` and `__pycache__` compilation artifacts from the repository index.
  - Hardened `.gitignore` with comprehensive exclusions for `*.cpython-*.pyc`, `*.pyo`, `*.pyd`, `**/__pycache__/`, and type checker/test caches (`.mypy_cache/`, `.ruff_cache/`).
  - Removed erroneous `examples/` ignore rule in `.gitignore` so example projects remain tracked.

## [2.1.0] - 2026-10-07

### Fixed (Major Bug Fixes)
- **POSIX Binary Collision on Console Scripts**:
  - Removed `test` and `cli` entry points from `[project.scripts]` in `pyproject.toml`.
  - Exclusively registers `aimlite = "aimlite.cli.main:main"`, completely preventing overrides of the core POSIX `/bin/test` utility on macOS/Linux and avoiding breakage in developer shells and CI/CD pipelines.
- **Root `site-packages` Namespace Pollution**:
  - Relocated CLI source tree from root `src/cli/` to namespaced `src/aimlite/cli/`.
  - Updated Hatchling wheel configuration (`packages = ["src/aimlite"]`) so `import cli` in user code is never hijacked by AIMLite.
- **LoRA Adapter Flexible Dataset Vectorization**:
  - Fixed training crash when fitting LoRA adapters on unstructured text or instruction datasets.
  - Eliminated rigid requirements for tabular numeric columns and hardcoded `target` columns.
  - Implemented hash-based vector projection (`_text_to_feature_vector`) and scalar target extraction (`_text_to_scalar_target`), supporting prompt-response instruction pairs (`instruction`/`response`, `prompt`/`completion`, `input`/`output`, `question`/`answer`), arbitrary dicts, raw text strings, and tabular data without zero-gradient dropouts.
  - Updated `Dataset.validate()` to dynamically discover schema columns for in-memory and custom-loaded datasets.
- **`aimlite init --clean` Paradigm Selection & Empty Code Scaffolding**:
  - Fixed `--clean` flag behavior to prompt developers for paradigm selection (`scratch`, `rag`, `adapter`) instead of defaulting unconditionally to scratch.
  - Clean mode now generates pristine empty code files (`0` bytes) for the chosen paradigm rather than pre-populated demo classes or dummy data.
- **Canonical 3-Paradigm Alignment (`scratch`, `rag`, `adapter`)**:
  - Standardized CLI `--type` choices strictly to the three official paradigms: `scratch`, `rag`, and `adapter`.
  - Added `UnknownParadigmError` providing explicit feedback when an unrecognized paradigm type is provided.
  - Aligned interactive prompts, docstrings, and agent instructions.
- **Python Version Recommendation in `doctor.py`**:
  - Replaced the recommendation for Python `>= 3.14` with `3.10–3.12` to maintain compatibility with PyTorch, CUDA, and TensorFlow binary wheels.
- **Duplicate Agent Skills Folder Creation**:
  - Removed duplicate folder generation in `skills/aimlite/SKILL.md` during `aimlite init`, retaining canonical `.agents/skills/aimlite/SKILL.md`.

## [2.0.0] - 2026-09-29

### Added
- **Production ML & Real Gradient Fine-Tuning Engine (`aimlite.adapters`)**:
  - Replaced simulated loss curves and heuristic updates in `AdapterTrainer.fit()` with true mathematical backpropagation across LoRA low-rank decomposition matrices $A$ and $B$.
  - Forward pass computes intermediate activations $a_x = A \cdot x$; backward passes compute and apply analytical gradients ($\nabla_B \mathcal{L}$ via `layer.backward_B`, $\nabla_A \mathcal{L}$ via `layer.backward_A`).
  - Implemented MSE loss computation, gradient clipping ($[-1.0, 1.0]$), and prediction clamping to protect against numerical instability and exploding gradients.
  - Parameter change verification: asserts and validates that parameters measurably updated during training, raising a `RuntimeError` if a zero-gradient condition is encountered.
  - Enforced strict validation: requires valid `AdapterModel` instances, non-empty datasets (`ValueError`), and initial LoRA layers.
  - Returns true loss progression history and genuine parameter efficiency statistics via `model.get_trainable_parameters()`.
- **Semantic & Hierarchical Text Chunker (`aimlite.rag.SmartChunker`)**:
  - Implemented recursive multi-tier splitting engine (`_split_into_chunks`) that decomposes long text along natural semantic boundaries: Markdown headings $\rightarrow$ paragraphs $\rightarrow$ sentence boundaries $\rightarrow$ word tokens $\rightarrow$ character windowing.
  - Guarantees strict `max_chunk_size` compliance on arbitrary input (including continuous unpunctuated text or long monolithic blocks).
  - Supports configurable `chunk_overlap` while preserving parent document title and section heading metadata.
- **Strict Chat Provider Enforcement (`aimlite.rag.KnowledgeModel`, `aimlite.rag.RAGModel`)**:
  - `KnowledgeModel` and `RAGModel` now strictly require an explicit `chat_provider` (e.g. `OpenAIChatProvider`, `AnthropicChatProvider`, `GeminiChatProvider`, `OllamaChatProvider`, `LocalChatProvider`, or `MockChatProvider`).
  - Eliminated silent fallback to mock providers in production code paths, providing clear configuration guidance when missing.
- **Zero Silent Fallbacks & Explicit Error Handling (`aimlite.rag`)**:
  - `SentenceTransformerEmbedding` strictly raises an explicit `ImportError` with actionable installation instructions when the `sentence-transformers` package is missing, eliminating silent fallback to TF-IDF.
  - `DocumentLoader.load_directory()` collects and reports file corruption or reading failures in a comprehensive `RuntimeError` rather than silently swallowing exceptions.
- **Deterministic Dataset Partitioning (`aimlite.data.Dataset.split`)**:
  - Added seeded random shuffling (`seed=42`, `shuffle=True`) for reproducible train/val/test splits.
  - Added validation ensuring split ratios sum to 1.0 ($\pm 0.01$) and `train > 0.0`.
  - Added support for pre-split datasets (`train.csv`, `test.csv`, `val.csv`) as well as in-memory record lists.
- **Agent-Native Scaffolding & LLM Standards (`.agents/`, `AGENTS.md`, `llms.txt`)**:
  - `aimlite init` now automatically scaffolds `.agents/skills/aimlite/SKILL.md` and `skills/aimlite/SKILL.md` into newly created projects, equipping coding agents (Antigravity, Cursor, Windsurf, Claude Code, GitHub Copilot) to understand zero-path execution, conventions, and class contracts immediately.
  - Generates tailored `AGENTS.md` and `llms.txt` files containing project metadata, CLI commands, and directory layouts in every scaffolded project.
  - Added project root `llms.txt` and static documentation endpoint `/llms.txt` for coding agents and web crawlers.
- **Markdown Rendering in Served Chat Interfaces**:
  - Integrated `marked.js` markdown parsing with light and dark mode styles into `app.html` and `chat.html`.
  - Implemented a zero-dependency pure-JS regex fallback parser to guarantee markdown formatting works in offline and air-gapped environments.
- **108-Test Production Verification Suite**:
  - Added comprehensive test suites: `test/test_adapters.py`, `test/test_dataset.py`, `test/test_rag.py`, `test/test_serving.py`, `test/test_init_clean.py`, verifying zero mock behaviors and complete agent asset generation.

---

## [1.0.8] - 2026-09-29

### Added
- **RAG & Fine-Tuning Template Sample Assets**:
  - Added built-in sample markdown articles in `data/samples/` (`sample_faq.md`, `company_handbook.md`, `product_specs.md`) when initializing standard RAG projects via `aimlite init`.
  - Added sample instruction-following datasets in `data/samples/` (`sample_instructions.json`, `alpaca_sample.jsonl`) and fine-tuning README guidelines for adapter projects.
- **Clean vs Standard Mode Isolation**:
  - Expanded `test/test_init_clean.py` to verify that standard mode includes curated samples while `--clean` mode strictly provisions pristine skeletons with zero sample data.

---

## [1.0.7] - 2026-09-28

### Added
- **Zero-Path Multi-Model Serving & Inventory Architecture (`aimlite serve`)**:
  - Automatically scans `model.py` and registers all trained model classes across any paradigm (Classical ML, RAG Knowledge, and LoRA Fine-Tuning) in a unified server model inventory.
  - Added class-specific targeting: `aimlite serve [ModelName]` (e.g. `aimlite serve ChurnClassifier`) to serve only that class exclusively.
  - Added Model Inventory API endpoint `GET /models` returning model metadata, active status, discovered features, and per-model prediction endpoints (`POST /models/{name}/predict`).
  - Added per-model inference dispatch: `POST /predict` accepts optional `"model": "<name>"` payload, and `POST /models/{name}/predict` directly executes the target model.
- **Paradigm-Adaptive Web Application (`src/aimlite/templates/app.html`)**:
  - Replaced static chat interface with an adaptive, monochromatic, responsive web application that tailors its layout to the active model's paradigm:
    - **Classical & Deep ML (`Model`)**: Renders an interactive Feature Form with auto-discovered dataset columns, 1-click Preset Buttons (`Sample 1`, `Sample 2`, `Zero Vector`, `Randomize`), Form vs JSON mode switcher, live prediction decision cards, confidence meters, and execution latency gauges.
    - **RAG & Knowledge Retrieval (`KnowledgeModel`, `RAGModel`)**: Renders a conversational chat assistant with grounded context citations, similarity percentages, expandable chunks, Top-K context slider, and query suggestion chips.
    - **LoRA Fine-Tuning (`AdapterModel`)**: Renders a prompt completion playground with temperature, max token controls, and token generation stream view.
  - **Dynamic Model Switcher**: Top navbar dropdown enables instant switching between all registered models and their dedicated paradigm controls on the fly without page reloads.
- **Clean Scratch Scaffolding (`aimlite init --clean`)**:
  - Added `--clean` flag and interactive "Clean Scratch Project" option to scaffold pristine machine learning projects with zero sample code or dummy datasets.
  - Generates clean class contracts (`Dataset`, `Model`, `BaseTrainer`, `BaseEvaluator`, `BaseInference`) with `NotImplementedError` and structured docstrings/comments guiding custom implementations.
  - Keeps `data/` strictly clean (zero dummy CSV records), includes clean `.env.example`, `experiments/benchmark.py`, and `client.py` templates.
- **Headless REST API Mode (`--api`)**:
  - Added `--api` flag to `aimlite serve` to disable HTML web interfaces and run as a lightweight, headless JSON REST microservice, ideal for containerized Docker and Kubernetes deployments.
- **Full Developer Control & Customization Priority**:
  - Added template priority loader: placing `app.html` or `index.html` inside `<project_root>/templates/` automatically overrides the built-in package templates.
  - Added SPA client-side routing fallback to `--frontend <dir>`: non-asset routes automatically resolve to `index.html`, supporting React Router, Vue Router, Next.js static exports, and SvelteKit without 404s.
  - Zero-CORS single-port deployment: custom frontend bundles and REST APIs share the exact same port.

---

## [1.0.6] - 2026-09-24

### Added
- **Developer-Editable `KnowledgeModel` & `RAGModel` Architecture (`aimlite.rag`)**:
  - Added modular lifecycle extension hooks across `KnowledgeModel` and `RAGModel` so developers can customize, override, and extend every part of the retrieval and generation pipeline:
    - `preprocess_query(self, query: str) -> str`: Hook for developer query normalization, spell checking, acronym resolution, or expansion.
    - `retrieve(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[Document]`: Direct access retrieval hook with runtime `top_k` and metadata filtering.
    - `rerank(self, query: str, documents: List[Document]) -> List[Document]`: Pluggable reranking hook for cross-encoders, reciprocal rank fusion, score thresholding, or custom filtering.
    - `format_prompt(self, query: str, context_docs: List[Document]) -> str`: Configurable prompt formatting supporting custom `prompt_template` property.
    - `synthesize(self, query: str, context_docs: List[Document], system_prompt: Optional[str] = None, **kwargs: Any) -> str`: Modular response generation and LLM synthesis hook.
    - `postprocess_answer(self, answer: str, context_docs: List[Document]) -> str`: Output post-processing hook for citations, disclaimers, and response sanitization.
    - `chunk_documents(self, documents: Sequence[Document]) -> List[Document]`: Override hook for custom document splitting strategies.
    - `search(self, query: str, top_k: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]`: Direct semantic vector search returning ranked document dictionaries without invoking LLM synthesis.
    - `add_document(self, content: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> int`: Convenience method to dynamically index ad-hoc text records into the active knowledge store.
    - `add_documents(self, documents: ...)`: Alias to `index_documents`.
  - Added `before_index(self, documents)` and `after_index(self, model, index_path, count)` lifecycle hooks to `RAGTrainer` for pre-filtering and post-indexing caching, logging, or notifications.
  - Added `fixed_response` parameter to `MockChatProvider` for test harnesses and baseline evaluations.

- **Fully Developer-Editable Generated RAG Codebase (`_scaffold_rag`)**:
  - Re-architected all scaffolded files generated by `aimlite init` for the RAG paradigm to be 100% modular, documented, transparent, and developer-editable:
    - `chat_provider.py`: Starter with editable `SYSTEM_PROMPT`, `QA_PROMPT_TEMPLATE`, `get_chat_provider()` factory supporting OpenAI, Gemini, Anthropic, Ollama, Local, and Mock, plus `get_query_analyzer()` for HyDE & search expansion.
    - `store.py`: Editable `get_embedding_model()` and `get_vector_store()` factories supporting PostgreSQL pgvector ORM, `MemoryVectorStore`, and third-party vector backends.
    - `data.py`: `KnowledgeDocsDataset` with document loader hooks (`parse_custom_file`), configurable chunk sizes/overlap, and dynamic document addition.
    - `model.py`: `SupportDocRAG(KnowledgeModel)` showcasing all overrideable extension hooks (`preprocess_query`, `rerank`, `synthesize`, `postprocess_answer`).
    - `trainer.py`: `IndexBuilderTrainer(RAGTrainer)` with `before_index` and `after_index` lifecycle hooks.
    - `evaluator.py`: `RAGEvaluator` with retrieval accuracy, latency checks, and grounding evaluation.
    - `inference.py`: `RAGInference` with `POST /predict`, `POST /search` (search-only endpoint), `GET /health`, and custom route extensibility.
    - `experiments/benchmark.py` & `client.py`: Fully editable evaluation scripts and pre-flight health checks.

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
