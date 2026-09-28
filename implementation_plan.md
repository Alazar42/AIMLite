# Multi-Model Serving, Type-Adaptive Web Application & Headless --api Architecture

This document records the major overhaul to AIMLite's CLI and runtime server (`aimlite serve`), introducing dynamic multi-model discovery, model-type-driven web interfaces, full developer customization, and headless REST mode.

## 1. Overview of Changes

1. **Zero-Path Multi-Model Discovery & Registration**:
   - In `model.py`, projects can declare multiple classes inheriting from `aimlite.models.Model` (e.g. `ChurnClassifier`, `FraudDetector`, `SupportKnowledgeModel`).
   - `aimlite serve` automatically scans `model.py`, discovers all trained model classes, loads their weights, and registers them in a unified server model inventory.
   - Positional target argument `aimlite serve [ModelName]` allows serving a specific class exclusively.

2. **The 3 Paradigm Extended Classes & Their Tailored Web Interfaces**:
   Every model in AIMLite belongs to one of 3 primary paradigms, and the served website (`src/aimlite/templates/app.html`) renders its own specialized frontend controls:
   - **`Model` (Classical / Deep ML)**:
     - Renders an interactive **Feature Form Playground**.
     - Auto-discovers feature column names from `dataset.py` or CSV headers (e.g. `AccountWeeks`, `ContractRenewal`, `MonthlyCharge`).
     - Includes 1-click **Preset Buttons** (`Sample 1`, `Sample 2`, `Zero Vector`, `Randomize`).
     - Toggles between visual **Form Mode** and raw **JSON Mode**.
     - Live **Results Panel** showing predicted class decisions, confidence meters, latency gauges, and raw JSON response.
   - **`KnowledgeModel` / `RAGModel` (Retrieval-Augmented Generation)**:
     - Renders an **AI Conversational Playground**.
     - Real-time chat stream with user and AI message bubbles.
     - **Source Citations Accordion**: displays document titles, cosine similarity percentages, and expandable chunk text.
     - **Top-K Slider** (1 to 10) to adjust context density on the fly.
     - Suggested query chips.
   - **`AdapterModel` (Fine-Tuned LoRA Models)**:
     - Renders a **LoRA Generation Playground**.
     - Prompt text editor with temperature (0.0–1.5) and max tokens sliders.
     - Token generation stream and real-time generation output inspection.
   - **Dynamic Model Switcher**:
     - When `model.py` contains multiple models (even across different paradigms!), the top navigation bar displays an interactive **Model Switcher**.
     - Selecting any model instantly switches the UI controls and endpoints without page reload or port changes.

3. **Using `--frontend` with ANY Frontend Framework**:
   Developers have 100% control over the user interface and can connect any frontend framework (React, Vue, Next.js, SvelteKit, Angular, or Vanilla JS):
   - **How `--frontend <dir>` works under the hood**:
     - Mounts the static build directory as the root web server.
     - Serves static assets (`.js`, `.css`, `.png`, `.svg`) with proper MIME types and CORS headers.
     - **SPA Client-Side Routing**: Automatically falls back non-asset paths to `index.html`, allowing React Router, Vue Router, or TanStack Router to work seamlessly without 404 errors.
     - **Unified Port & Zero CORS**: Keeps all backend REST endpoints (`/predict`, `/models`, `/models/{name}/predict`, `/health`, `/docs`) running on the exact same port! No reverse proxy (Nginx) or CORS setup is required.
   - **Integration Workflow (3 Steps)**:
     1. **Build your frontend**:
        - React / Vite: `npm run build` (outputs to `dist/` or `frontend/dist/`).
        - Next.js (Static Export): Add `output: 'export'` to `next.config.js`, then `npm run build` (outputs to `out/`).
        - Vue 3 / Nuxt: `npm run build` or `npx nuxt generate` (outputs to `dist/`).
        - SvelteKit: Use `@sveltejs/adapter-static`, then `npm run build` (outputs to `build/`).
     2. **Call AIMLite APIs via relative paths**:
        ```javascript
        // Fetch all registered models
        const models = await fetch('/models').then(r => r.json());

        // Submit ML or RAG inference
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: [128.0, 1.0, 2.7, ...] })
        }).then(r => r.json());
        ```
     3. **Launch with AIMLite**:
        - Run `aimlite serve --frontend ./frontend/dist`
        - Or if inside the project directory you have `frontend/dist` or `dist/`, simply running `aimlite serve` auto-detects it!

4. **Developer Template Overrides**:
   - For customization without a separate build step, create `<project_root>/templates/app.html` or `<project_root>/templates/index.html`. AIMLite prioritizes your project templates over the package default.

5. **Headless Mode (`--api`)**:
   - CLI flag `aimlite serve --api` launches a pure JSON REST server without HTML web pages, ideal for Kubernetes, Docker, and microservice architectures.

6. **Documentation & API Reference in `api_docs`**:
   - Updated `api_docs` with:
     - `cli-serve`: Multi-model options, `--api`, and type-adaptive UI details.
     - `endpoint-models`: Documentation for `GET /models` and `POST /models/{name}/predict`.
     - `endpoint-chat`: Details on the type-adaptive web playground at `/app` and `/chat`.
     - `guide-serving`: Comprehensive step-by-step developer guide on serving, multi-model workflows, `--frontend` framework integration, and template overrides.

## 2. Modified & Created Files

- `src/cli/main.py`: Added `--api` argument to `serve_parser` and forwarded `api_only` to `run_serve`.
- `src/cli/commands/evaluate.py`: Enhanced `_discover_checkpoint` to find multi-format checkpoints, RAG indexes, and fixed variable scope.
- `src/cli/commands/serve.py`: Added `detect_model_type`, `discover_feature_names`, multi-model candidate scanning, and template path discovery.
- `src/cli/server.py`: Added `GET /models`, `POST /models/{name}/predict`, template override priority loader, and headless mode routing.
- `src/aimlite/templates/app.html`: Built responsive, monochromatic, self-contained adaptive web playground supporting ML, RAG, and Adapter models.
- `api_docs/src/data/aimliteDocs.ts`: Added documentation for `/models`, adaptive `/app`, and developer serving guide.
- `test/test_serve_multimodel.py`: Added automated test suite verifying type detection, endpoints, and headless mode.
