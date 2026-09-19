"""Scaffolds a new AIMLite project with interactive paradigm and config selection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cli.ui import C, check, cross, next_steps


def _prompt_choice(question: str, options: List[Tuple[str, str]], default_idx: int = 0) -> str:
    """Interactively prompts the user to select from a numbered list of options."""
    if not sys.stdin.isatty():
        return options[default_idx][0]

    print(f"\n  {C.GREEN}?{C.RESET} {C.BOLD}{question}{C.RESET}")
    for idx, (key, label) in enumerate(options, start=1):
        indicator = f"{C.CYAN}{idx}){C.RESET}"
        print(f"    {indicator} {label}")

    prompt_str = f"  {C.DIM}Select option [1-{len(options)}] (default {default_idx + 1}):{C.RESET} » "
    try:
        raw = input(prompt_str).strip()
        if not raw:
            return options[default_idx][0]
        choice_num = int(raw)
        if 1 <= choice_num <= len(options):
            return options[choice_num - 1][0]
    except Exception:
        pass
    return options[default_idx][0]


def _prompt_confirm(question: str, default: bool = True) -> bool:
    """Interactively prompts for a yes/no confirmation."""
    if not sys.stdin.isatty():
        return default

    yn = "Y/n" if default else "y/N"
    prompt_str = f"  {C.GREEN}?{C.RESET} {C.BOLD}{question}{C.RESET} {C.DIM}({yn}):{C.RESET} » "
    try:
        raw = input(prompt_str).strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes", "true", "1")
    except Exception:
        return default


def run_init(
    project_name: Optional[str] = None,
    target_dir: Optional[str] = None,
    create_venv: bool = True,
    template_type: Optional[str] = None,
    chat_provider: Optional[str] = None,
    vector_db: Optional[str] = None,
    embedding_engine: Optional[str] = None,
    interactive: Optional[bool] = None,
) -> int:
    """Initializes a new AIMLite project with paradigm selection and follow-up configs.

    Args:
        project_name: Project name. If None, prompts interactively when interactive is True.
        target_dir: Optional custom parent target directory.
        create_venv: Whether to automatically create .venv and install aimlite.
        template_type: Preselected template ('rag', 'fine-tuning', 'scratch').
        chat_provider: Preselected chat provider ('openai', 'anthropic', 'gemini', 'ollama', 'local', 'mock').
        vector_db: Preselected vector store ('postgres', 'memory').
        embedding_engine: Preselected embedding ('sentence-transformers', 'api', 'tfidf').
        interactive: Whether to prompt for options if not passed.

    Returns:
        Process exit code (0 for success).
    """
    if interactive is None:
        # Only prompt interactively if running in a terminal without a preselected name and template
        interactive = sys.stdin.isatty() and project_name is None and template_type is None

    is_tty = sys.stdin.isatty() and interactive

    # 1. Project Name
    if not project_name:
        if is_tty:
            try:
                print(f"\n  {C.BOLD}{C.BRIGHT_CYAN}AIMLite{C.RESET} {C.DIM}create project{C.RESET}")
                prompt_str = f"  {C.GREEN}?{C.RESET} {C.BOLD}Project name:{C.RESET} {C.DIM}»{C.RESET} "
                prompted = input(prompt_str).strip()
                project_name = prompted or "my_ai"
            except (EOFError, KeyboardInterrupt):
                project_name = "my_ai"
        else:
            project_name = "my_ai"

    # Resolve target directory
    if target_dir:
        dest_root = Path(target_dir).resolve() / project_name
    elif project_name == ".":
        dest_root = Path.cwd()
        project_name = dest_root.name
    else:
        dest_root = Path.cwd() / project_name

    dest_root.mkdir(parents=True, exist_ok=True)

    # 2. System Paradigm / Template Selection
    if not template_type and is_tty:
        template_type = _prompt_choice(
            "Choose a system paradigm to start with:",
            [
                ("rag", "RAG & Knowledge Base (Semantic Search, Grounded LLM Chat, Smart Chunking)"),
                ("fine-tuning", "Fine-Tuning & Adapters (LoRA & PEFT Low-Rank Parameter Adaptation)"),
                ("scratch", "Custom / Scratch ML (Tabular, Supervised, Deep Learning & Classification)"),
            ],
            default_idx=0,
        )
    template_type = (template_type or "scratch").lower()
    if template_type in ("adapter", "peft", "lora"):
        template_type = "fine-tuning"

    rag_config: Dict[str, Any] = {}
    adapter_config: Dict[str, Any] = {}

    # 3. Follow-up configs for RAG
    if template_type == "rag":
        if is_tty and not chat_provider:
            chat_provider = _prompt_choice(
                "Select Chat Provider for LLM synthesis:",
                [
                    ("openai", "OpenAI (GPT-4o, GPT-4o-mini, o1, o3)"),
                    ("gemini", "Google Gemini (gemini-1.5-flash, gemini-1.5-pro, gemini-2.0)"),
                    ("anthropic", "Anthropic Claude (claude-3-5-sonnet, claude-3-opus)"),
                    ("ollama", "Ollama Local (llama3.2, mistral, deepseek-r1, gemma)"),
                    ("local", "Local Hugging Face Pipeline or Custom Python Callable"),
                    ("mock", "Mock / Baseline Provider (Zero external dependencies)"),
                ],
                default_idx=0,
            )
        chat_provider = (chat_provider or "openai").lower()

        if is_tty and not vector_db:
            vector_db = _prompt_choice(
                "Select Vector Database backend:",
                [
                    ("memory", "In-Memory & JSON Serialized (MemoryVectorStore - zero external DB)"),
                    ("postgres", "PostgreSQL with pgvector (PostgresVectorStore & SQL ORM)"),
                ],
                default_idx=0,
            )
        vector_db = (vector_db or "memory").lower()

        if is_tty and not embedding_engine:
            embedding_engine = _prompt_choice(
                "Select Embedding Engine:",
                [
                    ("sentence-transformers", "Dense Neural Vectors (SentenceTransformers - all-MiniLM-L6-v2)"),
                    ("api", "Chat Provider Native API Embeddings"),
                    ("tfidf", "Pure-Python Hash TF-IDF (Fast baseline, zero dependencies)"),
                ],
                default_idx=0,
            )
        embedding_engine = (embedding_engine or "sentence-transformers").lower()

        enable_hyde = _prompt_confirm("Enable Query Intelligence (Intent decomposition, expansion & HyDE)?", default=True) if is_tty else True
        enable_smart_chunker = _prompt_confirm("Enable Structure-Aware Smart Chunker (Markdown heading hierarchy)?", default=True) if is_tty else True

        rag_config = {
            "chat_provider": chat_provider,
            "vector_db": vector_db,
            "embedding_engine": embedding_engine,
            "enable_hyde": enable_hyde,
            "enable_smart_chunker": enable_smart_chunker,
        }

    elif template_type == "fine-tuning":
        adapter_config = {
            "r": 8,
            "alpha": 16.0,
            "target_modules": ["q_proj", "v_proj"],
            "base_model": "meta-llama/Llama-3.2-3B",
        }

    # 4. Create standard directory structure
    convention_dirs = ["data", "models", "experiments", "artifacts", "checkpoints"]
    for d in convention_dirs:
        (dest_root / d).mkdir(parents=True, exist_ok=True)

    # 5. Populate Manifest dependencies & config
    dependencies = []
    if template_type == "rag":
        if embedding_engine == "sentence-transformers":
            dependencies.extend(["numpy", "sentence-transformers"])
        elif embedding_engine == "api":
            dependencies.append("numpy")
        if vector_db == "postgres":
            dependencies.append("psycopg2-binary")
    elif template_type == "fine-tuning":
        dependencies.append("numpy")

    manifest_data: Dict[str, Any] = {
        "name": project_name,
        "version": "0.1.0",
        "entrypoint": project_name,
        "template": template_type,
        "dependencies": dependencies,
        "config": {
            "device": "auto",
            "batch_size": 32,
            **({"rag": rag_config} if rag_config else {}),
            **({"adapter": adapter_config} if adapter_config else {}),
        },
        "paths": {
            "data": "data",
            "models": "models",
            "experiments": "experiments",
            "artifacts": "artifacts",
            "checkpoints": "checkpoints",
        },
    }

    manifest_path = dest_root / "aimlite.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 6. Scaffold Python application package
    package_dir = dest_root / project_name
    package_dir.mkdir(parents=True, exist_ok=True)

    _scaffold_paradigm_files(package_dir, dest_root, project_name, template_type, rag_config, adapter_config)

    print(f"\n  {C.DIM}Scaffolding {template_type.upper()} project in{C.RESET} {dest_root}...")

    # 7. Create .venv and install aimlite
    if create_venv:
        _setup_project_venv(dest_root)

    # 8. Print next steps
    steps = []
    if dest_root != Path.cwd():
        steps.append(f"cd {project_name}")

    if dependencies:
        steps.append(f"aimlite install {' '.join(dependencies)}")
    else:
        steps.append("aimlite install <packages...>")

    steps.extend([
        "aimlite doctor",
        "aimlite train",
        "aimlite serve --port 8000",
    ])
    print(next_steps(steps))

    return 0


def _setup_project_venv(dest_root: Path) -> bool:
    """Creates an isolated virtual environment (.venv) and installs aimlite into it."""
    venv_dir = dest_root / ".venv"
    uv_bin = shutil.which("uv")

    if not venv_dir.is_dir():
        created = False
        if uv_bin:
            res = subprocess.run([uv_bin, "venv", str(venv_dir)], cwd=str(dest_root), capture_output=True)
            if res.returncode == 0:
                created = True
        if not created:
            try:
                import venv

                venv.create(str(venv_dir), with_pip=True)
                created = True
            except Exception:
                res = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=str(dest_root), capture_output=True)
                created = res.returncode == 0
        if not created:
            print(f"  {cross('Could not create virtual environment (.venv) automatically.')}")
            return False
        print(f"  {check('Created virtual environment (.venv).')}")

    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_dir / "Scripts" / "python.exe"

    if not venv_python.exists():
        return False

    installed = False
    if uv_bin:
        res = subprocess.run([uv_bin, "pip", "install", "--python", str(venv_python), "aimlite"], cwd=str(dest_root), capture_output=True)
        if res.returncode == 0:
            installed = True

    if not installed:
        res = subprocess.run([str(venv_python), "-m", "pip", "install", "aimlite"], cwd=str(dest_root), capture_output=True)
        if res.returncode == 0:
            installed = True

    if not installed:
        dist_dir = Path(__file__).resolve().parents[3] / "dist"
        wheels = list(dist_dir.glob("aimlite-*.whl")) if dist_dir.is_dir() else []
        if wheels:
            latest_wheel = sorted(wheels)[-1]
            if uv_bin:
                res = subprocess.run([uv_bin, "pip", "install", "--python", str(venv_python), str(latest_wheel)], cwd=str(dest_root), capture_output=True)
                if res.returncode == 0:
                    installed = True
            if not installed:
                res = subprocess.run([str(venv_python), "-m", "pip", "install", str(latest_wheel)], cwd=str(dest_root), capture_output=True)
                if res.returncode == 0:
                    installed = True

    if installed:
        print(f"  {check('Installed aimlite into .venv.')}")
    else:
        print(f"  {C.DIM}Note: aimlite will be installed into .venv once connected to PyPI.{C.RESET}")

    return True


def _scaffold_paradigm_files(
    package_dir: Path,
    dest_root: Path,
    project_name: str,
    template_type: str,
    rag_config: Dict[str, Any],
    adapter_config: Dict[str, Any],
) -> None:
    """Generates customized Python source files and starter datasets according to the paradigm."""
    config_content = f'''"""AIMLite App Configuration: config.py"""

from pathlib import Path
from aimlite import BaseConfig

BASE_DIR = Path(__file__).resolve().parent.parent


class Config(BaseConfig):
    """Project configuration declaring paths and hardware settings."""

    name: str = "{project_name}"
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    experiments_dir: Path = BASE_DIR / "experiments"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    checkpoints_dir: Path = BASE_DIR / "checkpoints"
    device: str = "auto"
    batch_size: int = 32
'''
    (package_dir / "config.py").write_text(config_content, encoding="utf-8")
    (package_dir / "__init__.py").write_text('"""AIMLite Package."""\n', encoding="utf-8")

    if template_type == "rag":
        _scaffold_rag(package_dir, dest_root, project_name, rag_config)
    elif template_type == "fine-tuning":
        _scaffold_fine_tuning(package_dir, dest_root, project_name, adapter_config)
    else:
        _scaffold_scratch(package_dir, dest_root, project_name)


def _scaffold_rag(package_dir: Path, dest_root: Path, project_name: str, rag_config: Dict[str, Any]) -> None:
    """Generates RAG & Knowledge Model files."""
    chat_p = rag_config.get("chat_provider", "openai")
    vector_db = rag_config.get("vector_db", "memory")
    embed_engine = rag_config.get("embedding_engine", "sentence-transformers")

    # Determine ChatProvider constructor
    provider_cls = {
        "openai": 'OpenAIChatProvider(model="gpt-4o-mini")',
        "gemini": 'GeminiChatProvider(model="gemini-1.5-flash")',
        "anthropic": 'AnthropicChatProvider(model="claude-3-5-sonnet-20241022")',
        "ollama": 'OllamaChatProvider(model="llama3.2")',
        "local": "LocalChatProvider()",
        "mock": "MockChatProvider()",
    }.get(chat_p, 'OpenAIChatProvider(model="gpt-4o-mini")')

    # Determine VectorStore constructor
    store_cls = "PostgresVectorStore()" if vector_db == "postgres" else "MemoryVectorStore(embedding_fn=embedding_fn)"

    # Determine Embedder constructor
    if embed_engine == "sentence-transformers":
        embed_code = """        try:
            return SentenceTransformerEmbedding(model_name_or_path="all-MiniLM-L6-v2")
        except Exception:
            return TfidfEmbedding()"""
    elif embed_engine == "api":
        embed_code = "        return self.chat_provider.get_embedder()"
    else:
        embed_code = "        return TfidfEmbedding()"

    data_py = '''"""Document & Knowledge QA (RAG Paradigm): data.py"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from aimlite import Dataset
from aimlite.rag import Document, SmartChunker


class KnowledgeDocsDataset(Dataset):
    """Ingests documentation articles and splits them into retrievable passages using SmartChunker."""

    filename: str = "knowledge_base.md"

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        max_chunk_size: int = 400,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(source=src, **kwargs)
        self.chunker = SmartChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        docs = self.load_documents()
        return [doc.to_dict() for doc in docs]

    def load_documents(self) -> List[Document]:
        raw_documents: List[Document] = []
        data_dir = Path("data")

        if data_dir.is_dir():
            for p in data_dir.glob("*.*"):
                if p.suffix.lower() in (".md", ".txt", ".markdown", ".rst"):
                    try:
                        text = p.read_text(encoding="utf-8")
                        raw_documents.append(
                            Document(content=text, metadata={"source": p.name, "title": p.stem})
                        )
                    except Exception:
                        continue

        return self.chunker.split_documents(raw_documents)
'''
    (package_dir / "data.py").write_text(data_py, encoding="utf-8")

    model_py = f'''"""Document & Knowledge QA (RAG Paradigm): model.py"""

from typing import Any, Dict, Optional
from aimlite.rag import (
    AnthropicChatProvider,
    BaseChatProvider,
    BaseEmbedding,
    BaseVectorStore,
    GeminiChatProvider,
    KnowledgeModel,
    LocalChatProvider,
    MemoryVectorStore,
    MockChatProvider,
    OllamaChatProvider,
    OpenAIChatProvider,
    PostgresVectorStore,
    QueryAnalyzer,
    SentenceTransformerEmbedding,
    SmartChunker,
    TfidfEmbedding,
)


class SupportDocRAG(KnowledgeModel):
    """Enterprise Knowledge Base Model with Query Intelligence, Smart Chunking, and Grounded Chat."""

    def __init__(
        self,
        name: str = "{project_name}_rag",
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        chat_provider: Optional[BaseChatProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        **kwargs: Any,
    ) -> None:
        provider = chat_provider or {provider_cls}
        self.chat_provider = provider
        embedding_fn = self._init_embedding()
        analyzer = QueryAnalyzer(chat_provider=provider, enable_hyde=True)
        store = vector_store or {store_cls}
        chunker = SmartChunker(max_chunk_size=400)

        super().__init__(
            name=name,
            config=config,
            chat_provider=provider,
            embedding_fn=embedding_fn,
            vector_store=store,
            chunker=chunker,
            query_analyzer=analyzer,
            top_k=top_k,
            **kwargs,
        )

    def _init_embedding(self) -> BaseEmbedding:
{embed_code}
'''
    (package_dir / "model.py").write_text(model_py, encoding="utf-8")

    trainer_py = '''"""Document & Knowledge QA (RAG Paradigm): trainer.py"""

from aimlite.rag import RAGTrainer


class IndexBuilderTrainer(RAGTrainer):
    """Trainer orchestrator building and persisting the semantic vector index."""
    pass
'''
    (package_dir / "trainer.py").write_text(trainer_py, encoding="utf-8")

    evaluator_py = '''"""Document & Knowledge QA (RAG Paradigm): evaluator.py"""

from typing import Any, Dict
from aimlite import BaseEvaluator, Model


class RAGEvaluator(BaseEvaluator):
    """Evaluates knowledge retrieval coverage and grounded answer synthesis."""

    def evaluate(self, model: Model, test_data: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "retrieval_status": "ready",
            "model_name": getattr(model, "name", "rag_model"),
        }
'''
    (package_dir / "evaluator.py").write_text(evaluator_py, encoding="utf-8")

    inference_py = '''"""Document & Knowledge QA (RAG Paradigm): inference.py"""

from pathlib import Path
from typing import Any, Dict
from aimlite import BaseInference, Model


class RAGInference(BaseInference):
    """Production inference endpoint for semantic knowledge retrieval and answer synthesis."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.index_path = Path("artifacts") / "rag_index.json"

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        if self.index_path.is_file():
            model.load(self.index_path)

        query = raw_input.get("query", "") if isinstance(raw_input, dict) else str(raw_input)
        if not query:
            return {"error": "Missing 'query' field in request body", "status": "failed"}

        result = model.predict(query)
        return {
            **result,
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
'''
    (package_dir / "inference.py").write_text(inference_py, encoding="utf-8")

    # Sample knowledge documentation in data/
    sample_md = f"""# {project_name.replace('_', ' ').title()} Knowledge Base

## Overview & Architecture
AIMLite provides zero-path convention-over-configuration execution for machine learning and AI.
All project directories (`data/`, `models/`, `experiments/`, `artifacts/`) are automatically resolved.

## Authentication & Security
- API Endpoints require standard Bearer token authentication in production.
- Multi-factor authentication (MFA) is enforced for administrative model registry operations.
- Session tokens expire after 24 hours of inactivity.

## Deployments & Scaling
AIMLite models are served via `aimlite serve --port 8000`.
For production deployments, containerize with Docker and scale horizontally with Kubernetes.
"""
    (dest_root / "data" / "knowledge_base.md").write_text(sample_md, encoding="utf-8")

    # Client starter script in root
    client_py = f'''"""Client test script for {project_name} RAG Knowledge Base."""

from {project_name}.data import KnowledgeDocsDataset
from {project_name}.model import SupportDocRAG


def main() -> None:
    print("[*] Initializing {project_name} Knowledge Model...")
    model = SupportDocRAG()
    dataset = KnowledgeDocsDataset()

    # Ingest and chunk documents
    docs = dataset.load_documents()
    print(f"[*] Ingested and chunked {{len(docs)}} passage(s) from data/.")
    model.index_documents(docs)

    # Execute test query
    test_query = "How do session tokens expire?"
    print(f"\\n[?] Query: {{test_query}}")
    result = model.predict(test_query, top_k=2)

    print(f"[!] Answer: {{result['answer']}}")
    print(f"[!] Sources Citations:")
    for idx, s in enumerate(result.get("sources", []), start=1):
        meta = s.get("metadata", {{}})
        source_label = meta.get("source", "knowledge_base.md")
        if meta.get("heading_path"):
            source_label = f"{{source_label}} ({{meta.get('heading_path')}})"
        score_str = f"{{s.get('score', 0):.4f}}" if isinstance(s.get("score"), (int, float)) else str(s.get("score"))
        print(f"    [{{idx}}] {{source_label}} (score: {{score_str}})")


if __name__ == "__main__":
    main()
'''
    (dest_root / "client.py").write_text(client_py, encoding="utf-8")

    # Project README
    readme_md = f"""# {project_name.replace('_', ' ').title()} (RAG Knowledge Engine)

Built with [AIMLite](https://github.com/Alazar42/aimlite) — The Django for AI & Machine Learning.

## Quickstart

```bash
# 1. Install dependencies
aimlite install

# 2. Validate data
aimlite data validate

# 3. Build & persist vector index
aimlite train

# 4. Run local client test
python client.py

# 5. Serve HTTP API
aimlite serve --port 8000
```
"""
    (dest_root / "README.md").write_text(readme_md, encoding="utf-8")


def _scaffold_fine_tuning(package_dir: Path, dest_root: Path, project_name: str, adapter_config: Dict[str, Any]) -> None:
    """Generates Fine-Tuning & LoRA Adapter files."""
    data_py = '''"""Fine-Tuning & LoRA (Adapter Paradigm): data.py"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from aimlite import Dataset


class InstructionDataset(Dataset):
    """Loads prompt/response instruction pairs for parameter-efficient adaptation."""

    filename: str = "instructions.jsonl"

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        file_path = Path("data") / self.filename
        if not file_path.is_file():
            return [
                {"instruction": "Explain LoRA fine-tuning.", "response": "LoRA freezes base weights and trains low-rank delta matrices."},
                {"instruction": "What is zero-path execution?", "response": "Zero-path execution automatically discovers ML conventions."},
            ]
        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
'''
    (package_dir / "data.py").write_text(data_py, encoding="utf-8")

    model_py = f'''"""Fine-Tuning & LoRA (Adapter Paradigm): model.py"""

from typing import Any, Dict, Optional
from aimlite import AdapterConfig, AdapterModel


class LoRAInstructionModel(AdapterModel):
    """Parameter-efficient LoRA adapter model with low-rank matrix decomposition."""

    def __init__(
        self,
        name: str = "{project_name}_adapter",
        config: Optional[Dict[str, Any]] = None,
        r: int = {adapter_config.get("r", 8)},
        alpha: float = {adapter_config.get("alpha", 16.0)},
        **kwargs: Any,
    ) -> None:
        adapter_cfg = AdapterConfig(
            r=r,
            alpha=alpha,
            target_modules={adapter_config.get("target_modules", ["q_proj", "v_proj"])},
        )
        super().__init__(name=name, adapter_config=adapter_cfg, config=config, **kwargs)

    def predict(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        prompt = inputs.get("instruction", str(inputs)) if isinstance(inputs, dict) else str(inputs)
        active_name = self.adapter_manager.active_adapter_name if hasattr(self, "adapter_manager") else "default"
        return {{
            "instruction": prompt,
            "response": f"Adapter-tuned response for: '{{prompt}}'",
            "adapter_active": active_name,
        }}
'''
    (package_dir / "model.py").write_text(model_py, encoding="utf-8")

    trainer_py = '''"""Fine-Tuning & LoRA (Adapter Paradigm): trainer.py"""

from aimlite import AdapterTrainer


class AdapterInstructionTrainer(AdapterTrainer):
    """Orchestrates parameter-efficient LoRA optimization with trainable parameter diagnostics."""
    pass
'''
    (package_dir / "trainer.py").write_text(trainer_py, encoding="utf-8")

    evaluator_py = '''"""Fine-Tuning & LoRA (Adapter Paradigm): evaluator.py"""

from typing import Any, Dict
from aimlite import BaseEvaluator, Model


class AdapterEvaluator(BaseEvaluator):
    """Assesses parameter efficiency and loss convergence on held-out instructions."""

    def evaluate(self, model: Model, test_data: Any, **kwargs: Any) -> Dict[str, Any]:
        diag = model.get_trainable_parameters() if hasattr(model, "get_trainable_parameters") else {}
        return {
            "evaluation_status": "passed",
            "trainable_percent": diag.get("trainable_percent", 0.12),
        }
'''
    (package_dir / "evaluator.py").write_text(evaluator_py, encoding="utf-8")

    inference_py = '''"""Fine-Tuning & LoRA (Adapter Paradigm): inference.py"""

from typing import Any, Dict
from aimlite import BaseInference, Model


class AdapterInference(BaseInference):
    """Production endpoint serving fine-tuned adapter responses."""

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            **model.predict(raw_input),
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
'''
    (package_dir / "inference.py").write_text(inference_py, encoding="utf-8")

    # Sample instructions in data/
    sample_jsonl = """{"instruction": "What is AIMLite?", "response": "AIMLite is the Django for AI & Machine Learning with zero-path CLI execution."}
{"instruction": "How does LoRA reduce checkpoint size?", "response": "LoRA freezes foundation parameters and trains low-rank delta matrices, reducing weights from 14GB down to under 50MB."}
{"instruction": "How do you evaluate adapter models in AIMLite?", "response": "Run aimlite evaluate to compute parameter efficiency, rank sparsity, and loss convergence."}
"""
    (dest_root / "data" / "instructions.jsonl").write_text(sample_jsonl, encoding="utf-8")

    # Client starter script in root
    client_py = f'''"""Client test script for {project_name} Fine-Tuning / LoRA Adapter Model."""

from {project_name}.data import InstructionDataset
from {project_name}.model import LoRAInstructionModel


def main() -> None:
    print("[*] Initializing {project_name} LoRA Adapter Model...")
    model = LoRAInstructionModel()
    dataset = InstructionDataset()

    # Load instruction dataset
    records = dataset.load()
    print(f"[*] Loaded {{len(records)}} training instruction pair(s) from data/.")

    # Inspect adapter parameter efficiency
    params = model.get_trainable_parameters() if hasattr(model, "get_trainable_parameters") else {{}}
    print(f"[*] Adapter Diagnostics: {{params}}")

    # Run inference test on prompt instructions
    test_prompts = [
        "What is AIMLite?",
        "How does LoRA reduce checkpoint size?",
        "How to deploy this model to production?",
    ]

    print("\\n[>] Running Client Inference Tests:")
    for prompt in test_prompts:
        result = model.predict({{"instruction": prompt}})
        print(f"\\n  [Prompt]   {{prompt}}")
        print(f"  [Response] {{result.get('response', result)}}")
        print(f"  [Adapter]  {{result.get('adapter_active', 'default')}}")


if __name__ == "__main__":
    main()
'''
    (dest_root / "client.py").write_text(client_py, encoding="utf-8")

    # Project README
    readme_md = f"""# {project_name.replace('_', ' ').title()} (LoRA / Fine-Tuning)

Built with [AIMLite](https://github.com/Alazar42/aimlite) — The Django for AI & Machine Learning.

## Quickstart

```bash
# 1. Install dependencies
aimlite install

# 2. Validate data
aimlite data validate

# 3. Train LoRA adapter weights
aimlite train

# 4. Assess parameter efficiency and loss
aimlite evaluate

# 5. Run local client test
python client.py

# 6. Serve HTTP API
aimlite serve --port 8000
```
"""
    (dest_root / "README.md").write_text(readme_md, encoding="utf-8")


def _scaffold_scratch(package_dir: Path, dest_root: Path, project_name: str) -> None:
    """Generates standard supervised / custom ML starter files from app_template."""
    try:
        from aimlite import templates

        template_dir = Path(templates.__file__).parent / "app_template"
    except Exception:
        template_dir = Path(__file__).resolve().parent.parent.parent / "aimlite" / "templates" / "app_template"

    if template_dir.is_dir():
        for filename in ["data.py", "model.py", "trainer.py", "evaluator.py", "inference.py"]:
            src = template_dir / filename
            dst = package_dir / filename
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
    else:
        _write_fallback_scratch_files(package_dir)

    # Sample dataset in data/
    sample_csv = """feature_1,feature_2,feature_3,target
1.2,3.4,0.5,10.2
2.1,1.8,1.4,14.8
0.9,4.2,0.8,12.1
3.5,2.1,2.0,19.4
1.8,3.1,1.1,13.5
"""
    (dest_root / "data" / "dataset.csv").write_text(sample_csv, encoding="utf-8")

    # Client starter script in root
    client_py = f'''"""Client test script for {project_name} Custom ML Model."""

from {project_name}.data import AppDataset
from {project_name}.model import AppModel


def main() -> None:
    print("[*] Initializing {project_name} Model...")
    model = AppModel()
    dataset = AppDataset()

    print(f"[*] App dataset ready (file: {{getattr(dataset, 'filename', 'dataset.csv')}}).")

    # Test sample inference
    sample_inputs = [
        {{"feature_1": 1.2, "feature_2": 3.4, "feature_3": 0.5}},
        {{"feature_1": 2.1, "feature_2": 1.8, "feature_3": 1.4}},
    ]

    print("\\n[>] Running Client Inference Tests:")
    for sample in sample_inputs:
        output = model.predict(sample)
        print(f"  [Input]  {{sample}}")
        print(f"  [Output] {{output}}")


if __name__ == "__main__":
    main()
'''
    (dest_root / "client.py").write_text(client_py, encoding="utf-8")

    # Project README
    readme_md = f"""# {project_name.replace('_', ' ').title()} (AIMLite Custom ML)

Built with [AIMLite](https://github.com/Alazar42/aimlite) — The Django for AI & Machine Learning.

## Quickstart

```bash
# 1. Install dependencies
aimlite install

# 2. Validate data
aimlite data validate

# 3. Train model
aimlite train

# 4. Evaluate performance
aimlite evaluate

# 5. Run local client test
python client.py

# 6. Serve HTTP API
aimlite serve --port 8000
```
"""
    (dest_root / "README.md").write_text(readme_md, encoding="utf-8")


def _write_fallback_scratch_files(package_dir: Path) -> None:
    """Fallback generator for standard starter files."""
    data_py = '''"""AIMLite App: data.py"""

from aimlite import Dataset


class AppDataset(Dataset):
    """Application dataset definition."""
    filename = "dataset.csv"
'''
    (package_dir / "data.py").write_text(data_py, encoding="utf-8")

    model_py = '''"""AIMLite App: model.py"""

from typing import Any
from aimlite import Model


class AppModel(Model):
    """Application model definition."""

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs
'''
    (package_dir / "model.py").write_text(model_py, encoding="utf-8")

    trainer_py = '''"""AIMLite App: trainer.py"""

from typing import Any, Dict
from aimlite import BaseTrainer, Dataset, Model


class AppTrainer(BaseTrainer):
    """Application training orchestrator."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "completed"}
'''
    (package_dir / "trainer.py").write_text(trainer_py, encoding="utf-8")

    evaluator_py = '''"""AIMLite App: evaluator.py"""

from typing import Any, Dict
from aimlite import BaseEvaluator, Model


class AppEvaluator(BaseEvaluator):
    """Application evaluation benchmark."""

    def evaluate(self, model: Model, test_data: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"accuracy": 1.0}
'''
    (package_dir / "evaluator.py").write_text(evaluator_py, encoding="utf-8")

    inference_py = '''"""AIMLite App: inference.py"""

from typing import Any, Dict
from aimlite import BaseInference, Model


class AppInference(BaseInference):
    """Application inference handler."""

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"result": model.predict(raw_input)}

    def get_routes(self) -> Dict[str, Any]:
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
'''
    (package_dir / "inference.py").write_text(inference_py, encoding="utf-8")
