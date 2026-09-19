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


def _prompt_text(question: str, default: str) -> str:
    """Interactively prompts the user for text input with a default value."""
    if not sys.stdin.isatty():
        return default
    print(f"\n  {C.GREEN}?{C.RESET} {C.BOLD}{question}{C.RESET} {C.DIM}(default: {default}){C.RESET}")
    prompt_str = f"  {C.DIM}»{C.RESET} "
    try:
        raw = input(prompt_str).strip()
        return raw or default
    except Exception:
        return default


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
    model_name: Optional[str] = None,
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
        model_name: Preselected LLM model identifier.
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

        # Medium-level model defaults
        default_model = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-1.5-flash",
            "anthropic": "claude-3-5-sonnet-20241022",
            "ollama": "llama3.2",
            "local": "meta-llama/Llama-3.2-3B",
            "mock": "mock-gpt",
        }.get(chat_provider, "gpt-4o-mini")

        if is_tty and not model_name:
            model_name = _prompt_text(f"Specify model identifier for {chat_provider.upper()}:", default=default_model)
        else:
            model_name = model_name or default_model

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

        default_embed_model = "all-MiniLM-L6-v2" if embedding_engine == "sentence-transformers" else (
            "text-embedding-3-small" if chat_provider == "openai" else ("nomic-embed-text" if chat_provider == "ollama" else "tfidf")
        )

        enable_hyde = _prompt_confirm("Enable Query Intelligence (Intent decomposition, expansion & HyDE)?", default=True) if is_tty else True
        enable_smart_chunker = _prompt_confirm("Enable Structure-Aware Smart Chunker (Markdown heading hierarchy)?", default=True) if is_tty else True

        rag_config = {
            "chat_provider": chat_provider,
            "model_name": model_name,
            "embedding_model": default_embed_model,
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
    """Generates RAG & Knowledge Model files with modular chat provider and storage starters."""
    chat_p = rag_config.get("chat_provider", "openai")
    model_name = rag_config.get("model_name", "gpt-4o-mini")
    vector_db = rag_config.get("vector_db", "memory")
    embed_engine = rag_config.get("embedding_engine", "sentence-transformers")
    embed_model = rag_config.get("embedding_model", "all-MiniLM-L6-v2")

    # 1. Chat Provider Starter Code: chat_provider.py
    chat_provider_py = f'''"""Chat Provider & Query Intelligence: chat_provider.py

Starter code for configuring LLM synthesis, system prompts, and query analysis (HyDE, intent decomposition, expansion).
"""

import os
from typing import Any, Dict, Optional
from aimlite.rag import (
    AnthropicChatProvider,
    BaseChatProvider,
    GeminiChatProvider,
    LocalChatProvider,
    MockChatProvider,
    OllamaChatProvider,
    OpenAIChatProvider,
    QueryAnalyzer,
    PROMPT_CONVERSATIONAL_RAG,
    PROMPT_QUERY_ANALYZER,
    PROMPT_RAG_QA,
    PROMPT_SMART_CHUNKER,
)


def get_chat_provider(
    provider_name: str = "{chat_p}",
    model_name: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> BaseChatProvider:
    """Instantiates and returns the configured LLM Chat Provider reading models and endpoints from environment."""
    prompt = system_prompt or PROMPT_RAG_QA

    if provider_name == "openai":
        return OpenAIChatProvider(
            model=model_name or os.environ.get("OPENAI_MODEL", "{model_name}"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "gemini":
        return GeminiChatProvider(
            model=model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "anthropic":
        return AnthropicChatProvider(
            model=model_name or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "ollama":
        return OllamaChatProvider(
            model=model_name or os.environ.get("OLLAMA_MODEL", "{model_name}"),
            base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            system_prompt=prompt,
            temperature=temperature,
            **kwargs,
        )
    elif provider_name == "local":
        return LocalChatProvider(
            model_name=model_name or os.environ.get("LOCAL_MODEL", "meta-llama/Llama-3.2-3B"),
            system_prompt=prompt,
            **kwargs,
        )
    else:
        return MockChatProvider(
            model=model_name or os.environ.get("MOCK_MODEL", "mock-gpt"),
            system_prompt=prompt,
            **kwargs,
        )


def get_query_analyzer(
    chat_provider: Optional[BaseChatProvider] = None,
    enable_hyde: bool = True,
) -> QueryAnalyzer:
    """Configures Query Intelligence for intent parsing, expansion, and hypothetical document generation."""
    provider = chat_provider or get_chat_provider()
    return QueryAnalyzer(chat_provider=provider, enable_hyde=enable_hyde)
'''
    (package_dir / "chat_provider.py").write_text(chat_provider_py, encoding="utf-8")

    # 2. Storage & Vector Store Starter Code: store.py
    store_py = f'''"""Vector Store & Database Storage: store.py

Starter code for semantic vector storage, embeddings, and PostgreSQL ORM records.
"""

import os
from typing import Any, Dict, Optional
from aimlite.rag import (
    APIEmbedding,
    BaseEmbedding,
    BaseVectorStore,
    KnowledgeChunkRecord,
    KnowledgeDocumentRecord,
    MemoryVectorStore,
    OllamaEmbedding,
    PostgresVectorStore,
    SentenceTransformerEmbedding,
    TfidfEmbedding,
)


def get_embedding_model(
    engine: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseEmbedding:
    """Instantiates embedding model (Dense Neural, Ollama API, OpenAI API, or Pure-Python TF-IDF)."""
    resolved_engine = (engine or os.environ.get("EMBEDDING_ENGINE") or "{embed_engine}").lower()
    resolved_model = model_name or os.environ.get("EMBEDDING_MODEL", "{embed_model}")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    if resolved_engine == "ollama" or "nomic" in resolved_model.lower():
        return OllamaEmbedding(model=resolved_model, host=ollama_host)
    elif resolved_engine == "openai":
        return APIEmbedding(provider="openai", model=resolved_model)
    elif resolved_engine in ("sentence-transformers", "local"):
        return SentenceTransformerEmbedding(model_name_or_path=resolved_model)
    elif resolved_engine == "tfidf":
        return TfidfEmbedding()
    return SentenceTransformerEmbedding(model_name_or_path=resolved_model)


def get_vector_store(
    backend: Optional[str] = None,
    embedding_fn: Optional[BaseEmbedding] = None,
    db_url: Optional[str] = None,
    **kwargs: Any,
) -> BaseVectorStore:
    """Instantiates vector storage backend (PostgreSQL pgvector ORM or MemoryVectorStore)."""
    resolved_backend = (backend or os.environ.get("VECTOR_STORE") or "{vector_db}").lower()
    resolved_db = db_url or os.environ.get("DATABASE_URL")
    embed_fn = embedding_fn or get_embedding_model()

    if resolved_backend == "postgres" or (resolved_db and resolved_backend != "memory"):
        return PostgresVectorStore(
            db_url=resolved_db,
            embedding_fn=embed_fn,
            **kwargs,
        )
    return MemoryVectorStore(
        embedding_fn=embed_fn,
        **kwargs,
    )
'''
    (package_dir / "store.py").write_text(store_py, encoding="utf-8")

    # 3. Data & Chunking: data.py
    data_py = '''"""Document & Knowledge QA (RAG Paradigm): data.py"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from aimlite import Dataset
from aimlite.rag import Document, DocumentLoader, SmartChunker


class KnowledgeDocsDataset(Dataset):
    """Ingests documentation articles, datasets, and custom user files into chunked passages."""

    filename: str = "knowledge_base.md"

    def __init__(
        self,
        source: Optional[Union[str, Path]] = None,
        data_path: Optional[Union[str, Path]] = None,
        max_chunk_size: int = 400,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        src = source or data_path or "data"
        super().__init__(source=src, **kwargs)
        self.chunker = SmartChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        self.additional_documents: List[Document] = []

    def add_document(self, content: str, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Dynamically appends a new document or user-populated text record to the dataset."""
        meta = metadata or {}
        if title:
            meta["title"] = title
        self.additional_documents.append(Document(content=content, metadata=meta))

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        docs = self.load_documents()
        return [doc.to_dict() for doc in docs]

    def load_documents(self) -> List[Document]:
        """Loads and chunks all prepared files from the data directory and user-added documents."""
        raw_documents: List[Document] = list(self.additional_documents)
        data_target = Path(self.source) if self.source else Path("data")

        valid_extensions = {".md", ".txt", ".markdown", ".rst", ".json", ".csv"}

        if data_target.is_file():
            raw_documents.extend(DocumentLoader.load_file(data_target))
        elif data_target.is_dir():
            for p in sorted(data_target.rglob("*")):
                if p.is_file() and p.suffix.lower() in valid_extensions and not p.name.startswith("."):
                    try:
                        raw_documents.extend(DocumentLoader.load_file(p))
                    except Exception:
                        continue

        if not raw_documents:
            raise ValueError(
                f"No knowledge documents found in target path '{data_target}'. "
                "Please place your .md, .txt, .json, or .csv files inside the 'data/' folder or pass a custom path."
            )

        return self.chunker.split_documents(raw_documents)
'''
    (package_dir / "data.py").write_text(data_py, encoding="utf-8")

    # 4. Model Architecture: model.py
    model_py = f'''"""Document & Knowledge QA (RAG Paradigm): model.py"""

from typing import Any, Dict, Optional
from aimlite.rag import (
    BaseChatProvider,
    BaseEmbedding,
    BaseVectorStore,
    KnowledgeModel,
    QueryAnalyzer,
    SmartChunker,
)
from {project_name}.chat_provider import get_chat_provider, get_query_analyzer
from {project_name}.store import get_embedding_model, get_vector_store


class SupportDocRAG(KnowledgeModel):
    """Enterprise Knowledge Base Model with Query Intelligence, Smart Chunking, and Grounded Chat."""

    def __init__(
        self,
        name: str = "{project_name}_rag",
        config: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        chat_provider: Optional[BaseChatProvider] = None,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_fn: Optional[BaseEmbedding] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 40,
        **kwargs: Any,
    ) -> None:
        provider = chat_provider or get_chat_provider()
        self.chat_provider = provider

        embed_model = embedding_fn or get_embedding_model()
        store = vector_store or get_vector_store(embedding_fn=embed_model)
        analyzer = query_analyzer or get_query_analyzer(chat_provider=provider)
        chunker = SmartChunker(max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        super().__init__(
            name=name,
            config=config,
            chat_provider=provider,
            embedding_fn=embed_model,
            vector_store=store,
            chunker=chunker,
            query_analyzer=analyzer,
            top_k=top_k,
            **kwargs,
        )
'''
    (package_dir / "model.py").write_text(model_py, encoding="utf-8")

    # 5. Trainer, Evaluator, Inference
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

    # 6. Sample knowledge documentation in data/
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

    faq_md = """# Frequently Asked Questions (FAQ)

## How does AIMLite manage model checkpoints?
AIMLite automatically persists model weights, optimizer states, and configuration metadata into the `checkpoints/` directory.

## Can I switch from SQLite/Memory to PostgreSQL in production?
Yes! Simply configure `PostgresVectorStore(db_url="postgresql://user:pass@localhost:5432/rag_db")` in `store.py`.

## How do I customize LLM System Prompts?
Edit `chat_provider.py` to change prompt templates or pass custom instructions directly to `get_chat_provider()`.
"""
    (dest_root / "data" / "faq.md").write_text(faq_md, encoding="utf-8")

    # 7. Experiments benchmark starter script
    benchmark_py = f'''"""Benchmark and evaluation experiment script for {project_name} RAG Knowledge Base."""

import time
from {project_name}.data import KnowledgeDocsDataset
from {project_name}.model import SupportDocRAG


def run_benchmark() -> None:
    print("=" * 60)
    print("  AIMLite RAG Knowledge Base Retrieval & Latency Benchmark")
    print("=" * 60)

    model = SupportDocRAG()
    dataset = KnowledgeDocsDataset()

    # 1. Ingestion Benchmark
    t0 = time.perf_counter()
    docs = dataset.load_documents()
    model.index_documents(docs)
    ingest_time = (time.perf_counter() - t0) * 1000
    print(f"\\n[*] Ingestion & Indexing: {{len(docs)}} passages indexed in {{ingest_time:.2f}} ms")

    # 2. Query Latency & Grounding Tests
    benchmark_queries = [
        "What are the authentication and session expiration rules?",
        "How do you serve and scale AIMLite in production?",
        "How do I customize LLM System Prompts?",
    ]

    print("\\n[*] Evaluating Benchmark Queries:")
    for idx, query in enumerate(benchmark_queries, start=1):
        t0 = time.perf_counter()
        result = model.predict(query, top_k=2)
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"\\n[{{idx}}] Query: '{{query}}' ({{latency_ms:.2f}} ms)")
        print(f"    Answer: {{result.get('answer', '')[:120]}}...")
        print(f"    Retrieved Sources: {{len(result.get('sources', []))}} chunks")

    print("\\n[OK] Benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
'''
    (dest_root / "experiments" / "benchmark.py").write_text(benchmark_py, encoding="utf-8")

    # 8. Directory placeholders & Git configuration
    for d in ["models", "artifacts", "checkpoints"]:
        (dest_root / d / ".gitkeep").write_text("", encoding="utf-8")

    gitignore_content = """# Environments & Virtual Envs
.venv/
env/
venv/
ENV/

# Python cache & artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
build/
dist/
*.egg-info/

# Sensitive Environment Variables & Secrets
.env
.env.local
.env.*.local

# ML Artifacts & Checkpoints
artifacts/*.json
artifacts/*.pkl
checkpoints/*.pt
checkpoints/*.safetensors
checkpoints/*.bin
"""
    (dest_root / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    # 9. Sensitive variables starter template: .env.example and auto-created .env
    env_content = f"""# ==============================================================================
# AIMLite RAG Knowledge Engine: Environment Variables & Secrets (.env)
# ==============================================================================

# --- Chat LLM Model Identifiers ---
OPENAI_MODEL={"gpt-4o-mini" if chat_p != "openai" else model_name}
ANTHROPIC_MODEL={"claude-3-5-sonnet-20241022" if chat_p != "anthropic" else model_name}
GEMINI_MODEL={"gemini-1.5-flash" if chat_p != "gemini" else model_name}
OLLAMA_MODEL={"llama3.2" if chat_p != "ollama" else model_name}
LOCAL_MODEL=meta-llama/Llama-3.2-3B

# --- Embedding Engine & Identifier ---
EMBEDDING_ENGINE={"ollama" if "nomic" in embed_model.lower() or chat_p == "ollama" else ("openai" if "openai" in embed_model.lower() else "sentence-transformers")}
EMBEDDING_MODEL={embed_model}

# --- LLM API Credentials ---
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=

# --- Local LLM & Ollama ---
OLLAMA_HOST=http://localhost:11434

# --- Storage & Database Vector Storage ---
VECTOR_STORE={vector_db}
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/atoai

# --- Hardware Acceleration & Serving ---
AIMLITE_DEVICE=auto
AIMLITE_PORT=8000
"""
    (dest_root / ".env.example").write_text(env_content, encoding="utf-8")
    if not (dest_root / ".env").exists():
        (dest_root / ".env").write_text(env_content, encoding="utf-8")

    # 10. Client starter script in root with health check
    client_py = f'''"""Client test script for {project_name} RAG Knowledge Base."""

import os
import urllib.request
from {project_name}.data import KnowledgeDocsDataset
from {project_name}.model import SupportDocRAG


def check_provider_health(model: SupportDocRAG) -> None:
    """Pre-flight check verifying whether local servers (Ollama) or required API keys are configured."""
    provider = getattr(model, "chat_provider", None)
    if not provider:
        return

    provider_name = provider.__class__.__name__

    if "Ollama" in provider_name:
        host = getattr(provider, "base_url", "http://localhost:11434")
        model_name = getattr(provider, "model", "llama3.2")
        try:
            req = urllib.request.Request(f"{{host}}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                pass
            print(f"[*] Ollama server connected at {{host}} (model: {{model_name}})")
        except Exception:
            print(f"\\n[!] Note: Ollama daemon is not responding at {{host}}.")
            print(f"    To run Ollama locally:")
            print(f"      1. Start server:   ollama serve")
            print(f"      2. Pull model:     ollama pull {{model_name}}")
            print(f"      3. Set OLLAMA_HOST or OLLAMA_MODEL in .env if using custom settings\\n")

    elif "OpenAI" in provider_name:
        if not getattr(provider, "api_key", None) and not os.environ.get("OPENAI_API_KEY"):
            print("\\n[!] Note: OPENAI_API_KEY is not set in .env. LLM generation requires an active key.\\n")
    elif "Anthropic" in provider_name:
        if not getattr(provider, "api_key", None) and not os.environ.get("ANTHROPIC_API_KEY"):
            print("\\n[!] Note: ANTHROPIC_API_KEY is not set in .env. LLM generation requires an active key.\\n")
    elif "Gemini" in provider_name:
        if not getattr(provider, "api_key", None) and not os.environ.get("GEMINI_API_KEY"):
            print("\\n[!] Note: GEMINI_API_KEY is not set in .env. LLM generation requires an active key.\\n")


def main() -> None:
    print("[*] Initializing {project_name} Knowledge Model...")
    model = SupportDocRAG()
    dataset = KnowledgeDocsDataset()

    check_provider_health(model)

    # Ingest and chunk documents
    docs = dataset.load_documents()
    print(f"[*] Ingested and chunked {{len(docs)}} passage(s) from data/.")
    model.index_documents(docs)

    # Execute test queries
    test_queries = [
        "How do session tokens expire?",
        "How can I switch to PostgreSQL for vector storage?",
    ]

    for test_query in test_queries:
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

    # 11. Project README
    readme_md = f"""# {project_name.replace('_', ' ').title()} (RAG Knowledge Engine)

Built with [AIMLite](https://github.com/Alazar42/aimlite) — The Django for AI & Machine Learning.

## Project Structure
- `{project_name}/chat_provider.py`: Chat Provider & Query Intelligence LLM configuration.
- `{project_name}/store.py`: Vector store, PostgreSQL ORM, and embedding setup.
- `{project_name}/model.py`: High-level Knowledge Model connecting retrieval and generation.
- `{project_name}/data.py`: SmartChunker document ingestion dataset.
- `experiments/benchmark.py`: Latency & retrieval quality benchmark script.
- `client.py`: Ready-to-run interactive/batch client query starter.
- `.env`: Active local environment configuration with model identifiers.
- `.env.example`: Template for sensitive credentials, database URLs, and API keys.

---

## Environment Variables & Secrets

AIMLite reads sensitive configuration (API keys, database credentials, model names, host URLs) from system environment variables or a local `.env` file in the project root.

### Configured Environment Variables

| Variable | Description | Default / Example Value |
|---|---|---|
| `OPENAI_MODEL` | OpenAI Model Identifier | `gpt-4o-mini` |
| `OLLAMA_MODEL` | Ollama Local Model Identifier | `llama3.2` |
| `GEMINI_MODEL` | Google Gemini Model Identifier | `gemini-1.5-flash` |
| `ANTHROPIC_MODEL` | Anthropic Claude Model Identifier | `claude-3-5-sonnet-20241022` |
| `EMBEDDING_MODEL` | Semantic Dense Vector Embedding Model | `all-MiniLM-L6-v2` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-proj-abc123xyz456...` |
| `ANTHROPIC_API_KEY` | Anthropic API Key | `sk-ant-api03-abc...` |
| `GEMINI_API_KEY` | Google Gemini API Key | `AIzaSyD...` |
| `OLLAMA_HOST` | Ollama daemon endpoint (local LLM) | `http://localhost:11434` |
| `DATABASE_URL` | PostgreSQL connection string for pgvector ORM | `postgresql://user:pass@localhost:5432/my_rag_db` |

#### Example `.env` file:
```ini
# --- Model Identifiers ---
OLLAMA_MODEL={model_name}
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL={embed_model}

# --- LLM API Credentials (if using cloud providers) ---
OPENAI_API_KEY=
GEMINI_API_KEY=

# --- Local Ollama Endpoint ---
OLLAMA_HOST=http://localhost:11434

# --- Database & Vector Storage (PostgreSQL ORM) ---
DATABASE_URL=postgresql://postgres:secretpassword@localhost:5432/knowledge_db

# --- Hardware Acceleration & Serving ---
AIMLITE_DEVICE=auto
AIMLITE_PORT=8000
```

> **Security Note:** Never commit `.env` containing sensitive credentials to Git. `.env` is already configured in `.gitignore`.

---

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

# 5. Run retrieval benchmark
python experiments/benchmark.py

# 6. Serve HTTP API
aimlite serve --port 8000
```
"""
    (dest_root / "README.md").write_text(readme_md, encoding="utf-8")


def _scaffold_fine_tuning(package_dir: Path, dest_root: Path, project_name: str, adapter_config: Dict[str, Any]) -> None:
    """Generates Fine-Tuning & LoRA Adapter files."""
    # 1. Adapter Configuration Starter: adapter.py
    adapter_py = f'''"""Fine-Tuning & Adapter Configuration: adapter.py

Starter code for LoRA low-rank decomposition matrices, target linear modules, and parameter diagnostics.
"""

from typing import Any, Dict, List, Optional
from aimlite import AdapterConfig


def get_adapter_config(
    r: int = {adapter_config.get("r", 8)},
    alpha: float = {adapter_config.get("alpha", 16.0)},
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> AdapterConfig:
    """Returns the LoRA hyperparameter configuration."""
    modules = target_modules or {adapter_config.get("target_modules", ["q_proj", "v_proj"])}
    return AdapterConfig(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=modules,
    )
'''
    (package_dir / "adapter.py").write_text(adapter_py, encoding="utf-8")

    # 2. Data loader: data.py
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

    # 3. Model: model.py
    model_py = f'''"""Fine-Tuning & LoRA (Adapter Paradigm): model.py"""

from typing import Any, Dict, Optional
from aimlite import AdapterModel
from {project_name}.adapter import get_adapter_config


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
        adapter_cfg = get_adapter_config(r=r, alpha=alpha)
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

    # 4. Trainer, Evaluator, Inference
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

    # 5. Sample instructions in data/
    sample_jsonl = """{"instruction": "What is AIMLite?", "response": "AIMLite is the Django for AI & Machine Learning with zero-path CLI execution."}
{"instruction": "How does LoRA reduce checkpoint size?", "response": "LoRA freezes foundation parameters and trains low-rank delta matrices, reducing weights from 14GB down to under 50MB."}
{"instruction": "How do you evaluate adapter models in AIMLite?", "response": "Run aimlite evaluate to compute parameter efficiency, rank sparsity, and loss convergence."}
"""
    (dest_root / "data" / "instructions.jsonl").write_text(sample_jsonl, encoding="utf-8")

    # 6. Experiments benchmark script
    benchmark_py = f'''"""Benchmark script evaluating parameter efficiency and latency for {project_name} LoRA Adapter."""

import time
from {project_name}.data import InstructionDataset
from {project_name}.model import LoRAInstructionModel


def run_benchmark() -> None:
    print("=" * 60)
    print("  AIMLite LoRA Adapter Parameter Efficiency & Latency Benchmark")
    print("=" * 60)

    model = LoRAInstructionModel()
    dataset = InstructionDataset()
    records = dataset.load()

    diag = model.get_trainable_parameters() if hasattr(model, "get_trainable_parameters") else {{}}
    print(f"\\n[*] Parameter Efficiency Diagnostics:")
    for k, v in diag.items():
        print(f"    • {{k}}: {{v}}")

    print(f"\\n[*] Evaluating Inference Latency across {{len(records)}} instruction(s):")
    for idx, item in enumerate(records, start=1):
        t0 = time.perf_counter()
        result = model.predict(item)
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"[{{idx}}] Latency: {{latency_ms:.3f}} ms | Output: {{result.get('response', '')}}")

    print("\\n[OK] Benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
'''
    (dest_root / "experiments" / "benchmark.py").write_text(benchmark_py, encoding="utf-8")

    # 7. Directory placeholders & Git configuration
    for d in ["models", "artifacts", "checkpoints"]:
        (dest_root / d / ".gitkeep").write_text("", encoding="utf-8")

    gitignore_content = """# Environments & Virtual Envs
.venv/
env/
venv/
ENV/

# Python cache & artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
build/
dist/
*.egg-info/

# Sensitive Environment Variables & Secrets
.env
.env.local
.env.*.local

# ML Checkpoints & Adapters
checkpoints/*.pt
checkpoints/*.safetensors
checkpoints/*.bin
"""
    (dest_root / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    # 8. Sensitive variables starter template: .env.example
    env_example = """# ==============================================================================
# AIMLite LoRA / Fine-Tuning: Environment Variables & Secrets
# ==============================================================================
# Copy this file to .env and fill in your actual credentials:
#   cp .env.example .env
# Never commit your .env file containing private keys to version control!
# ==============================================================================

# --- Model Registry & Hub Access Tokens ---
# Required for downloading gated foundation weights (e.g. meta-llama/Llama-3.2)
HF_TOKEN=hf_your_huggingface_access_token_here

# --- Experiment Tracking & Monitoring ---
# Optional Weights & Biases API key
WANDB_API_KEY=your_wandb_api_key_here

# --- Hardware Acceleration & Serving ---
AIMLITE_DEVICE=cuda
AIMLITE_PORT=8000
"""
    (dest_root / ".env.example").write_text(env_example, encoding="utf-8")

    # 9. Client starter script in root
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

    # 10. Project README
    readme_md = f"""# {project_name.replace('_', ' ').title()} (LoRA / Fine-Tuning)

Built with [AIMLite](https://github.com/Alazar42/aimlite) — The Django for AI & Machine Learning.

## Project Structure
- `{project_name}/adapter.py`: LoRA configuration, target modules, and low-rank matrices.
- `{project_name}/model.py`: Parameter-efficient AdapterModel architecture.
- `{project_name}/data.py`: InstructionDataset loader for prompt/response pairs.
- `experiments/benchmark.py`: Parameter efficiency & inference latency benchmark.
- `client.py`: Ready-to-run interactive/batch client inference test.
- `.env.example`: Template for API keys, Hugging Face tokens, and runtime settings.

---

## Environment Variables & Secrets

AIMLite reads sensitive credentials from system environment variables or a local `.env` file in the project root.

### 1. Create your `.env` file
Copy the provided `.env.example` template:
```bash
cp .env.example .env
```

### 2. Configure Credentials

| Variable | Description | Example Value |
|---|---|---|
| `HF_TOKEN` | Hugging Face Hub token for downloading gated foundation models | `hf_abc123xyz...` |
| `WANDB_API_KEY` | Weights & Biases API key for loss & parameter logging | `wandb_api_key...` |
| `AIMLITE_DEVICE` | Hardware accelerator target (`auto`, `cuda`, `mps`, `cpu`) | `cuda` |

#### Example `.env` file:
```ini
# --- Model Hub Credentials ---
HF_TOKEN=hf_your_actual_token_here

# --- Hardware Acceleration ---
AIMLITE_DEVICE=cuda
AIMLITE_PORT=8000
```

> **Security Note:** Never commit `.env` containing sensitive credentials to Git. `.env` is already configured in `.gitignore`.

---

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

# 6. Run benchmark
python experiments/benchmark.py

# 7. Serve HTTP API
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

    # Experiments benchmark
    benchmark_py = f'''"""Benchmark script evaluating latency and throughput for {project_name} Model."""

import time
from {project_name}.data import AppDataset
from {project_name}.model import AppModel


def run_benchmark() -> None:
    print("=" * 60)
    print("  AIMLite Model Benchmark & Latency Evaluation")
    print("=" * 60)

    model = AppModel()
    dataset = AppDataset()

    sample = {{"feature_1": 1.2, "feature_2": 3.4, "feature_3": 0.5}}

    t0 = time.perf_counter()
    for _ in range(100):
        _ = model.predict(sample)
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"\\n[*] 100 Forward passes executed in {{total_ms:.2f}} ms ({{total_ms/100:.3f}} ms/query)")
    print("\\n[OK] Benchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()
'''
    (dest_root / "experiments" / "benchmark.py").write_text(benchmark_py, encoding="utf-8")

    # Directory placeholders & Git configuration
    for d in ["models", "artifacts", "checkpoints"]:
        (dest_root / d / ".gitkeep").write_text("", encoding="utf-8")

    gitignore_content = """# Environments & Virtual Envs
.venv/
env/
venv/
ENV/

# Python cache & artifacts
__pycache__/
*.py[cod]
*$py.class
*.so
build/
dist/
*.egg-info/

# Sensitive Environment Variables & Secrets
.env
.env.local
.env.*.local

# ML Artifacts & Checkpoints
artifacts/*.json
artifacts/*.pkl
checkpoints/*.pt
"""
    (dest_root / ".gitignore").write_text(gitignore_content, encoding="utf-8")

    # Sensitive variables starter template: .env.example
    env_example = """# ==============================================================================
# AIMLite Custom ML: Environment Variables & Secrets
# ==============================================================================
# Copy this file to .env and fill in your actual credentials:
#   cp .env.example .env
# Never commit your .env file containing private keys to version control!
# ==============================================================================

# --- Feature Store / Database Connection ---
DATABASE_URL=postgresql://user:password@localhost:5432/ml_db

# --- Hardware Acceleration & Serving ---
AIMLITE_DEVICE=auto
AIMLITE_PORT=8000
"""
    (dest_root / ".env.example").write_text(env_example, encoding="utf-8")

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

## Project Structure
- `{project_name}/model.py`: Model architecture and forward prediction.
- `{project_name}/data.py`: Application dataset ingestion.
- `{project_name}/trainer.py`: Model training orchestration.
- `experiments/benchmark.py`: Forward throughput & latency benchmark.
- `client.py`: Ready-to-run test client.
- `.env.example`: Template for environment variables and secrets.

---

## Environment Variables & Secrets

AIMLite reads sensitive credentials from system environment variables or a local `.env` file in the project root.

### 1. Create your `.env` file
Copy the provided `.env.example` template:
```bash
cp .env.example .env
```

### 2. Configure Credentials

| Variable | Description | Example Value |
|---|---|---|
| `DATABASE_URL` | Feature store or SQL database connection string | `postgresql://user:pass@localhost:5432/ml_db` |
| `AIMLITE_DEVICE` | Hardware acceleration target (`auto`, `cuda`, `cpu`) | `auto` |
| `AIMLITE_PORT` | HTTP REST API server port | `8000` |

#### Example `.env` file:
```ini
# --- Database / Feature Store ---
DATABASE_URL=postgresql://postgres:secretpassword@localhost:5432/ml_db

# --- Serving ---
AIMLITE_DEVICE=auto
AIMLITE_PORT=8000
```

> **Security Note:** Never commit `.env` containing sensitive credentials to Git. `.env` is already configured in `.gitignore`.

---

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

# 6. Run benchmark
python experiments/benchmark.py

# 7. Serve HTTP API
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
