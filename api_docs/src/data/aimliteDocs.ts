import {
  type GuideStep,
  SCRATCH_FILES,
  SCRATCH_GUIDE_STEPS,
  RAG_FILES,
  RAG_GUIDE_STEPS,
  ADAPTER_FILES,
  ADAPTER_GUIDE_STEPS,
} from './paradigmGuides';

export interface DocParameter {
  name: string;
  type: string;
  required: boolean;
  defaultValue?: string;
  description: string;
  validation?: string;
  children?: DocParameter[];
}

export interface DocSection {
  id: string;
  category: string;
  title: string;
  subtitle: string;
  badge: {
    label: string;
    variant: 'pillar' | 'cli' | 'get' | 'post' | 'util';
  };
  signatureOrPath: string;
  breadcrumbs: string[];
  overview: string;
  djangoAnalogy: string;
  whyCode?: {
    component: string;
    reason: string;
  }[];
  parametersTitle?: string;
  parameters?: DocParameter[];
  conventions?: { title: string; description: string }[];
  snippets: {
    python?: string;
    cli?: string;
    curl?: string;
    typescript?: string;
    files?: Record<string, string>;
  };
  guideSteps?: GuideStep[];
  defaultPayload?: string;
  defaultResponse?: any;
}

export interface NavItem {
  id: string;
  label: string;
  badge?: string;
  badgeVariant?: 'pillar' | 'cli' | 'get' | 'post' | 'util';
}

export interface NavCategory {
  name: string;
  items: NavItem[];
}

export const NAVIGATION_CATEGORIES: NavCategory[] = [
  {
    name: 'The 3 AI Paradigms',
    items: [
      { id: 'paradigm-scratch', label: '1. Scratch (Customer Churn)', badge: 'TRAIN', badgeVariant: 'pillar' },
      { id: 'paradigm-rag', label: '2. RAG (Knowledge QA)', badge: 'RAG', badgeVariant: 'pillar' },
      { id: 'paradigm-adapters', label: '3. Adapters (LoRA & PEFT)', badge: 'PEFT', badgeVariant: 'pillar' },
    ],
  },
  {
    name: 'The 3 Pillars of AIMLite',
    items: [
      { id: 'pillar-data', label: '1. Data Pillar (Dataset)', badge: 'PILLAR', badgeVariant: 'pillar' },
      { id: 'pillar-model', label: '2. Model Pillar (Model)', badge: 'PILLAR', badgeVariant: 'pillar' },
      { id: 'pillar-lifecycle', label: '3. Lifecycle Pillar (Trainer)', badge: 'PILLAR', badgeVariant: 'pillar' },
    ],
  },
  {
    name: 'AIMLite RAG Deep Dive',
    items: [
      { id: 'rag-hooks', label: 'KnowledgeModel & Lifecycle Hooks', badge: 'HOOKS', badgeVariant: 'pillar' },
    ],
  },
  {
    name: 'Supporting Foundations',
    items: [
      { id: 'foundations-config', label: 'BaseConfig & Hardware', badge: 'UTIL', badgeVariant: 'util' },
      { id: 'foundations-registry', label: 'Component Registry', badge: 'UTIL', badgeVariant: 'util' },
    ],
  },
  {
    name: 'Zero-Path CLI Commands',
    items: [
      { id: 'cli-init', label: 'aimlite init', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-install', label: 'aimlite install', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-train', label: 'aimlite train', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-evaluate', label: 'aimlite evaluate', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-serve', label: 'aimlite serve', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-benchmark', label: 'aimlite benchmark', badge: 'NEW', badgeVariant: 'cli' },
      { id: 'cli-data', label: 'aimlite data validate', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-doctor', label: 'aimlite doctor', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-update', label: 'aimlite update', badge: 'NEW', badgeVariant: 'cli' },
    ],
  },
  {
    name: 'HTTP Inference Server',
    items: [
      { id: 'endpoint-predict', label: 'Execute Inference (/predict)', badge: 'POST', badgeVariant: 'post' },
      { id: 'endpoint-models', label: 'Models Inventory (/models)', badge: 'GET', badgeVariant: 'get' },
      { id: 'endpoint-chat', label: 'Adaptive Web App (/app)', badge: 'WEB', badgeVariant: 'get' },
      { id: 'endpoint-search', label: 'Execute Search (/search)', badge: 'POST', badgeVariant: 'post' },
      { id: 'endpoint-health', label: 'Server Health (/health)', badge: 'GET', badgeVariant: 'get' },
      { id: 'endpoint-openapi', label: 'OpenAPI 3.0 Schema', badge: 'GET', badgeVariant: 'get' },
      { id: 'endpoint-docs', label: 'Swagger UI Docs', badge: 'GET', badgeVariant: 'get' },
      { id: 'guide-serving', label: 'Serving & Customization', badge: 'GUIDE', badgeVariant: 'cli' },
    ],
  },
  {
    name: 'Releases & Changelog',
    items: [
      { id: 'changelog', label: 'Version Changelog', badge: 'v2.1.2', badgeVariant: 'util' },
    ],
  },
];

export const DOC_SECTIONS: Record<string, DocSection> = {
  'pillar-data': {
    id: 'pillar-data',
    category: 'The 3 Pillars of AIMLite',
    title: 'Pillar 1: Data (Dataset)',
    subtitle: 'Standardized dataset ingestion, schema validation, and partition contracts.',
    badge: { label: 'PILLAR 1', variant: 'pillar' },
    signatureOrPath: 'from aimlite import Dataset',
    breadcrumbs: ['3 Pillars', 'Data Pillar'],
    overview:
      'The Data pillar standardizes how raw data enters the machine learning lifecycle. Just like Django models.Model, defining an empty class does nothing at import time. When data or source is configured, Dataset provides automatic validation and partition contracts.',
    djangoAnalogy:
      'In Django, you subclass models.Model in models.py. An empty class does not load or touch the database. In AIMLite, you subclass Dataset in data.py. Zero side effects occur on import; when loaded, it validates and partitions your data.',
    conventions: [
      {
        title: 'Zero-Code Data Discovery',
        description: 'Drop files into data/. Writing `class AIDataset(Dataset): pass` automatically discovers and validates data when loaded.',
      },
      {
        title: 'Pre-Split Convention',
        description: 'If you have data/train.csv and data/test.csv, Dataset automatically maps them to training and testing sets without slicing.',
      },
      {
        title: 'Multi-File & Concatenation',
        description: 'Target a specific file with `filename = "customers.csv"`, or merge all files using `combine_all = True`.',
      },
      {
        title: 'Strict Execution Rule',
        description: 'If the class is empty and data/ has no files, `aimlite train` strictly halts with exit code 1.',
      },
    ],
    parametersTitle: 'Dataset Class Attributes & Contracts',
    parameters: [
      {
        name: 'load(source=None)',
        type: 'method',
        required: true,
        description: 'Ingests data into memory. Uses standard library csv (or pandas if installed).',
      },
      {
        name: 'split(train=0.8, validation=0.1, test=0.1, seed=42, shuffle=True)',
        type: 'method',
        required: true,
        description: 'Partitions rows into (train_data, val_data, test_data) tuples with deterministic seeded shuffling and ratio sum validation.',
        defaultValue: '0.8, 0.1, 0.1, seed=42, shuffle=True',
      },
      {
        name: 'validate()',
        type: 'method',
        required: true,
        description: 'Verifies that files exist and contain non-zero valid rows.',
      },
      {
        name: 'source',
        type: 'string | Path',
        required: false,
        description: 'Optional file path or URI anchor to the dataset.',
        defaultValue: 'None (auto-resolves from data/)',
      },
      {
        name: 'filename',
        type: 'string',
        required: false,
        description: 'Optional class attribute to target a specific file within data/.',
        defaultValue: 'None',
      },
      {
        name: 'combine_all',
        type: 'boolean',
        required: false,
        description: 'When true, merges/concatenates all CSV files found in data/ into a single dataset.',
        defaultValue: 'false',
      },
    ],
    snippets: {
      files: {
        'data.py': `from aimlite import Dataset

# Option 1: Standard dataset with file source
class AIDataset(Dataset):
    source = "data/train.csv"

# Option 2: Pre-split convention in data/ (train.csv and test.csv)
class CustomerDataset(Dataset):
    pass  # Auto-discovers train.csv & test.csv when loaded

# Option 3: Concatenate multiple shards
class ShardedDataset(Dataset):
    combine_all = True`,
      },
      cli: `aimlite data validate`,
    },
    defaultPayload: '{\n  "source": "data/housing_prices.csv",\n  "test_split": 0.2\n}',
    defaultResponse: {
      status: 'success',
      dataset: 'AIDataset',
      records_loaded: 2500,
      columns: ['area', 'bedrooms', 'bathrooms', 'price'],
      partitions: {
        train: 2000,
        validation: 250,
        test: 250,
      },
    },
  },

  'pillar-model': {
    id: 'pillar-model',
    category: 'The 3 Pillars of AIMLite',
    title: 'Pillar 2: Model (Model)',
    subtitle: 'Framework-agnostic model wrapper with built-in pickle persistence.',
    badge: { label: 'PILLAR 2', variant: 'pillar' },
    signatureOrPath: 'from aimlite import Model',
    breadcrumbs: ['3 Pillars', 'Model Pillar'],
    overview:
      'The Model pillar standardizes execution and serialization across any machine learning framework (PyTorch, TensorFlow, Scikit-learn, XGBoost, or pure Python). It includes built-in pickle persistence (save/load) that directs model weights directly to the models/ folder.',
    djangoAnalogy:
      'In Django, Model instances have built-in .save() and .delete() database persistence. In AIMLite, Model instances have built-in .save() and .load() weight serialization into models/model.pkl.',
    conventions: [
      {
        title: 'Built-in Persistence (save & load)',
        description: 'Calling model.save(destination) writes weights to models/model.pkl automatically unless overridden.',
      },
      {
        title: 'Framework Agnostic (BYOF)',
        description: 'Implement predict(inputs, **kwargs) with PyTorch tensors, Scikit-learn arrays, or pure Python lists.',
      },
      {
        title: 'Config Binding',
        description: 'Hyperparameters in aimlite.json are automatically bound to self.config during zero-path discovery.',
      },
    ],
    parametersTitle: 'Model Class Methods & Contracts',
    parameters: [
      {
        name: 'dataset',
        type: 'Type[Dataset] | string | None',
        required: false,
        description: 'Explicitly binds a Dataset class to this model (like Django ModelForm). If None, auto-discovers data.py by convention.',
        defaultValue: 'None (auto-discovers data.py)',
      },
      {
        name: 'predict(inputs, **kwargs)',
        type: 'abstractmethod',
        required: true,
        description: 'Standard forward pass contract. Accepts raw batches or feature lists and returns predictions.',
      },
      {
        name: 'save(destination)',
        type: 'method',
        required: false,
        description: 'Serializes model weights into models/model.pkl using pickle (can be overridden for torch.save/onnx).',
        defaultValue: 'models/model.pkl',
      },
      {
        name: 'load(source)',
        type: 'method',
        required: false,
        description: 'Restores model weights from models/model.pkl automatically during evaluate and serve cycles.',
      },
    ],
    snippets: {
      files: {
        'model.py': `from aimlite import Model, Dataset

# Option 1: Automatic convention (auto-discovers default Dataset in data.py)
class AIModel(Model):
    def predict(self, inputs, **kwargs):
        return [x * 2.5 for x in inputs]

# Option 2: Explicitly declare which Dataset belongs to this Model (Django-style)
class ChurnModel(Model):
    dataset = CustomerDataset  # Direct model-to-dataset binding

    def predict(self, inputs, **kwargs):
        return [1 if sum(x) > 0.5 else 0 for x in inputs]`,
      },
      cli: `aimlite train AIModel`,
    },
    defaultPayload: '{\n  "inputs": [1.0, 2.0, 3.0, 4.0]\n}',
    defaultResponse: {
      status: 'success',
      model: 'AIModel',
      checkpoint: 'models/model.pkl',
      predictions: [2.5, 5.0, 7.5, 10.0],
    },
  },

  'pillar-lifecycle': {
    id: 'pillar-lifecycle',
    category: 'The 3 Pillars of AIMLite',
    title: 'Pillar 3: Lifecycle (Trainer, Evaluator, Inference)',
    subtitle: 'Execution cycles with strict separation of weights and metadata.',
    badge: { label: 'PILLAR 3', variant: 'pillar' },
    signatureOrPath: 'from aimlite import BaseTrainer, BaseEvaluator, BaseInference',
    breadcrumbs: ['3 Pillars', 'Lifecycle Pillar'],
    overview:
      'The Lifecycle pillar governs training, evaluation, and live inference. It guarantees strict architectural hygiene: serialized model weights live strictly in models/, and experiment metadata snapshots live strictly in experiments/.',
    djangoAnalogy:
      'In Django, manage.py test and views handle the request-response lifecycle. In AIMLite, BaseTrainer, BaseEvaluator, and BaseInference handle the machine learning lifecycle.',
    conventions: [
      {
        title: 'Strict Directory Separation',
        description: 'Weights go to models/model.pkl. Metadata snapshots go to experiments/experiment_snapshot.json.',
      },
      {
        title: 'Zero-Path Execution',
        description: 'mlkit train, mlkit evaluate, and mlkit serve automatically locate your lifecycle classes without path arguments.',
      },
    ],
    parametersTitle: 'Lifecycle Contracts',
    parameters: [
      {
        name: 'BaseTrainer.fit(model, dataset)',
        type: 'abstractmethod',
        required: true,
        description: 'Executes the training optimization loop and returns metrics dictionary (loss, step, accuracy).',
      },
      {
        name: 'BaseTrainer.checkpoint(model, destination, experiments_dir)',
        type: 'method',
        required: false,
        description: 'Serializes weights to models/ and writes JSON snapshot to experiments/.',
      },
      {
        name: 'BaseEvaluator.evaluate(model, dataset)',
        type: 'abstractmethod',
        required: true,
        description: 'Computes validation or test set metrics without altering model weights.',
      },
      {
        name: 'BaseInference.run(model, raw_input)',
        type: 'abstractmethod',
        required: true,
        description: 'Transforms incoming HTTP JSON payload into model input and returns response object.',
      },
    ],
    snippets: {
      files: {
        'trainer.py': `from aimlite import BaseTrainer

class AITrainer(BaseTrainer):
    def fit(self, model, dataset, **kwargs):
        # Optimization loop here
        return {"status": "success", "loss": 0.035, "step": 100}`,
        'evaluator.py': `from aimlite import BaseEvaluator

class AIEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        return {"accuracy": 0.965}`,
        'inference.py': `from aimlite import BaseInference

class AIInference(BaseInference):
    def run(self, model, raw_input, **kwargs):
        features = raw_input.get("features", [])
        return model.predict(features)`,
      },
      cli: `aimlite train AIModel && aimlite evaluate AIModel && aimlite serve AIModel`,
    },
    defaultPayload: '{\n  "features": [0.45, 1.28, 3.14]\n}',
    defaultResponse: {
      status: 'success',
      weights_path: 'models/model.pkl',
      snapshot_path: 'experiments/experiment_snapshot.json',
      training_metrics: {
        loss: 0.035,
        step: 100,
        device: 'cuda',
      },
    },
  },

  'foundations-config': {
    id: 'foundations-config',
    category: 'Supporting Foundations',
    title: 'BaseConfig & Hardware Auto-Probe',
    subtitle: 'Hardware accelerator discovery (CUDA / MPS / CPU) and manifest resolution.',
    badge: { label: 'FOUNDATION', variant: 'util' },
    signatureOrPath: 'from aimlite import BaseConfig',
    breadcrumbs: ['Foundations', 'BaseConfig'],
    overview:
      'BaseConfig parses aimlite.json and provides automatic hardware device resolution. It probes for NVIDIA CUDA GPUs, Apple Silicon MPS (Metal Performance Shaders), or gracefully falls back to CPU.',
    djangoAnalogy:
      'Equivalent to Django settings.py. Automatically resolves paths and hardware accelerators.',
    parametersTitle: 'BaseConfig Properties & Methods',
    parameters: [
      {
        name: 'resolve_device()',
        type: 'method',
        required: true,
        description: 'Returns "cuda", "mps", or "cpu" based on available local hardware.',
      },
      {
        name: 'data_dir, models_dir, experiments_dir',
        type: 'property (Path)',
        required: false,
        description: 'Resolved absolute paths to project directories defined in aimlite.json.',
      },
      {
        name: 'to_dict()',
        type: 'method',
        required: false,
        description: 'Exports active configuration dictionary for experiment tracking snapshots.',
      },
    ],
    snippets: {
      python: `from aimlite import BaseConfig

config = BaseConfig()
device = config.resolve_device()  # 'cuda', 'mps', or 'cpu'
data_path = config.data_dir       # /project/data`,
    },
    defaultPayload: '{\n  "hardware": { "device": "auto" }\n}',
    defaultResponse: {
      device: 'cuda',
      platform: 'Linux x86_64',
      gpu_name: 'NVIDIA RTX 4090',
      status: 'Accelerated',
    },
  },

  'foundations-registry': {
    id: 'foundations-registry',
    category: 'Supporting Foundations',
    title: 'Component Registry',
    subtitle: 'Automatic under-the-hood component discovery and zero-decorator registration.',
    badge: { label: 'FOUNDATION', variant: 'util' },
    signatureOrPath: 'from aimlite import get, get_all, register',
    breadcrumbs: ['Foundations', 'Registry'],
    overview:
      'The Registry automatically tracks every Model, Dataset, BaseTrainer, BaseEvaluator, BaseInference, and BaseConfig subclass under the hood via class extension hooks (__init_subclass__). Developers never need to write explicit @register decorators.',
    djangoAnalogy:
      "Just like Django's AppRegistry automatically discovers and registers every models.Model subclass on definition without decorators, AIMLite auto-registers your ML classes under the hood.",
    snippets: {
      python: `from aimlite import Model, Dataset, get, get_all

# 1. Automatic registration under the hood on class extension!
# No @register decorator needed!
class CustomerDataset(Dataset):
    filename = "customers.csv"

class ChurnModel(Model):
    dataset = CustomerDataset

    def predict(self, inputs, **kwargs):
        return [0]

# 2. Dynamic lookup & reflection
model_cls = get("model", "ChurnModel")       # Exact or case-insensitive
all_datasets = get_all("dataset")            # {'CustomerDataset': <class>}
`,
    },
    defaultPayload: '{\n  "category": "model",\n  "name": "ChurnModel"\n}',
    defaultResponse: {
      auto_registered: true,
      category: 'model',
      name: 'ChurnModel',
      mechanism: '__init_subclass__',
      decorator_required: false,
    },
  },

  'cli-init': {
    id: 'cli-init',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite init',
    subtitle: 'Vite-inspired interactive wizard scaffolding modular AI projects in <50ms.',
    badge: { label: 'CLI v2.1.2', variant: 'cli' },
    signatureOrPath: 'aimlite init [project_name | .] [--type <scratch|rag|adapter>] [--clean] [--install] [-y]',
    breadcrumbs: ['CLI', 'init'],
    overview:
      'Scaffolds a new project directory with Vite-inspired terminal prompts powered by questionary and rich, rendered with a pyfiglet ASCII banner. Generates standard layout: data/ (strictly clean), artifacts/, experiments/, and modular starter files. Supports --clean to generate a pristine project with empty code files (0 bytes) and zero sample datasets for the selected paradigm (scratch, rag, or adapter). Features an interactive package search prompt (+) to select additional dependencies, instantaneous (<50ms) project generation (heavy downloads are deferred), and generates aimlite.json as the single source of truth.',
    djangoAnalogy:
      'Direct equivalent of create-vite or django-admin startproject <name>. Scaffolds 100% developer-editable code with zero hidden boilerplate.',
    parametersTitle: 'CLI Flags & Options',
    parameters: [
      {
        name: 'project_name | .',
        type: 'string (positional)',
        required: false,
        defaultValue: 'interactive prompt',
        description: 'Directory name for the new project, or . to scaffold into the current folder.',
      },
      {
        name: '--type, -t',
        type: 'scratch | rag | adapter',
        required: false,
        description: 'Selects the canonical AI paradigm template: scratch, rag, or adapter.',
      },
      {
        name: '--clean',
        type: 'boolean flag',
        required: false,
        defaultValue: 'false',
        description: 'Generates clean project with empty code files (0 bytes) and zero sample datasets for the chosen paradigm.',
      },
      {
        name: '--chat-provider',
        type: 'openai | gemini | anthropic | ollama | local | mock',
        required: false,
        defaultValue: 'openai',
        description: 'RAG paradigm: Configures LLM chat provider for synthesis.',
      },
      {
        name: '--vector-db',
        type: 'memory | postgres',
        required: false,
        defaultValue: 'memory',
        description: 'RAG paradigm: Configures vector database storage engine.',
      },
      {
        name: '--install',
        type: 'boolean flag',
        required: false,
        defaultValue: 'false',
        description: 'Automatically creates .venv and installs dependencies immediately during scaffolding.',
      },
      {
        name: '-y, --non-interactive',
        type: 'boolean flag',
        required: false,
        description: 'Skips interactive prompts and applies standard defaults (ideal for CI/CD).',
      },
    ],
    snippets: {
      cli: `# Clean project with empty code files for adapter paradigm:
aimlite init my_adapter --clean --type adapter

# Interactive Vite-style wizard:
aimlite init my_rag

# Non-interactive CLI flag setup for RAG:
aimlite init my_rag --type rag --chat-provider gemini --vector-db memory

# Scaffold in current directory with immediate dependency install:
aimlite init . --type adapter --install`,
    },
    defaultPayload: '{\n  "command": "aimlite init my_rag",\n  "type": "rag",\n  "chat_provider": "openai"\n}',
    defaultResponse: {
      status: 'success',
      project: 'my_rag',
      paradigm: 'rag',
      files_generated: [
        'my_rag/chat_provider.py',
        'my_rag/store.py',
        'my_rag/data.py',
        'my_rag/model.py',
        'my_rag/trainer.py',
        'my_rag/evaluator.py',
        'my_rag/inference.py',
        'experiments/benchmark.py',
        'client.py',
        'aimlite.json',
      ],
      elapsed_time: '42ms',
    },
  },

  'cli-install': {
    id: 'cli-install',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite install',
    subtitle: 'Installs project dependencies into managed .venv directly from aimlite.json or CLI arguments.',
    badge: { label: 'CLI v2.1.2', variant: 'cli' },
    signatureOrPath: 'aimlite install [package_name ...] [-r requirements.txt]',
    breadcrumbs: ['CLI', 'install'],
    overview:
      'Manages your project virtual environment (.venv) using uv (preferred) or pip. When run without arguments (aimlite install), it reads dependencies directly from aimlite.json as the single source of truth — no -r requirements.txt needed. Automatically creates the .venv if it does not exist.',
    djangoAnalogy:
      'Like npm install or poetry install — automatically synchronizes your virtual environment with your project manifest (aimlite.json).',
    parametersTitle: 'Arguments & Options',
    parameters: [
      {
        name: 'package_name ...',
        type: 'string[] (optional)',
        required: false,
        description: 'One or more packages to install and automatically add to aimlite.json dependencies.',
      },
      {
        name: '-r, --requirement',
        type: 'Path (optional)',
        required: false,
        description: 'Path to a legacy requirements.txt file to install packages from.',
      },
      {
        name: '--upgrade',
        type: 'boolean flag',
        required: false,
        defaultValue: 'false',
        description: 'Upgrades installed packages to their latest versions.',
      },
    ],
    snippets: {
      cli: `# Install all project dependencies defined in aimlite.json (default):
aimlite install

# Install specific packages into the active project .venv:
aimlite install sentence-transformers psycopg2-binary

# Install with package upgrade:
aimlite install --upgrade`,
    },
    defaultPayload: '{\n  "command": "aimlite install",\n  "source": "aimlite.json"\n}',
    defaultResponse: {
      status: 'success',
      venv_path: '.venv',
      manifest: 'aimlite.json',
      installed: ['sentence-transformers', 'psycopg2-binary'],
      backend: 'uv pip install',
    },
  },

  'cli-train': {
    id: 'cli-train',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite train',
    subtitle: 'Runs model training cycle through zero-path discovery or targeted model class.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'aimlite train [ModelName]',
    breadcrumbs: ['CLI', 'train'],
    overview:
      'Discovers and executes your training cycle. Supports targeted training by passing the model class name (e.g. aimlite train ChurnClassifier). Strictly validates that classes are implemented and that data files exist in data/. If multiple models exist in model.py, prompts you to select one.',
    djangoAnalogy:
      'Equivalent to running database migrations or test runner. Halts cleanly if models or data are missing.',
    snippets: {
      cli: `# Train single or default model
aimlite train

# Targeted training by model class name
aimlite train ChurnClassifier`,
    },
    defaultPayload: '{\n  "target_model": "ChurnClassifier"\n}',
    defaultResponse: {
      status: 'success',
      model: 'ChurnClassifier',
      elapsed_time: '2.14s',
      model_weights: 'artifacts/churn_classifier.pkl',
      experiment_snapshot: 'experiments/experiment_snapshot.json',
      metrics: {
        train_accuracy: 0.984,
        val_accuracy: 0.962,
      },
    },
  },

  'cli-serve': {
    id: 'cli-serve',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite serve',
    subtitle: 'Zero-path multi-model inference server with type-adaptive Web Playgrounds, Swagger UI, and headless --api mode.',
    badge: { label: 'CLI v2.1.2', variant: 'cli' },
    signatureOrPath: 'aimlite serve [ModelName] [--api] [--checkpoint <path>] [--port 8000] [--host 127.0.0.1] [--frontend <dir>]',
    breadcrumbs: ['CLI', 'serve'],
    overview:
      'Starts a high-performance HTTP inference server. Automatically discovers and registers all trained model classes in model.py (or serves a specific class if passed as an argument). The served web application dynamically adapts its UI based on the active model type: Classical ML models render an interactive Feature Form with sample presets; RAG models render a grounded conversational assistant with source citations; and LoRA adapters render a prompt generation playground. Developers can run in pure headless mode with --api, override templates by creating a templates/ directory in their project, or serve custom SPA builds via --frontend.',
    djangoAnalogy:
      'Direct equivalent of python manage.py runserver, extended with multi-model dispatching, automatic type-adaptive UI playgrounds, and headless REST mode.',
    parametersTitle: 'Server Arguments & Flags',
    parameters: [
      {
        name: 'ModelName',
        type: 'string (optional)',
        required: false,
        defaultValue: 'auto-discover all trained models',
        description: 'Optional specific Model class name to serve (e.g. ChurnClassifier). If omitted, all trained models in model.py are served with dynamic switching.',
      },
      {
        name: '--api',
        type: 'boolean flag',
        required: false,
        defaultValue: 'false',
        description: 'Headless API-only mode. Disables all HTML web interfaces and serves pure JSON REST endpoints (/models, /predict, /health, /openapi.json).',
      },
      {
        name: '--checkpoint',
        type: 'Path (optional)',
        required: false,
        description: 'Explicit path to a model weights checkpoint or index artifact to load (defaults to automatic discovery in models/ and artifacts/).',
      },
      {
        name: '--port',
        type: 'integer',
        required: false,
        defaultValue: '8000',
        description: 'TCP port number to bind the server.',
      },
      {
        name: '--host',
        type: 'string',
        required: false,
        defaultValue: '127.0.0.1',
        description: 'Network interface IP address to listen on (e.g. 0.0.0.0 for external/container access).',
      },
      {
        name: '--frontend',
        type: 'Path (optional)',
        required: false,
        description: 'Path to a custom frontend build directory (e.g. frontend/dist) to serve at the root URL.',
      },
    ],
    conventions: [
      {
        title: 'Model (Classical / Deep ML)',
        description: 'Auto-generates interactive Feature Form inputs with presets (Sample 1/2, Zero Vector, Randomize), confidence scores, and raw JSON editor.',
      },
      {
        title: 'KnowledgeModel (RAG)',
        description: 'Auto-generates conversational chat assistant with grounded context citations, similarity scores, and Top-K context slider.',
      },
      {
        title: 'AdapterModel (LoRA)',
        description: 'Auto-generates prompt completion playground with temperature, max tokens, and real-time generation output.',
      },
      {
        title: 'Dynamic Model Switcher',
        description: 'Navbar dropdown lets you switch between all trained classes in model.py without page reload or restarting the server.',
      },
      {
        title: 'Any Frontend Framework (--frontend)',
        description: 'Mounts React, Next.js, Vue, or SvelteKit build folder at / with SPA client-side routing fallback and zero CORS issues.',
      },
      {
        title: 'Template Overrides',
        description: 'Place templates/app.html or templates/index.html in your project root to completely replace the built-in interface.',
      },
    ],
    snippets: {
      cli: `# 1. Auto-discover & serve all trained models with type-adaptive UI:
aimlite serve

# 2. Serve a specific model class exclusively:
aimlite serve ChurnClassifier

# 3. Headless REST API mode for microservice deployment (no HTML UI):
aimlite serve --api --host 0.0.0.0 --port 8000

# 4. Custom developer templates or frontend build:
aimlite serve --frontend ./frontend/dist`,
    },
    defaultPayload: '{\n  "port": 8000,\n  "target_model": "All Trained Models",\n  "api_only": false\n}',
    defaultResponse: {
      status: 'server_active',
      endpoints: {
        'GET /': 'Interactive Adaptive Web Playground (browser) or API Directory (JSON)',
        'GET /models': 'Registered Models Inventory List (JSON)',
        'POST /predict': 'Default Model Forward Inference',
        'POST /models/{name}/predict': 'Targeted Model Forward Inference',
        'GET /health': 'Server & Active Model Health Probe',
        'GET /docs': 'Interactive Swagger UI Documentation',
        'GET /openapi.json': 'OpenAPI 3.0 Schema',
      },
    },
  },

  'cli-benchmark': {
    id: 'cli-benchmark',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite benchmark',
    subtitle: 'Runs experiment and benchmark scripts with the project root automatically injected into sys.path.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'aimlite benchmark [script.py] [args...]',
    breadcrumbs: ['CLI', 'benchmark'],
    overview:
      'Eliminates the common ModuleNotFoundError: No module named <project> error when running benchmark or experiment scripts directly from subdirectories. Discovers the project root via aimlite.json, extends PYTHONPATH, resolves the project .venv Python interpreter, and executes the target script. If no script path is provided, automatically discovers the first script in experiments/. Extra arguments are forwarded to the script unchanged.',
    djangoAnalogy:
      'Like python manage.py test or running custom management commands where Django automatically configures sys.path and settings without manual environment setup.',
    parametersTitle: 'Command Arguments & Options',
    parameters: [
      {
        name: 'script.py',
        type: 'Path (optional)',
        required: false,
        defaultValue: 'auto-discover in experiments/*.py',
        description: 'Path to benchmark or experiment script (e.g. experiments/benchmark.py). Auto-discovers first script in experiments/ if omitted.',
      },
      {
        name: 'args...',
        type: 'string[] (optional)',
        required: false,
        description: 'Extra arguments forwarded directly to the benchmark script.',
      },
    ],
    snippets: {
      cli: `# Run default auto-discovered benchmark in experiments/:
aimlite benchmark

# Run specific experiment script:
aimlite benchmark experiments/benchmark.py

# Forward arguments to the benchmark script:
aimlite benchmark experiments/benchmark.py --iterations 100 --batch-size 16`,
    },
    defaultPayload: '{\n  "command": "aimlite benchmark",\n  "script": "experiments/benchmark.py"\n}',
    defaultResponse: {
      status: 'success',
      root: '/workspace/support_rag',
      script: 'experiments/benchmark.py',
      python: '.venv/bin/python',
      output: [
        'Running 3 benchmark queries:',
        "  Query: 'What are the 3 pillars...' -> 24.2ms (sources: 3)",
        'Benchmark Results: Average Latency = 26.14ms',
        'Benchmark completed successfully.',
      ],
    },
  },

  'cli-data': {
    id: 'cli-data',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite data validate',
    subtitle: 'Verifies dataset schema, integrity, and partition readiness.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run aimlite data validate',
    breadcrumbs: ['CLI', 'data validate'],
    overview:
      'Executes Dataset.load() followed by Dataset.validate() and partition analysis. Outputs a clean schema verification report with zero emojis.',
    djangoAnalogy:
      'Equivalent to python manage.py check.',
    snippets: {
      cli: 'uv run aimlite data validate',
    },
    defaultPayload: '{}',
    defaultResponse: {
      validation: 'PASSED',
      records: 2500,
      columns: 4,
    },
  },

  'cli-evaluate': {
    id: 'cli-evaluate',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite evaluate',
    subtitle: 'Assesses model performance against held-out validation or test partitions.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run aimlite evaluate',
    breadcrumbs: ['CLI', 'evaluate'],
    overview:
      'Automatically discovers the latest checkpoint in models/ and executes BaseEvaluator.evaluate(model, dataset).',
    djangoAnalogy:
      'Equivalent to running python manage.py test.',
    snippets: {
      cli: 'uv run aimlite evaluate',
    },
    defaultPayload: '{}',
    defaultResponse: {
      accuracy: 0.965,
      f1_score: 0.958,
      val_loss: 0.051,
    },
  },

  'cli-doctor': {
    id: 'cli-doctor',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite doctor',
    subtitle: 'Inspects runtime environment health, hardware accelerators, and directory permissions.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run aimlite doctor',
    breadcrumbs: ['CLI', 'doctor'],
    overview:
      'Diagnoses Python runtime version (verifying Python >= 3.10 and recommending 3.10–3.12 for ML ecosystem compatibility), hardware accelerators (CUDA/MPS/CPU), manifest location, and directory read/write permissions.',
    djangoAnalogy:
      'Comprehensive environment diagnostic check.',
    snippets: {
      cli: 'uv run aimlite doctor',
    },
    defaultPayload: '{}',
    defaultResponse: {
      python_version: '3.12.8 (x86_64 linux)',
      recommended_python: '3.10–3.12',
      device: 'CPU',
      directories_ok: true,
    },
  },

  'cli-update': {
    id: 'cli-update',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite update',
    subtitle: 'Cross-platform self-updater synchronizing global, user-level, or virtualenv AIMLite CLI & package directly from PyPI.',
    badge: { label: 'CLI v2.1.2', variant: 'cli' },
    signatureOrPath: 'aimlite update [--user] [--global] [--force] [--version <version>]',
    breadcrumbs: ['CLI', 'update'],
    overview:
      'Checks the installed AIMLite version against the latest PyPI release and performs safe, non-interactive installation and upgrades. Intelligently adapts to virtual environments (.venv), Windows User site (%APPDATA%), Linux/macOS user directories (~/.local), and system environments, handling PEP 668 externally-managed barriers automatically.',
    djangoAnalogy:
      'Self-contained package and CLI upgrade command, ensuring your local and global environments stay up to date without manual pip flag juggling.',
    parametersTitle: 'CLI Flags & Options',
    parameters: [
      {
        name: '--user',
        type: 'boolean flag (optional)',
        required: false,
        defaultValue: 'auto',
        description: 'Force installation into Python user directory (~/.local or %APPDATA%), avoiding root/administrator requirements.',
      },
      {
        name: '--global',
        type: 'boolean flag (optional)',
        required: false,
        defaultValue: 'false',
        description: 'Force system-wide installation without --user.',
      },
      {
        name: '--force',
        type: 'boolean flag (optional)',
        required: false,
        defaultValue: 'false',
        description: 'Force reinstallation even if AIMLite is already at the latest version.',
      },
      {
        name: '--version, --target-version',
        type: 'string (optional)',
        required: false,
        description: 'Install a specific release version (e.g. 2.1.2).',
      },
    ],
    snippets: {
      cli: `# Check and update to latest release:
aimlite update

# Force reinstallation:
aimlite update --force

# Install specific version:
aimlite update --version 2.1.2`,
    },
    defaultPayload: '{\n  "command": "aimlite update"\n}',
    defaultResponse: {
      status: 'up_to_date',
      installed_version: '2.1.2',
      latest_version: '2.1.2',
      message: 'AIMLite is already up to date (v2.1.2).',
    },
  },

  'endpoint-predict': {
    id: 'endpoint-predict',
    category: 'HTTP Inference Server',
    title: 'Execute Inference (POST /predict)',
    subtitle: 'Submit input features to execute live model predictions.',
    badge: { label: 'POST', variant: 'post' },
    signatureOrPath: 'POST http://127.0.0.1:8000/predict',
    breadcrumbs: ['HTTP Server', 'POST /predict'],
    overview:
      'Executes zero-path inference against the trained model loaded from models/model.pkl. Input JSON is routed through BaseInference.run(model, raw_input) to model.predict().',
    djangoAnalogy:
      'The primary inference view in AIMLite.',
    parametersTitle: 'Request Payload Parameters',
    parameters: [
      {
        name: 'features',
        type: 'array or object',
        required: true,
        description: 'Input feature tensor, dictionary, or array to feed into the model forward pass.',
      },
    ],
    snippets: {
      curl: `curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [1.5, 2.7, 3.2, 4.0]}'`,
      python: `import requests

resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"features": [1.5, 2.7, 3.2, 4.0]}
)
print(resp.json())`,
      typescript: `const resp = await fetch("http://127.0.0.1:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ features: [1.5, 2.7, 3.2, 4.0] }),
});
const data = await resp.json();
console.log(data);`,
    },
    defaultPayload: '{\n  "features": [1.5, 2.7, 3.2, 4.0]\n}',
    defaultResponse: {
      status: 'success',
      model: 'my_ai',
      latency_ms: 8.4,
      result: {
        prediction: [3.22, 5.8, 6.88, 8.6],
      },
    },
  },

  'endpoint-search': {
    id: 'endpoint-search',
    category: 'HTTP Inference Server',
    title: 'Execute Semantic Search (POST /search)',
    subtitle: 'Direct vector similarity retrieval returning ranked document passages without LLM synthesis.',
    badge: { label: 'POST v1.0.6', variant: 'post' },
    signatureOrPath: 'POST http://127.0.0.1:8000/search',
    breadcrumbs: ['HTTP Server', 'POST /search'],
    overview:
      'Direct semantic vector search endpoint added in v1.0.6. Queries the underlying vector store (PostgreSQL pgvector or MemoryVectorStore) and returns ranked passage snippets, scores, and metadata without triggering generative LLM synthesis. Ideal for custom UI search inputs, citation lookups, and multi-agent retrieval pipelines.',
    djangoAnalogy:
      'Like querying an API endpoint that performs a database filter or full-text query and returns raw serialized model records without template rendering.',
    parametersTitle: 'JSON Request Body Schema',
    parameters: [
      {
        name: 'query',
        type: 'string',
        required: true,
        description: 'Natural language search query string to embed and match against stored knowledge chunks.',
      },
      {
        name: 'top_k',
        type: 'integer',
        required: false,
        defaultValue: '5',
        description: 'Maximum number of most relevant document passages to return.',
      },
      {
        name: 'filters',
        type: 'object',
        required: false,
        description: 'Metadata key-value filters to narrow search scope (e.g. {"category": "billing"}).',
      },
    ],
    snippets: {
      curl: `curl -X POST http://127.0.0.1:8000/search \\
  -H "Content-Type: application/json" \\
  -d '{"query": "authentication policy and session expiration", "top_k": 3}'`,
      python: `import urllib.request
import json

payload = {"query": "authentication policy and session expiration", "top_k": 3}
req = urllib.request.Request(
    "http://127.0.0.1:8000/search",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as resp:
    print(json.loads(resp.read().decode()))`,
    },
    defaultPayload: '{\n  "query": "authentication policy and session expiration",\n  "top_k": 3\n}',
    defaultResponse: {
      query: 'authentication policy and session expiration',
      count: 2,
      results: [
        {
          id: 'auth-01',
          content: 'AIMLite supports API key and Bearer token authentication. Session tokens expire after 24 hours of inactivity.',
          score: 0.9412,
          metadata: { source: 'auth_policy.md', section: 'Authentication' },
        },
        {
          id: 'auth-02',
          content: 'Multi-factor authentication (MFA) is required for administrative access to production endpoints.',
          score: 0.8875,
          metadata: { source: 'auth_policy.md', section: 'Multi-Factor' },
        },
      ],
      status: 'success',
    },
  },

  'endpoint-health': {
    id: 'endpoint-health',
    category: 'HTTP Inference Server',
    title: 'Server Health (GET /health)',
    subtitle: 'Inspect server readiness and active model status.',
    badge: { label: 'GET', variant: 'get' },
    signatureOrPath: 'GET http://127.0.0.1:8000/health',
    breadcrumbs: ['HTTP Server', 'GET /health'],
    overview:
      'Health probe endpoint for Kubernetes liveness/readiness probes or load balancer health monitors.',
    djangoAnalogy:
      'Health check view.',
    snippets: {
      curl: 'curl http://127.0.0.1:8000/health',
    },
    defaultPayload: '{}',
    defaultResponse: {
      status: 'healthy',
      model: 'SupportDocRAG',
      endpoints: {
        'GET /': 'Interactive web playground',
        'GET /chat': 'Interactive RAG chat interface',
        'GET /docs': 'Swagger API documentation',
        'GET /openapi.json': 'OpenAPI 3.0 schema',
        'POST /predict': 'Submit prediction request',
        'POST /search': 'Semantic document search',
        'GET /health': 'Server health status',
      },
    },
  },

  'endpoint-models': {
    id: 'endpoint-models',
    category: 'HTTP Inference Server',
    title: 'Models Inventory (GET /models)',
    subtitle: 'List all registered and trained model classes with types and feature metadata.',
    badge: { label: 'GET', variant: 'get' },
    signatureOrPath: 'GET http://127.0.0.1:8000/models',
    breadcrumbs: ['HTTP Server', 'GET /models'],
    overview:
      'Returns a JSON array of all model classes discovered in model.py that have trained checkpoints or ready weights. Each entry includes the model name, type (ml, rag, or adapter), descriptive docstring, checkpoint path, active status, and discovered feature names for ML models. Also supports targeting specific models at POST /models/{name}/predict.',
    djangoAnalogy:
      'Like an automated model introspection registry or Django apps/models catalog.',
    snippets: {
      curl: `# List all registered models:
curl http://127.0.0.1:8000/models

# Inspect specific model metadata:
curl http://127.0.0.1:8000/models/ChurnClassifier

# Execute prediction on a specific model:
curl -X POST http://127.0.0.1:8000/models/ChurnClassifier/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [128.0, 1.0, 1.0, 2.7, 1.0, 265.1, 110.0, 89.0, 9.8, 10.0]}'`,
    },
    defaultPayload: '{\n  "method": "GET",\n  "endpoint": "/models"\n}',
    defaultResponse: [
      {
        name: 'ChurnClassifier',
        type: 'ml',
        description: 'Customer Churn classifier model predicting whether a customer will churn.',
        checkpoint: 'models/churn_classifier.pkl',
        features: ['AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage', 'CustServCalls', 'DayMins'],
        active: true,
      },
    ],
  },

  'endpoint-chat': {
    id: 'endpoint-chat',
    category: 'HTTP Inference Server',
    title: 'Adaptive Web Playground (GET /app, /chat)',
    subtitle: 'Embedded web interface automatically tailored to your model type (ML, RAG, or LoRA Adapter).',
    badge: { label: 'WEB UI', variant: 'get' },
    signatureOrPath: 'GET http://127.0.0.1:8000/app',
    breadcrumbs: ['HTTP Server', 'GET /app'],
    overview:
      'AIMLite serves a production-ready, responsive web application at /app (and /chat). The interface dynamically inspects your registered models and adapts based on their type: Classical ML models render an interactive Feature Form with sample presets and real-time prediction output; RAG models render an AI chat playground with grounded context citations; and LoRA models render prompt completion sliders. Supports light/dark mode, instant model switching, and 100% offline self-contained execution.',
    djangoAnalogy:
      'Like Django admin and Django debug toolbar combined — an instant interactive UI that automatically understands your schema and model types.',
    snippets: {
      cli: `# Start server with embedded adaptive playground:
aimlite serve --port 8000

# Open in browser:
# http://localhost:8000/
# http://localhost:8000/docs (Swagger UI)`,
    },
    defaultPayload: '{\n  "browser_url": "http://127.0.0.1:8000/app",\n  "method": "GET"\n}',
    defaultResponse: {
      endpoint: '/app',
      ui_type: 'Type-Adaptive Playground',
      supported_types: {
        ml: 'Interactive feature input grid, presets, confidence scores, and JSON mode',
        rag: 'Conversational assistant, grounded context citations, top-k slider',
        adapter: 'LoRA prompt playground, temperature and token controls',
      },
      status: 'active',
    },
  },

  'guide-serving': {
    id: 'guide-serving',
    category: 'HTTP Inference Server',
    title: 'Serving & App Customization Guide',
    subtitle: 'Complete developer guide on multi-model serving, type-driven web apps, headless --api mode, and custom templates.',
    badge: { label: 'GUIDE', variant: 'cli' },
    signatureOrPath: 'aimlite serve [options]',
    breadcrumbs: ['HTTP Server', 'Serving Guide'],
    overview:
      'AIMLite makes model deployment effortless while giving developers 100% control over the application. When you launch aimlite serve, the framework automatically scans model.py, instantiates every trained model class, and launches a type-adaptive web playground. This guide covers multi-model serving, UI adaptation per model type, headless API mode, and custom template overrides.',
    djangoAnalogy:
      'Just like Django allows overriding default templates in a templates/ directory and toggling headless DRF API mode, AIMLite gives developers full customizability without boilerplate.',
    whyCode: [
      {
        component: '1. Classical / Deep ML Models (Model)',
        reason:
          'When the active model inherits from Model, the served website renders an interactive Feature Form Playground. It auto-discovers dataset feature names (e.g. AccountWeeks, MonthlyCharge), provides 1-click Preset Buttons (Sample 1, Sample 2, Zero Vector, Randomize), toggles between visual Form Mode and raw JSON Mode, and displays predictions with latency gauges and confidence meters.',
      },
      {
        component: '2. Knowledge / RAG Models (KnowledgeModel, RAGModel)',
        reason:
          'When the active model inherits from KnowledgeModel or RAGModel, the served website transforms into an AI Conversational Playground. It features multi-turn chat bubbles, Top-K context sliders (1-10), query suggestion chips, and an expandable Source Citations accordion showing document IDs, chunk excerpts, and cosine similarity percentages.',
      },
      {
        component: '3. Fine-Tuned LoRA Models (AdapterModel)',
        reason:
          'When the active model inherits from AdapterModel, the served website renders a LoRA Prompt Completion Playground. It provides temperature and max token sliders, dynamic adapter controls, and real-time generation output inspection.',
      },
      {
        component: '4. Dynamic Model Switcher',
        reason:
          'If model.py contains multiple trained classes (even across different paradigms!), all ready models are registered in the server inventory. The top navigation bar displays a live Model Switcher that flips the UI controls instantly without page reload or port changes.',
      },
      {
        component: '5. Using --frontend with ANY Framework (React, Vue, Next.js, Svelte)',
        reason:
          'Pass --frontend <build_dir> to serve your custom frontend directly. AIMLite acts as a high-performance static file server with SPA routing fallback (redirecting non-file routes to index.html for React/Vue/Svelte routers) while keeping all REST APIs (/predict, /models, /health) active on the same port. This eliminates CORS issues completely.',
      },
      {
        component: '6. Developer Template Overrides',
        reason:
          'For rapid UI customization without a build step, create a templates/ directory in your project root (e.g. templates/app.html or templates/index.html). AIMLite checks your project directory first before falling back to built-in templates.',
      },
      {
        component: '7. Headless REST API Mode (--api)',
        reason:
          'Add --api to run a lightweight headless JSON microservice without HTML pages, perfect for Docker containers, Kubernetes pods, and backend integrations.',
      },
    ],
    parametersTitle: 'Serving Arguments & Integration Options',
    parameters: [
      {
        name: 'ModelName',
        type: 'string (optional)',
        required: false,
        defaultValue: 'all trained models in model.py',
        description: 'Specific model class to serve. If omitted, all trained model classes in model.py are served with navbar model switching.',
      },
      {
        name: '--frontend <dir>',
        type: 'Path (optional)',
        required: false,
        description: 'Directory path to any compiled frontend build (React/Vite in frontend/dist, Next.js in out/, Vue in dist/, SvelteKit in build/).',
      },
      {
        name: '--api',
        type: 'boolean flag',
        required: false,
        defaultValue: 'false',
        description: 'Headless REST mode. Disables all HTML web pages and serves pure JSON REST endpoints (/models, /predict, /health, /openapi.json).',
      },
      {
        name: '--checkpoint <path>',
        type: 'Path (optional)',
        required: false,
        description: 'Explicit checkpoint weights or index artifact path (defaults to auto-discovering in models/ and artifacts/).',
      },
      {
        name: '--port <int>',
        type: 'integer',
        required: false,
        defaultValue: '8000',
        description: 'TCP port number to bind the server.',
      },
      {
        name: '--host <str>',
        type: 'string',
        required: false,
        defaultValue: '127.0.0.1',
        description: 'Network interface IP to listen on (use 0.0.0.0 for external/container access).',
      },
    ],
    conventions: [
      {
        title: '1. Model Paradigm: Classical & Deep ML (Model)',
        description: 'Renders an interactive feature form with auto-discovered dataset columns, 1-click preset sample buttons, visual form vs JSON mode, confidence badges, and latency tracking.',
      },
      {
        title: '2. Model Paradigm: Knowledge & RAG (KnowledgeModel, RAGModel)',
        description: 'Renders an AI conversational chat assistant with grounded context citations, similarity percentages, expandable chunks, Top-K slider (1-10), and sample prompt chips.',
      },
      {
        title: '3. Model Paradigm: LoRA Fine-Tuning (AdapterModel)',
        description: 'Renders a LoRA prompt completion runner with temperature slider, max token controls, and token generation stream view.',
      },
      {
        title: '4. Dynamic Multi-Model Switcher',
        description: 'Auto-registers all trained models in model.py. Top navbar switcher switches models and their dedicated paradigm controls on the fly without page reload.',
      },
      {
        title: '5. Framework-Agnostic Hosting (--frontend)',
        description: 'Works with React, Vue, Next.js, SvelteKit, Angular, or Vanilla JS. Mounts the build folder at / with SPA client routing fallback to index.html.',
      },
      {
        title: '6. Zero-CORS Unified Deployment',
        description: 'Serves your custom frontend and all backend REST APIs (/predict, /models, /health) on the same port, completely eliminating CORS problems.',
      },
      {
        title: '7. Template Overrides (<project>/templates/)',
        description: 'Place templates/app.html or templates/index.html in your project root to customize the web interface without any frontend build pipeline.',
      },
      {
        title: '8. Headless Microservices (--api)',
        description: 'Lightweight REST API container mode without HTML pages, returning structured JSON for Kubernetes, Docker, and backend services.',
      },
    ],
    snippets: {
      cli: `# 1. Auto-serve all models with type-adaptive UI:
aimlite serve

# 2. Serve a specific model only:
aimlite serve ChurnClassifier

# 3. Headless REST API mode (pure JSON, no HTML):
aimlite serve --api --port 8000

# 4. Serve with React / Vite:
# In frontend/: npm run build (creates frontend/dist)
aimlite serve --frontend ./frontend/dist

# 5. Serve with Next.js (Static Export):
# In next.config.js: { output: 'export' }, then npm run build (creates out/)
aimlite serve --frontend ./frontend/out

# 6. Serve with Vue 3 / Nuxt / SvelteKit:
aimlite serve --frontend ./frontend/dist

# 7. Custom developer template override:
# Put your custom HTML in templates/app.html
aimlite serve`,
      curl: `// Calling AIMLite from ANY frontend framework (React, Vue, Vanilla JS):
// Because AIMLite serves the frontend on the same port, use simple relative paths!

// 1. Fetch all registered models:
const models = await fetch('/models').then(r => r.json());
console.log(models); // [{ name: "ChurnClassifier", type: "ml", features: [...] }]

// 2. Submit ML inference:
const mlResult = await fetch('/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ features: [128.0, 1.0, 2.7, 1.0, 265.1, 110.0, 89.0, 9.8, 10.0] })
}).then(r => r.json());

// 3. Submit RAG Knowledge query:
const ragResult = await fetch('/models/SupportDocRAG/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'How do I configure database backups?', top_k: 3 })
}).then(r => r.json());`,
    },
    defaultPayload: '{\n  "supported_frameworks": ["React", "Vue", "Next.js", "SvelteKit", "Angular", "Vanilla JS"],\n  "paradigm_uis": ["ml: Feature Form", "rag: Chat & Citations", "adapter: LoRA Generator"]\n}',
    defaultResponse: {
      framework_integration: {
        'Step 1': 'Build frontend bundle: npm run build (into dist/, out/, or build/)',
        'Step 2': 'Make relative API calls in frontend code: fetch("/predict"), fetch("/models")',
        'Step 3': 'Serve with single command: aimlite serve --frontend ./frontend/dist',
        'Benefit': 'Zero CORS configuration, single-port deployment, SPA client routing supported',
      },
      paradigm_controllers: {
        'Model (ML)': 'Interactive input form, auto-populated feature labels, presets, class decision badge',
        'KnowledgeModel (RAG)': 'Multi-turn chat, context citation accordion, top-k slider, query chips',
        'AdapterModel (LoRA)': 'Prompt editor, temperature slider, max token controls, response streaming',
      },
    },
  },

  'endpoint-openapi': {
    id: 'endpoint-openapi',
    category: 'HTTP Inference Server',
    title: 'OpenAPI 3.0 Schema (GET /openapi.json)',
    subtitle: 'Standardized machine-readable OpenAPI 3.0 specification.',
    badge: { label: 'GET', variant: 'get' },
    signatureOrPath: 'GET http://127.0.0.1:8000/openapi.json',
    breadcrumbs: ['HTTP Server', 'GET /openapi.json'],
    overview:
      'Exports OpenAPI 3.0 compliant schema for automated client generation and Swagger UI rendering.',
    djangoAnalogy:
      'Automatic schema generator like drf-spectacular.',
    snippets: {
      curl: 'curl http://127.0.0.1:8000/openapi.json',
    },
    defaultPayload: '{}',
    defaultResponse: {
      openapi: '3.0.0',
      info: {
        title: 'AIMLite API - my_ai',
        version: '0.1.0',
      },
      paths: {
        '/predict': { post: { summary: 'Execute Model Inference' } },
        '/health': { get: { summary: 'Server Health Status' } },
      },
    },
  },

  'endpoint-docs': {
    id: 'endpoint-docs',
    category: 'HTTP Inference Server',
    title: 'Swagger UI API Docs (GET /docs)',
    subtitle: 'Interactive browser Swagger documentation powered by templates/swagger.html.',
    badge: { label: 'GET', variant: 'get' },
    signatureOrPath: 'GET http://127.0.0.1:8000/docs',
    breadcrumbs: ['HTTP Server', 'GET /docs'],
    overview:
      'Renders the embedded Swagger UI from aimlite/templates/swagger.html, backed by the live /openapi.json schema.',
    djangoAnalogy:
      'Interactive Swagger UI like Django REST Framework Swagger.',
    snippets: {
      curl: 'curl http://127.0.0.1:8000/docs',
    },
    defaultPayload: '{}',
    defaultResponse: {
      status: 'rendered_html',
      template: 'aimlite/templates/swagger.html',
    },
  },

  'paradigm-scratch': {
    id: 'paradigm-scratch',
    category: 'The 3 AI Paradigms',
    title: 'Paradigm 1: Training from Scratch (Customer Churn)',
    subtitle: 'End-to-end tabular classification pipeline using scikit-learn on the Kaggle Telecom Churn dataset.',
    badge: { label: 'PARADIGM 1', variant: 'pillar' },
    signatureOrPath: 'from aimlite import Dataset, Model, BaseTrainer, BaseEvaluator, BaseInference',
    breadcrumbs: ['Paradigms', 'From Scratch'],
    overview:
      'Training from scratch provides total algorithmic freedom over feature transformations, loss functions, and optimization schedules. In this real-world guide, we build a production customer churn classifier using the Kaggle Telecom Churn dataset (https://www.kaggle.com/datasets/barun2104/telecom-churn). Required dependencies: aimlite install scikit-learn pandas.',
    djangoAnalogy:
      'Like building custom Django models, managers, and service layers with full database index and query control, Training from Scratch gives you total ownership over weights, hyperparameters, and persistence.',
    whyCode: [
      {
        component: 'data.py (TelecomChurnDataset)',
        reason:
          'Handles tabular ingestion and normalization of customer account metrics. Decoupling data loading from model logic ensures you can switch from pandas to streaming loaders without modifying training or inference code.',
      },
      {
        component: 'model.py (ChurnClassifier)',
        reason:
          'Subclasses Model, inheriting standard predict(), predict_proba(), and save()/load() contracts. Wrapping scikit-learn RandomForest ensures automatic compatibility with AIMLite CLI discovery and REST serving.',
      },
      {
        component: 'trainer.py (ChurnTrainer)',
        reason:
          'Isolates data partitioning, classifier fitting, and metric logging from model definitions. Automatically logs train/val metrics and writes model weights to artifacts/churn_classifier.pkl.',
      },
      {
        component: 'evaluator.py (ChurnEvaluator)',
        reason:
          'Calculates precision, recall, and F1-score on evaluation splits. This provides objective validation benchmarks before deploying models to production.',
      },
      {
        component: 'inference.py (ChurnInference)',
        reason:
          'The production API gateway. Translates incoming raw JSON payloads into feature vectors, executes risk scoring, and returns actionable business recommendations (e.g. "Intervene (High Churn Risk)").',
      },
    ],
    conventions: [
      {
        title: 'Kaggle Telecom Churn Dataset',
        description:
          'Download telecom_churn.csv from https://www.kaggle.com/datasets/barun2104/telecom-churn and place into data/. Features: AccountWeeks, ContractRenewal, CustServCalls, MonthlyCharge, RoamMins, Churn.',
      },
      {
        title: 'Required Dependencies',
        description: 'Run `aimlite install scikit-learn pandas` (or `pip install scikit-learn pandas`).',
      },
      {
        title: 'Zero-Path Execution',
        description:
          '`aimlite train ChurnClassifier` fits the classifier and writes weights to artifacts/. `aimlite serve` exposes the prediction API.',
      },
    ],
    snippets: {
      files: SCRATCH_FILES,
      cli: `# 1. Install dependencies & auto-manage .venv
aimlite install scikit-learn pandas

# 2. Initialize project (scratch paradigm)
aimlite init telecom_churn --type scratch
cd telecom_churn

# 3. Download Kaggle dataset to data/telecom_churn.csv:
# https://www.kaggle.com/datasets/barun2104/telecom-churn

# 4. Validate and train
aimlite data validate
aimlite train ChurnClassifier

# 5. Evaluate benchmark metrics
aimlite evaluate ChurnClassifier

# 6. Serve inference API
aimlite serve ChurnClassifier --port 8000`,
      curl: `curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "AccountWeeks": 128,
    "ContractRenewal": 1,
    "DataPlan": 1,
    "DataUsage": 2.7,
    "CustServCalls": 1,
    "DayMins": 265.1,
    "DayCalls": 110,
    "MonthlyCharge": 89.0,
    "OverageFee": 9.87,
    "RoamMins": 10.0
  }'`,
    },
    guideSteps: SCRATCH_GUIDE_STEPS,
    defaultPayload: '{\n  "AccountWeeks": 128,\n  "ContractRenewal": 1,\n  "DataPlan": 1,\n  "DataUsage": 2.7,\n  "CustServCalls": 1,\n  "DayMins": 265.1,\n  "DayCalls": 110,\n  "MonthlyCharge": 89.0,\n  "OverageFee": 9.87,\n  "RoamMins": 10.0\n}',
    defaultResponse: {
      churn_prediction: 0,
      churn_risk: 0.12,
      decision: "Retain (Low Risk)",
      status: "success",
    },
  },

  'paradigm-rag': {
    id: 'paradigm-rag',
    category: 'The 3 AI Paradigms',
    title: 'Paradigm 2: RAG (Retrieval-Augmented Generation & Knowledge Models)',
    subtitle: 'Ground foundation models in enterprise documents with PostgreSQL ORM vector search, query intelligence, and smart chunking.',
    badge: { label: 'PARADIGM 2', variant: 'pillar' },
    signatureOrPath: 'from aimlite.rag import Document, SmartChunker, PostgresVectorStore, QueryAnalyzer, BaseChatProvider, KnowledgeModel, RAGModel, RAGTrainer',
    breadcrumbs: ['Paradigms', 'RAG & Knowledge Models'],
    overview:
      'The RAG paradigm turns enterprise knowledge into actionable intelligence with zero hallucinations and no expensive fine-tuning. AIMLite provides enterprise-grade abstractions: SmartChunker (Markdown hierarchy and semantic boundary awareness), QueryAnalyzer (intent decomposition, search expansion, and HyDE), BaseChatProvider (OpenAI, Anthropic, Gemini, Ollama, and local models), PostgresVectorStore (PostgreSQL ORM with native pgvector support), pre-defined task prompts, KnowledgeModel, and RAGTrainer orchestration.',
    djangoAnalogy:
      'In Django, you define Models backed by a PostgreSQL database and query them with QuerySets. In AIMLite RAG, you define a KnowledgeModel backed by PostgresVectorStore (or MemoryVectorStore) and query it with QueryAnalyzer and VectorRetriever to dynamically inject grounded context into chat prompts.',
    whyCode: [
      {
        component: 'data.py (KnowledgeDocsDataset & SmartChunker)',
        reason:
          'Ingests Markdown and documentation articles using SmartChunker. Preserves header hierarchies (#, ##), paragraphs, and injects contextual metadata into every chunk.',
      },
      {
        component: 'model.py (KnowledgeModel / SupportDocRAG)',
        reason:
          'Subclasses KnowledgeModel/RAGModel, coordinating QueryAnalyzer, PostgresVectorStore / MemoryVectorStore, and BaseChatProvider (OpenAI, Gemini, Claude, Ollama, or local LLMs).',
      },
      {
        component: 'Query Intelligence (QueryAnalyzer & HyDE)',
        reason:
          'Deconstructs search queries into user intent, keywords, expanded multi-query variations, and Hypothetical Document Embeddings (HyDE) for maximum semantic recall.',
      },
      {
        component: 'Database Vector Store (PostgresVectorStore & ORM)',
        reason:
          'Persists document vectors directly in PostgreSQL using native pgvector (<=> cosine / <-> L2 distance) or in-memory fallback, with declarative ORM records (KnowledgeDocumentRecord, KnowledgeChunkRecord).',
      },
      {
        component: 'trainer.py (RAGTrainer / IndexBuilderTrainer)',
        reason:
          'Standardized lifecycle trainer coordinating document loading -> smart chunking -> embedding calculation -> vector database persistence.',
      },
      {
        component: 'inference.py (RAGInference)',
        reason:
          'Production HTTP endpoint executing queries over the vector index and returning grounded responses alongside verifiable document citations and confidence scores.',
      },
    ],
    parametersTitle: 'KnowledgeModel & RAG Configuration Parameters',
    parameters: [
      {
        name: 'chat_provider',
        type: 'BaseChatProvider',
        required: false,
        defaultValue: 'MockChatProvider()',
        description: 'LLM chat completion provider: OpenAIChatProvider, AnthropicChatProvider, GeminiChatProvider, OllamaChatProvider, LocalChatProvider, or custom.',
      },
      {
        name: 'query_analyzer',
        type: 'QueryAnalyzer',
        required: false,
        defaultValue: 'QueryAnalyzer()',
        description: 'Query intelligence engine performing intent extraction, keyword tagging, multi-query expansion, and HyDE passage generation.',
      },
      {
        name: 'chunker',
        type: 'SmartChunker',
        required: false,
        defaultValue: 'SmartChunker(max_chunk_size=600)',
        description: 'Structure-aware text splitter that preserves Markdown headings, code blocks, and section hierarchies.',
      },
      {
        name: 'vector_store',
        type: 'BaseVectorStore',
        required: false,
        defaultValue: 'MemoryVectorStore()',
        description: 'Vector database engine: PostgresVectorStore (with pgvector support) or in-memory MemoryVectorStore.',
      },
      {
        name: 'embedding_fn',
        type: 'BaseEmbedding',
        required: false,
        defaultValue: 'TfidfEmbedding()',
        description: 'Dense or sparse embedding engine: SentenceTransformerEmbedding, APIEmbedding, or zero-dependency TfidfEmbedding.',
      },
      {
        name: 'top_k',
        type: 'int',
        required: false,
        defaultValue: '3',
        description: 'Number of top-ranked context document passages to retrieve for answer synthesis.',
      },
    ],
    conventions: [
      {
        title: 'Smart Markdown & Document Chunking',
        description:
          'SmartChunker splits documents along natural section and paragraph boundaries, retaining document title and section headings in chunk metadata.',
      },
      {
        title: 'Pre-Defined Task System Prompts',
        description:
          'Standardized system prompts are built-in: PROMPT_RAG_QA (grounded QA with citations), PROMPT_QUERY_ANALYZER (intent & HyDE), PROMPT_SMART_CHUNKER (semantic summarization), and PROMPT_CONVERSATIONAL_RAG (multi-turn chat).',
      },
      {
        title: 'PostgreSQL & ORM Database Integration',
        description:
          'PostgresVectorStore connects to any PostgreSQL database with pgvector support and provides standard ORM records (KnowledgeDocumentRecord, KnowledgeChunkRecord).',
      },
      {
        title: 'Universal Chat Providers',
        description:
          'Supports API models (OpenAI GPT-4o, Anthropic Claude 3.5, Gemini 1.5/2.0) and local models (Ollama, HuggingFace transformers, local callables) under a unified BaseChatProvider contract.',
      },
    ],
    snippets: {
      files: RAG_FILES,
      cli: `# 1. Install optional production dependencies
aimlite install sentence-transformers psycopg2-binary
# or pip install sentence-transformers psycopg2-binary

# 2. Scaffold RAG project (rag paradigm)
aimlite init support_rag --type rag
cd support_rag

# 3. Add knowledge documents to data/ (e.g. data/faq.md)

# 4. Build vector index & database embeddings
aimlite train SupportDocRAG

# 5. Serve knowledge API
aimlite serve SupportDocRAG --port 8000`,
      curl: `curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"query": "How do session tokens expire?", "top_k": 3}'`,
    },
    guideSteps: RAG_GUIDE_STEPS,
    defaultPayload: '{\n  "query": "How do session tokens expire?",\n  "top_k": 3\n}',
    defaultResponse: {
      query: "How do session tokens expire?",
      answer: "Based on auth_policy.md: Authentication and Security Policy: AIMLite supports API key and Bearer token authentication. Session tokens expire after 24 hours of inactivity...",
      sources: [
        {
          source: "auth_policy.md",
          snippet: "Authentication and Security Policy: AIMLite supports API key and Bearer token authentication. Session tokens expire after 24 hours...",
          score: 0.9412,
        },
      ],
      status: "success",
    },
  },

  'paradigm-adapters': {
    id: 'paradigm-adapters',
    category: 'The 3 AI Paradigms',
    title: 'Paradigm 3: Fine-Tuning (LoRA & PEFT Adapters)',
    subtitle: 'Parameter-efficient adaptation with mathematical low-rank matrix decomposition and zero-latency serving.',
    badge: { label: 'PARADIGM 3', variant: 'pillar' },
    signatureOrPath: 'from aimlite.adapters import AdapterConfig, AdapterModel, AdapterTrainer, LoRALayer, MultiAdapterManager',
    breadcrumbs: ['Paradigms', 'Fine-Tuning'],
    overview:
      'Fine-tuning specializes base foundation models for custom instruction-following using mathematical low-rank decomposition (LoRA/PEFT). AdapterModel freezes the base model and trains only low-rank matrices A and B (W = W0 + (alpha/r)*B*A). Supports zero-latency serving via in-place weight merging (merge_weights()), multi-adapter hot-swapping (MultiAdapterManager), and real-time parameter efficiency diagnostics (get_trainable_parameters()). Runs out-of-the-box in pure Python/NumPy, with full PyTorch/PEFT interop.',
    djangoAnalogy:
      'In Django, you use model inheritance or Proxy models to extend behavior without duplicating the underlying database table. In AIMLite, AdapterModel attaches trainable low-rank delta matrices to a frozen base model.',
    whyCode: [
      {
        component: 'data.py (InstructionDataset)',
        reason:
          'Loads instruction-following prompt/response pairs from any dataset file in data/ and formats them into standard instruction templates (### Instruction: ... ### Response:).',
      },
      {
        component: 'model.py (LoRALayer & MultiAdapterManager)',
        reason:
          'Applies low-rank decomposition h = W0*x + (alpha/r)*(B*A)*x. Enables zero-overhead weight merging (merge_weights()) and runtime multi-adapter routing without reloading foundation models.',
      },
      {
        component: 'trainer.py (AdapterTrainer / AdapterInstructionTrainer)',
        reason:
          'Optimizes exclusively adapter matrices with real-time parameter efficiency tracking and convergence loss history.',
      },
      {
        component: 'Parameter Diagnostics (print_trainable_parameters)',
        reason:
          'Calculates trainable parameters vs total parameters and memory reduction percentages directly in CLI output and metadata.',
      },
      {
        component: 'Hugging Face PEFT Checkpointing',
        reason:
          'Saves only low-rank delta weights (~50KB to 50MB instead of duplicating 14GB base weights) via standard adapter_config.json and adapter_model.pkl.',
      },
      {
        component: 'inference.py (AdapterInference)',
        reason:
          'Dynamically serves fine-tuned responses on top of the shared foundation model via POST /predict with zero runtime latency penalty.',
      },
    ],
    conventions: [
      {
        title: 'Flexible Instruction Data',
        description: 'Place your instruction file (e.g. instructions.json, prompts.jsonl) in data/ and declare filename on Dataset.',
      },
      {
        title: 'Zero Heavy Dependencies',
        description: 'Runs out-of-the-box in pure Python and NumPy. Optional deep learning backends: `aimlite install torch peft`.',
      },
      {
        title: 'Delta Persistence',
        description:
          'Calling model.save() writes adapter_config.json, adapter_model.pkl, and adapter_metadata.json without duplicating the foundation model.',
      },
    ],
    snippets: {
      files: ADAPTER_FILES,
      cli: `# 1. Install dependencies into managed .venv
aimlite install torch peft

# 2. Initialize project (adapter paradigm)
aimlite init lora_instructions --type adapter
cd lora_instructions

# 3. Add instruction dataset to data/instructions.json

# 4. Fine-tune adapter weights
aimlite train LoRAInstructionModel

# 5. Serve fine-tuned API
aimlite serve LoRAInstructionModel --port 8000`,
      curl: `curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"instruction": "Classify support ticket", "input": "Cannot access billing portal"}'`,
    },
    guideSteps: ADAPTER_GUIDE_STEPS,
    defaultPayload: '{\n  "instruction": "Summarize customer feedback.",\n  "input": "Great support response time."\n}',
    defaultResponse: {
      prompt: "### Instruction:\nSummarize customer feedback.\n\n### Response:",
      response: "[LoRA-Adapted Llama-3-8B (r=8)]: Completed successfully.",
      adapter_rank: 8,
      status: "success",
    },
  },

  'rag-hooks': {
    id: 'rag-hooks',
    category: 'AIMLite RAG Deep Dive',
    title: 'KnowledgeModel & RAG Lifecycle Hooks',
    subtitle: 'Complete guide to modular extension hooks in aimlite.rag introduced in v1.0.6.',
    badge: { label: 'HOOKS', variant: 'pillar' },
    signatureOrPath: 'from aimlite.rag import KnowledgeModel, RAGModel, RAGTrainer, Document',
    breadcrumbs: ['RAG Deep Dive', 'Extension Hooks'],
    overview:
      'In AIMLite v1.0.6, KnowledgeModel, RAGModel, and RAGTrainer were completely re-architected with modular, developer-editable lifecycle extension hooks. Instead of a rigid black-box pipeline, developers can cleanly override, extend, or replace every stage of query processing, candidate retrieval, cross-encoder reranking, prompt synthesis, answer post-processing, and index caching.',
    djangoAnalogy:
      'Like Django signals, custom QuerySet managers, or middleware — hook into the exact lifecycle events without hacking core framework internals.',
    whyCode: [
      {
        component: 'preprocess_query(query) -> str',
        reason: 'Rewrites user queries before vector embedding. Perfect for acronym expansion (e.g. MFA -> Multi-Factor Authentication), spell-checking, or query normalization.',
      },
      {
        component: 'retrieve(query, top_k, filters) -> List[Document]',
        reason: 'Direct candidate retrieval hook. Customize vector database search, hybrid sparse/dense retrieval, or dynamic metadata filtering.',
      },
      {
        component: 'rerank(query, documents) -> List[Document]',
        reason: 'Plug in cross-encoders (e.g. bge-reranker), Reciprocal Rank Fusion (RRF), or custom relevance thresholds to eliminate false positives.',
      },
      {
        component: 'format_prompt(query, context_docs) -> str',
        reason: 'Formats retrieved passages and user questions into the final synthesis prompt with configurable template support.',
      },
      {
        component: 'synthesize(query, context_docs, system_prompt, **kwargs) -> str',
        reason: 'Customizes LLM response generation, multi-turn conversational context, streaming generators, or fallback heuristics.',
      },
      {
        component: 'postprocess_answer(answer, context_docs) -> str',
        reason: 'Sanitizes LLM outputs, appends automated citation links/disclaimers, or runs safety guardrails before returning to the client.',
      },
      {
        component: 'search(query, top_k, filters) -> List[Dict[str, Any]]',
        reason: 'Direct semantic vector search returning ranked document dictionaries without invoking LLM synthesis (used by POST /search).',
      },
      {
        component: 'RAGTrainer.before_index & after_index',
        reason: 'Pre-filtering documents before embedding calculation and executing post-indexing cache invalidations, notifications, or logs.',
      },
    ],
    parametersTitle: 'Developer-Editable Hooks & Signatures',
    parameters: [
      {
        name: 'preprocess_query(self, query: str) -> str',
        type: 'method hook',
        required: false,
        defaultValue: 'query.strip()',
        description: 'Hook to clean, rewrite, expand, or spell-check user queries before retrieval.',
      },
      {
        name: 'retrieve(self, query: str, top_k=None, filters=None) -> List[Document]',
        type: 'method hook',
        required: false,
        description: 'Hook for candidate document retrieval from vector stores or hybrid search engines.',
      },
      {
        name: 'rerank(self, query: str, documents: List[Document]) -> List[Document]',
        type: 'method hook',
        required: false,
        description: 'Pluggable reranking hook for cross-encoders, reciprocal rank fusion, or score threshold filtering.',
      },
      {
        name: 'format_prompt(self, query: str, context_docs: List[Document]) -> str',
        type: 'method hook',
        required: false,
        description: 'Hook to assemble the prompt passed to LLM synthesis, supporting custom prompt_template.',
      },
      {
        name: 'synthesize(self, query, context_docs, system_prompt=None, **kwargs) -> str',
        type: 'method hook',
        required: false,
        description: 'Hook for generative LLM response synthesis. Override for streaming, custom prompts, or agent routing.',
      },
      {
        name: 'postprocess_answer(self, answer: str, context_docs: List[Document]) -> str',
        type: 'method hook',
        required: false,
        description: 'Hook for output sanitization, citation formatting, disclaimer appending, or guardrail validation.',
      },
      {
        name: 'chunk_documents(self, documents: Sequence[Document]) -> List[Document]',
        type: 'method hook',
        required: false,
        description: 'Override to implement custom document chunking or recursive splitting.',
      },
      {
        name: 'search(self, query, top_k=None, filters=None) -> List[Dict[str, Any]]',
        type: 'method hook',
        required: false,
        description: 'Direct semantic vector search returning ranked document dictionaries without LLM synthesis.',
      },
      {
        name: 'add_document(self, content: str, title=None, metadata=None) -> int',
        type: 'convenience method',
        required: false,
        description: 'Convenience method to dynamically index ad-hoc text records into the active knowledge store.',
      },
      {
        name: 'before_index(self, documents: List[Document]) -> List[Document]',
        type: 'trainer hook',
        required: false,
        description: 'RAGTrainer lifecycle hook called before indexing for pre-filtering or enrichment.',
      },
      {
        name: 'after_index(self, model, index_path, count: int) -> None',
        type: 'trainer hook',
        required: false,
        description: 'RAGTrainer lifecycle hook called after index serialization for notifications, logging, or caching.',
      },
    ],
    snippets: {
      python: `from typing import Any, Dict, List, Optional
from aimlite.rag import Document, KnowledgeModel, SmartChunker

class CustomSupportRAG(KnowledgeModel):
    """Production Knowledge Base with customized extension hooks."""

    def preprocess_query(self, query: str) -> str:
        # 1. Acronym expansion and query normalization
        cleaned = query.strip()
        replacements = {"mfa": "multi-factor authentication", "sla": "service level agreement"}
        for k, v in replacements.items():
            cleaned = cleaned.replace(k, v)
        return cleaned

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        # 2. Strict score thresholding: drop passages below 0.4 similarity
        filtered = [doc for doc in documents if (doc.score or 0.0) >= 0.4]
        # Return top 3 highest scoring passages
        return sorted(filtered, key=lambda d: d.score or 0.0, reverse=True)[:3]

    def postprocess_answer(self, answer: str, context_docs: List[Document]) -> str:
        # 3. Append verified citations footer
        sources = {d.metadata.get("source") for d in context_docs if "source" in d.metadata}
        citations = ", ".join(sources) if sources else "Knowledge Base"
        return f"{answer}\\n\\n---\\n[Grounded Sources: {citations}]"`,
    },
    defaultPayload: '{\n  "query": "How do session tokens expire?",\n  "top_k": 3\n}',
    defaultResponse: {
      query: 'How do session tokens expire?',
      answer: 'Session tokens expire after 24 hours of inactivity.\n\n---\n[Grounded Sources: auth_policy.md]',
      sources: [
        {
          source: 'auth_policy.md',
          snippet: 'Session tokens expire after 24 hours of inactivity.',
          score: 0.9412,
        },
      ],
      status: 'success',
    },
  },

  'changelog': {
    id: 'changelog',
    category: 'Releases & Changelog',
    title: 'Framework Changelog & Releases',
    subtitle: 'Release history and upgrade guide for AIMLite (v2.1.2 latest).',
    badge: { label: 'v2.1.2', variant: 'util' },
    signatureOrPath: 'pip install --upgrade aimlite==2.1.2',
    breadcrumbs: ['Releases', 'v2.1.2'],
    overview:
      'AIMLite adheres strictly to Semantic Versioning (SemVer). The latest stable release is v2.1.2, published live on PyPI. Below is the full chronological record of changes, new features, and upgrade instructions across all releases.',
    djangoAnalogy:
      'Comprehensive release notes detailing architectural improvements and new lifecycle hooks.',
    conventions: [
      {
        title: 'Release [2.1.2] - 2026-10-08 (LoRA Adapter Scaffolding Fix & Documentation Overhaul)',
        description: 'LoRA Adapter Scaffolding & Prompt Resolution: Enhanced predict() in LoRAInstructionModel to resolve prompt, instruction, text, and input payload keys, fixing dictionary stringification during Web UI and API serving. Documentation & Paradigm Guides Indentation Overhaul: Fixed indentation and code formatting across all 16 template snippets in api_docs (SCRATCH_FILES, RAG_FILES, ADAPTER_FILES), ensuring error-free copy-paste into local Python modules. Corrected CLI command spacing in guide steps (scikit-learn, sentence-transformers, --type, --port). Interactive Documentation App: Synchronized api_docs to v2.1.2 with production build.',
      },
      {
        title: 'Release [2.1.1] - 2026-10-08 (Dynamic Config Sync, Ollama Robustness & Git Hygiene)',
        description: 'Dynamic Model Configuration Synchronization: Model parameters and active LLMs in aimlite.json are hot-reloaded during serving via KnowledgeModel.sync_config() and BaseConfig.load_active(). Ollama Provider Robustness: Configurable timeout (OLLAMA_TIMEOUT, default 300s) with explicit TimeoutError, max_tokens capping, and graceful REST fallback. Package Sanitization & Auto-Healing: sanitize_package_name prevents invalid Python package names (leading dots, hyphens, keywords) and auto-heals legacy project entrypoints. Web UI Error Display: app.html renders informative error bubbles on server inference failures. Git Repository Hygiene: Untracked all tracked CPython .pyc / __pycache__ artifacts from git and hardened .gitignore rules.',
      },
      {
        title: 'Release [2.1.0] - 2026-10-07 (Major Packaging & LoRA Flexibility Fixes)',
        description: 'POSIX Binary Collision Resolved: Removed global test and cli script entries in pyproject.toml; only aimlite is registered to prevent overrides of POSIX /bin/test. Root Namespace Pollution Eliminated: Moved cli into aimlite.cli and configured Hatchling packaging. Flexible LoRA Dataset Training: Vectorizer supports instruction pairs (instruction/response, prompt/completion), arbitrary dictionaries, and raw text strings without crashing on non-numeric columns. Canonical 3-Paradigm Alignment: Standardized choices strictly to scratch, rag, and adapter with UnknownParadigmError. aimlite init --clean Paradigm Scaffolding: --clean prompts for paradigm and provisions 0-byte pristine empty code files. doctor Python Version Check: Recommends Python 3.10–3.12 for ML ecosystem wheel compatibility. 113 passing tests.',
      },
      {
        title: 'Release [2.0.0] - 2026-09-29 (Production-Readiness Implementation)',
        description: 'Real LoRA Gradient Training Engine: Replaced simulated curves with true analytical backpropagation (MSE loss, backward_A & backward_B updates, gradient clipping, zero-gradient assertion). Semantic & Hierarchical SmartChunker: Recursive multi-tier document splitting across Markdown headers, paragraphs, sentences, words, and character fallback with max_chunk_size guarantees and chunk_overlap. Strict Chat Provider Requirement: Mandates explicit chat_provider for KnowledgeModel and RAGModel (no silent Mock fallbacks in production). Zero Silent Fallbacks: SentenceTransformerEmbedding raises explicit ImportError when dependencies are missing; DocumentLoader.load_directory raises RuntimeError for corrupt files. Deterministic Dataset Partitioning: Dataset.split with seeded random shuffling and ratio validation. Threaded Concurrent Server: ThreadingHTTPServer (ThreadingMixIn) for non-blocking concurrent inference and health checks. 107 passing tests.',
      },
      {
        title: 'Release [1.0.8] - 2026-09-29',
        description: 'RAG & Fine-Tuning Template Sample Assets: Curated sample markdown documentation in data/samples/ for standard RAG projects, and instruction datasets in data/samples/ with fine-tuning README for adapter projects. Expanded test_init_clean suite verifying clean vs standard scaffolding modes.',
      },
      {
        title: 'Release [1.0.7] - 2026-09-28 (Major)',
        description: 'Zero-Path Multi-Model Serving & Inventory: Auto-scans model.py for all trained classes, registers them in GET /models, supports class targeting (aimlite serve [ModelName]), and POST /models/{name}/predict. Paradigm-Adaptive Web Application (app.html) with tailored controls for ML (Feature Form, presets, JSON mode), RAG (Grounded Chat, Citations, Top-K), and LoRA (Prompt tester), plus live top navbar Model Switcher. Clean Scratch Scaffolding (aimlite init --clean) generating pure project skeletons with zero sample code or dummy datasets (comments & class contracts only). Headless REST API mode (--api). Framework-agnostic static hosting (--frontend) with SPA routing fallback to index.html and zero-CORS single-port deployment. Project template priority overrides (<project>/templates/).',
      },
      {
        title: 'Release [1.0.6] - 2026-09-24',
        description: 'Developer-Editable KnowledgeModel & RAGModel architecture (preprocess_query, retrieve, rerank, format_prompt, synthesize, postprocess_answer, chunk_documents, search, add_document). before_index & after_index hooks on RAGTrainer. Modular scaffolded RAG codebase. POST /search endpoint.',
      },
      {
        title: 'Release [1.0.5] - 2026-09-23',
        description: 'aimlite benchmark command: Auto-injects project root into PYTHONPATH, resolves .venv Python, eliminates ModuleNotFoundError. pyfiglet ASCII art banner. Interactive + package search prompt. aimlite.json as single dependency source of truth.',
      },
      {
        title: 'Release [1.0.4] - 2026-09-23',
        description: 'Vite-inspired modern terminal UI (questionary & rich) with interactive arrows and spinners. Instant (<50ms) project initialization. --install CLI flag. Root .gitignore. Fixed ML template background pip freeze bug.',
      },
      {
        title: 'Release [1.0.3] - 2026-09-20',
        description: 'Production Modular RAG & Enterprise Knowledge Base engine (aimlite.rag). Multi-provider chat integrations (OpenAI, Gemini, Anthropic, Ollama, Local, Mock). SmartChunker, QueryAnalyzer (HyDE), PostgresVectorStore (pgvector ORM). Interactive Web Chat Playground at /chat.',
      },
      {
        title: 'Release [0.1.2] - 2026-09-18',
        description: 'Production-Grade LoRA & PEFT Adaptation Engine (aimlite.adapters). Mathematical low-rank decomposition, zero-latency weight merging (merge_weights), MultiAdapterManager hot-swapping, parameter efficiency diagnostics (<50MB vs 14GB).',
      },
      {
        title: 'Release [0.1.1] - 2026-09-18',
        description: 'Developer freedom for arbitrary data formats (CSV, Parquet, JSON, JSONL, TSV) and filenames. Reference template flexibility across all ML frameworks.',
      },
      {
        title: 'Release [0.1.0] - 2026-09-18',
        description: 'Initial release. Ecosystem rebranded to AIMLite. 3 AI paradigms (Scratch, LoRA Adapters, RAG). Zero-path execution, aimlite doctor, aimlite serve.',
      },
    ],
    snippets: {
      cli: `# Upgrade to latest AIMLite release:
pip install --upgrade aimlite

# Verify installation & diagnostic health:
aimlite doctor`,
    },
    defaultPayload: '{\n  "package": "aimlite",\n  "version": "2.1.2",\n  "channel": "pypi"\n}',
    defaultResponse: {
      package: 'aimlite',
      installed_version: '2.1.2',
      latest_pypi_version: '2.1.2',
      release_date: '2026-10-08',
      status: 'up_to_date',
      highlights: [
        'Dynamic model configuration sync: hot-reload aimlite.json model and top_k changes during serve',
        'Ollama chat provider robustness: configurable timeout (OLLAMA_TIMEOUT) and token limits',
        'Package name sanitization and project auto-healing for valid Python identifiers',
        'Web UI interactive chat error handling and clear error bubbles',
        'Repository hygiene: untracked all .pyc bytecode and hardened .gitignore rules',
      ],
    },
  },
};

