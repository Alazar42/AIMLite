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
      { id: 'pillar-data', label: '1. Data Pillar', badge: 'PILLAR', badgeVariant: 'pillar' },
      { id: 'pillar-model', label: '2. Model Pillar', badge: 'PILLAR', badgeVariant: 'pillar' },
      { id: 'pillar-lifecycle', label: '3. Lifecycle Pillar', badge: 'PILLAR', badgeVariant: 'pillar' },
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
      { id: 'cli-data', label: 'aimlite data validate', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-train', label: 'aimlite train', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-evaluate', label: 'aimlite evaluate', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-serve', label: 'aimlite serve', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-doctor', label: 'aimlite doctor', badge: 'CLI', badgeVariant: 'cli' },
    ],
  },
  {
    name: 'HTTP Inference Server',
    items: [
      { id: 'endpoint-predict', label: 'Execute Inference', badge: 'POST', badgeVariant: 'post' },
      { id: 'endpoint-health', label: 'Server Health', badge: 'GET', badgeVariant: 'get' },
      { id: 'endpoint-voxide', label: 'Voxide AI Helper', badge: 'AI', badgeVariant: 'post' },
      { id: 'endpoint-openapi', label: 'OpenAPI 3.0 Schema', badge: 'GET', badgeVariant: 'get' },
      { id: 'endpoint-docs', label: 'Swagger UI Docs', badge: 'GET', badgeVariant: 'get' },
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
        name: 'split(train=0.8, validation=0.1, test=0.1)',
        type: 'method',
        required: true,
        description: 'Partitions rows into (train_data, val_data, test_data) tuples.',
        defaultValue: '0.8, 0.1, 0.1',
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
    subtitle: 'Scaffolds a new project directory or initializes inside the current folder.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'aimlite init [project_name | .]',
    breadcrumbs: ['CLI', 'init'],
    overview:
      'Scaffolds a new project directory with convention layout: data/ (strictly empty), models/, experiments/, artifacts/, checkpoints/, and starter package files. Supports both new directory creation (aimlite init my_ai) and in-place current folder initialization (aimlite init .).',
    djangoAnalogy:
      'Direct equivalent of django-admin startproject <name> or npm init. Generates clean boilerplate with instructional contracts.',
    snippets: {
      cli: `# Option 1: Create a new project directory
aimlite init my_ai
cd my_ai

# Option 2: Initialize directly inside current folder with parent folder name
aimlite init .`,
    },
    defaultPayload: '{\n  "command": "aimlite init",\n  "project": "."\n}',
    defaultResponse: {
      status: 'success',
      output: [
        "[OK] Successfully initialized AIMLite project 'my_ai'",
        'Package files: data.py, model.py, trainer.py, evaluator.py, inference.py, config.py',
        'Directory data/ created empty (ready for your dataset files)',
      ],
    },
  },

  'cli-install': {
    id: 'cli-install',
    category: 'Zero-Path CLI Commands',
    title: 'aimlite install',
    subtitle: 'Installs libraries safely into project .venv using uv (preferred) or pip.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'aimlite install <package_name ...> [--upgrade]',
    breadcrumbs: ['CLI', 'install'],
    overview:
      'Manages your project virtual environment (.venv). If .venv does not exist, automatically creates it. Uses uv pip install --python <venv> for lightning fast, conflict-free dependency installation.',
    djangoAnalogy:
      'Eliminates environment mismatches and global package collisions automatically.',
    snippets: {
      cli: `# Paradigm 1 (Scratch):
aimlite install scikit-learn pandas

# Paradigm 2 (RAG):
aimlite install sentence-transformers numpy

# Paradigm 3 (Adapters):
aimlite install torch peft`,
    },
    defaultPayload: '{\n  "packages": ["scikit-learn", "pandas"],\n  "upgrade": false\n}',
    defaultResponse: {
      status: 'success',
      venv_path: '.venv',
      installed: ['scikit-learn', 'pandas'],
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
    subtitle: 'Headless JSON inference API server with optional custom frontend hosting.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'aimlite serve [ModelName] [--port 8000] [--frontend <dir>]',
    breadcrumbs: ['CLI', 'serve'],
    overview:
      'Starts a high-performance, headless JSON inference server on port 8000. Serves API root metadata at GET /, prediction endpoint at POST /predict, server health at GET /health, Swagger UI at GET /docs, and OpenAPI specification at GET /openapi.json. When you build a custom frontend (e.g. React/Vite in frontend/dist/), pass --frontend frontend/dist to host it directly alongside the API.',
    djangoAnalogy:
      'Direct equivalent of python manage.py runserver 8000, with Swagger UI and developer-customizable endpoints.',
    snippets: {
      cli: `# Run pure headless JSON API server
aimlite serve --port 8000

# Serve targeted model class
aimlite serve ChurnClassifier --port 8000

# Serve custom frontend bundle alongside API
aimlite serve --frontend ./frontend/dist`,
    },
    defaultPayload: '{\n  "port": 8000,\n  "target_model": "ChurnClassifier"\n}',
    defaultResponse: {
      status: 'server_active',
      endpoints: {
        'GET /': 'AIMLite API Root & Metadata',
        'GET /docs': 'Swagger UI Documentation',
        'GET /openapi.json': 'OpenAPI 3.0 Schema',
        'POST /predict': 'Inference API',
        'GET /health': 'Health Status',
      },
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
      'Diagnoses Python runtime version, hardware accelerators (CUDA/MPS/CPU), and directory read/write permissions for data/, models/, experiments/, and checkpoints/.',
    djangoAnalogy:
      'Comprehensive environment diagnostic check.',
    snippets: {
      cli: 'uv run aimlite doctor',
    },
    defaultPayload: '{}',
    defaultResponse: {
      python_version: '3.14.4',
      device: 'CUDA',
      directories_ok: true,
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
      model: 'my_ai',
      endpoints: {
        'GET /': 'Interactive web playground',
        'GET /docs': 'Swagger API documentation',
        'GET /openapi.json': 'OpenAPI 3.0 schema',
        'POST /predict': 'Submit prediction request',
        'GET /health': 'Server health status',
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

  'endpoint-voxide': {
    id: 'endpoint-voxide',
    category: 'HTTP Inference Server',
    title: 'Voxide AI Assistant & Tool Integration',
    subtitle: 'Voice & text AI helper on the served UI with extensible developer tool calling.',
    badge: { label: 'AI HELPER', variant: 'post' },
    signatureOrPath: 'npm install @voxide/react@latest',
    breadcrumbs: ['HTTP Server', 'Voxide AI'],
    overview:
      'AIMLite 1.0.0 integrates Voxide AI (@voxide/react) into served frontends as an interactive voice and chat helper. Out-of-the-box, the widget provides live "predict" and "health" tools. Developers can easily register additional custom tools in their frontend or inference handlers.',
    djangoAnalogy:
      'Like an intelligent administrative assistant wired directly into your Django REST views, able to execute actions from spoken phrases or chat prompts.',
    parametersTitle: 'Default Tools Registered on VoxideClient',
    parameters: [
      {
        name: 'predict',
        type: 'tool (POST /predict)',
        required: true,
        description: 'Runs inference forward pass given feature arrays (e.g. [1.5, 2.7, 3.2, 4.0]).',
      },
      {
        name: 'health',
        type: 'tool (GET /health)',
        required: true,
        description: 'Queries server readiness, uptime, and loaded model status.',
      },
      {
        name: 'navigate',
        type: 'tool (UI Router)',
        required: false,
        description: 'Voice or text navigation between documentation sections and endpoints.',
      },
    ],
    snippets: {
      python: `"""inference.py - Add custom endpoints for Voxide AI tools"""
from aimlite import BaseInference, Model

class AppInference(BaseInference):
    def run(self, model: Model, raw_input):
        return model.predict(raw_input)

    def custom_analyze(self, model: Model, payload):
        # Additional custom capability callable by Voxide tool
        return {"sentiment": "positive", "score": 0.94}

    def get_routes(self):
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
            "POST /api/v1/analyze": self.custom_analyze,
        }`,
      typescript: `import { VoxideClient, VoxideWidget } from '@voxide/react';

// 1. Create client
const ai = new VoxideClient({
  publicKey: "vox_pub_your_key",
});

// 2. Register basic tools & custom developer tools
ai.register({
  predict: {
    description: "Run model inference with input features.",
    params: { features: { type: "array", required: true } },
    handler: async ({ features }) => {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });
      return res.json();
    },
  },
  health: {
    description: "Check model server health.",
    params: {},
    handler: async () => {
      const res = await fetch("/health");
      return res.json();
    },
  },
  // Developers can add custom capabilities here:
  analyzeText: {
    description: "Analyze sentiment using custom endpoint.",
    params: { text: { type: "string", required: true } },
    handler: async ({ text }) => {
      const res = await fetch("/api/v1/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      return res.json();
    },
  },
});

// 3. Drop in the widget
export default function App() {
  return <VoxideWidget client={ai} theme="dark" accentColor="#0284c7" />;
}`,
      cli: 'npm install @voxide/react@latest',
    },
    defaultPayload: '{\n  "tool": "predict",\n  "params": {\n    "features": [1.5, 2.7, 3.2, 4.0]\n  }\n}',
    defaultResponse: {
      status: 'success',
      voxide_version: '^0.8.0',
      active_tools: ['predict', 'health', 'navigate'],
      widget_mounted: true,
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

# 2. Initialize project (or in current folder with .)
aimlite init telecom_churn
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
    title: 'Paradigm 2: RAG (Retrieval-Augmented Generation)',
    subtitle: 'Ground foundation models in private documents with semantic vector search and zero hallucination.',
    badge: { label: 'PARADIGM 2', variant: 'pillar' },
    signatureOrPath: 'from aimlite.rag import Document, TextSplitter, MemoryVectorStore, VectorRetriever, RAGModel',
    breadcrumbs: ['Paradigms', 'RAG'],
    overview:
      'The RAG paradigm turns enterprise knowledge into actionable intelligence without expensive retraining. AIMLite provides first-class Document loaders, sliding-window text chunkers, zero-dependency TF-IDF or dense embeddings, in-memory vector stores, and RAGModel. Required dependencies: aimlite install sentence-transformers numpy.',
    djangoAnalogy:
      'In Django, you query the database using ORM QuerySets. In AIMLite RAG, you query your knowledge base using VectorRetriever to dynamically inject relevant context passages into generation prompts.',
    whyCode: [
      {
        component: 'data.py (KnowledgeDocsDataset)',
        reason:
          'Ingests Markdown and text documents from data/ and chunks them with TextSplitter into overlapping passages, ensuring passages fit embedding model context limits without truncation.',
      },
      {
        component: 'model.py (SupportDocRAG)',
        reason:
          'Subclasses RAGModel and binds MemoryVectorStore with VectorRetriever. Ranks documents using cosine similarity and synthesizes grounded answers with source citations.',
      },
      {
        component: 'trainer.py (IndexBuilderTrainer)',
        reason:
          'Indexes document passages and calculates semantic embeddings as an offline background step, ensuring production queries execute with sub-10ms latency.',
      },
      {
        component: 'inference.py (RAGInference)',
        reason:
          'Production HTTP endpoint executing queries over the vector index and returning answers alongside verifiable document citations and confidence scores.',
      },
    ],
    conventions: [
      {
        title: 'Document Ingestion',
        description:
          'Place Markdown (.md) or Text (.txt) knowledge articles in data/. TextSplitter chunks them into sliding windows automatically.',
      },
      {
        title: 'Required Dependencies',
        description: 'Run `aimlite install sentence-transformers numpy` (or `pip install sentence-transformers numpy`).',
      },
      {
        title: 'Built-in Vector Store',
        description:
          'MemoryVectorStore supports cosine similarity search and built-in save/load serialization to artifacts/rag_index.json.',
      },
    ],
    snippets: {
      files: RAG_FILES,
      cli: `# 1. Install dependencies into managed .venv
aimlite install sentence-transformers numpy

# 2. Scaffold project
aimlite init support_rag
cd support_rag

# 3. Add knowledge documents to data/ (e.g. data/faq.md)

# 4. Build vector index
aimlite train SupportDocRAG

# 5. Serve knowledge API
aimlite serve SupportDocRAG --port 8000`,
      curl: `curl -X POST http://127.0.0.1:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"query": "How does zero-path execution work?", "top_k": 3}'`,
    },
    guideSteps: RAG_GUIDE_STEPS,
    defaultPayload: '{\n  "query": "How does zero-path execution work?",\n  "top_k": 3\n}',
    defaultResponse: {
      query: "How does zero-path execution work?",
      answer: "Based on faq.md: AIMLite supports zero-path CLI execution by resolving conventions...",
      sources: [
        {
          source: "faq.md",
          snippet: "AIMLite supports zero-path CLI execution...",
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

# 2. Initialize project
aimlite init lora_instructions
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
};

