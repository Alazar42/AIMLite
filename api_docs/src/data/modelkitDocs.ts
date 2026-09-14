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
  parametersTitle?: string;
  parameters?: DocParameter[];
  conventions?: { title: string; description: string }[];
  snippets: {
    python?: string;
    cli?: string;
    curl?: string;
    typescript?: string;
  };
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
      { id: 'paradigm-rag', label: '1. RAG (Retrieval-Augmented)', badge: 'RAG', badgeVariant: 'pillar' },
      { id: 'paradigm-adapters', label: '2. Fine-Tuning (LoRA & Adapters)', badge: 'PEFT', badgeVariant: 'pillar' },
      { id: 'paradigm-scratch', label: '3. Training from Scratch', badge: 'TRAIN', badgeVariant: 'pillar' },
    ],
  },
  {
    name: 'Multi-App Architecture',
    items: [
      { id: 'arch-multiapp', label: 'Django-Style Multi-App Layout', badge: 'ARCH', badgeVariant: 'util' },
      { id: 'cli-startapp', label: 'modelkit startapp', badge: 'CLI', badgeVariant: 'cli' },
    ],
  },
  {
    name: 'The 3 Pillars of ModelKit',
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
      { id: 'cli-init', label: 'modelkit init', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-startapp', label: 'modelkit startapp', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-install', label: 'modelkit install', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-data', label: 'modelkit data validate', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-train', label: 'modelkit train', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-evaluate', label: 'modelkit evaluate', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-serve', label: 'modelkit serve', badge: 'CLI', badgeVariant: 'cli' },
      { id: 'cli-doctor', label: 'modelkit doctor', badge: 'CLI', badgeVariant: 'cli' },
    ],
  },
  {
    name: 'HTTP Inference Server',
    items: [
      { id: 'endpoint-predict', label: 'Execute Inference', badge: 'POST', badgeVariant: 'post' },
      { id: 'endpoint-health', label: 'Server Health', badge: 'GET', badgeVariant: 'get' },
      { id: 'endpoint-openapi', label: 'OpenAPI 3.0 Schema', badge: 'GET', badgeVariant: 'get' },
      { id: 'endpoint-docs', label: 'Swagger UI Docs', badge: 'GET', badgeVariant: 'get' },
    ],
  },
];

export const DOC_SECTIONS: Record<string, DocSection> = {
  'pillar-data': {
    id: 'pillar-data',
    category: 'The 3 Pillars of ModelKit',
    title: 'Pillar 1: Data (Dataset)',
    subtitle: 'Standardized dataset ingestion, schema validation, and partition contracts.',
    badge: { label: 'PILLAR 1', variant: 'pillar' },
    signatureOrPath: 'from modelkit import Dataset',
    breadcrumbs: ['3 Pillars', 'Data Pillar'],
    overview:
      'The Data pillar standardizes how raw data enters the machine learning lifecycle. Just like Django models.Model, defining an empty class does nothing at import time. When data or source is configured, Dataset provides automatic validation and partition contracts.',
    djangoAnalogy:
      'In Django, you subclass models.Model in models.py. An empty class does not load or touch the database. In ModelKit, you subclass Dataset in data.py. Zero side effects occur on import; when loaded, it validates and partitions your data.',
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
        description: 'If the class is empty and data/ has no files, `modelkit train` strictly halts with exit code 1.',
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
      python: `from modelkit import Dataset

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
    category: 'The 3 Pillars of ModelKit',
    title: 'Pillar 2: Model (Model)',
    subtitle: 'Framework-agnostic model wrapper with built-in pickle persistence.',
    badge: { label: 'PILLAR 2', variant: 'pillar' },
    signatureOrPath: 'from modelkit import Model',
    breadcrumbs: ['3 Pillars', 'Model Pillar'],
    overview:
      'The Model pillar standardizes execution and serialization across any machine learning framework (PyTorch, TensorFlow, Scikit-learn, XGBoost, or pure Python). It includes built-in pickle persistence (save/load) that directs model weights directly to the models/ folder.',
    djangoAnalogy:
      'In Django, Model instances have built-in .save() and .delete() database persistence. In ModelKit, Model instances have built-in .save() and .load() weight serialization into models/model.pkl.',
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
        description: 'Hyperparameters in mlkit.json are automatically bound to self.config during zero-path discovery.',
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
      python: `from modelkit import Model, Dataset

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
    category: 'The 3 Pillars of ModelKit',
    title: 'Pillar 3: Lifecycle (Trainer, Evaluator, Inference)',
    subtitle: 'Execution cycles with strict separation of weights and metadata.',
    badge: { label: 'PILLAR 3', variant: 'pillar' },
    signatureOrPath: 'from modelkit import BaseTrainer, BaseEvaluator, BaseInference',
    breadcrumbs: ['3 Pillars', 'Lifecycle Pillar'],
    overview:
      'The Lifecycle pillar governs training, evaluation, and live inference. It guarantees strict architectural hygiene: serialized model weights live strictly in models/, and experiment metadata snapshots live strictly in experiments/.',
    djangoAnalogy:
      'In Django, manage.py test and views handle the request-response lifecycle. In ModelKit, BaseTrainer, BaseEvaluator, and BaseInference handle the machine learning lifecycle.',
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
      python: `from modelkit import BaseTrainer, BaseEvaluator, BaseInference

class AITrainer(BaseTrainer):
    def fit(self, model, dataset, **kwargs):
        # Optimization loop here
        return {"status": "success", "loss": 0.035, "step": 100}

class AIEvaluator(BaseEvaluator):
    def evaluate(self, model, dataset, **kwargs):
        return {"accuracy": 0.965}

class AIInference(BaseInference):
    def run(self, model, raw_input, **kwargs):
        features = raw_input.get("features", [])
        return model.predict(features)`,
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
    signatureOrPath: 'from modelkit import BaseConfig',
    breadcrumbs: ['Foundations', 'BaseConfig'],
    overview:
      'BaseConfig parses mlkit.json and provides automatic hardware device resolution. It probes for NVIDIA CUDA GPUs, Apple Silicon MPS (Metal Performance Shaders), or gracefully falls back to CPU.',
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
        description: 'Resolved absolute paths to project directories defined in mlkit.json.',
      },
      {
        name: 'to_dict()',
        type: 'method',
        required: false,
        description: 'Exports active configuration dictionary for experiment tracking snapshots.',
      },
    ],
    snippets: {
      python: `from modelkit import BaseConfig

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
    signatureOrPath: 'from modelkit import get, get_all, register',
    breadcrumbs: ['Foundations', 'Registry'],
    overview:
      'The Registry automatically tracks every Model, Dataset, BaseTrainer, BaseEvaluator, BaseInference, and BaseConfig subclass under the hood via class extension hooks (__init_subclass__). Developers never need to write explicit @register decorators.',
    djangoAnalogy:
      "Just like Django's AppRegistry automatically discovers and registers every models.Model subclass on definition without decorators, ModelKit auto-registers your ML classes under the hood.",
    snippets: {
      python: `from modelkit import Model, Dataset, get, get_all

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
    title: 'modelkit init',
    subtitle: 'Scaffolds a new project directory with Django-style zero-code starter files.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'modelkit init <project_name>',
    breadcrumbs: ['CLI', 'init'],
    overview:
      'Scaffolds a new project directory with convention layout: data/ (strictly empty, no starter CSV), models/, experiments/, artifacts/, and starter files with comments only (no dummy code or fake math).',
    djangoAnalogy:
      'Direct equivalent of django-admin startproject <name>. Generates clean boilerplate with instructional comments.',
    snippets: {
      cli: 'modelkit init my_ai',
    },
    defaultPayload: '{\n  "command": "modelkit init",\n  "project": "my_ai"\n}',
    defaultResponse: {
      status: 'success',
      output: [
        "[OK] Successfully initialized MLKit project 'my_ai'",
        'Starter files: data.py, model.py, trainer.py, evaluator.py, inference.py',
        'Directory data/ created empty (no starter CSV provided)',
      ],
    },
  },

  'cli-install': {
    id: 'cli-install',
    category: 'Zero-Path CLI Commands',
    title: 'modelkit install',
    subtitle: 'Installs libraries safely into project .venv using uv (preferred) or pip.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run modelkit install <package_name ...> [--upgrade]',
    breadcrumbs: ['CLI', 'install'],
    overview:
      'Manages your project virtual environment (.venv). If .venv does not exist, automatically creates it. Uses uv pip install --python <venv> for lightning fast, conflict-free dependency installation.',
    djangoAnalogy:
      'Eliminates environment mismatches and global package collisions automatically.',
    snippets: {
      cli: 'uv run modelkit install torch torchvision pandas scikit-learn',
    },
    defaultPayload: '{\n  "packages": ["torch", "pandas"],\n  "upgrade": false\n}',
    defaultResponse: {
      status: 'success',
      venv_path: '.venv',
      installed: ['torch', 'pandas'],
      backend: 'uv pip install',
    },
  },

  'cli-train': {
    id: 'cli-train',
    category: 'Zero-Path CLI Commands',
    title: 'modelkit train',
    subtitle: 'Runs model training cycle through zero-path discovery.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run modelkit train',
    breadcrumbs: ['CLI', 'train'],
    overview:
      'Discovers and executes your training cycle. Strictly validates that classes are implemented and that data files exist in data/. If the user did nothing, it halts with error code 1.',
    djangoAnalogy:
      'Equivalent to running database migrations or test runner. Halts cleanly if models or data are missing.',
    snippets: {
      cli: 'uv run modelkit train',
    },
    defaultPayload: '{\n  "device": "auto"\n}',
    defaultResponse: {
      status: 'success',
      elapsed_time: '3.42s',
      model_weights: 'models/model.pkl',
      experiment_snapshot: 'experiments/experiment_snapshot.json',
      metrics: {
        loss: 0.042,
        accuracy: 0.984,
      },
    },
  },

  'cli-serve': {
    id: 'cli-serve',
    category: 'Zero-Path CLI Commands',
    title: 'modelkit serve',
    subtitle: 'Launches zero-path HTTP inference server with web playground & Swagger UI.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run modelkit serve [--port 8000]',
    breadcrumbs: ['CLI', 'serve'],
    overview:
      'Starts the built-in HTTP inference server on port 8000. Serves interactive web playground at GET / and GET /predict, Swagger documentation at GET /docs, OpenAPI specification at GET /openapi.json, and executes predictions at POST /predict.',
    djangoAnalogy:
      'Direct equivalent of python manage.py runserver 8000, with Swagger UI built in!',
    snippets: {
      cli: 'uv run modelkit serve --port 8000',
    },
    defaultPayload: '{\n  "port": 8000\n}',
    defaultResponse: {
      status: 'server_active',
      endpoints: {
        'GET /': 'Interactive Web Playground',
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
    title: 'modelkit data validate',
    subtitle: 'Verifies dataset schema, integrity, and partition readiness.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run modelkit data validate',
    breadcrumbs: ['CLI', 'data validate'],
    overview:
      'Executes Dataset.load() followed by Dataset.validate() and partition analysis. Outputs a clean schema verification report with zero emojis.',
    djangoAnalogy:
      'Equivalent to python manage.py check.',
    snippets: {
      cli: 'uv run modelkit data validate',
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
    title: 'modelkit evaluate',
    subtitle: 'Assesses model performance against held-out validation or test partitions.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run modelkit evaluate',
    breadcrumbs: ['CLI', 'evaluate'],
    overview:
      'Automatically discovers the latest checkpoint in models/ and executes BaseEvaluator.evaluate(model, dataset).',
    djangoAnalogy:
      'Equivalent to running python manage.py test.',
    snippets: {
      cli: 'uv run modelkit evaluate',
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
    title: 'modelkit doctor',
    subtitle: 'Inspects runtime environment health, hardware accelerators, and directory permissions.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'uv run modelkit doctor',
    breadcrumbs: ['CLI', 'doctor'],
    overview:
      'Diagnoses Python runtime version, hardware accelerators (CUDA/MPS/CPU), and directory read/write permissions for data/, models/, experiments/, and checkpoints/.',
    djangoAnalogy:
      'Comprehensive environment diagnostic check.',
    snippets: {
      cli: 'uv run modelkit doctor',
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
      'The primary inference view in ModelKit.',
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
        title: 'ModelKit API - my_ai',
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
      'Renders the embedded Swagger UI from modelkit/templates/swagger.html, backed by the live /openapi.json schema.',
    djangoAnalogy:
      'Interactive Swagger UI like Django REST Framework Swagger.',
    snippets: {
      curl: 'curl http://127.0.0.1:8000/docs',
    },
    defaultPayload: '{}',
    defaultResponse: {
      status: 'rendered_html',
      template: 'modelkit/templates/swagger.html',
    },
  },

  'paradigm-rag': {
    id: 'paradigm-rag',
    category: 'The 3 AI Paradigms',
    title: 'Paradigm 1: RAG (Retrieval-Augmented Generation)',
    subtitle: 'Ground foundation models in your private documents without retraining.',
    badge: { label: 'PARADIGM 1', variant: 'pillar' },
    signatureOrPath: 'from modelkit.rag import Document, RAGModel, MemoryVectorStore',
    breadcrumbs: ['Paradigms', 'RAG'],
    overview:
      'The RAG paradigm turns enterprise knowledge into actionable intelligence. ModelKit provides first-class Document loaders, text chunkers, zero-dependency embeddings, vector stores, and RAGModel. Because RAGModel subclasses Model, it works out of the box with modelkit evaluate and modelkit serve.',
    djangoAnalogy:
      'In Django, you query the database using ORM QuerySets. In ModelKit RAG, you query your knowledge base using BaseRetriever and VectorStores to inject context passages into generation prompts.',
    snippets: {
      python: `from modelkit import Document, RAGModel, MemoryVectorStore, VectorRetriever, TfidfEmbedding

# 1. Index your knowledge base
embedder = TfidfEmbedding()
store = MemoryVectorStore(embedding_fn=embedder)
docs = [
    Document(content="Refunds are processed within 14 business days.", metadata={"topic": "billing"}),
    Document(content="Remote work requires manager pre-approval.", metadata={"topic": "hr"})
]
store.add_documents(docs)

# 2. Bind Retriever to RAGModel
retriever = VectorRetriever(vector_store=store, embedding_fn=embedder)

class KnowledgeBot(RAGModel):
    pass

bot = KnowledgeBot("kb_bot", retriever=retriever)
result = bot.predict("What is the refund timeline?")
print(result["answer"])
print(result["sources"])
`,
    },
    defaultPayload: '{\n  "query": "What is the refund timeline?",\n  "top_k": 3\n}',
    defaultResponse: {
      query: "What is the refund timeline?",
      answer: "Based on billing: Refunds are processed within 14 business days...",
      sources: [
        {
          id: "doc-1",
          content: "Refunds are processed within 14 business days.",
          metadata: { topic: "billing" },
          score: 0.9412
        }
      ]
    },
  },

  'paradigm-adapters': {
    id: 'paradigm-adapters',
    category: 'The 3 AI Paradigms',
    title: 'Paradigm 2: Fine-Tuning (LoRA & Adapters)',
    subtitle: 'Parameter-efficient adaptation with lightweight delta checkpoints.',
    badge: { label: 'PARADIGM 2', variant: 'pillar' },
    signatureOrPath: 'from modelkit.adapters import AdapterConfig, AdapterModel, AdapterTrainer',
    breadcrumbs: ['Paradigms', 'Fine-Tuning'],
    overview:
      'Fine-tuning adapts a base foundation model to your specific domain using low-rank adapters (LoRA/QLoRA). AdapterModel freezes the base model and saves only lightweight delta weights (~50MB instead of 14GB), saving gigabytes of storage and bandwidth.',
    djangoAnalogy:
      'In Django, you use model inheritance or Proxy models to extend behavior without duplicating the database table. In ModelKit, AdapterModel attaches trainable delta matrices to a frozen base model.',
    snippets: {
      python: `from modelkit.adapters import AdapterConfig, AdapterModel, AdapterTrainer

# 1. Configure LoRA hyperparameters
config = AdapterConfig(
    r=16,
    alpha=32.0,
    target_modules=["q_proj", "v_proj"],
    base_model_path="meta-llama/Llama-3-8B"
)

# 2. Attach adapter to base foundation model
class DomainAdapter(AdapterModel):
    pass

model = DomainAdapter("customer_support_lora", adapter_config=config, base_model=base_llm)

# 3. Train adapter weights (base model remains frozen)
trainer = AdapterTrainer()
trainer.fit(model, dataset, epochs=3)

# 4. Saves ONLY the 50MB adapter weights!
model.save("models/adapter_checkpoint")
`,
    },
    defaultPayload: '{\n  "adapter_type": "lora",\n  "rank": 16,\n  "alpha": 32.0\n}',
    defaultResponse: {
      status: "trained",
      adapter_size: "48.2 MB",
      base_model_frozen: true,
      checkpoint: "models/adapter_checkpoint/adapter_model.pkl"
    },
  },

  'paradigm-scratch': {
    id: 'paradigm-scratch',
    category: 'The 3 AI Paradigms',
    title: 'Paradigm 3: Training from Scratch',
    subtitle: 'Bespoke neural architectures, full optimization loops, and custom weights.',
    badge: { label: 'PARADIGM 3', variant: 'pillar' },
    signatureOrPath: 'from modelkit import Model, Dataset, BaseTrainer',
    breadcrumbs: ['Paradigms', 'From Scratch'],
    overview:
      'Training from scratch provides full control over custom architectures, optimizers, learning rate schedules, and loss functions. Ideal for custom tabular, vision, or specialized neural models.',
    djangoAnalogy:
      'Writing custom Django ORM models and custom managers from scratch when generic solutions do not fit.',
    snippets: {
      python: `from modelkit import Model, Dataset, BaseTrainer

class CustomerDataset(Dataset):
    filename = "customers.csv"

class ChurnClassifier(Model):
    dataset = CustomerDataset

    def predict(self, inputs, **kwargs):
        return [0.85]  # Probability of churn

class ChurnTrainer(BaseTrainer):
    def fit(self, model, dataset, **kwargs):
        # Full training loop
        return {"status": "completed", "final_loss": 0.042}
`,
    },
    defaultPayload: '{\n  "epochs": 10,\n  "learning_rate": 0.001\n}',
    defaultResponse: {
      status: "completed",
      loss: 0.042,
      accuracy: 0.962
    },
  },

  'arch-multiapp': {
    id: 'arch-multiapp',
    category: 'Multi-App Architecture',
    title: 'Django-Style Multi-App Project Architecture',
    subtitle: 'Organize enterprise AI projects into modular, decoupled applications.',
    badge: { label: 'ARCHITECTURE', variant: 'util' },
    signatureOrPath: 'my_ai/ [mlkit.json, knowledge/, classifier/]',
    breadcrumbs: ['Architecture', 'Multi-App'],
    overview:
      'Just like a Django project consists of multiple apps (users/, blog/, billing/), an enterprise ModelKit project can contain multiple AI apps (knowledge/ for RAG, classifier/ for fine-tuning) sharing the same config and CLI lifecycle.',
    djangoAnalogy:
      'Direct equivalent of Django INSTALLED_APPS in settings.py: mlkit.json declares "apps": ["knowledge", "classifier"].',
    snippets: {
      cli: `# 1. Scaffold project
modelkit init enterprise_ai
cd enterprise_ai

# 2. Add modular AI apps
modelkit startapp knowledge      # RAG app
modelkit startapp classifier     # Classifier / fine-tune app
`,
    },
    defaultPayload: '{\n  "name": "enterprise_ai",\n  "apps": ["knowledge", "classifier"]\n}',
    defaultResponse: {
      project: "enterprise_ai",
      apps: ["knowledge", "classifier"],
      layout: "modular_multi_app"
    },
  },

  'cli-startapp': {
    id: 'cli-startapp',
    category: 'Zero-Path CLI Commands',
    title: 'modelkit startapp <app_name>',
    subtitle: 'Scaffolds a new modular AI application directory within the project.',
    badge: { label: 'CLI', variant: 'cli' },
    signatureOrPath: 'modelkit startapp <app_name>',
    breadcrumbs: ['CLI', 'startapp'],
    overview:
      'Scaffolds a new modular AI application directory with standard contracts (data.py, model.py, trainer.py, evaluator.py, inference.py) and registers the app in mlkit.json.',
    djangoAnalogy:
      'Direct equivalent of python manage.py startapp <app_name> in Django.',
    snippets: {
      cli: 'modelkit startapp knowledge',
    },
    defaultPayload: '{\n  "command": "modelkit startapp",\n  "app_name": "knowledge"\n}',
    defaultResponse: {
      status: "success",
      app_created: "knowledge",
      files: ["data.py", "model.py", "trainer.py", "evaluator.py", "inference.py"]
    },
  },
};

