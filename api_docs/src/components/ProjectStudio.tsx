import { useState, useMemo, useEffect } from 'react';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-bash';
import {
  Cpu,
  BookOpen,
  Sliders,
  Copy,
  Check,
  Folder,
  FileCode,
  Sparkles,
  Package,
  CheckCircle2,
  AlertCircle,
  Plus,
  X,
  ArrowRight,
} from 'lucide-react';

export type TemplateType = 'scratch' | 'rag' | 'adapter';
export type ChatProvider = 'openai' | 'gemini' | 'anthropic' | 'ollama' | 'local' | 'mock';
export type VectorDB = 'memory' | 'postgres';
export type EmbeddingEngine = 'sentence-transformers' | 'api' | 'tfidf';

interface ProjectStudioProps {
  onNavigateToDocs: (sectionId?: string) => void;
  initialParams?: {
    name?: string;
    template_type?: string;
    chat_provider?: string;
    vector_db?: string;
    embedding_engine?: string;
    model_name?: string;
    clean?: boolean;
    install?: boolean;
    no_venv?: boolean;
  };
}

interface InitResponse {
  status: 'success' | 'error';
  project_name?: string;
  dest_dir?: string;
  files?: string[];
  message?: string;
  dependencies?: string[];
  template?: string;
  commands?: string[];
}

const DEFAULT_MODELS: Record<ChatProvider, string> = {
  openai: 'gpt-4o-mini',
  gemini: 'gemini-1.5-flash',
  anthropic: 'claude-3-5-sonnet-20241022',
  ollama: 'llama3.2',
  local: 'meta-llama/Llama-3.2-3B',
  mock: 'mock-gpt',
};

const SUGGESTED_PACKAGES = [
  'pandas',
  'numpy',
  'scikit-learn',
  'torch',
  'transformers',
  'fastapi',
  'rich',
  'matplotlib',
];

export default function ProjectStudio({ onNavigateToDocs, initialParams }: ProjectStudioProps) {
  // Read URL params if available
  const urlParams = useMemo(() => {
    if (typeof window === 'undefined') return new URLSearchParams();
    return new URLSearchParams(window.location.search);
  }, []);

  // Form State
  const [projectName, setProjectName] = useState<string>(
    initialParams?.name || urlParams.get('name') || 'my_ai'
  );
  const [targetDir, setTargetDir] = useState<string>('.');
  const [templateType, setTemplateType] = useState<TemplateType>(
    (initialParams?.template_type as TemplateType) ||
      (urlParams.get('type') as TemplateType) ||
      'scratch'
  );

  // RAG Configs
  const [chatProvider, setChatProvider] = useState<ChatProvider>(
    (initialParams?.chat_provider as ChatProvider) || 'openai'
  );
  const [modelName, setModelName] = useState<string>(
    initialParams?.model_name || DEFAULT_MODELS.openai
  );
  const [vectorDb, setVectorDb] = useState<VectorDB>(
    (initialParams?.vector_db as VectorDB) || 'memory'
  );
  const [embeddingEngine, setEmbeddingEngine] = useState<EmbeddingEngine>(
    (initialParams?.embedding_engine as EmbeddingEngine) || 'sentence-transformers'
  );
  const [enableHyde, setEnableHyde] = useState<boolean>(true);
  const [enableSmartChunker, setEnableSmartChunker] = useState<boolean>(true);

  // Adapter Configs
  const [loraRank, setLoraRank] = useState<number>(8);
  const [loraAlpha, setLoraAlpha] = useState<number>(16);
  const [targetModules, setTargetModules] = useState<string[]>(['q_proj', 'v_proj']);
  const [newModuleInput, setNewModuleInput] = useState<string>('');
  const [baseModel, setBaseModel] = useState<string>('meta-llama/Llama-3.2-3B');

  // General Options
  const [clean, setClean] = useState<boolean>(initialParams?.clean || false);
  const [createVenv, setCreateVenv] = useState<boolean>(!(initialParams?.no_venv || false));
  const [installDeps, setInstallDeps] = useState<boolean>(initialParams?.install || false);
  const [extraPackages, setExtraPackages] = useState<string[]>([]);
  const [packageInput, setPackageInput] = useState<string>('');

  // UI state
  const [previewTab, setPreviewTab] = useState<'manifest' | 'structure' | 'cli'>('manifest');
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<InitResponse | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [serverStatus, setServerStatus] = useState<{ active: boolean; cwd?: string }>({
    active: true,
  });

  // Sync model default when chatProvider changes
  const handleChatProviderChange = (provider: ChatProvider) => {
    setChatProvider(provider);
    setModelName(DEFAULT_MODELS[provider]);
  };

  // Check backend server status
  useEffect(() => {
    fetch('/api/status')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          setServerStatus({ active: true, cwd: data.cwd });
          if (data.cwd) {
            setTargetDir((prev) => (prev === '.' ? data.cwd : prev));
          }
        }
      })
      .catch(() => {
        // Standalone or preview mode
        setServerStatus({ active: false });
      });
  }, []);

  // Sanitized package name
  const packageName = useMemo(() => {
    return (
      projectName
        .toLowerCase()
        .replace(/[^a-z0-9_]/g, '_')
        .replace(/^_+|_+$/g, '') || 'my_ai'
    );
  }, [projectName]);

  // Computed dependencies
  const computedDependencies = useMemo(() => {
    const deps: string[] = [];
    if (templateType === 'rag') {
      if (chatProvider === 'ollama') deps.push('ollama');
      else if (chatProvider === 'openai') deps.push('openai');
      else if (chatProvider === 'anthropic') deps.push('anthropic');
      else if (chatProvider === 'gemini') deps.push('google-genai');
      else if (chatProvider === 'local') deps.push('transformers', 'torch');

      if (embeddingEngine === 'sentence-transformers') {
        deps.push('sentence-transformers', 'numpy');
      } else if (chatProvider === 'ollama' && !deps.includes('ollama')) {
        deps.push('ollama');
      } else if (embeddingEngine === 'api' && chatProvider === 'openai' && !deps.includes('openai')) {
        deps.push('openai');
      }

      if (vectorDb === 'postgres') deps.push('psycopg2-binary');
    } else if (templateType === 'adapter') {
      deps.push('torch', 'transformers', 'peft', 'datasets', 'numpy');
    }

    for (const pkg of extraPackages) {
      if (!deps.includes(pkg)) deps.push(pkg);
    }
    return deps;
  }, [templateType, chatProvider, embeddingEngine, vectorDb, extraPackages]);

  // Computed aimlite.json manifest
  const computedManifest = useMemo(() => {
    const manifest: Record<string, any> = {
      name: projectName || 'my_ai',
      version: '0.1.0',
      entrypoint: packageName,
      template: templateType,
      clean,
      dependencies: computedDependencies,
      config: {
        device: 'auto',
        batch_size: 32,
      },
      paths: {
        data: 'data',
        models: 'models',
        experiments: 'experiments',
        artifacts: 'artifacts',
        checkpoints: 'checkpoints',
      },
    };

    if (templateType === 'rag') {
      manifest.config.rag = {
        chat_provider: chatProvider,
        model_name: modelName,
        embedding_model:
          embeddingEngine === 'sentence-transformers'
            ? 'all-MiniLM-L6-v2'
            : chatProvider === 'openai'
              ? 'text-embedding-3-small'
              : 'tfidf',
        vector_db: vectorDb,
        embedding_engine: embeddingEngine,
        enable_hyde: enableHyde,
        enable_smart_chunker: enableSmartChunker,
      };
    } else if (templateType === 'adapter') {
      manifest.config.adapter = {
        r: loraRank,
        alpha: loraAlpha,
        target_modules: targetModules,
        base_model: baseModel,
      };
    }

    return manifest;
  }, [
    projectName,
    packageName,
    templateType,
    clean,
    computedDependencies,
    chatProvider,
    modelName,
    embeddingEngine,
    vectorDb,
    enableHyde,
    enableSmartChunker,
    loraRank,
    loraAlpha,
    targetModules,
    baseModel,
  ]);

  // Computed CLI command equivalent
  const computedCliCommand = useMemo(() => {
    let cmd = `aimlite init ${projectName || 'my_ai'} --cli`;
    if (templateType !== 'scratch') {
      cmd += ` --type ${templateType}`;
    }
    if (clean) {
      cmd += ` --clean`;
    }
    if (!createVenv) {
      cmd += ` --no-venv`;
    }
    if (installDeps) {
      cmd += ` --install`;
    }

    if (templateType === 'rag') {
      if (chatProvider !== 'openai') cmd += ` --chat-provider ${chatProvider}`;
      if (modelName !== DEFAULT_MODELS[chatProvider]) cmd += ` --model ${modelName}`;
      if (vectorDb !== 'memory') cmd += ` --vector-db ${vectorDb}`;
      if (embeddingEngine !== 'sentence-transformers') cmd += ` --embedding ${embeddingEngine}`;
    }

    return cmd;
  }, [
    projectName,
    templateType,
    clean,
    createVenv,
    installDeps,
    chatProvider,
    modelName,
    vectorDb,
    embeddingEngine,
  ]);

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const handleAddPackage = (pkgToAdd?: string) => {
    const pkg = (pkgToAdd || packageInput).trim().toLowerCase();
    if (pkg && !extraPackages.includes(pkg)) {
      setExtraPackages((prev) => [...prev, pkg]);
      if (!pkgToAdd) setPackageInput('');
    }
  };

  const handleRemovePackage = (pkgToRemove: string) => {
    setExtraPackages((prev) => prev.filter((p) => p !== pkgToRemove));
  };

  const handleAddTargetModule = () => {
    const mod = newModuleInput.trim();
    if (mod && !targetModules.includes(mod)) {
      setTargetModules((prev) => [...prev, mod]);
      setNewModuleInput('');
    }
  };

  const handleRemoveTargetModule = (modToRemove: string) => {
    setTargetModules((prev) => prev.filter((m) => m !== modToRemove));
  };

  // Submit project creation
  const handleCreateProject = async () => {
    setIsSubmitting(true);
    setResult(null);

    const payload = {
      project_name: projectName || 'my_ai',
      target_dir: targetDir === '.' ? undefined : targetDir,
      template_type: templateType,
      chat_provider: templateType === 'rag' ? chatProvider : undefined,
      model_name: templateType === 'rag' ? modelName : undefined,
      vector_db: templateType === 'rag' ? vectorDb : undefined,
      embedding_engine: templateType === 'rag' ? embeddingEngine : undefined,
      clean,
      create_venv: createVenv,
      install_deps: installDeps,
      extra_packages: extraPackages,
      adapter_config:
        templateType === 'adapter'
          ? {
              r: loraRank,
              alpha: loraAlpha,
              target_modules: targetModules,
              base_model: baseModel,
            }
          : undefined,
    };

    try {
      const response = await fetch('/api/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || `Creation failed with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setResult({
        status: 'error',
        message:
          err.message ||
          'Failed to connect to local AIMLite studio server. Ensure "aimlite" or "aimlite init" is running in terminal.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8 animate-in fade-in duration-300">
      {/* Top Hero Banner */}
      <div className="relative rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-xl p-6 sm:p-8 shadow-xs overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-zinc-200/50 to-transparent dark:from-zinc-800/30 dark:to-transparent rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full text-[11px] font-mono font-medium border border-zinc-200 dark:border-zinc-800 bg-zinc-100/80 dark:bg-zinc-900/80 text-zinc-700 dark:text-zinc-300">
              <span className={`w-2 h-2 rounded-full ${serverStatus.active ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
              <span>AIMLite Studio • Local Web Mode</span>
              {serverStatus.cwd && (
                <>
                  <span className="text-zinc-400 dark:text-zinc-500">|</span>
                  <span className="text-zinc-500 dark:text-zinc-400 truncate max-w-xs">{serverStatus.cwd}</span>
                </>
              )}
            </div>

            <h1 className="text-2xl sm:text-3xl font-heading font-extrabold text-zinc-950 dark:text-white tracking-tight">
              Create New AIMLite Project
            </h1>
            <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
              Scaffold production AI and Machine Learning architectures with strict type contracts,
              instant .venv isolation, and zero-path model training and serving.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <button
              onClick={() => onNavigateToDocs('pillar-data')}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs font-medium border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 text-zinc-700 dark:text-zinc-300 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors shadow-xs"
            >
              <BookOpen size={14} className="text-zinc-400" />
              <span>Explore Docs</span>
            </button>
            <div className="px-3 py-2 rounded-xl text-xs font-mono font-semibold border border-zinc-200 dark:border-zinc-800 bg-zinc-100/50 dark:bg-zinc-900/50 text-zinc-600 dark:text-zinc-400">
              $ aimlite init --cli
            </div>
          </div>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Form: Steps & Controls (7 Cols) */}
        <div className="lg:col-span-7 space-y-6">
          {/* Step 1: Project Identity */}
          <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-5 sm:p-6 shadow-xs space-y-4">
            <div className="flex items-center gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
              <span className="flex items-center justify-center w-5 h-5 rounded-md bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 text-[11px] font-bold font-mono">
                1
              </span>
              <h2 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                Project Identity
              </h2>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Project Name
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={projectName}
                    onChange={(e) => setProjectName(e.target.value)}
                    placeholder="my_ai"
                    className="w-full px-3 py-2 rounded-xl text-xs font-mono bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:focus:ring-zinc-600 transition-all"
                  />
                </div>
                <div className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400 flex items-center gap-1.5">
                  <Package size={12} className="text-zinc-400" />
                  <span>Module:</span>
                  <code className="text-zinc-900 dark:text-zinc-200 font-semibold">
                    {packageName}
                  </code>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Target Directory
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={targetDir}
                    onChange={(e) => setTargetDir(e.target.value)}
                    placeholder="."
                    className="w-full px-3 py-2 rounded-xl text-xs font-mono bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:focus:ring-zinc-600 transition-all"
                  />
                </div>
                <div className="text-[11px] font-mono text-zinc-500 dark:text-zinc-400 flex items-center gap-1.5">
                  <Folder size={12} className="text-zinc-400" />
                  <span>Location:</span>
                  <span className="truncate max-w-[200px]" title={targetDir}>
                    {targetDir === '.' ? './' + (projectName || 'my_ai') : targetDir}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Step 2: Architecture Paradigm Selection */}
          <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-5 sm:p-6 shadow-xs space-y-4">
            <div className="flex items-center gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
              <span className="flex items-center justify-center w-5 h-5 rounded-md bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 text-[11px] font-bold font-mono">
                2
              </span>
              <h2 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                Select Architecture Paradigm
              </h2>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {/* Scratch / Custom ML */}
              <button
                type="button"
                onClick={() => setTemplateType('scratch')}
                className={`text-left p-3.5 rounded-xl border transition-all flex flex-col justify-between gap-3 ${
                  templateType === 'scratch'
                    ? 'border-zinc-900 dark:border-zinc-100 bg-zinc-900/5 dark:bg-white/5 ring-1 ring-zinc-900 dark:ring-white shadow-xs'
                    : 'border-zinc-200 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-900/30'
                }`}
              >
                <div className="flex items-start justify-between w-full">
                  <div className="p-2 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100">
                    <Cpu size={16} />
                  </div>
                  {templateType === 'scratch' && (
                    <CheckCircle2 size={16} className="text-zinc-900 dark:text-white shrink-0" />
                  )}
                </div>
                <div>
                  <h3 className="text-xs font-heading font-bold text-zinc-950 dark:text-white">
                    Scratch / Custom ML
                  </h3>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-1 line-clamp-2">
                    Tabular, supervised & PyTorch models with zero-path contracts.
                  </p>
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                    Tabular
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                    PyTorch
                  </span>
                </div>
              </button>

              {/* RAG Knowledge Base */}
              <button
                type="button"
                onClick={() => setTemplateType('rag')}
                className={`text-left p-3.5 rounded-xl border transition-all flex flex-col justify-between gap-3 ${
                  templateType === 'rag'
                    ? 'border-zinc-900 dark:border-zinc-100 bg-zinc-900/5 dark:bg-white/5 ring-1 ring-zinc-900 dark:ring-white shadow-xs'
                    : 'border-zinc-200 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-900/30'
                }`}
              >
                <div className="flex items-start justify-between w-full">
                  <div className="p-2 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100">
                    <BookOpen size={16} />
                  </div>
                  {templateType === 'rag' && (
                    <CheckCircle2 size={16} className="text-zinc-900 dark:text-white shrink-0" />
                  )}
                </div>
                <div>
                  <h3 className="text-xs font-heading font-bold text-zinc-950 dark:text-white">
                    RAG Knowledge Base
                  </h3>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-1 line-clamp-2">
                    Enterprise vector retrieval, grounded LLM synthesis & HyDE.
                  </p>
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                    HyDE
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                    Vectors
                  </span>
                </div>
              </button>

              {/* LoRA / Adapter Fine-Tuning */}
              <button
                type="button"
                onClick={() => setTemplateType('adapter')}
                className={`text-left p-3.5 rounded-xl border transition-all flex flex-col justify-between gap-3 ${
                  templateType === 'adapter'
                    ? 'border-zinc-900 dark:border-zinc-100 bg-zinc-900/5 dark:bg-white/5 ring-1 ring-zinc-900 dark:ring-white shadow-xs'
                    : 'border-zinc-200 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-900/30'
                }`}
              >
                <div className="flex items-start justify-between w-full">
                  <div className="p-2 rounded-lg bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100">
                    <Sliders size={16} />
                  </div>
                  {templateType === 'adapter' && (
                    <CheckCircle2 size={16} className="text-zinc-900 dark:text-white shrink-0" />
                  )}
                </div>
                <div>
                  <h3 className="text-xs font-heading font-bold text-zinc-950 dark:text-white">
                    LoRA Adapter Tuning
                  </h3>
                  <p className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-1 line-clamp-2">
                    Low-rank parameter adaptation (PEFT) & multi-adapter routing.
                  </p>
                </div>
                <div className="flex flex-wrap gap-1 mt-1">
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                    LoRA
                  </span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400">
                    PEFT
                  </span>
                </div>
              </button>
            </div>
          </div>

          {/* Step 3: Paradigm Specific Configs */}
          {templateType === 'rag' && (
            <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-5 sm:p-6 shadow-xs space-y-4">
              <div className="flex items-center gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
                <span className="flex items-center justify-center w-5 h-5 rounded-md bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 text-[11px] font-bold font-mono">
                  3
                </span>
                <h2 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                  RAG Architecture Settings
                </h2>
              </div>

              {/* Chat Provider selection */}
              <div className="space-y-2">
                <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Chat Provider for LLM Synthesis
                </label>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {(
                    [
                      { id: 'openai', label: 'OpenAI', desc: 'GPT-4o, o1, mini' },
                      { id: 'gemini', label: 'Google Gemini', desc: 'gemini-1.5-flash' },
                      { id: 'anthropic', label: 'Anthropic', desc: 'claude-3-5' },
                      { id: 'ollama', label: 'Ollama Local', desc: 'llama3.2, mistral' },
                      { id: 'local', label: 'Local Pipeline', desc: 'Hugging Face' },
                      { id: 'mock', label: 'Mock Provider', desc: 'Zero dependencies' },
                    ] as const
                  ).map((p) => (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => handleChatProviderChange(p.id)}
                      className={`text-left p-2.5 rounded-xl border text-xs transition-all ${
                        chatProvider === p.id
                          ? 'border-zinc-900 dark:border-white bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold'
                          : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40 text-zinc-700 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-zinc-700'
                      }`}
                    >
                      <div className="font-heading font-bold">{p.label}</div>
                      <div
                        className={`text-[10px] truncate ${
                          chatProvider === p.id
                            ? 'text-zinc-300 dark:text-zinc-600'
                            : 'text-zinc-400 dark:text-zinc-500'
                        }`}
                      >
                        {p.desc}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Model identifier */}
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Model Identifier
                </label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder={DEFAULT_MODELS[chatProvider]}
                  className="w-full px-3 py-2 rounded-xl text-xs font-mono bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:focus:ring-zinc-600 transition-all"
                />
              </div>

              {/* Vector DB & Embedding Engine */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    Vector Database Backend
                  </label>
                  <div className="space-y-1.5">
                    {[
                      { id: 'memory', label: 'In-Memory (JSON)', desc: 'Zero external database' },
                      { id: 'postgres', label: 'PostgreSQL + pgvector', desc: 'Enterprise SQL ORM' },
                    ].map((db) => (
                      <button
                        key={db.id}
                        type="button"
                        onClick={() => setVectorDb(db.id as VectorDB)}
                        className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all flex items-center justify-between ${
                          vectorDb === db.id
                            ? 'border-zinc-900 dark:border-white bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold'
                            : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40 text-zinc-700 dark:text-zinc-300'
                        }`}
                      >
                        <div>
                          <div>{db.label}</div>
                          <div
                            className={`text-[10px] ${
                              vectorDb === db.id
                                ? 'text-zinc-300 dark:text-zinc-600'
                                : 'text-zinc-400 dark:text-zinc-500'
                            }`}
                          >
                            {db.desc}
                          </div>
                        </div>
                        {vectorDb === db.id && <Check size={14} />}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    Embedding Engine
                  </label>
                  <div className="space-y-1.5">
                    {[
                      {
                        id: 'sentence-transformers',
                        label: 'SentenceTransformers',
                        desc: 'all-MiniLM-L6-v2',
                      },
                      { id: 'api', label: 'Native Provider API', desc: 'Embeddings via API' },
                      { id: 'tfidf', label: 'Pure Hash TF-IDF', desc: 'Zero dependency baseline' },
                    ].map((emb) => (
                      <button
                        key={emb.id}
                        type="button"
                        onClick={() => setEmbeddingEngine(emb.id as EmbeddingEngine)}
                        className={`w-full text-left p-2 rounded-xl border text-xs transition-all flex items-center justify-between ${
                          embeddingEngine === emb.id
                            ? 'border-zinc-900 dark:border-white bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold'
                            : 'border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/40 text-zinc-700 dark:text-zinc-300'
                        }`}
                      >
                        <div>
                          <div>{emb.label}</div>
                          <div
                            className={`text-[10px] ${
                              embeddingEngine === emb.id
                                ? 'text-zinc-300 dark:text-zinc-600'
                                : 'text-zinc-400 dark:text-zinc-500'
                            }`}
                          >
                            {emb.desc}
                          </div>
                        </div>
                        {embeddingEngine === emb.id && <Check size={14} />}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Feature Toggles */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-zinc-100 dark:border-zinc-800/80">
                <label className="flex items-center gap-2 cursor-pointer text-xs text-zinc-700 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={enableHyde}
                    onChange={(e) => setEnableHyde(e.target.checked)}
                    className="rounded border-zinc-300 dark:border-zinc-700 text-zinc-900 focus:ring-zinc-500"
                  />
                  <span>Enable HyDE (Hypothetical Document Embeddings)</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-xs text-zinc-700 dark:text-zinc-300">
                  <input
                    type="checkbox"
                    checked={enableSmartChunker}
                    onChange={(e) => setEnableSmartChunker(e.target.checked)}
                    className="rounded border-zinc-300 dark:border-zinc-700 text-zinc-900 focus:ring-zinc-500"
                  />
                  <span>Enable Smart Semantic Chunker</span>
                </label>
              </div>
            </div>
          )}

          {templateType === 'adapter' && (
            <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-5 sm:p-6 shadow-xs space-y-4">
              <div className="flex items-center gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
                <span className="flex items-center justify-center w-5 h-5 rounded-md bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 text-[11px] font-bold font-mono">
                  3
                </span>
                <h2 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                  LoRA Hyperparameters & Architecture
                </h2>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    Rank r: <span className="font-mono font-bold">{loraRank}</span>
                  </label>
                  <div className="flex items-center gap-1.5">
                    {[4, 8, 16, 32, 64].map((rVal) => (
                      <button
                        key={rVal}
                        type="button"
                        onClick={() => setLoraRank(rVal)}
                        className={`flex-1 py-1.5 rounded-lg text-xs font-mono font-semibold border transition-all ${
                          loraRank === rVal
                            ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 border-zinc-900 dark:border-white'
                            : 'bg-zinc-50 dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300'
                        }`}
                      >
                        {rVal}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    Alpha α: <span className="font-mono font-bold">{loraAlpha}</span>
                  </label>
                  <div className="flex items-center gap-1.5">
                    {[8, 16, 32, 64].map((aVal) => (
                      <button
                        key={aVal}
                        type="button"
                        onClick={() => setLoraAlpha(aVal)}
                        className={`flex-1 py-1.5 rounded-lg text-xs font-mono font-semibold border transition-all ${
                          loraAlpha === aVal
                            ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 border-zinc-900 dark:border-white'
                            : 'bg-zinc-50 dark:bg-zinc-900 border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300'
                        }`}
                      >
                        {aVal}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Base Model Identifier
                </label>
                <input
                  type="text"
                  value={baseModel}
                  onChange={(e) => setBaseModel(e.target.value)}
                  placeholder="meta-llama/Llama-3.2-3B"
                  className="w-full px-3 py-2 rounded-xl text-xs font-mono bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:focus:ring-zinc-600 transition-all"
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                  Target Attention Modules
                </label>
                <div className="flex flex-wrap gap-1.5 items-center">
                  {targetModules.map((mod) => (
                    <span
                      key={mod}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-mono bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-900 dark:text-zinc-100"
                    >
                      {mod}
                      <button
                        type="button"
                        onClick={() => handleRemoveTargetModule(mod)}
                        className="text-zinc-400 hover:text-red-500 transition-colors ml-0.5"
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                  <div className="flex items-center gap-1">
                    <input
                      type="text"
                      value={newModuleInput}
                      onChange={(e) => setNewModuleInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault();
                          handleAddTargetModule();
                        }
                      }}
                      placeholder="+ add module"
                      className="px-2 py-1 rounded-lg text-xs font-mono bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-900 dark:text-zinc-100 w-28 focus:outline-none focus:ring-1 focus:ring-zinc-500"
                    />
                    <button
                      type="button"
                      onClick={handleAddTargetModule}
                      className="p-1 rounded-md bg-zinc-200 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 hover:text-zinc-950 dark:hover:text-white"
                    >
                      <Plus size={14} />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Code Generation Mode (--clean) */}
          <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-5 sm:p-6 shadow-xs space-y-4">
            <div className="flex items-center gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
              <span className="flex items-center justify-center w-5 h-5 rounded-md bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 text-[11px] font-bold font-mono">
                4
              </span>
              <h2 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                Code Generation Mode (--clean)
              </h2>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => setClean(false)}
                className={`text-left p-3.5 rounded-xl border transition-all ${
                  !clean
                    ? 'border-zinc-900 dark:border-zinc-100 bg-zinc-900/5 dark:bg-white/5 ring-1 ring-zinc-900 dark:ring-white shadow-xs'
                    : 'border-zinc-200 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-900/30'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="text-xs font-heading font-bold text-zinc-950 dark:text-white">
                    Starter Boilerplate
                  </div>
                  {!clean && <CheckCircle2 size={15} className="text-zinc-900 dark:text-white" />}
                </div>
                <div className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-1">
                  Includes working starter model, data pipelines, client script, and sample data.
                </div>
              </button>

              <button
                type="button"
                onClick={() => setClean(true)}
                className={`text-left p-3.5 rounded-xl border transition-all ${
                  clean
                    ? 'border-zinc-900 dark:border-zinc-100 bg-zinc-900/5 dark:bg-white/5 ring-1 ring-zinc-900 dark:ring-white shadow-xs'
                    : 'border-zinc-200 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700 bg-zinc-50/50 dark:bg-zinc-900/30'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="text-xs font-heading font-bold text-zinc-950 dark:text-white">
                    Clean Skeleton (--clean)
                  </div>
                  {clean && <CheckCircle2 size={15} className="text-zinc-900 dark:text-white" />}
                </div>
                <div className="text-[11px] text-zinc-500 dark:text-zinc-400 mt-1">
                  Provisions 0-byte empty code files with class contracts only and zero sample data.
                </div>
              </button>
            </div>
          </div>

          {/* Step 5: Extra Packages Search (CLI Step 3b) */}
          <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-5 sm:p-6 shadow-xs space-y-4">
            <div className="flex items-center gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
              <span className="flex items-center justify-center w-5 h-5 rounded-md bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 text-[11px] font-bold font-mono">
                5
              </span>
              <h2 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                Add Extra Packages (pip)
              </h2>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={packageInput}
                  onChange={(e) => setPackageInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      handleAddPackage();
                    }
                  }}
                  placeholder="e.g. pandas, scikit-learn, fastapi, rich..."
                  className="flex-1 px-3 py-2 rounded-xl text-xs font-mono bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-900 dark:text-zinc-100 placeholder:text-zinc-400 focus:outline-none focus:ring-2 focus:ring-zinc-400 dark:focus:ring-zinc-600 transition-all"
                />
                <button
                  type="button"
                  onClick={() => handleAddPackage()}
                  className="px-3.5 py-2 rounded-xl text-xs font-medium bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 hover:opacity-90 transition-opacity shrink-0"
                >
                  Add Package
                </button>
              </div>

              {/* Quick Suggestion Chips */}
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                <span className="text-[10px] font-mono text-zinc-400 dark:text-zinc-500">
                  Quick Add:
                </span>
                {SUGGESTED_PACKAGES.map((pkg) => {
                  const isAdded =
                    computedDependencies.includes(pkg) || extraPackages.includes(pkg);
                  return (
                    <button
                      key={pkg}
                      type="button"
                      disabled={isAdded}
                      onClick={() => handleAddPackage(pkg)}
                      className={`px-2 py-0.5 rounded-md text-[10px] font-mono transition-colors ${
                        isAdded
                          ? 'bg-zinc-200/50 dark:bg-zinc-800/50 text-zinc-400 cursor-not-allowed'
                          : 'bg-zinc-100 dark:bg-zinc-800/80 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-700'
                      }`}
                    >
                      +{pkg}
                    </button>
                  );
                })}
              </div>

              {/* Added extra packages list */}
              {extraPackages.length > 0 && (
                <div className="flex flex-wrap gap-1.5 pt-2">
                  {extraPackages.map((pkg) => (
                    <span
                      key={pkg}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-mono bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-900 dark:text-zinc-100"
                    >
                      {pkg}
                      <button
                        type="button"
                        onClick={() => handleRemovePackage(pkg)}
                        className="text-zinc-400 hover:text-red-500 transition-colors ml-0.5"
                      >
                        <X size={12} />
                      </button>
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Step 6: Virtual Environment & Dependency Installation */}
          <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-md p-5 sm:p-6 shadow-xs space-y-4">
            <div className="flex items-center gap-2 pb-3 border-b border-zinc-100 dark:border-zinc-800/80">
              <span className="flex items-center justify-center w-5 h-5 rounded-md bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 text-[11px] font-bold font-mono">
                6
              </span>
              <h2 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                Virtual Environment & Dependency Installation
              </h2>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="flex items-start gap-2.5 p-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30 cursor-pointer">
                <input
                  type="checkbox"
                  checked={createVenv}
                  onChange={(e) => setCreateVenv(e.target.checked)}
                  className="mt-0.5 rounded border-zinc-300 dark:border-zinc-700 text-zinc-900 focus:ring-zinc-500"
                />
                <div className="space-y-0.5">
                  <div className="text-xs font-medium text-zinc-900 dark:text-zinc-100">
                    Create Isolated .venv
                  </div>
                  <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                    Provisions an isolated local virtual environment for project modules
                  </div>
                </div>
              </label>

              <label className="flex items-start gap-2.5 p-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30 cursor-pointer">
                <input
                  type="checkbox"
                  checked={installDeps}
                  onChange={(e) => setInstallDeps(e.target.checked)}
                  className="mt-0.5 rounded border-zinc-300 dark:border-zinc-700 text-zinc-900 focus:ring-zinc-500"
                />
                <div className="space-y-0.5">
                  <div className="text-xs font-medium text-zinc-900 dark:text-zinc-100">
                    Install Dependencies Now (--install)
                  </div>
                  <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                    Installs requirements into .venv immediately instead of deferring
                  </div>
                </div>
              </label>
            </div>
          </div>

          {/* Primary Create Button */}
          <div className="pt-2">
            <button
              type="button"
              disabled={isSubmitting}
              onClick={handleCreateProject}
              className="w-full py-3.5 px-6 rounded-2xl bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 font-heading font-extrabold text-sm shadow-xl hover:scale-[1.01] active:scale-[0.99] disabled:opacity-50 disabled:pointer-events-none transition-all flex items-center justify-center gap-2.5"
            >
              {isSubmitting ? (
                <>
                  <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  <span>Scaffolding Architecture...</span>
                </>
              ) : (
                <>
                  <Sparkles size={16} />
                  <span>Create Project ({projectName || 'my_ai'})</span>
                </>
              )}
            </button>
          </div>

          {/* Result Alert / Celebration Box */}
          {result && (
            <div
              className={`rounded-2xl border p-5 sm:p-6 transition-all space-y-4 ${
                result.status === 'success'
                  ? 'border-emerald-500/30 bg-emerald-500/5 dark:bg-emerald-500/10'
                  : 'border-red-500/30 bg-red-500/5 dark:bg-red-500/10'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2.5">
                  {result.status === 'success' ? (
                    <div className="p-2 rounded-xl bg-emerald-500 text-white">
                      <CheckCircle2 size={18} />
                    </div>
                  ) : (
                    <div className="p-2 rounded-xl bg-red-500 text-white">
                      <AlertCircle size={18} />
                    </div>
                  )}
                  <div>
                    <h3 className="text-sm font-heading font-bold text-zinc-950 dark:text-white">
                      {result.status === 'success'
                        ? 'Project Initialized Successfully!'
                        : 'Initialization Error'}
                    </h3>
                    <p className="text-xs text-zinc-600 dark:text-zinc-400 mt-0.5">
                      {result.status === 'success'
                        ? `Scaffolded ${result.template} architecture in ${result.dest_dir || './' + (result.project_name || projectName)}`
                        : result.message}
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setResult(null)}
                  className="p-1 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
                >
                  <X size={16} />
                </button>
              </div>

              {result.status === 'success' && (
                <div className="space-y-3 pt-2 border-t border-emerald-500/20">
                  <div className="text-xs font-semibold text-zinc-900 dark:text-zinc-100">
                    Next Steps:
                  </div>
                  <div className="space-y-1.5">
                    {(
                      result.commands || [
                        `cd ${result.project_name || projectName || 'my_ai'}`,
                        `aimlite train`,
                        `aimlite serve --port 8000`,
                      ]
                    ).map((cmd, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-2.5 rounded-xl bg-zinc-900 text-zinc-100 font-mono text-xs border border-zinc-800"
                      >
                        <code>$ {cmd}</code>
                        <button
                          type="button"
                          onClick={() => handleCopy(cmd, `cmd-${idx}`)}
                          className="p-1 rounded-md text-zinc-400 hover:text-white transition-colors"
                          title="Copy command"
                        >
                          {copiedKey === `cmd-${idx}` ? (
                            <Check size={13} className="text-emerald-400" />
                          ) : (
                            <Copy size={13} />
                          )}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Live Dynamic Preview Pane (5 Cols, Sticky) */}
        <div className="lg:col-span-5 sticky top-20 space-y-4">
          <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-xl shadow-xs overflow-hidden">
            {/* Header with Switcher Tabs */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200/80 dark:border-zinc-800/80 bg-zinc-50/50 dark:bg-zinc-900/50">
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() => setPreviewTab('manifest')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
                    previewTab === 'manifest'
                      ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 shadow-xs'
                      : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  aimlite.json
                </button>
                <button
                  type="button"
                  onClick={() => setPreviewTab('structure')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
                    previewTab === 'structure'
                      ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 shadow-xs'
                      : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  File Tree
                </button>
                <button
                  type="button"
                  onClick={() => setPreviewTab('cli')}
                  className={`px-2.5 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
                    previewTab === 'cli'
                      ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 shadow-xs'
                      : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                  }`}
                >
                  CLI Equiv
                </button>
              </div>

              <div className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-[10px] font-mono text-zinc-400">Live</span>
              </div>
            </div>

            {/* Tab 1: Manifest JSON */}
            {previewTab === 'manifest' && (
              <div className="relative p-4 max-h-[500px] overflow-y-auto font-mono text-xs">
                <button
                  type="button"
                  onClick={() =>
                    handleCopy(JSON.stringify(computedManifest, null, 2), 'manifest-json')
                  }
                  className="absolute top-3 right-3 p-1.5 rounded-lg border border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-900/80 text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-100 transition-colors shadow-xs"
                  title="Copy aimlite.json"
                >
                  {copiedKey === 'manifest-json' ? (
                    <Check size={13} className="text-emerald-500" />
                  ) : (
                    <Copy size={13} />
                  )}
                </button>

                <pre className="text-zinc-800 dark:text-zinc-200 leading-relaxed overflow-x-auto">
                  <code>{JSON.stringify(computedManifest, null, 2)}</code>
                </pre>
              </div>
            )}

            {/* Tab 2: File Structure Tree */}
            {previewTab === 'structure' && (
              <div className="p-4 max-h-[500px] overflow-y-auto font-mono text-xs space-y-1 text-zinc-700 dark:text-zinc-300">
                <div className="flex items-center gap-1.5 font-bold text-zinc-950 dark:text-white">
                  <Folder size={14} className="text-zinc-400" />
                  <span>{projectName || 'my_ai'}/</span>
                </div>

                <div className="pl-4 space-y-1">
                  <div className="flex items-center gap-1.5 text-zinc-500">
                    <Folder size={13} />
                    <span>data/</span>
                    <span className="text-[10px] text-zinc-400">(raw & processed partitions)</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-zinc-500">
                    <Folder size={13} />
                    <span>models/</span>
                    <span className="text-[10px] text-zinc-400">(serialized checkpoints)</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-zinc-500">
                    <Folder size={13} />
                    <span>experiments/</span>
                    <span className="text-[10px] text-zinc-400">(benchmarks & metrics)</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-zinc-500">
                    <Folder size={13} />
                    <span>artifacts/</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-zinc-500">
                    <Folder size={13} />
                    <span>checkpoints/</span>
                  </div>
                  {createVenv && (
                    <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
                      <Folder size={13} />
                      <span>.venv/</span>
                      <span className="text-[10px] opacity-75">(isolated python venv)</span>
                    </div>
                  )}

                  {/* Root files */}
                  <div className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300">
                    <FileCode size={13} className="text-zinc-400" />
                    <span>aimlite.json</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300">
                    <FileCode size={13} className="text-zinc-400" />
                    <span>README.md</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300">
                    <FileCode size={13} className="text-zinc-400" />
                    <span>.gitignore</span>
                  </div>
                  <div className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300">
                    <FileCode size={13} className="text-zinc-400" />
                    <span>.env.example</span>
                  </div>

                  {/* Python package directory */}
                  <div className="pt-2">
                    <div className="flex items-center gap-1.5 font-semibold text-zinc-900 dark:text-zinc-100">
                      <Folder size={13} className="text-zinc-400" />
                      <span>{packageName}/</span>
                    </div>
                    <div className="pl-4 space-y-1 pt-1 text-zinc-600 dark:text-zinc-400">
                      <div className="flex items-center gap-1.5">
                        <FileCode size={12} />
                        <span>__init__.py</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <FileCode size={12} />
                        <span>config.py</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <FileCode size={12} />
                        <span>data.py</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <FileCode size={12} />
                        <span>model.py</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <FileCode size={12} />
                        <span>trainer.py</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <FileCode size={12} />
                        <span>evaluator.py</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <FileCode size={12} />
                        <span>inference.py</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 3: CLI Equivalent Command */}
            {previewTab === 'cli' && (
              <div className="p-4 space-y-3 font-mono text-xs">
                <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                  Only the flag{' '}
                  <code className="text-zinc-900 dark:text-zinc-200 font-bold">--cli</code> starts
                  the creation steps from the command-line interface. You can run this exact
                  command:
                </p>

                <div className="relative p-3 rounded-xl bg-zinc-900 text-zinc-100 border border-zinc-800 leading-relaxed break-all">
                  <button
                    type="button"
                    onClick={() => handleCopy(computedCliCommand, 'cli-command')}
                    className="absolute top-2.5 right-2.5 p-1 rounded-md text-zinc-400 hover:text-white transition-colors"
                    title="Copy command"
                  >
                    {copiedKey === 'cli-command' ? (
                      <Check size={13} className="text-emerald-400" />
                    ) : (
                      <Copy size={13} />
                    )}
                  </button>
                  <code>$ {computedCliCommand}</code>
                </div>

                <div className="p-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/50 dark:bg-zinc-900/30 text-[11px] text-zinc-600 dark:text-zinc-400 space-y-1">
                  <div className="font-semibold text-zinc-900 dark:text-zinc-200">
                    Pro-tip: Interactive Mode vs Web
                  </div>
                  <div>
                    Running <code className="text-zinc-900 dark:text-zinc-300">aimlite init</code>{' '}
                    launches this Web Studio. Running{' '}
                    <code className="text-zinc-900 dark:text-zinc-300">aimlite init --cli</code>{' '}
                    starts the interactive prompt wizard in your terminal!
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Quick Pillar & Paradigm Links */}
          <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-950/60 backdrop-blur-xl p-4 shadow-xs space-y-2 text-xs">
            <div className="font-heading font-bold text-zinc-900 dark:text-zinc-100">
              Need Architectural Guidance?
            </div>
            <div className="space-y-1 text-zinc-600 dark:text-zinc-400">
              <button
                type="button"
                onClick={() => onNavigateToDocs('paradigm-rag')}
                className="w-full text-left p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900 flex items-center justify-between transition-colors group"
              >
                <span>Read Enterprise RAG Architecture Guide</span>
                <ArrowRight
                  size={12}
                  className="group-hover:translate-x-0.5 transition-transform"
                />
              </button>
              <button
                type="button"
                onClick={() => onNavigateToDocs('paradigm-adapter')}
                className="w-full text-left p-2 rounded-lg hover:bg-zinc-100 dark:hover:bg-zinc-900 flex items-center justify-between transition-colors group"
              >
                <span>Read LoRA Fine-Tuning Guide</span>
                <ArrowRight
                  size={12}
                  className="group-hover:translate-x-0.5 transition-transform"
                />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
