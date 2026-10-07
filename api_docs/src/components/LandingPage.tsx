import { useState, useMemo, useEffect, useRef } from 'react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import {
  Terminal,
  ArrowRight,
  Copy,
  Check,
  Cpu,
  Database,
  RefreshCw,
  BookOpen,
  Layers,
  Server,
  ShieldCheck,
  Rocket,
  Code2,
  ExternalLink,
  CheckCircle2,
  Sliders,
  Boxes,
} from 'lucide-react';

const GithubIcon = ({ size = 16, className = '' }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
    />
  </svg>
);

interface LandingPageProps {
  onNavigateToDocs: (sectionId?: string) => void;
}

const RAG_SNIPPET = `from aimlite.rag import KnowledgeModel, Document

class SupportDocRAG(KnowledgeModel):
    """Enterprise Knowledge Base with modular extension hooks."""

    def preprocess_query(self, query: str) -> str:
        # Hook: Normalize query and expand acronyms
        return query.strip().replace("MFA", "Multi-Factor Authentication")

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        # Hook: Filter candidates with cosine similarity >= 0.4
        return [d for d in documents if (d.score or 0) >= 0.4]

    def postprocess_answer(self, answer: str, context_docs: list[Document]) -> str:
        # Hook: Append verified source citations
        sources = {d.metadata.get("source") for d in context_docs}
        return f"{answer}\\n\\n[Verified Sources: {', '.join(sources)}]"

# Forward inference handles preprocessing, retrieval, reranking, synthesis:
# response = model.predict("How does MFA work?", top_k=3)`;

const ADAPTER_SNIPPET = `from aimlite.adapters import AdapterModel, MultiAdapterManager

# Initialize base foundation model with low-rank adapter
model = AdapterModel(base_model, r=8, lora_alpha=16)

# Parameter efficiency diagnostics:
model.print_trainable_parameters()
# Output: Trainable: 294,912 / 6,738,415,616 (0.0044% memory)

# Zero-overhead inference via weight merging:
model.merge_weights()
predictions = model.predict(inputs)
model.unmerge_weights()  # Revert for dynamic adapter swapping

# Multi-adapter registry hosting multiple tenant models:
manager = MultiAdapterManager(base_model)
manager.add_adapter("billing", adapter_billing)
manager.add_adapter("support", adapter_support)
manager.set_active_adapter("billing")`;

const SCRATCH_SNIPPET = `from aimlite import Dataset, Model, BaseTrainer

class CustomerChurnDataset(Dataset):
    filename = "telecom_churn.csv"  # Target file in data/

class ChurnModel(Model):
    def fit(self, X, y):
        # Implement scikit-learn, PyTorch, or custom logic
        self.weights = train_classifier(X, y)
        return self

    def predict(self, inputs):
        return {"churn_probability": calculate_score(inputs)}

# Train and serve via zero-path commands:
# $ aimlite train ChurnModel
# $ aimlite serve ChurnModel --port 8000`;

const renderTerminalLine = (line: string, idx: number) => {
  if (line.startsWith('$ ')) {
    return (
      <div key={idx} className="flex items-center gap-1.5 text-zinc-100 font-semibold py-0.5">
        <span className="text-zinc-500 select-none">$</span>
        <span className="text-zinc-100 font-bold">{line.slice(2)}</span>
      </div>
    );
  }
  // ASCII Figlet art: Vibrant Electric Bright Cyan (#00d8ff) matching CLI big_header
  if (
    line.includes('.d8b.') ||
    line.includes('d888888b') ||
    line.includes("d8'") ||
    line.includes('88ooo88') ||
    line.includes('88~~~88') ||
    line.includes('88booo') ||
    line.includes('Y88888P') ||
    line.includes('YP   YP')
  ) {
    return (
      <div
        key={idx}
        className="text-[#00d8ff] font-extrabold select-none whitespace-pre tracking-normal leading-[1.15] font-mono text-[7px] min-[360px]:text-[8px] min-[400px]:text-[9.5px] sm:text-[12px] md:text-[13px] drop-shadow-[0_0_12px_rgba(0,216,255,0.4)]"
      >
        {line}
      </div>
    );
  }
  // Vite Header title in blue
  if (line.trim().startsWith('AIMLITE v1.')) {
    const parts = line.trim().split(/\s+/);
    return (
      <div key={idx} className="flex items-center gap-2 py-1 text-xs font-mono">
        <span className="text-[#00d8ff] font-extrabold tracking-wider">AIMLITE</span>
        <span className="text-zinc-500">{parts[1] || 'v2.1.0'}</span>
        <span className="text-emerald-400 font-semibold">{parts[2]}</span>
        <span className="text-zinc-400">{parts.slice(3).join(' ')}</span>
      </div>
    );
  }
  if (line.includes('The Django for AI & Machine Learning')) {
    return (
      <div key={idx} className="flex items-center gap-2 pt-1 pb-0.5 text-xs font-mono">
        <span className="text-zinc-500">v2.1.0</span>
        <span className="text-[#00d8ff] font-bold">❯</span>
        <span className="text-zinc-100 font-bold">The Django for AI & Machine Learning</span>
      </div>
    );
  }
  if (line.trim() === 'create project') {
    return (
      <div key={idx} className="text-emerald-400 font-semibold pb-1.5 font-mono text-xs">
        {line}
      </div>
    );
  }
  if (line.startsWith('✔ ') || line.includes('✔ ')) {
    return (
      <div key={idx} className="flex items-center gap-1.5 text-zinc-200">
        <span className="text-emerald-400 font-bold select-none">✔</span>
        <span className="text-emerald-300 font-medium">{line.replace(/^.*✔\s*/, '')}</span>
      </div>
    );
  }
  if (line.startsWith('❯ ')) {
    return (
      <div key={idx} className="flex items-center gap-1.5 text-[#00d8ff]">
        <span className="text-[#00d8ff] font-bold select-none">❯</span>
        <span className="text-[#00d8ff] font-bold">{line.slice(2)}</span>
      </div>
    );
  }
  if (line.startsWith('? ')) {
    return (
      <div key={idx} className="text-zinc-100 font-semibold pt-1">
        <span className="text-[#00d8ff] font-bold select-none">? </span>
        <span className="text-white font-bold">{line.slice(2)}</span>
      </div>
    );
  }
  if (line.startsWith('INFO:')) {
    return (
      <div key={idx} className="text-zinc-400">
        <span className="text-blue-400 font-semibold">INFO:</span>
        <span className="text-zinc-300">{line.slice(5)}</span>
      </div>
    );
  }
  if (line.includes('🚀 Endpoints Ready:')) {
    return (
      <div key={idx} className="text-emerald-400 font-bold pt-1">
        {line}
      </div>
    );
  }
  if (line.trim().startsWith('• GET')) {
    const parts = line.split('->');
    return (
      <div key={idx} className="pl-2">
        <span className="text-sky-400 font-semibold">• GET</span>
        <span className="text-zinc-200">{parts[0].replace('• GET', '')}</span>
        {parts[1] && <span className="text-zinc-400">→{parts[1]}</span>}
      </div>
    );
  }
  if (line.trim().startsWith('• POST')) {
    const parts = line.split('->');
    return (
      <div key={idx} className="pl-2">
        <span className="text-emerald-400 font-semibold">• POST</span>
        <span className="text-zinc-200">{parts[0].replace('• POST', '')}</span>
        {parts[1] && <span className="text-zinc-400">→{parts[1]}</span>}
      </div>
    );
  }
  if (line.includes('[========================================]')) {
    return (
      <div key={idx} className="text-zinc-300">
        <span>Embedding Passages: </span>
        <span className="text-emerald-400 font-bold">[========================================]</span>
        <span className="text-emerald-300 font-semibold"> 48/48 (100%)</span>
      </div>
    );
  }
  if (line.includes('Average Latency =')) {
    return (
      <div key={idx} className="text-zinc-200">
        <span className="text-emerald-400 font-bold select-none">✔ </span>
        <span>Benchmark Results: </span>
        <span className="text-sky-300 font-bold">Average Latency = 21.50ms</span>
      </div>
    );
  }
  return <div key={idx}>{line}</div>;
};

function ScrollReveal({
  children,
  className = '',
  delay = 0,
}: {
  children: React.ReactNode;
  className?: string;
  delay?: number;
}) {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`transition-all duration-700 ease-out transform ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        } ${className}`}
    >
      {children}
    </div>
  );
}

export default function LandingPage({ onNavigateToDocs }: LandingPageProps) {
  const [installTab, setInstallTab] = useState<'pip' | 'uv' | 'poetry'>('pip');
  const [hasCopiedInstall, setHasCopiedInstall] = useState(false);
  const [terminalTab, setTerminalTab] = useState<'init' | 'train' | 'serve' | 'benchmark'>('init');
  const [hasCopiedTerminal, setHasCopiedTerminal] = useState(false);
  const [activeParadigmTab, setActiveParadigmTab] = useState<'rag' | 'adapters' | 'scratch'>('rag');

  const highlightedRAG = useMemo(() => {
    try {
      return Prism.highlight(RAG_SNIPPET, Prism.languages.python, 'python');
    } catch {
      return RAG_SNIPPET;
    }
  }, []);

  const highlightedAdapter = useMemo(() => {
    try {
      return Prism.highlight(ADAPTER_SNIPPET, Prism.languages.python, 'python');
    } catch {
      return ADAPTER_SNIPPET;
    }
  }, []);

  const highlightedScratch = useMemo(() => {
    try {
      return Prism.highlight(SCRATCH_SNIPPET, Prism.languages.python, 'python');
    } catch {
      return SCRATCH_SNIPPET;
    }
  }, []);

  const installCommands = {
    pip: 'pip install aimlite',
    uv: 'uv pip install aimlite',
    poetry: 'poetry add aimlite',
  };

  const handleCopyInstall = () => {
    navigator.clipboard.writeText(installCommands[installTab]);
    setHasCopiedInstall(true);
    setTimeout(() => setHasCopiedInstall(false), 2000);
  };

  const terminalSnippets = {
    init: `$ aimlite init my_knowledge_base

   .d8b.  d888888b .88b  d88. db      d888888b d888888b d88888b 
  d8' \`8b   \`88'   88'YbdP\`88 88        \`88'   \`~~88~~' 88'     
  88ooo88    88    88  88  88 88         88       88    88ooooo 
  88~~~88    88    88  88  88 88         88       88    88~~~~~ 
  88   88   .88.   88  88  88 88booo.   .88.      88    88.     
  YP   YP Y888888P YP  YP  YP Y88888P Y888888P    YP    Y88888P

  v2.1.0  ❯  The Django for AI & Machine Learning
  create project

  ✔  Project name    my_knowledge_base

? Select Project Paradigm:
  Scratch Training (Custom ML, PyTorch, Scikit-Learn)
❯ RAG & Knowledge Base (PostgreSQL / Memory + Multi-Provider)
  LoRA Fine-Tuning (Parameter-Efficient Adapters)

? Select LLM Chat Provider:
❯ OpenAI (GPT-4o, GPT-4o-mini)
  Google Gemini (1.5 Flash, 2.0 Pro)
  Anthropic Claude (3.5 Sonnet)
  Ollama (Local Llama 3.2, DeepSeek-R1)

✔ Project 'my_knowledge_base' scaffolded in 42ms.
✔ Editable starter code: chat_provider.py, store.py, model.py, trainer.py`,

    train: `$ aimlite train SupportDocRAG

  AIMLITE v2.1.0 train  SupportDocRAG

❯ Discovering project root: /workspace/my_knowledge_base
❯ Inspecting data/: found 14 knowledge articles (.md, .txt)
❯ Applying SmartChunker: 48 coherent semantic passages generated
❯ Embedding Passages: [========================================] 48/48 (100%)
❯ Persisting Vector Index: artifacts/rag_index.json (192 KB)
✔ Indexing completed successfully in 0.84s.`,

    serve: `$ aimlite serve --port 8000

  AIMLITE v2.1.0 serve

  ➜  Primary Model:    ChurnClassifier (ML)
  ➜  Registered:       ChurnClassifier [ml], SupportDocRAG [rag]
  ➜  Web App:          http://127.0.0.1:8000/
  ➜  Models API:       http://127.0.0.1:8000/models
  ➜  Inference:        POST http://127.0.0.1:8000/predict
  ➜  Docs:             http://127.0.0.1:8000/docs

  press Ctrl+C to terminate`,

    benchmark: `$ aimlite benchmark experiments/benchmark.py

  AIMLITE v2.1.0 benchmark  experiments/benchmark.py

❯ Project Root: /workspace/my_knowledge_base
❯ Extended PYTHONPATH with project root
❯ Using Python: .venv/bin/python

Initializing SupportDocRAG model for benchmark...
Running 3 benchmark queries:
  Query: 'What are the 3 pillars of AIMLite?...' -> 21.4ms (sources: 3)
  Query: 'How do I customize the reranking hook?...' -> 18.9ms (sources: 3)
  Query: 'How does PostgresVectorStore handle conn?...' -> 24.2ms (sources: 3)

✔ Benchmark Results: Average Latency = 21.50ms
✔ Benchmark completed successfully with 0 errors.`,
  };

  const handleCopyTerminal = () => {
    navigator.clipboard.writeText(terminalSnippets[terminalTab]);
    setHasCopiedTerminal(true);
    setTimeout(() => setHasCopiedTerminal(false), 2000);
  };

  return (
    <div className="w-full flex flex-col items-center select-text">
      {/* ================================================================= */}
      {/* 1. HERO SECTION                                                   */}
      {/* ================================================================= */}
      <section className="relative w-full max-w-6xl px-4 sm:px-6 pt-10 sm:pt-20 pb-12 sm:pb-16 flex flex-col items-center text-center">
        {/* Version Pill / Changelog link */}
        <button
          onClick={() => onNavigateToDocs('changelog')}
          className="group inline-flex items-center gap-1.5 sm:gap-2 px-3 py-1 rounded-full bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-900/90 dark:hover:bg-zinc-800 border border-zinc-300/80 dark:border-zinc-800 text-zinc-800 dark:text-zinc-200 text-xs font-medium transition-all mb-6 sm:mb-7 shadow-xs backdrop-blur-sm max-w-full"
        >
          <span className="flex h-1.5 w-1.5 rounded-full bg-zinc-900 dark:bg-zinc-100 animate-pulse shrink-0" />
          <span className="font-semibold text-zinc-900 dark:text-zinc-100 shrink-0">v2.1.0 Live</span>
          <span className="text-zinc-400 dark:text-zinc-600 shrink-0">•</span>
          <span className="hidden sm:inline text-zinc-600 dark:text-zinc-400 group-hover:text-zinc-950 dark:group-hover:text-white transition-colors truncate">
            Production ML Engine & Multi-Model Serving
          </span>
          <span className="sm:hidden text-zinc-600 dark:text-zinc-400 truncate">
            Multi-Model & Web Serving
          </span>
          <ArrowRight size={13} className="text-zinc-500 dark:text-zinc-400 group-hover:translate-x-0.5 transition-transform shrink-0" />
        </button>

        {/* Big Bold Headline */}
        <h1 className="text-3xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-zinc-950 dark:text-white max-w-4xl leading-[1.15] sm:leading-[1.1] drop-shadow-xs">
          The{' '}
          <span className="bg-gradient-to-r from-zinc-950 via-zinc-800 to-zinc-600 dark:from-white dark:via-zinc-100 dark:to-zinc-300 bg-clip-text text-transparent">
            Django for AI
          </span>
        </h1>

        <p className="mt-3 sm:mt-6 text-sm sm:text-xl text-zinc-600 dark:text-zinc-300 max-w-2xl font-light leading-relaxed px-1 sm:px-0">
          Standardized, zero-path machine learning, LoRA fine-tuning, and enterprise RAG for Python.
          <br className="hidden sm:inline" /> Convention over configuration meets 100% developer extensibility.
        </p>

        {/* Quick Install Bar */}
        <div className="mt-6 sm:mt-8 flex flex-col sm:flex-row items-center gap-3 w-full max-w-sm sm:max-w-md">
          <div className="w-full flex items-center justify-between px-2.5 sm:px-3 py-2 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-800 shadow-md font-mono text-xs text-zinc-800 dark:text-zinc-200">
            <div className="flex items-center gap-2 truncate">
              {/* Tab Selector */}
              <div className="flex bg-zinc-100 dark:bg-zinc-950 rounded-lg p-0.5 border border-zinc-200 dark:border-zinc-800 shrink-0">
                {(['pip', 'uv', 'poetry'] as const).map((mgr) => (
                  <button
                    key={mgr}
                    onClick={() => setInstallTab(mgr)}
                    className={`px-1.5 sm:px-2 py-0.5 rounded text-[10px] uppercase font-semibold transition-colors ${installTab === mgr
                        ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 shadow-sm'
                        : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                      }`}
                  >
                    {mgr}
                  </button>
                ))}
              </div>
              <span className="text-zinc-400 dark:text-zinc-500 select-none">$</span>
              <span className="font-medium select-all truncate text-[11px] sm:text-xs">
                {installCommands[installTab]}
              </span>
            </div>

            <button
              onClick={handleCopyInstall}
              className="p-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-zinc-100 hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors ml-1 sm:ml-2 shrink-0"
              title="Copy install command"
            >
              {hasCopiedInstall ? (
                <Check size={14} className="text-zinc-950 dark:text-white" />
              ) : (
                <Copy size={14} />
              )}
            </button>
          </div>
        </div>

        {/* Primary Call To Actions */}
        <div className="mt-6 sm:mt-8 flex flex-col sm:flex-row flex-wrap items-stretch sm:items-center justify-center gap-2.5 sm:gap-3 w-full sm:w-auto max-w-sm sm:max-w-none">
          <button
            onClick={() => onNavigateToDocs('pillar-data')}
            className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-zinc-950 hover:bg-zinc-800 text-white dark:bg-white dark:text-zinc-950 dark:hover:bg-zinc-200 font-semibold text-sm shadow-md transition-all hover:scale-[1.02] w-full sm:w-auto"
          >
            <span>Explore Documentation</span>
            <ArrowRight size={16} />
          </button>

          <button
            onClick={() => onNavigateToDocs('paradigm-rag')}
            className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-white hover:bg-zinc-100 dark:bg-zinc-900 dark:hover:bg-zinc-800 border border-zinc-300 dark:border-zinc-800 text-zinc-900 dark:text-zinc-200 font-medium text-sm transition-all w-full sm:w-auto"
          >
            <BookOpen size={16} className="text-zinc-500 dark:text-zinc-400" />
            <span>Interactive RAG Guide</span>
          </button>

          <div className="flex items-center justify-center gap-2.5 w-full sm:w-auto">
            <a
              href="https://github.com/Alazar42/AIMLite"
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 sm:flex-none flex items-center justify-center gap-2 px-4 py-2.5 sm:py-3 rounded-xl bg-white hover:bg-zinc-100 dark:bg-zinc-900/60 dark:hover:bg-zinc-800/80 border border-zinc-300 dark:border-zinc-800 text-zinc-700 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-zinc-200 font-medium text-sm transition-colors"
            >
              <GithubIcon size={16} />
              <span>GitHub</span>
            </a>

            <a
              href="https://pypi.org/project/aimlite/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-3 py-2.5 sm:py-3 rounded-xl bg-white hover:bg-zinc-100 dark:bg-zinc-900/60 dark:hover:bg-zinc-800/80 border border-zinc-300 dark:border-zinc-800 text-zinc-700 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-zinc-200 font-mono text-xs transition-colors"
            >
              <span>pypi: v2.1.0</span>
              <ExternalLink size={12} />
            </a>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* 2. INTERACTIVE TERMINAL SHOWCASE                                  */}
      {/* ================================================================= */}
      <section className="w-full max-w-5xl px-3 sm:px-6 py-6">
        <ScrollReveal>
          <div className="rounded-2xl border border-zinc-300 dark:border-zinc-800 bg-[#0d0e12] shadow-2xl overflow-hidden">
            {/* macOS Terminal Titlebar & Command Tabs */}
            <div className="flex items-center justify-between bg-zinc-900 border-b border-zinc-800 px-3 sm:px-4 py-2.5 gap-2 overflow-hidden">
              {/* Window Controls */}
              <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
                <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-zinc-600 inline-block" />
                <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-zinc-700 inline-block" />
                <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-zinc-700 inline-block" />
                <span className="text-[11px] font-mono text-zinc-500 ml-2 hidden sm:inline">
                  aimlite terminal session
                </span>
              </div>

              {/* Interactive Step Switcher Tabs */}
              <div className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5 max-w-[calc(100vw-120px)] sm:max-w-none">
                {(
                  [
                    { id: 'init', label: '1. aimlite init' },
                    { id: 'train', label: '2. aimlite train' },
                    { id: 'serve', label: '3. aimlite serve' },
                    { id: 'benchmark', label: '4. aimlite benchmark' },
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setTerminalTab(tab.id)}
                    className={`shrink-0 px-2 sm:px-2.5 py-1 rounded-md text-[10px] sm:text-[11px] font-mono transition-colors whitespace-nowrap ${terminalTab === tab.id
                        ? 'bg-zinc-800 text-white font-semibold border border-zinc-700'
                        : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
                      }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Copy Button */}
              <button
                onClick={handleCopyTerminal}
                className="flex items-center gap-1 px-2 py-1 rounded text-[10px] sm:text-[11px] font-mono text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors shrink-0"
                title="Copy terminal session"
              >
                {hasCopiedTerminal ? (
                  <>
                    <Check size={12} className="text-zinc-200" />
                    <span className="text-zinc-200">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy size={12} />
                    <span>Copy</span>
                  </>
                )}
              </button>
            </div>

            {/* Terminal Body */}
            <div className="p-3 sm:p-6 font-mono text-xs sm:text-[13px] leading-relaxed text-zinc-300 bg-[#090a10] overflow-x-auto min-h-[250px]">
              <div className="space-y-0.5">
                {terminalSnippets[terminalTab].split('\n').map((line, idx) =>
                  renderTerminalLine(line, idx)
                )}
              </div>
            </div>
          </div>
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 3. THE 3 AI PARADIGMS SHOWCASE                                    */}
      {/* ================================================================= */}
      <section className="w-full max-w-6xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="text-center space-y-3 mb-8 sm:mb-10 px-1 sm:px-0">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-100 dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-800 text-zinc-800 dark:text-zinc-300 text-xs font-semibold uppercase tracking-wider">
              <Layers size={13} />
              <span>Multi-Paradigm Framework</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold text-zinc-950 dark:text-white tracking-tight">
              One Architecture. 3 Modern AI Paradigms.
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 text-xs sm:text-base max-w-2xl mx-auto leading-relaxed">
              Whether training classical models from scratch, fine-tuning LLMs with LoRA deltas, or deploying enterprise RAG, AIMLite standardizes the entire lifecycle.
            </p>
          </div>

          {/* Tab Switcher */}
          <div className="flex items-center justify-center w-full mb-6 sm:mb-8 overflow-hidden">
            <div className="flex max-w-full overflow-x-auto no-scrollbar p-1 rounded-xl bg-zinc-200 dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-800 gap-1">
              <button
                onClick={() => setActiveParadigmTab('rag')}
                className={`shrink-0 flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${activeParadigmTab === 'rag'
                    ? 'bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-sm'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white'
                  }`}
              >
                <BookOpen size={14} className="shrink-0" />
                <span>1. Production RAG</span>
              </button>
              <button
                onClick={() => setActiveParadigmTab('adapters')}
                className={`shrink-0 flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${activeParadigmTab === 'adapters'
                    ? 'bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-sm'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white'
                  }`}
              >
                <Cpu size={14} className="shrink-0" />
                <span>2. LoRA Fine-Tuning</span>
              </button>
              <button
                onClick={() => setActiveParadigmTab('scratch')}
                className={`shrink-0 flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${activeParadigmTab === 'scratch'
                    ? 'bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-sm'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white'
                  }`}
              >
                <Boxes size={14} className="shrink-0" />
                <span>3. Scratch ML</span>
              </button>
            </div>
          </div>

          {/* Paradigm Detail Card */}
          {activeParadigmTab === 'rag' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-4 sm:p-8 rounded-2xl border border-zinc-300 dark:border-zinc-800 bg-white dark:bg-zinc-900/50 shadow-xl">
              <div className="lg:col-span-6 space-y-5">
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-200 font-mono text-[11px] font-semibold border border-zinc-300 dark:border-zinc-700">
                    v2.1.0
                  </span>
                  <span className="text-xs text-zinc-500 font-mono">aimlite.rag</span>
                </div>

                <h3 className="text-2xl font-bold text-zinc-950 dark:text-white tracking-tight">
                  Enterprise Knowledge Base & Modular RAG
                </h3>

                <p className="text-sm text-zinc-600 dark:text-zinc-300 leading-relaxed">
                  Connect enterprise documents with unified multi-provider chat integrations (OpenAI, Gemini, Anthropic, Ollama, Local). All scaffolded code is 100% developer-editable with clean lifecycle extension hooks.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Developer Extension Hooks</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Override <code className="text-zinc-800 dark:text-zinc-200">preprocess_query</code>, <code className="text-zinc-800 dark:text-zinc-200">rerank</code>, <code className="text-zinc-800 dark:text-zinc-200">synthesize</code>, and <code className="text-zinc-800 dark:text-zinc-200">postprocess_answer</code>.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Smart Chunking & HyDE</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Preserves header context with <code className="text-zinc-800 dark:text-zinc-200">SmartChunker</code>; decomposes complex queries via <code className="text-zinc-800 dark:text-zinc-200">QueryAnalyzer</code>.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Vector DB & pgvector</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Seamless swap between <code className="text-zinc-800 dark:text-zinc-200">MemoryVectorStore</code> and production <code className="text-zinc-800 dark:text-zinc-200">PostgresVectorStore</code>.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Web Chat Playground</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Embedded browser test console at <code className="text-zinc-800 dark:text-zinc-200">/chat</code> with citation inspectors and streaming responses.
                    </p>
                  </div>
                </div>

                <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5 sm:gap-3">
                  <button
                    onClick={() => onNavigateToDocs('paradigm-rag')}
                    className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 hover:bg-zinc-800 dark:hover:bg-zinc-200 font-semibold text-xs transition-colors shadow-md w-full sm:w-auto"
                  >
                    <span>Read RAG Architecture Guide</span>
                    <ArrowRight size={14} />
                  </button>
                  <button
                    onClick={() => onNavigateToDocs('rag-hooks')}
                    className="flex items-center justify-center gap-1.5 px-3 py-2.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 border border-zinc-300 dark:border-zinc-700 text-zinc-800 dark:text-zinc-200 text-xs font-medium transition-colors w-full sm:w-auto"
                  >
                    <Code2 size={13} />
                    <span>Inspect Lifecycle Hooks</span>
                  </button>
                </div>
              </div>

              {/* Code Snippet Column */}
              <div className="lg:col-span-6 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-[#090a0f] p-3 sm:p-4 font-mono text-[10px] sm:text-[11px] leading-relaxed text-zinc-900 dark:text-zinc-200 overflow-x-auto shadow-inner transition-colors">
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-zinc-200 dark:border-zinc-800 text-zinc-500 text-[10px]">
                  <span>model.py (Developer-Editable KnowledgeModel)</span>
                  <span className="text-zinc-700 dark:text-zinc-400 font-semibold">Python 3.10+</span>
                </div>
                <pre className="m-0 p-0 font-mono whitespace-pre text-zinc-900 dark:text-zinc-200">
                  <code dangerouslySetInnerHTML={{ __html: highlightedRAG }} />
                </pre>
              </div>
            </div>
          )}

          {activeParadigmTab === 'adapters' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-4 sm:p-8 rounded-2xl border border-zinc-300 dark:border-zinc-800 bg-white dark:bg-zinc-900/50 shadow-xl">
              <div className="lg:col-span-6 space-y-5">
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-200 font-mono text-[11px] font-semibold border border-zinc-300 dark:border-zinc-700">
                    PEFT & LoRA
                  </span>
                  <span className="text-xs text-zinc-500 font-mono">aimlite.adapters</span>
                </div>

                <h3 className="text-xl sm:text-2xl font-bold text-zinc-950 dark:text-white tracking-tight">
                  Parameter-Efficient Fine-Tuning & Hot-Swapping
                </h3>

                <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-300 leading-relaxed">
                  Train lightweight low-rank delta matrices without duplicating multi-gigabyte base model weights. Reduce checkpoints from 14GB down to &lt;50MB with zero runtime serving latency.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Low-Rank Math Decomposition</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      <code className="text-zinc-800 dark:text-zinc-200">h = W0*x + (alpha/r)*(B*A)*x</code> with Gaussian A and zero-initialized B.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Zero-Latency Weight Merging</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      In-place <code className="text-zinc-800 dark:text-zinc-200">merge_weights()</code> folds delta weights into base matrices for 0ms inference overhead.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>MultiAdapterManager</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Host and route dozens of specialized adapters on a single running base model in sub-millisecond time.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Zero Heavy Dependencies</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Runs in pure Python & NumPy; seamlessly connects with PyTorch and Hugging Face PEFT when installed.
                    </p>
                  </div>
                </div>

                <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5 sm:gap-3">
                  <button
                    onClick={() => onNavigateToDocs('paradigm-adapters')}
                    className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 hover:bg-zinc-800 dark:hover:bg-zinc-200 font-semibold text-xs transition-colors shadow-md w-full sm:w-auto"
                  >
                    <span>Explore Adapter & LoRA Docs</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>

              {/* Code Snippet Column */}
              <div className="lg:col-span-6 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-[#090a0f] p-3 sm:p-4 font-mono text-[10px] sm:text-[11px] leading-relaxed text-zinc-900 dark:text-zinc-200 overflow-x-auto shadow-inner transition-colors">
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-zinc-200 dark:border-zinc-800 text-zinc-500 text-[10px]">
                  <span>model.py (LoRALayer & MultiAdapterManager)</span>
                  <span className="text-zinc-700 dark:text-zinc-400 font-semibold">PEFT Compatible</span>
                </div>
                <pre className="m-0 p-0 font-mono whitespace-pre text-zinc-900 dark:text-zinc-200">
                  <code dangerouslySetInnerHTML={{ __html: highlightedAdapter }} />
                </pre>
              </div>
            </div>
          )}

          {activeParadigmTab === 'scratch' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-4 sm:p-8 rounded-2xl border border-zinc-300 dark:border-zinc-800 bg-white dark:bg-zinc-900/50 shadow-xl">
              <div className="lg:col-span-6 space-y-5">
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-200 font-mono text-[11px] font-semibold border border-zinc-300 dark:border-zinc-700">
                    CLASSICAL & DEEP ML
                  </span>
                  <span className="text-xs text-zinc-500 font-mono">aimlite.data & aimlite.models</span>
                </div>

                <h3 className="text-xl sm:text-2xl font-bold text-zinc-950 dark:text-white tracking-tight">
                  Scratch Training & Custom Architectures
                </h3>

                <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-300 leading-relaxed">
                  Build tabular predictors, neural networks, or scikit-learn pipelines with zero boilerplate. Define your data contract and model class — AIMLite handles training execution, evaluation, and production serving.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Zero-Path Execution</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      No imports or routing boilerplate. <code className="text-zinc-800 dark:text-zinc-200">aimlite train</code> automatically discovers models in <code className="text-zinc-800 dark:text-zinc-200">model.py</code>.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Strict Data Contracts</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Dataset subclasses explicitly declare <code className="text-zinc-800 dark:text-zinc-200">filename</code>. Validates data readiness and auto-splits rows.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Any Framework</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Use PyTorch, scikit-learn, XGBoost, TensorFlow, or pure Python with identical interfaces.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-900/80 border border-zinc-200 dark:border-zinc-800 space-y-1">
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 flex items-center gap-1.5">
                      <CheckCircle2 size={13} className="text-zinc-700 dark:text-zinc-300" />
                      <span>Instant Production API</span>
                    </div>
                    <p className="text-zinc-500 dark:text-zinc-400 text-[11px]">
                      Deploy forward passes immediately with <code className="text-zinc-800 dark:text-zinc-200">aimlite serve</code> with Swagger UI at <code className="text-zinc-800 dark:text-zinc-200">/docs</code>.
                    </p>
                  </div>
                </div>

                <div className="pt-2 flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5 sm:gap-3">
                  <button
                    onClick={() => onNavigateToDocs('paradigm-scratch')}
                    className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 hover:bg-zinc-800 dark:hover:bg-zinc-200 font-semibold text-xs transition-colors shadow-md w-full sm:w-auto"
                  >
                    <span>Explore Scratch Training Docs</span>
                    <ArrowRight size={14} />
                  </button>
                </div>
              </div>

              {/* Code Snippet Column */}
              <div className="lg:col-span-6 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-[#090a0f] p-3 sm:p-4 font-mono text-[10px] sm:text-[11px] leading-relaxed text-zinc-900 dark:text-zinc-200 overflow-x-auto shadow-inner transition-colors">
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-zinc-200 dark:border-zinc-800 text-zinc-500 text-[10px]">
                  <span>model.py & data.py (Customer Churn Example)</span>
                  <span className="text-zinc-700 dark:text-zinc-400 font-semibold">Zero Boilerplate</span>
                </div>
                <pre className="m-0 p-0 font-mono whitespace-pre text-zinc-900 dark:text-zinc-200">
                  <code dangerouslySetInnerHTML={{ __html: highlightedScratch }} />
                </pre>
              </div>
            </div>
          )}
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 4. THE 3 ARCHITECTURAL PILLARS                                    */}
      {/* ================================================================= */}
      <section className="w-full max-w-6xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="text-center space-y-3 mb-8 sm:mb-12 px-1 sm:px-0">
            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-zinc-100 dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-800 text-zinc-800 dark:text-zinc-300 text-xs font-semibold uppercase tracking-wider">
              <Boxes size={13} />
              <span>Core Architecture</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold text-zinc-950 dark:text-white tracking-tight">
              The 3 Pillars of AIMLite
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 text-xs sm:text-base max-w-xl mx-auto leading-relaxed">
              Just like Django separates Models, Views, and Templates, AIMLite organizes AI development into three strictly defined pillars.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
            {/* Pillar 1: Data */}
            <div
              onClick={() => onNavigateToDocs('pillar-data')}
              className="group cursor-pointer rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-5 sm:p-6 space-y-3 sm:space-y-4 hover:border-zinc-400 dark:hover:border-zinc-600 hover:shadow-lg transition-all relative overflow-hidden"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center text-zinc-900 dark:text-white group-hover:scale-105 transition-transform">
                <Database size={18} />
              </div>

              <div className="space-y-1">
                <div className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider font-semibold">
                  Pillar 1
                </div>
                <h3 className="text-base sm:text-lg font-bold text-zinc-900 dark:text-white group-hover:underline transition-colors">
                  Data (Dataset)
                </h3>
              </div>

              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Standardized data ingestion, explicit file declarations (<code className="text-zinc-900 dark:text-zinc-200">filename = "data.csv"</code>), schema validation, and automatic partition contracts (<code className="text-zinc-900 dark:text-zinc-200">split()</code>).
              </p>

              <div className="pt-2 flex items-center gap-1.5 text-xs font-semibold text-zinc-900 dark:text-white group-hover:translate-x-1 transition-transform">
                <span>Inspect Data Pillar</span>
                <ArrowRight size={13} />
              </div>
            </div>

            {/* Pillar 2: Model */}
            <div
              onClick={() => onNavigateToDocs('pillar-model')}
              className="group cursor-pointer rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-5 sm:p-6 space-y-3 sm:space-y-4 hover:border-zinc-400 dark:hover:border-zinc-600 hover:shadow-lg transition-all relative overflow-hidden"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center text-zinc-900 dark:text-white group-hover:scale-105 transition-transform">
                <Cpu size={18} />
              </div>

              <div className="space-y-1">
                <div className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider font-semibold">
                  Pillar 2
                </div>
                <h3 className="text-base sm:text-lg font-bold text-zinc-900 dark:text-white group-hover:underline transition-colors">
                  Model (Model & KnowledgeModel)
                </h3>
              </div>

              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Unified interface for classical ML, LoRA adaptation, and Knowledge Base RAG. Features developer-editable lifecycle hooks with zero hidden boilerplate.
              </p>

              <div className="pt-2 flex items-center gap-1.5 text-xs font-semibold text-zinc-900 dark:text-white group-hover:translate-x-1 transition-transform">
                <span>Inspect Model Pillar</span>
                <ArrowRight size={13} />
              </div>
            </div>

            {/* Pillar 3: Lifecycle */}
            <div
              onClick={() => onNavigateToDocs('pillar-lifecycle')}
              className="group cursor-pointer rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-5 sm:p-6 space-y-3 sm:space-y-4 hover:border-zinc-400 dark:hover:border-zinc-600 hover:shadow-lg transition-all relative overflow-hidden"
            >
              <div className="w-10 h-10 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center text-zinc-900 dark:text-white group-hover:scale-105 transition-transform">
                <RefreshCw size={18} />
              </div>

              <div className="space-y-1">
                <div className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider font-semibold">
                  Pillar 3
                </div>
                <h3 className="text-base sm:text-lg font-bold text-zinc-900 dark:text-white group-hover:underline transition-colors">
                  Lifecycle (Trainer & Inference)
                </h3>
              </div>

              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Standardized training loops (<code className="text-zinc-900 dark:text-zinc-200">BaseTrainer</code>, <code className="text-zinc-900 dark:text-zinc-200">RAGTrainer</code>), metric evaluations, and embedded FastAPI serving with Swagger and Web Chat Playground.
              </p>

              <div className="pt-2 flex items-center gap-1.5 text-xs font-semibold text-zinc-900 dark:text-white group-hover:translate-x-1 transition-transform">
                <span>Inspect Lifecycle Pillar</span>
                <ArrowRight size={13} />
              </div>
            </div>
          </div>
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 5. WHY DEVELOPERS LOVE AIMLITE (FEATURE GRID)                     */}
      {/* ================================================================= */}
      <section className="w-full max-w-6xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="text-center space-y-3 mb-8 sm:mb-12 px-1 sm:px-0">
            <h2 className="text-2xl sm:text-4xl font-bold text-zinc-950 dark:text-white tracking-tight">
              Engineered for Developer Velocity
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 text-xs sm:text-base max-w-xl mx-auto leading-relaxed">
              Everything you need to go from an idea in a terminal to an enterprise-ready inference service.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
            <div className="p-4 sm:p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/30 space-y-2.5 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white">
                <Rocket size={17} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Zero-Path Execution</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Never configure <code className="text-zinc-900 dark:text-zinc-200">sys.path.append()</code> or deal with <code className="text-zinc-900 dark:text-zinc-200">ModuleNotFoundError</code>. AIMLite resolves roots automatically via <code className="text-zinc-900 dark:text-zinc-200">aimlite.json</code>.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/30 space-y-2.5 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white">
                <ShieldCheck size={17} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Zero Heavy-Dependency Guarantee</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Core package runs in pure Python and NumPy. Heavy frameworks (PyTorch, transformers, CUDA) are loaded only when explicitly requested.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/30 space-y-2.5 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white">
                <Boxes size={17} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Instant Project Scaffolding</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                <code className="text-zinc-900 dark:text-zinc-200">aimlite init</code> completes in under 50ms. Never wait for silent multi-gigabyte downloads during project creation.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/30 space-y-2.5 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white">
                <Sliders size={17} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Developer-Editable Code</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                No closed black boxes. Every file generated by the CLI is clear, commented, and checked directly into your git repository.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/30 space-y-2.5 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white">
                <Server size={17} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Multi-Model & Adaptive Serving</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Zero-path server with <code className="text-zinc-900 dark:text-zinc-200">/models</code> inventory, type-adaptive web playgrounds (ML forms, RAG chat, LoRA runner), and headless <code className="text-zinc-900 dark:text-zinc-200">--api</code> mode.
              </p>
            </div>

            <div className="p-4 sm:p-5 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/30 space-y-2.5 hover:border-zinc-400 dark:hover:border-zinc-600 transition-colors">
              <div className="w-8 h-8 rounded-lg bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center text-zinc-900 dark:text-white">
                <Terminal size={17} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Built-in Benchmark Command</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                <code className="text-zinc-900 dark:text-zinc-200">aimlite benchmark</code> executes scripts in <code className="text-zinc-900 dark:text-zinc-200">experiments/</code> with project context injected into Python's path automatically.
              </p>
            </div>
          </div>
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 6. 3-MINUTE QUICKSTART WALKTHROUGH                                */}
      {/* ================================================================= */}
      <section className="w-full max-w-4xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/40 p-4 sm:p-8 space-y-5 sm:space-y-6">
            <div className="space-y-1">
              <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider font-semibold">
                Quickstart Guide
              </span>
              <h3 className="text-xl sm:text-2xl font-bold text-zinc-950 dark:text-white">
                Launch Your First AIMLite Project in 3 Minutes
              </h3>
            </div>

            <div className="space-y-3 sm:space-y-4 font-mono text-xs">
              {/* Step 1 */}
              <div className="p-3 sm:p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950/80 border border-zinc-200 dark:border-zinc-800 space-y-1.5">
                <div className="flex items-center justify-between text-zinc-500">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">1. Install AIMLite via pip</span>
                  <span className="text-[10px]">Terminal</span>
                </div>
                <div className="text-zinc-900 dark:text-zinc-100 font-semibold overflow-x-auto select-all">$ pip install aimlite</div>
              </div>

              {/* Step 2 */}
              <div className="p-3 sm:p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950/80 border border-zinc-200 dark:border-zinc-800 space-y-1.5">
                <div className="flex items-center justify-between text-zinc-500">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">2. Scaffold your RAG Knowledge Base</span>
                  <span className="text-[10px]">Terminal</span>
                </div>
                <div className="text-zinc-900 dark:text-zinc-100 font-semibold overflow-x-auto select-all">$ aimlite init my_rag --type rag</div>
                <div className="text-zinc-500 text-[11px] overflow-x-auto select-all">$ cd my_rag && aimlite install</div>
              </div>

              {/* Step 3 */}
              <div className="p-3 sm:p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950/80 border border-zinc-200 dark:border-zinc-800 space-y-1.5">
                <div className="flex items-center justify-between text-zinc-500">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">3. Index documents & start the server</span>
                  <span className="text-[10px]">Terminal</span>
                </div>
                <div className="text-zinc-900 dark:text-zinc-100 font-semibold overflow-x-auto select-all">$ aimlite train && aimlite serve --port 8000</div>
                <div className="text-zinc-500 dark:text-zinc-400 text-[10px] sm:text-[11px] leading-relaxed break-words">
                  # Open http://localhost:8000/chat in your browser for the live Web Chat Playground!
                </div>
              </div>
            </div>

            <div className="pt-2 flex justify-center sm:justify-end">
              <button
                onClick={() => onNavigateToDocs('cli-init')}
                className="w-full sm:w-auto flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-zinc-950 hover:bg-zinc-800 text-white dark:bg-white dark:text-zinc-950 dark:hover:bg-zinc-200 font-semibold text-xs transition-colors"
              >
                <span>View Complete CLI Commands</span>
                <ArrowRight size={14} />
              </button>
            </div>
          </div>
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 7. FOOTER                                                         */}
      {/* ================================================================= */}
      <footer className="w-full border-t border-zinc-200 dark:border-zinc-800 bg-white dark:bg-[#09090b] py-8 sm:py-12 px-4 sm:px-6 mt-8 sm:mt-12 text-xs text-zinc-500 dark:text-zinc-400">
        <div className="w-full max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-lg bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 flex items-center justify-center font-bold text-xs shadow-sm">
              <Boxes size={14} />
            </div>
            <span className="font-bold text-zinc-900 dark:text-zinc-200 text-sm">AIMLite</span>
            <span className="text-[11px] text-zinc-400 dark:text-zinc-500 font-mono">v2.1.0 (Apache 2.0)</span>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-x-4 sm:gap-x-5 gap-y-2 text-xs text-zinc-600 dark:text-zinc-400">
            <button
              onClick={() => onNavigateToDocs('pillar-data')}
              className="hover:text-zinc-950 dark:hover:text-zinc-200 transition-colors"
            >
              Documentation
            </button>
            <button
              onClick={() => onNavigateToDocs('paradigm-rag')}
              className="hover:text-zinc-950 dark:hover:text-zinc-200 transition-colors"
            >
              RAG Guide
            </button>
            <button
              onClick={() => onNavigateToDocs('cli-benchmark')}
              className="hover:text-zinc-950 dark:hover:text-zinc-200 transition-colors"
            >
              Benchmark CLI
            </button>
            <button
              onClick={() => onNavigateToDocs('endpoint-chat')}
              className="hover:text-zinc-950 dark:hover:text-zinc-200 transition-colors"
            >
              Chat Playground
            </button>
            <button
              onClick={() => onNavigateToDocs('changelog')}
              className="hover:text-zinc-950 dark:hover:text-zinc-200 transition-colors"
            >
              Changelog
            </button>
            <a
              href="https://github.com/Alazar42/AIMLite"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-950 dark:hover:text-zinc-200 transition-colors inline-flex items-center gap-1"
            >
              <span>GitHub</span>
              <ExternalLink size={11} />
            </a>
            <a
              href="https://pypi.org/project/aimlite/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-zinc-950 dark:hover:text-zinc-200 transition-colors inline-flex items-center gap-1"
            >
              <span>PyPI</span>
              <ExternalLink size={11} />
            </a>
          </div>
        </div>

        <div className="w-full max-w-6xl mx-auto text-center mt-6 text-[11px] text-zinc-400 dark:text-zinc-600">
          AIMLite — Convention over configuration meets 100% developer extensibility for modern AI & Python.
        </div>
      </footer>
    </div>
  );
}
