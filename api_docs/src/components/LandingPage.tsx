import { useState, useMemo, useEffect, useRef } from 'react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-docker';
import 'prismjs/components/prism-yaml';
import {
  Terminal,
  ArrowRight,
  Copy,
  Check,
  Cpu,
  BookOpen,
  Server,
  ShieldCheck,
  Rocket,
  CheckCircle2,
  Sliders,
  Boxes,
  Box,
  Zap,
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
        # Hook: Normalize query and expand domain acronyms
        return query.strip().replace("MFA", "Multi-Factor Authentication")

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        # Hook: Strict similarity thresholding (cosine >= 0.40)
        return [d for d in documents if (d.score or 0) >= 0.4]

    def postprocess_answer(self, answer: str, context_docs: list[Document]) -> str:
        # Hook: Append verifiable audit sources to citations footer
        sources = {d.metadata.get("source") for d in context_docs}
        return f"{answer}\\n\\n[Verified Sources: {', '.join(sources)}]"

# Forward inference coordinates preprocessing, retrieval, reranking & synthesis:
# response = model.predict("How does MFA work?", top_k=3)`;

const ADAPTER_SNIPPET = `from aimlite.adapters import AdapterModel, MultiAdapterManager

# Low-rank adapter linear layer (mathematical decomposition: W = W0 + (alpha/r)*B*A)
model = AdapterModel(base_model, r=8, lora_alpha=16)

# Real-time parameter accounting diagnostics:
model.print_trainable_parameters()
# Output: Trainable: 294,912 / 6,738,415,616 (0.0044% memory footprint)

# Zero-overhead inference via in-place weight merging:
model.merge_weights()
predictions = model.predict(inputs)
model.unmerge_weights()  # Revert for multi-tenant dynamic swapping

# Multi-adapter hot-swapper hosting specialized tenant models:
manager = MultiAdapterManager(base_model)
manager.add_adapter("billing", adapter_billing)
manager.add_adapter("support", adapter_support)
manager.set_active_adapter("billing")`;

const SCRATCH_SNIPPET = `from aimlite import Dataset, Model, BaseTrainer

class CustomerChurnDataset(Dataset):
    filename = "telecom_churn.csv"  # Target file auto-discovered in data/

class ChurnModel(Model):
    def fit(self, X, y):
        # Full tabular optimization (scikit-learn, XGBoost, PyTorch)
        self.weights = train_classifier(X, y)
        return self

    def predict(self, inputs):
        return {"churn_probability": calculate_score(inputs)}

# Train and serve via zero-path commands:
# $ aimlite train ChurnModel
# $ aimlite serve ChurnModel --port 8000`;

const DOCKER_SNIPPET = `# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Set environment configuration
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PORT=8000 \\
    HOST=0.0.0.0 \\
    PATH="/app/.venv/bin:$PATH"

# Install system dependencies (curl for healthchecks & network tooling)
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 1. Install uv (blazing fast backend) and AIMLite framework
RUN pip install --no-cache-dir uv aimlite

# 2. Set project working directory
WORKDIR /app

# 3. Copy project manifest
COPY aimlite.json .

# 4. Install all project dependencies into managed .venv using aimlite CLI
RUN aimlite install

# 5. Copy the remaining application files, data, and frontend assets
COPY . .

# 6. Train and calibrate model weights for serving
RUN aimlite train

# 7. Expose serving port
EXPOSE 8000

# 8. Health check verifying inference server status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:\${PORT:-8000}/health || exit 1

# 9. Host multi-model inference server with custom frontend (dynamically binds to $PORT on cloud hosts)
CMD ["sh", "-c", "aimlite serve --host 0.0.0.0 --port \${PORT:-8000} --frontend frontend"]`;

const DOCKER_COMPOSE_SNIPPET = `version: "3.8"
services:
  aimlite:
    build: .
    ports:
      - "\${PORT:-8000}:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - OPENAI_API_KEY=\${OPENAI_API_KEY:-}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3`;

const DOCKER_COMMANDS_SNIPPET = `# 1. Build production image with layer caching
docker build -t my-aimlite-app .

# 2. Run container with host port 8000 mapped
docker run -d --name aimlite-srv -p 8000:8000 my-aimlite-app

# 3. Verify server health probe (/health)
curl http://localhost:8000/health

# 4. Submit prediction request
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [128.0, 1.0, 2.7, 1.0, 265.1, 110.0, 89.0, 9.8, 10.0]}'`;

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
  // Vite / CLI Header title in cyan
  if (line.trim().startsWith('AIMLITE v')) {
    const parts = line.trim().split(/\s+/);
    return (
      <div key={idx} className="flex items-center gap-2 py-1 text-xs font-mono">
        <span className="text-[#00d8ff] font-extrabold tracking-wider">AIMLITE</span>
        <span className="text-zinc-500">{parts[1] || 'v2.1.2'}</span>
        <span className="text-emerald-400 font-semibold">{parts[2]}</span>
        <span className="text-zinc-400">{parts.slice(3).join(' ')}</span>
      </div>
    );
  }
  if (line.includes('The Django for AI & Machine Learning')) {
    return (
      <div key={idx} className="flex items-center gap-2 pt-1 pb-0.5 text-xs font-mono">
        <span className="text-zinc-500">v2.1.2</span>
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
  if (line.includes('[========================================]')) {
    return (
      <div key={idx} className="flex items-center gap-1.5 text-zinc-300">
        <span className="text-[#00d8ff] font-bold select-none">❯</span>
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
  if (line.trim().startsWith('➜')) {
    const content = line.trim().replace(/^➜\s*/, '');
    const colonIdx = content.indexOf(':');
    if (colonIdx !== -1) {
      const label = content.slice(0, colonIdx + 1);
      const val = content.slice(colonIdx + 1);
      return (
        <div key={idx} className="flex items-center gap-2 py-0.5 pl-2 font-mono text-xs">
          <span className="text-emerald-400 font-bold select-none">➜</span>
          <span className="text-zinc-400">{label}</span>
          <span className="text-cyan-300 font-semibold">{val}</span>
        </div>
      );
    }
    return (
      <div key={idx} className="flex items-center gap-2 py-0.5 pl-2 font-mono text-xs">
        <span className="text-emerald-400 font-bold select-none">➜</span>
        <span className="text-zinc-200">{content}</span>
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
  if (line.includes('Query:') && line.includes('->')) {
    const [queryPart, msPart] = line.split('->');
    return (
      <div key={idx} className="pl-2 text-zinc-300">
        <span className="text-zinc-400">{queryPart}</span>
        <span className="text-zinc-500">→</span>
        <span className="text-emerald-400 font-semibold">{msPart}</span>
      </div>
    );
  }
  if (line.startsWith('[+] Building')) {
    return (
      <div key={idx} className="text-sky-400 font-semibold py-0.5">
        <span>[+] </span>
        <span className="text-zinc-200">Building 4.2s (10/10) </span>
        <span className="text-emerald-400 font-bold">FINISHED</span>
      </div>
    );
  }
  if (line.trim().startsWith('=>')) {
    return (
      <div key={idx} className="text-zinc-400 py-0.5 pl-1">
        <span className="text-sky-400 font-bold select-none">=&gt; </span>
        <span className="text-zinc-300">{line.replace(/^.*=>\s*/, '')}</span>
      </div>
    );
  }
  if (line.includes('press Ctrl+C to terminate')) {
    return (
      <div key={idx} className="text-zinc-500 italic text-[11px] pt-1">
        {line}
      </div>
    );
  }
  return <div key={idx} className="text-zinc-300">{line}</div>;
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
      className={`transition-all duration-700 ease-out transform ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
      } ${className}`}
    >
      {children}
    </div>
  );
}

export default function LandingPage({ onNavigateToDocs }: LandingPageProps) {
  const [installTab, setInstallTab] = useState<'pip' | 'uv' | 'curl' | 'docker'>('pip');
  const [hasCopiedInstall, setHasCopiedInstall] = useState(false);
  const [terminalTab, setTerminalTab] = useState<'init' | 'train' | 'serve' | 'benchmark' | 'docker'>('init');
  const [hasCopiedTerminal, setHasCopiedTerminal] = useState(false);
  const [activeParadigmTab, setActiveParadigmTab] = useState<'rag' | 'adapters' | 'scratch'>('rag');
  const [activeDockerTab, setActiveDockerTab] = useState<'dockerfile' | 'compose' | 'cli'>('dockerfile');
  const [hasCopiedDocker, setHasCopiedDocker] = useState(false);

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

  const highlightedDocker = useMemo(() => {
    try {
      return Prism.highlight(DOCKER_SNIPPET, Prism.languages.docker || Prism.languages.bash, 'docker');
    } catch {
      return DOCKER_SNIPPET;
    }
  }, []);

  const highlightedDockerCompose = useMemo(() => {
    try {
      return Prism.highlight(DOCKER_COMPOSE_SNIPPET, Prism.languages.yaml || Prism.languages.bash, 'yaml');
    } catch {
      return DOCKER_COMPOSE_SNIPPET;
    }
  }, []);

  const highlightedDockerCommands = useMemo(() => {
    try {
      return Prism.highlight(DOCKER_COMMANDS_SNIPPET, Prism.languages.bash, 'bash');
    } catch {
      return DOCKER_COMMANDS_SNIPPET;
    }
  }, []);

  const handleCopyDocker = () => {
    const code =
      activeDockerTab === 'dockerfile'
        ? DOCKER_SNIPPET
        : activeDockerTab === 'compose'
        ? DOCKER_COMPOSE_SNIPPET
        : DOCKER_COMMANDS_SNIPPET;
    navigator.clipboard.writeText(code);
    setHasCopiedDocker(true);
    setTimeout(() => setHasCopiedDocker(false), 2000);
  };

  const installCommands = {
    pip: 'pip install aimlite',
    uv: 'uv add aimlite',
    curl: 'curl -fsSL https://raw.githubusercontent.com/Alazar42/AIMLite/main/install.sh | bash',
    docker: 'docker run -p 8000:8000 alazar42/aimlite:latest',
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

  v2.1.2  ❯  The Django for AI & Machine Learning
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

  AIMLITE v2.1.2 train  SupportDocRAG

❯ Discovering project root: /workspace/my_knowledge_base
❯ Inspecting data/: found 14 knowledge articles (.md, .txt)
❯ Applying SmartChunker: 48 coherent semantic passages generated
❯ Embedding Passages: [========================================] 48/48 (100%)
❯ Persisting Vector Index: artifacts/rag_index.json (192 KB)
✔ Indexing completed successfully in 0.84s.`,

    serve: `$ aimlite serve --port 8000

  AIMLITE v2.1.2 serve

  ➜  Primary Model:    ChurnClassifier (ML)
  ➜  Registered:       ChurnClassifier [ml], SupportDocRAG [rag]
  ➜  Web App:          http://127.0.0.1:8000/
  ➜  Models API:       http://127.0.0.1:8000/models
  ➜  Inference:        POST http://127.0.0.1:8000/predict
  ➜  Docs:             http://127.0.0.1:8000/docs

  press Ctrl+C to terminate`,

    benchmark: `$ aimlite benchmark experiments/benchmark.py

  AIMLITE v2.1.2 benchmark  experiments/benchmark.py

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

    docker: `$ docker build -t my-aimlite-app .
[+] Building 4.2s (10/10) FINISHED
 => [2/7] RUN pip install --no-cache-dir uv aimlite
 => [4/7] RUN aimlite install (cached layer from aimlite.json)
 => [6/7] RUN aimlite train (baked weights: models/SupportDocRAG.pkl)
 => EXPOSE 8000
 => HEALTHCHECK curl -f http://localhost:8000/health

$ docker run -d -p 8000:8000 my-aimlite-app
✔ Container started (ID: 8f4e21a). Liveness probe: HEALTHY.
✔ Multi-model inference server online on http://localhost:8000/`,
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
      <section className="relative w-full max-w-6xl px-4 sm:px-6 pt-12 sm:pt-24 pb-14 sm:pb-20 flex flex-col items-center text-center">
        {/* Floating Release Banner */}
        <button
          onClick={() => onNavigateToDocs('deployment-docker')}
          className="group inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/80 dark:bg-zinc-900/80 hover:bg-zinc-100 dark:hover:bg-zinc-800 border border-zinc-200 dark:border-zinc-800/90 text-zinc-800 dark:text-zinc-200 text-xs font-medium transition-all mb-7 shadow-xs backdrop-blur-md max-w-full hover:scale-105 active:scale-95"
        >
          <span className="flex h-2 w-2 rounded-full bg-zinc-400 dark:bg-zinc-500 animate-pulse shrink-0" />
          <span className="font-bold text-zinc-950 dark:text-white shrink-0 font-mono">v2.1.2 Live</span>
          <span className="text-zinc-300 dark:text-zinc-700 shrink-0">•</span>
          <span className="text-zinc-600 dark:text-zinc-400 group-hover:text-zinc-950 dark:group-hover:text-white transition-colors truncate">
            Now with 1-Click Production Docker & Cloud Deploy
          </span>
          <ArrowRight size={13} className="text-zinc-400 group-hover:text-zinc-950 dark:group-hover:text-white group-hover:translate-x-0.5 transition-all shrink-0" />
        </button>

        {/* Big Bold Headline with Shimmer Typography */}
        <h1 className="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-zinc-950 dark:text-white max-w-4xl leading-[1.12] sm:leading-[1.08] font-heading">
          The{' '}
          <span className="bg-gradient-to-r from-zinc-950 via-zinc-700 to-zinc-950 dark:from-white dark:via-zinc-200 dark:to-zinc-400 bg-clip-text text-transparent">
            Django for AI
          </span>
          <br />
          & Machine Learning
        </h1>

        <p className="mt-4 sm:mt-6 text-sm sm:text-lg text-zinc-600 dark:text-zinc-300 max-w-2xl font-light leading-relaxed px-1 sm:px-0">
          An opinionated, convention-over-configuration Python framework with{' '}
          <strong className="font-semibold text-zinc-900 dark:text-white">zero-path CLI execution</strong>,
          modular <strong className="font-semibold text-zinc-900 dark:text-white">RAG knowledge bases</strong>,
          and parameter-efficient <strong className="font-semibold text-zinc-900 dark:text-white">LoRA adapters</strong>.
        </p>

        {/* 1-Click Install Bar */}
        <div className="mt-7 sm:mt-9 flex flex-col items-center gap-2.5 w-full max-w-md sm:max-w-lg">
          <div className="w-full flex items-center justify-between px-3 py-2 rounded-2xl bg-white/90 dark:bg-[#0c0d12]/90 border border-zinc-200 dark:border-zinc-800 shadow-md backdrop-blur-md font-mono text-xs text-zinc-800 dark:text-zinc-200">
            <div className="flex items-center gap-2.5 truncate">
              {/* Manager Tab Selector */}
              <div className="flex bg-zinc-100 dark:bg-zinc-950 rounded-xl p-0.5 border border-zinc-200 dark:border-zinc-800 shrink-0">
                {(['pip', 'uv', 'curl', 'docker'] as const).map((mgr) => (
                  <button
                    key={mgr}
                    onClick={() => setInstallTab(mgr)}
                    className={`px-2 py-0.5 rounded-lg text-[10px] uppercase font-bold transition-all ${
                      installTab === mgr
                        ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 shadow-xs'
                        : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200'
                    }`}
                  >
                    {mgr}
                  </button>
                ))}
              </div>
              <span className="text-zinc-400 dark:text-zinc-500 select-none">$</span>
              <span className="font-semibold select-all truncate text-[11px] sm:text-xs">
                {installCommands[installTab]}
              </span>
            </div>

            <button
              onClick={handleCopyInstall}
              className="p-1.5 rounded-lg text-zinc-500 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-800 transition-colors ml-2 shrink-0"
              title="Copy install command"
            >
              {hasCopiedInstall ? (
                <Check size={14} className="text-zinc-900 dark:text-white" />
              ) : (
                <Copy size={14} />
              )}
            </button>
          </div>
        </div>

        {/* Primary Call To Actions */}
        <div className="mt-7 sm:mt-9 flex flex-col sm:flex-row flex-wrap items-center justify-center gap-3 w-full sm:w-auto">
          <button
            onClick={() => onNavigateToDocs('pillar-data')}
            className="flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-zinc-950 hover:bg-zinc-800 text-white dark:bg-white dark:text-zinc-950 dark:hover:bg-zinc-200 font-semibold text-sm shadow-md transition-all hover:scale-[1.02] active:scale-[0.98] w-full sm:w-auto"
          >
            <span>Explore Documentation</span>
            <ArrowRight size={16} />
          </button>

          <button
            onClick={() => onNavigateToDocs('deployment-docker')}
            className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-white/80 hover:bg-zinc-100 dark:bg-zinc-900/80 dark:hover:bg-zinc-800 border border-zinc-200 dark:border-zinc-800 text-zinc-900 dark:text-zinc-100 font-medium text-sm transition-all hover:scale-[1.02] active:scale-[0.98] w-full sm:w-auto"
          >
            <Box size={16} className="text-zinc-700 dark:text-zinc-300" />
            <span>Docker & Cloud Deploy</span>
          </button>

          <a
            href="https://github.com/Alazar42/AIMLite"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 px-4 py-3 rounded-xl text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white text-sm font-medium transition-colors"
          >
            <GithubIcon size={16} />
            <span>GitHub (Apache 2.0)</span>
          </a>
        </div>
      </section>

      {/* ================================================================= */}
      {/* 2. METRICS & PROOF BAR                                            */}
      {/* ================================================================= */}
      <section className="w-full max-w-5xl px-4 sm:px-6 py-4">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 p-3 rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/60 dark:bg-zinc-900/40 backdrop-blur-md shadow-xs">
          <div className="flex flex-col items-center justify-center p-2 text-center">
            <span className="font-heading font-extrabold text-lg sm:text-xl text-zinc-950 dark:text-white">Zero-Path</span>
            <span className="text-[10px] text-zinc-500 font-mono">CLI Auto-Discovery</span>
          </div>
          <div className="flex flex-col items-center justify-center p-2 text-center">
            <span className="font-heading font-extrabold text-lg sm:text-xl text-zinc-950 dark:text-white">&lt;50ms</span>
            <span className="text-[10px] text-zinc-500 font-mono">Project Scaffolding</span>
          </div>
          <div className="flex flex-col items-center justify-center p-2 text-center">
            <span className="font-heading font-extrabold text-lg sm:text-xl text-zinc-950 dark:text-white">3 AI</span>
            <span className="text-[10px] text-zinc-500 font-mono">Native Paradigms</span>
          </div>
          <div className="flex flex-col items-center justify-center p-2 text-center">
            <span className="font-heading font-extrabold text-lg sm:text-xl text-zinc-950 dark:text-white">128 / 128</span>
            <span className="text-[10px] text-zinc-500 font-mono">Tests Verified</span>
          </div>
          <div className="col-span-2 sm:col-span-1 flex flex-col items-center justify-center p-2 text-center">
            <span className="font-heading font-extrabold text-lg sm:text-xl text-zinc-950 dark:text-white">1-Click</span>
            <span className="text-[10px] text-zinc-500 font-mono">Docker Container</span>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/* 3. INTERACTIVE TERMINAL & CLI SHOWCASE                            */}
      {/* ================================================================= */}
      <section className="w-full max-w-5xl px-3 sm:px-6 py-8 sm:py-12">
        <ScrollReveal>
          <div className="relative rounded-2xl border border-zinc-300 dark:border-zinc-800 bg-[#090b10] shadow-2xl overflow-hidden font-mono">
            {/* Terminal Window Header Chrome */}
            <div className="flex items-center justify-between px-3.5 py-2.5 bg-[#12151e] border-b border-zinc-800/80 select-none overflow-hidden gap-2">
              <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
                <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#ff5f56] inline-block shadow-xs" />
                <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#ffbd2e] inline-block shadow-xs" />
                <span className="w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#27c93f] inline-block shadow-xs" />
                <span className="text-[11px] text-zinc-400 font-mono ml-2 hidden sm:inline">
                  aimlite terminal session
                </span>
              </div>

              {/* Command Tabs */}
              <div className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5 max-w-[calc(100vw-120px)] sm:max-w-none">
                {(
                  [
                    { id: 'init', label: '1. aimlite init' },
                    { id: 'train', label: '2. aimlite train' },
                    { id: 'serve', label: '3. aimlite serve' },
                    { id: 'benchmark', label: '4. aimlite benchmark' },
                    { id: 'docker', label: '5. docker' },
                  ] as const
                ).map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setTerminalTab(tab.id)}
                    className={`shrink-0 px-2.5 py-1 rounded-md text-[10px] sm:text-[11px] font-mono transition-all whitespace-nowrap font-semibold ${
                      terminalTab === tab.id
                        ? 'bg-zinc-800 text-[#00d8ff] border border-zinc-700 shadow-xs'
                        : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <button
                onClick={handleCopyTerminal}
                className="flex items-center gap-1 px-2 py-1 rounded-lg text-[11px] text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors shrink-0"
                title="Copy terminal commands"
              >
                {hasCopiedTerminal ? (
                  <>
                    <Check size={12} className="text-emerald-400" />
                    <span className="text-emerald-400 text-[10px]">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy size={12} />
                    <span className="text-[10px]">Copy</span>
                  </>
                )}
              </button>
            </div>

            {/* Terminal Content Screen */}
            <div className="p-4 sm:p-6 overflow-x-auto text-[11px] sm:text-xs leading-[1.65] max-h-[460px] overflow-y-auto">
              <div className="space-y-0.5">
                {terminalSnippets[terminalTab]
                  .split('\n')
                  .map((line, idx) => renderTerminalLine(line, idx))}
              </div>
            </div>
          </div>
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 4. THE 3 AI PARADIGMS                                             */}
      {/* ================================================================= */}
      <section className="w-full max-w-6xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="text-center space-y-3 mb-8 sm:mb-12">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-zinc-500">
              Architectural Versatility
            </span>
            <h2 className="text-2xl sm:text-4xl font-bold text-zinc-950 dark:text-white tracking-tight font-heading">
              The 3 AI Paradigms
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 text-xs sm:text-base max-w-2xl mx-auto leading-relaxed">
              Classical ML, Enterprise RAG, and LoRA Fine-Tuning. Built into the core framework with identical lifecycle contracts.
            </p>
          </div>

          {/* Paradigm Selector Bar */}
          <div className="flex items-center justify-center w-full mb-6 sm:mb-8">
            <div className="flex p-1 rounded-2xl bg-zinc-200/80 dark:bg-zinc-900 border border-zinc-300 dark:border-zinc-800 gap-1 shadow-xs">
              <button
                onClick={() => setActiveParadigmTab('rag')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all ${
                  activeParadigmTab === 'rag'
                    ? 'bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 font-bold shadow-xs'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white'
                }`}
              >
                <BookOpen size={14} />
                <span>1. Enterprise RAG</span>
              </button>
              <button
                onClick={() => setActiveParadigmTab('adapters')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all ${
                  activeParadigmTab === 'adapters'
                    ? 'bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 font-bold shadow-xs'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white'
                }`}
              >
                <Cpu size={14} />
                <span>2. LoRA Fine-Tuning</span>
              </button>
              <button
                onClick={() => setActiveParadigmTab('scratch')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all ${
                  activeParadigmTab === 'scratch'
                    ? 'bg-zinc-950 text-white dark:bg-white dark:text-zinc-950 font-bold shadow-xs'
                    : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white'
                }`}
              >
                <Boxes size={14} />
                <span>3. Scratch ML</span>
              </button>
            </div>
          </div>

          {/* Active Paradigm Card */}
          {activeParadigmTab === 'rag' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-5 sm:p-8 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/50 shadow-xl">
              <div className="lg:col-span-5 space-y-4">
                <span className="px-2.5 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 font-mono text-[11px] font-bold border border-zinc-200 dark:border-zinc-700">
                  aimlite.rag
                </span>
                <h3 className="text-xl sm:text-2xl font-bold text-zinc-950 dark:text-white font-heading">
                  Enterprise Knowledge Base & Modular RAG
                </h3>
                <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  Turn enterprise documents into verifiable intelligence with hierarchical markdown chunking, multi-provider LLM synthesis, and PostgreSQL <code className="text-zinc-900 dark:text-zinc-200">pgvector</code> storage.
                </p>
                <div className="space-y-2 pt-2">
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>Hierarchical <strong className="font-semibold">SmartChunker</strong> preserving headers</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>Native PostgreSQL <strong className="font-semibold">pgvector</strong> ORM integration</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>Multi-Provider: OpenAI, Gemini, Claude, Ollama</span>
                  </div>
                </div>
                <button
                  onClick={() => onNavigateToDocs('paradigm-rag')}
                  className="inline-flex items-center gap-2 text-xs font-bold text-zinc-900 dark:text-white hover:underline pt-2"
                >
                  <span>Explore RAG Implementation Guide</span>
                  <ArrowRight size={13} />
                </button>
              </div>

              <div className="lg:col-span-7 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-900 dark:bg-[#0c0d12] p-4 text-xs font-mono text-zinc-100 max-h-96 overflow-y-auto">
                <pre className="m-0 p-0 whitespace-pre">
                  <code dangerouslySetInnerHTML={{ __html: highlightedRAG }} />
                </pre>
              </div>
            </div>
          )}

          {activeParadigmTab === 'adapters' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-5 sm:p-8 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/50 shadow-xl">
              <div className="lg:col-span-5 space-y-4">
                <span className="px-2.5 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 font-mono text-[11px] font-bold border border-zinc-200 dark:border-zinc-700">
                  aimlite.adapters
                </span>
                <h3 className="text-xl sm:text-2xl font-bold text-zinc-950 dark:text-white font-heading">
                  LoRA & Parameter-Efficient Fine-Tuning
                </h3>
                <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  Low-rank decomposition linear layers with zero-overhead weight folding, multi-adapter hot-swapping, and lightweight delta checkpoints (~50MB vs 14GB).
                </p>
                <div className="space-y-2 pt-2">
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>Mathematical decomposition: <strong className="font-semibold">W = W0 + (alpha/r)*B*A</strong></span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>In-place <strong className="font-semibold">merge_weights()</strong> for zero latency overhead</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>MultiAdapterManager: runtime tenant hot-swapping</span>
                  </div>
                </div>
                <button
                  onClick={() => onNavigateToDocs('paradigm-adapters')}
                  className="inline-flex items-center gap-2 text-xs font-bold text-zinc-900 dark:text-white hover:underline pt-2"
                >
                  <span>Explore LoRA Implementation Guide</span>
                  <ArrowRight size={13} />
                </button>
              </div>

              <div className="lg:col-span-7 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-900 dark:bg-[#0c0d12] p-4 text-xs font-mono text-zinc-100 max-h-96 overflow-y-auto">
                <pre className="m-0 p-0 whitespace-pre">
                  <code dangerouslySetInnerHTML={{ __html: highlightedAdapter }} />
                </pre>
              </div>
            </div>
          )}

          {activeParadigmTab === 'scratch' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-5 sm:p-8 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/50 shadow-xl">
              <div className="lg:col-span-5 space-y-4">
                <span className="px-2.5 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-800 dark:text-zinc-200 font-mono text-[11px] font-bold border border-zinc-200 dark:border-zinc-700">
                  aimlite.scratch
                </span>
                <h3 className="text-xl sm:text-2xl font-bold text-zinc-950 dark:text-white font-heading">
                  Training from Scratch (Tabular & Deep ML)
                </h3>
                <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  Total algorithmic freedom for scikit-learn, PyTorch, XGBoost, or bespoke mathematical models. Production customer churn classification walkthrough.
                </p>
                <div className="space-y-2 pt-2">
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>Deterministic 80/10/10 splits with fixed evaluation seed</span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>Automatic weights persistence to <strong className="font-semibold">models/*.pkl</strong></span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-zinc-700 dark:text-zinc-300">
                    <CheckCircle2 size={14} className="text-zinc-400 dark:text-zinc-500 shrink-0" />
                    <span>Interactive Feature Form auto-generated in Web UI</span>
                  </div>
                </div>
                <button
                  onClick={() => onNavigateToDocs('paradigm-scratch')}
                  className="inline-flex items-center gap-2 text-xs font-bold text-zinc-900 dark:text-white hover:underline pt-2"
                >
                  <span>Explore Scratch ML Guide</span>
                  <ArrowRight size={13} />
                </button>
              </div>

              <div className="lg:col-span-7 rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-900 dark:bg-[#0c0d12] p-4 text-xs font-mono text-zinc-100 max-h-96 overflow-y-auto">
                <pre className="m-0 p-0 whitespace-pre">
                  <code dangerouslySetInnerHTML={{ __html: highlightedScratch }} />
                </pre>
              </div>
            </div>
          )}
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 5. PRODUCTION CONTAINERIZATION & CLOUD DEPLOYMENT                 */}
      {/* ================================================================= */}
      <section className="w-full max-w-6xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="text-center space-y-3 mb-8 sm:mb-12">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 text-zinc-700 dark:text-zinc-300 text-[11px] font-mono font-bold tracking-wider uppercase mb-1">
              <Box size={13} />
              <span>Container Ready • Zero Cold Starts</span>
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold text-zinc-950 dark:text-white tracking-tight font-heading">
              Production Containerization in 1 Click
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 text-xs sm:text-base max-w-2xl mx-auto leading-relaxed">
              Bake models directly into lightweight, production-grade Docker containers with layer-cached dependencies, pre-baked weights, active health probes, and dynamic cloud port binding.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-5 sm:p-8 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900/50 shadow-xl">
            {/* Left Column: Key DevOps Highlights */}
            <div className="lg:col-span-5 space-y-4">
              <div className="space-y-1">
                <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider font-semibold">
                  Zero-Friction DevOps
                </span>
                <h3 className="text-xl font-bold text-zinc-950 dark:text-white tracking-tight font-heading">
                  Engineered for Cloud Hosts
                </h3>
                <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                  Optimized for Kubernetes, Google Cloud Run, Render, Railway, Fly.io, and AWS ECS.
                </p>
              </div>

              <div className="space-y-3 text-xs">
                <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950/70 border border-zinc-200 dark:border-zinc-800/80 space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-zinc-900 dark:text-zinc-100">
                    <CheckCircle2 size={14} className="text-zinc-700 dark:text-zinc-300 shrink-0" />
                    <span>UV Layer Caching</span>
                  </div>
                  <p className="text-zinc-600 dark:text-zinc-400 text-[11px] leading-relaxed">
                    Manifest copied first; dependencies are pre-compiled and cached. Code edits rebuild in under 5 seconds.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950/70 border border-zinc-200 dark:border-zinc-800/80 space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-zinc-900 dark:text-zinc-100">
                    <CheckCircle2 size={14} className="text-zinc-700 dark:text-zinc-300 shrink-0" />
                    <span>Baked Weights (0ms Cold Start)</span>
                  </div>
                  <p className="text-zinc-600 dark:text-zinc-400 text-[11px] leading-relaxed">
                    <code className="text-zinc-900 dark:text-zinc-200">RUN aimlite train</code> fits models at build time. Containers start instantaneously with zero latency lag.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950/70 border border-zinc-200 dark:border-zinc-800/80 space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-zinc-900 dark:text-zinc-100">
                    <CheckCircle2 size={14} className="text-zinc-700 dark:text-zinc-300 shrink-0" />
                    <span>Automated Healthcheck Probe</span>
                  </div>
                  <p className="text-zinc-600 dark:text-zinc-400 text-[11px] leading-relaxed">
                    Native <code className="text-zinc-900 dark:text-zinc-200">GET /health</code> probe verified every 30s. Self-healing restarts in Kubernetes & Cloud Run.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-zinc-50 dark:bg-zinc-950/70 border border-zinc-200 dark:border-zinc-800/80 space-y-1">
                  <div className="flex items-center gap-2 font-semibold text-zinc-900 dark:text-zinc-100">
                    <CheckCircle2 size={14} className="text-zinc-700 dark:text-zinc-300 shrink-0" />
                    <span>Dynamic Cloud Port Binding</span>
                  </div>
                  <p className="text-zinc-600 dark:text-zinc-400 text-[11px] leading-relaxed">
                    Binds to <code className="text-zinc-900 dark:text-zinc-200">${"{PORT:-8000}"}</code> dynamically. Works out-of-the-box on Render, Railway, Fly.io & Cloud Run.
                  </p>
                </div>
              </div>

              <div className="pt-2">
                <button
                  onClick={() => onNavigateToDocs('deployment-docker')}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-zinc-950 hover:bg-zinc-800 text-white dark:bg-white dark:text-zinc-950 dark:hover:bg-zinc-200 font-semibold text-xs transition-colors shadow-sm"
                >
                  <Box size={14} />
                  <span>View Full Docker Deployment Guide</span>
                  <ArrowRight size={14} />
                </button>
              </div>
            </div>

            {/* Right Column: Interactive Code Tabs */}
            <div className="lg:col-span-7 flex flex-col rounded-xl overflow-hidden border border-zinc-200 dark:border-zinc-800 bg-zinc-900 dark:bg-[#0c0d12] shadow-2xl">
              {/* Header Bar */}
              <div className="flex items-center justify-between bg-zinc-800/80 dark:bg-[#15161f] px-3 py-2 border-b border-zinc-700/60 dark:border-zinc-800">
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setActiveDockerTab('dockerfile')}
                    className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
                      activeDockerTab === 'dockerfile'
                        ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-bold'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    Dockerfile
                  </button>
                  <button
                    onClick={() => setActiveDockerTab('compose')}
                    className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
                      activeDockerTab === 'compose'
                        ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-bold'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    docker-compose.yml
                  </button>
                  <button
                    onClick={() => setActiveDockerTab('cli')}
                    className={`px-3 py-1 rounded-lg text-xs font-mono font-medium transition-colors ${
                      activeDockerTab === 'cli'
                        ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-bold'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    deploy.sh
                  </button>
                </div>

                <button
                  onClick={handleCopyDocker}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-mono text-zinc-300 hover:text-white hover:bg-zinc-700/60 transition-colors"
                >
                  {hasCopiedDocker ? (
                    <>
                      <Check size={12} className="text-zinc-200" />
                      <span className="text-zinc-200 font-bold">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy size={12} />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              </div>

              {/* Code Surface */}
              <div className="p-4 overflow-x-auto max-h-[460px] overflow-y-auto text-xs font-mono leading-relaxed text-zinc-100">
                <pre className="m-0 p-0 whitespace-pre">
                  <code
                    dangerouslySetInnerHTML={{
                      __html:
                        activeDockerTab === 'dockerfile'
                          ? highlightedDocker
                          : activeDockerTab === 'compose'
                          ? highlightedDockerCompose
                          : highlightedDockerCommands,
                    }}
                  />
                </pre>
              </div>
            </div>
          </div>
        </ScrollReveal>
      </section>

      {/* ================================================================= */}
      {/* 6. WHY DEVELOPERS LOVE AIMLITE (VELOCITY GRID)                     */}
      {/* ================================================================= */}
      <section className="w-full max-w-6xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="text-center space-y-3 mb-8 sm:mb-12">
            <h2 className="text-2xl sm:text-4xl font-bold text-zinc-950 dark:text-white tracking-tight font-heading">
              Engineered for Developer Velocity
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 text-xs sm:text-base max-w-xl mx-auto leading-relaxed">
              Everything you need to go from an idea in a terminal to an enterprise-grade inference microservice.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
            <div className="p-5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/40 space-y-3 hover:border-zinc-400 dark:hover:border-zinc-600 transition-all hover:shadow-md">
              <div className="w-9 h-9 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 border border-zinc-200/60 dark:border-zinc-700/60 flex items-center justify-center font-bold">
                <Rocket size={18} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Zero-Path Execution</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Never configure <code className="text-zinc-900 dark:text-zinc-200">sys.path.append()</code> or deal with <code className="text-zinc-900 dark:text-zinc-200">ModuleNotFoundError</code>. AIMLite resolves roots automatically via <code className="text-zinc-900 dark:text-zinc-200">aimlite.json</code>.
              </p>
            </div>

            <div className="p-5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/40 space-y-3 hover:border-zinc-400 dark:hover:border-zinc-600 transition-all hover:shadow-md">
              <div className="w-9 h-9 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 border border-zinc-200/60 dark:border-zinc-700/60 flex items-center justify-center font-bold">
                <ShieldCheck size={18} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Zero Heavy-Dependency Guarantee</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Core package runs in pure Python and NumPy. Heavy frameworks (PyTorch, transformers, CUDA) are loaded only when explicitly requested.
              </p>
            </div>

            <div className="p-5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/40 space-y-3 hover:border-zinc-400 dark:hover:border-zinc-600 transition-all hover:shadow-md">
              <div className="w-9 h-9 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 border border-zinc-200/60 dark:border-zinc-700/60 flex items-center justify-center font-bold">
                <Zap size={18} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Instant Project Scaffolding</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                <code className="text-zinc-900 dark:text-zinc-200">aimlite init</code> completes in under 50ms. Never wait for silent multi-gigabyte downloads during project creation.
              </p>
            </div>

            <div className="p-5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/40 space-y-3 hover:border-zinc-400 dark:hover:border-zinc-600 transition-all hover:shadow-md">
              <div className="w-9 h-9 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 border border-zinc-200/60 dark:border-zinc-700/60 flex items-center justify-center font-bold">
                <Sliders size={18} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Developer-Editable Code</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                No closed black boxes. Every file generated by the CLI is clear, commented, and checked directly into your git repository.
              </p>
            </div>

            <div className="p-5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/40 space-y-3 hover:border-zinc-400 dark:hover:border-zinc-600 transition-all hover:shadow-md">
              <div className="w-9 h-9 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 border border-zinc-200/60 dark:border-zinc-700/60 flex items-center justify-center font-bold">
                <Server size={18} />
              </div>
              <h4 className="text-sm font-bold text-zinc-950 dark:text-white">Multi-Model & Adaptive Serving</h4>
              <p className="text-xs text-zinc-600 dark:text-zinc-400 leading-relaxed">
                Zero-path server with <code className="text-zinc-900 dark:text-zinc-200">/models</code> inventory, type-adaptive web playgrounds (ML forms, RAG chat, LoRA runner), and headless <code className="text-zinc-900 dark:text-zinc-200">--api</code> mode.
              </p>
            </div>

            <div className="p-5 rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-zinc-900/40 space-y-3 hover:border-zinc-400 dark:hover:border-zinc-600 transition-all hover:shadow-md">
              <div className="w-9 h-9 rounded-xl bg-zinc-100 dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 border border-zinc-200/60 dark:border-zinc-700/60 flex items-center justify-center font-bold">
                <Terminal size={18} />
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
      {/* 7. 3-MINUTE QUICKSTART WALKTHROUGH                                */}
      {/* ================================================================= */}
      <section className="w-full max-w-4xl px-3 sm:px-6 py-12 sm:py-16">
        <ScrollReveal>
          <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white/80 dark:bg-zinc-900/40 p-5 sm:p-8 space-y-6 shadow-xl">
            <div className="space-y-1">
              <span className="text-[11px] font-mono text-zinc-500 uppercase tracking-wider font-semibold">
                Quickstart Guide
              </span>
              <h3 className="text-xl sm:text-2xl font-bold text-zinc-950 dark:text-white font-heading">
                Launch Your First AIMLite Project in 3 Minutes
              </h3>
            </div>

            <div className="space-y-3 sm:space-y-4 font-mono text-xs">
              {/* Step 1 */}
              <div className="p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950/80 border border-zinc-200 dark:border-zinc-800 space-y-1.5">
                <div className="flex items-center justify-between text-zinc-500">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">1. Install AIMLite via pip / uv</span>
                  <span className="text-[10px]">Terminal</span>
                </div>
                <div className="text-zinc-900 dark:text-zinc-100 font-semibold select-all">$ pip install aimlite</div>
              </div>

              {/* Step 2 */}
              <div className="p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950/80 border border-zinc-200 dark:border-zinc-800 space-y-1.5">
                <div className="flex items-center justify-between text-zinc-500">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">2. Scaffold your RAG Knowledge Base</span>
                  <span className="text-[10px]">Terminal</span>
                </div>
                <div className="text-zinc-900 dark:text-zinc-100 font-semibold select-all">$ aimlite init my_rag --type rag</div>
                <div className="text-zinc-500 text-[11px] select-all">$ cd my_rag && aimlite install</div>
              </div>

              {/* Step 3 */}
              <div className="p-3.5 rounded-xl bg-zinc-50 dark:bg-zinc-950/80 border border-zinc-200 dark:border-zinc-800 space-y-1.5">
                <div className="flex items-center justify-between text-zinc-500">
                  <span className="font-semibold text-zinc-800 dark:text-zinc-200">3. Index documents & start the server</span>
                  <span className="text-[10px]">Terminal</span>
                </div>
                <div className="text-zinc-900 dark:text-zinc-100 font-semibold select-all">$ aimlite train && aimlite serve --port 8000</div>
                <div className="text-zinc-500 dark:text-zinc-400 text-[10px] sm:text-[11px] leading-relaxed">
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
      {/* 8. FOOTER                                                         */}
      {/* ================================================================= */}
      <footer className="w-full border-t border-zinc-200 dark:border-zinc-800 bg-white/70 dark:bg-[#07080a] py-8 sm:py-12 px-4 sm:px-6 mt-8 sm:mt-12 text-xs text-zinc-500 dark:text-zinc-400">
        <div className="w-full max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <span className="font-heading font-extrabold text-sm text-zinc-900 dark:text-white">AIMLite</span>
            <span>•</span>
            <span>Apache 2.0 Open Source</span>
            <span>•</span>
            <span>v2.1.2</span>
          </div>

          <div className="flex items-center gap-4 text-xs font-medium">
            <button onClick={() => onNavigateToDocs('pillar-data')} className="hover:text-zinc-900 dark:hover:text-white transition-colors">
              Pillars
            </button>
            <button onClick={() => onNavigateToDocs('paradigm-rag')} className="hover:text-zinc-900 dark:hover:text-white transition-colors">
              Paradigms
            </button>
            <button onClick={() => onNavigateToDocs('cli-init')} className="hover:text-zinc-900 dark:hover:text-white transition-colors">
              CLI
            </button>
            <button onClick={() => onNavigateToDocs('deployment-docker')} className="hover:text-zinc-900 dark:hover:text-white transition-colors">
              Docker
            </button>
            <button onClick={() => onNavigateToDocs('changelog')} className="hover:text-zinc-900 dark:hover:text-white transition-colors">
              Changelog
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
