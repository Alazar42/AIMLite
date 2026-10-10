import { useState, useMemo, useEffect } from 'react';
import { Copy, Check, FileCode, Terminal, Lock, PanelRightClose, Box, Layers } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-docker';
import 'prismjs/components/prism-yaml';
import { type DocSection } from '../data/aimliteDocs';

interface CodePlaygroundProps {
  section: DocSection;
  onClose?: () => void;
}

export default function CodePlayground({ section, onClose }: CodePlaygroundProps) {
  const [activeTab, setActiveTab] = useState<string>('');
  const [hasCopiedCode, setHasCopiedCode] = useState<boolean>(false);

  // Compute available tabs dynamically from section.snippets
  const availableTabs = useMemo(() => {
    const tabs: { id: string; label: string; type: 'py' | 'sh' | 'json' | 'docker' | 'yaml' }[] = [];

    // 1. Individual project files (Dockerfile, docker-compose.yml, data.py, model.py, etc.)
    if (section.snippets.files && Object.keys(section.snippets.files).length > 0) {
      for (const fname of Object.keys(section.snippets.files)) {
        let type: 'py' | 'sh' | 'json' | 'docker' | 'yaml' = 'py';
        if (fname === 'Dockerfile' || fname.endsWith('.dockerfile')) type = 'docker';
        else if (fname.endsWith('.yml') || fname.endsWith('.yaml')) type = 'yaml';
        else if (fname.endsWith('.json')) type = 'json';
        else if (fname.endsWith('.sh') || fname === '.dockerignore') type = 'sh';
        tabs.push({ id: fname, label: fname, type });
      }
    } else if (section.snippets.python) {
      // Resolve meaningful convention file name (never main.py)
      let defaultName = 'model.py';
      if (section.id === 'pillar-data') defaultName = 'data.py';
      else if (section.id === 'pillar-model') defaultName = 'model.py';
      else if (section.id === 'pillar-lifecycle') defaultName = 'trainer.py';
      else if (section.id.startsWith('foundations-')) defaultName = 'config.py';
      else if (section.id === 'paradigm-scratch') defaultName = 'model.py';
      else if (section.id === 'paradigm-rag') defaultName = 'model.py';
      else if (section.id === 'paradigm-adapters') defaultName = 'model.py';

      tabs.push({ id: defaultName, label: defaultName, type: 'py' });
    }

    // 2. CLI commands tab
    if (section.snippets.cli || section.id.startsWith('cli-')) {
      tabs.push({ id: 'terminal.sh', label: 'terminal.sh', type: 'sh' });
    }

    // 3. cURL request tab
    if (section.snippets.curl) {
      tabs.push({ id: 'request.sh', label: 'request.sh', type: 'sh' });
    }

    return tabs;
  }, [section.id, section.snippets]);

  // Set initial active tab when section changes
  useEffect(() => {
    if (availableTabs.length > 0) {
      setActiveTab(availableTabs[0].id);
    }
  }, [section.id, availableTabs]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setHasCopiedCode(true);
    setTimeout(() => setHasCopiedCode(false), 1500);
  };

  const getCodeSnippet = (): string => {
    if (section.snippets.files && activeTab in section.snippets.files) {
      return section.snippets.files[activeTab];
    }
    if (activeTab === 'terminal.sh') {
      return section.snippets.cli || `aimlite ${section.signatureOrPath}`;
    }
    if (activeTab === 'request.sh') {
      return (
        section.snippets.curl ||
        `curl -X POST http://127.0.0.1:8000/predict \\\n  -H "Content-Type: application/json" \\\n  -d '{"features": [1.5, 2.7, 3.2]}'`
      );
    }
    return section.snippets.python || '';
  };

  const currentSnippet = getCodeSnippet();

  const highlightedHtml = useMemo(() => {
    const raw = currentSnippet;
    let grammar = Prism.languages.python;
    let lang = 'python';

    if (activeTab === 'Dockerfile' || activeTab.endsWith('.dockerfile')) {
      grammar = Prism.languages.docker || Prism.languages.bash;
      lang = 'docker';
    } else if (activeTab.endsWith('.yml') || activeTab.endsWith('.yaml')) {
      grammar = Prism.languages.yaml || Prism.languages.bash;
      lang = 'yaml';
    } else if (activeTab.endsWith('.sh') || activeTab === '.dockerignore') {
      grammar = Prism.languages.bash;
      lang = 'bash';
    } else if (activeTab.endsWith('.json')) {
      grammar = Prism.languages.json;
      lang = 'json';
    }

    if (grammar) {
      try {
        return Prism.highlight(raw, grammar, lang);
      } catch {
        return raw;
      }
    }
    return raw;
  }, [currentSnippet, activeTab]);

  const lineCount = currentSnippet ? currentSnippet.split('\n').length : 1;
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1);

  const getLanguageLabel = (): string => {
    if (activeTab === 'Dockerfile') return 'Dockerfile';
    if (activeTab.endsWith('.yml') || activeTab.endsWith('.yaml')) return 'YAML';
    if (activeTab.endsWith('.sh') || activeTab === '.dockerignore') return 'Shell';
    if (activeTab.endsWith('.json')) return 'JSON';
    return 'Python';
  };

  return (
    <aside className="w-full h-full flex flex-col bg-white dark:bg-[#121318] border-l border-zinc-200 dark:border-zinc-800 font-mono text-xs select-text shadow-xl transition-colors">
      {/* Top Chrome Header with Mac Traffic Lights & Tab Bar */}
      <div className="flex items-center justify-between bg-zinc-100 dark:bg-[#1a1b22] border-b border-zinc-200 dark:border-zinc-800 px-2 shrink-0 select-none">
        {/* Left Mac-style Traffic Light Dots */}
        <div className="hidden sm:flex items-center gap-1.5 mr-2 shrink-0 pl-1">
          <span className="w-2.5 h-2.5 rounded-full bg-red-400/80 dark:bg-red-500/70 inline-block" />
          <span className="w-2.5 h-2.5 rounded-full bg-amber-400/80 dark:bg-amber-500/70 inline-block" />
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400/80 dark:bg-emerald-500/70 inline-block" />
        </div>

        {/* Tab List */}
        <div className="flex items-center flex-1 overflow-x-auto scrollbar-none">
          {availableTabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-2 text-xs border-r border-zinc-200 dark:border-zinc-800/80 transition-colors relative shrink-0 ${
                  isActive
                    ? 'bg-white dark:bg-[#121318] text-zinc-950 dark:text-white font-medium shadow-xs'
                    : 'text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/40'
                }`}
                title={`View and copy ${tab.label}`}
              >
                {isActive && (
                  <div className="absolute top-0 left-0 right-0 h-[2px] bg-zinc-900 dark:bg-white" />
                )}
                {tab.type === 'docker' && <Box size={13} className="text-zinc-500 dark:text-zinc-400" />}
                {tab.type === 'yaml' && <Layers size={13} className="text-zinc-500 dark:text-zinc-400" />}
                {tab.type === 'py' && <FileCode size={13} className="text-zinc-500 dark:text-zinc-400" />}
                {tab.type === 'sh' && <Terminal size={13} className="text-zinc-500 dark:text-zinc-400" />}
                {tab.type === 'json' && <FileCode size={13} className="text-zinc-500 dark:text-zinc-400" />}
                <span className="font-mono text-[11px]">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Right Action Icons: Read Only Badge, Copy, Hide */}
        <div className="flex items-center gap-1 shrink-0 ml-2 py-1">
          {/* Non-editable indicator */}
          <span className="hidden sm:inline-flex items-center gap-1 text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-zinc-200/80 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border border-zinc-300 dark:border-zinc-700">
            <Lock size={9} />
            <span>Read Only</span>
          </span>

          {/* Copy Button */}
          <button
            onClick={() => copyToClipboard(currentSnippet)}
            className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-zinc-600 dark:text-zinc-300 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-200/70 dark:hover:bg-zinc-800 transition-colors"
            title={`Copy ${activeTab || 'code'}`}
          >
            {hasCopiedCode ? (
              <>
                <Check size={12} className="text-zinc-900 dark:text-zinc-100" />
                <span className="text-zinc-900 dark:text-zinc-100">Copied</span>
              </>
            ) : (
              <>
                <Copy size={12} />
                <span>Copy</span>
              </>
            )}
          </button>

          {/* Hide/Collapse Console Button */}
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 rounded text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200 hover:bg-zinc-200/70 dark:hover:bg-zinc-800 transition-colors"
              title="Hide code console"
            >
              <PanelRightClose size={13} />
            </button>
          )}
        </div>
      </div>

      {/* Code Surface with Line Numbers - Read Only Non-Editable */}
      <div className="flex-1 overflow-y-auto p-3 flex font-mono text-[11px] leading-[1.6] bg-zinc-50/60 dark:bg-[#121318] select-text cursor-default">
        {/* Line Numbers Gutter */}
        <div className="select-none text-right pr-3 text-zinc-400 dark:text-zinc-500 border-r border-zinc-200 dark:border-zinc-800 mr-3 shrink-0">
          {lineNumbers.map((num) => (
            <div key={num}>{num}</div>
          ))}
        </div>

        {/* Highlighted Code Block */}
        <div className="flex-1 min-w-0 overflow-x-auto text-zinc-900 dark:text-zinc-200">
          <pre className="m-0 p-0 font-mono whitespace-pre">
            <code dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
          </pre>
        </div>
      </div>

      {/* Monochromatic Status Footer */}
      <div className="h-6 bg-zinc-100 dark:bg-zinc-950 text-zinc-600 dark:text-zinc-400 border-t border-zinc-200 dark:border-zinc-800 px-3 flex items-center justify-between text-[10px] font-mono select-none shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-zinc-900 dark:text-zinc-200 font-semibold">AIMLite</span>
          <span>{activeTab}</span>
          <span className="text-zinc-400 dark:text-zinc-600">•</span>
          <span>Read-Only</span>
        </div>
        <div className="flex items-center gap-2">
          <span>Ln {lineCount}, Col 1</span>
          <span>{getLanguageLabel()}</span>
        </div>
      </div>
    </aside>
  );
}
