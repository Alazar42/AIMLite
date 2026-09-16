import { useState, useMemo, useEffect } from 'react';
import { Copy, Check, FileCode, Terminal } from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import { type DocSection } from '../data/modelkitDocs';

interface CodePlaygroundProps {
  section: DocSection;
}

export default function CodePlayground({ section }: CodePlaygroundProps) {
  const [activeTab, setActiveTab] = useState<string>('');
  const [hasCopiedCode, setHasCopiedCode] = useState<boolean>(false);

  // Compute available tabs dynamically from section.snippets
  const availableTabs = useMemo(() => {
    const tabs: { id: string; label: string; type: 'py' | 'sh' | 'json' }[] = [];

    // 1. Individual project files (data.py, model.py, trainer.py, evaluator.py, inference.py, config.py)
    if (section.snippets.files && Object.keys(section.snippets.files).length > 0) {
      for (const fname of Object.keys(section.snippets.files)) {
        const type = fname.endsWith('.json') ? 'json' : fname.endsWith('.sh') ? 'sh' : 'py';
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
      return section.snippets.cli || `modelkit ${section.signatureOrPath}`;
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

    if (activeTab.endsWith('.sh')) {
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

  return (
    <aside className="w-full h-full flex flex-col bg-[#1e1e1e] border-l border-zinc-800 font-mono text-xs select-text shadow-2xl">
      {/* VS Code Tab Bar with Individual File Tabs */}
      <div className="flex items-center justify-between bg-[#252526] border-b border-[#191919] px-1 shrink-0 select-none overflow-x-auto">
        <div className="flex items-center overflow-x-auto scrollbar-none">
          {availableTabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-2 text-xs border-r border-[#191919] transition-colors relative shrink-0 ${
                  isActive
                    ? 'bg-[#1e1e1e] text-zinc-100 font-medium'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-[#2a2d2e]'
                }`}
                title={`View and copy ${tab.label}`}
              >
                {isActive && (
                  <div className="absolute top-0 left-0 right-0 h-[2px] bg-[#007acc]" />
                )}
                {tab.type === 'py' && <FileCode size={13} className="text-[#3572A5]" />}
                {tab.type === 'sh' && <Terminal size={13} className="text-[#89e051]" />}
                {tab.type === 'json' && <FileCode size={13} className="text-[#cbcb41]" />}
                <span className="font-mono text-[11px]">{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Copy Active File Button */}
        <button
          onClick={() => copyToClipboard(currentSnippet)}
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-zinc-400 hover:text-zinc-200 hover:bg-[#333333] transition-colors shrink-0 ml-2"
          title={`Copy ${activeTab || 'code'}`}
        >
          {hasCopiedCode ? (
            <>
              <Check size={12} className="text-[#4ec9b0]" />
              <span className="text-[#4ec9b0]">Copied</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy {activeTab || ''}</span>
            </>
          )}
        </button>
      </div>

      {/* VS Code Code Surface with Line Numbers */}
      <div className="flex-1 overflow-y-auto p-3 flex font-mono text-[11px] leading-[1.6] bg-[#1e1e1e]">
        {/* Line Numbers Gutter */}
        <div className="select-none text-right pr-3 text-[#858585] border-r border-[#2d2d2d] mr-3 shrink-0">
          {lineNumbers.map((num) => (
            <div key={num}>{num}</div>
          ))}
        </div>

        {/* Highlighted Code Block */}
        <div className="flex-1 min-w-0 overflow-x-auto text-[#d4d4d4]">
          <pre className="m-0 p-0 font-mono whitespace-pre">
            <code dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
          </pre>
        </div>
      </div>

      {/* VS Code Status Footer */}
      <div className="h-6 bg-[#007acc] text-white px-3 flex items-center justify-between text-[10px] font-mono select-none shrink-0">
        <div className="flex items-center gap-2">
          <span>ModelKit</span>
          <span>{activeTab}</span>
          <span>UTF-8</span>
        </div>
        <div className="flex items-center gap-2">
          <span>Ln {lineCount}, Col 1</span>
          <span>{activeTab.endsWith('.sh') ? 'Shell' : activeTab.endsWith('.json') ? 'JSON' : 'Python'}</span>
        </div>
      </div>
    </aside>
  );
}
