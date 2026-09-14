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

type TabType = 'python' | 'cli' | 'curl';

export default function CodePlayground({ section }: CodePlaygroundProps) {
  const [activeTab, setActiveTab] = useState<TabType>('python');
  const [hasCopiedCode, setHasCopiedCode] = useState<boolean>(false);

  useEffect(() => {
    if (section.snippets.python) {
      setActiveTab('python');
    } else if (section.snippets.cli) {
      setActiveTab('cli');
    } else if (section.snippets.curl) {
      setActiveTab('curl');
    }
  }, [section.id, section.snippets]);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setHasCopiedCode(true);
    setTimeout(() => setHasCopiedCode(false), 1500);
  };

  const getCodeSnippet = (): string => {
    switch (activeTab) {
      case 'python':
        return section.snippets.python || '# Python snippet';
      case 'cli':
        return section.snippets.cli || `modelkit ${section.signatureOrPath}`;
      case 'curl':
        return (
          section.snippets.curl ||
          `curl -X POST http://127.0.0.1:8000/predict \\\n  -H "Content-Type: application/json" \\\n  -d '{"features": [1.5, 2.7, 3.2]}'`
        );
      default:
        return section.snippets.python || '';
    }
  };

  const currentSnippet = getCodeSnippet();

  const highlightedHtml = useMemo(() => {
    const raw = currentSnippet;
    let grammar = Prism.languages.python;
    let lang = 'python';

    if (activeTab === 'cli' || activeTab === 'curl') {
      grammar = Prism.languages.bash;
      lang = 'bash';
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

  const lineCount = currentSnippet.split('\n').length;
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1);

  const getTabFileName = (tab: TabType) => {
    if (tab === 'python') {
      if (section.id === 'pillar-data') return 'data.py';
      if (section.id === 'pillar-model') return 'model.py';
      if (section.id === 'pillar-lifecycle') return 'trainer.py';
      if (section.id.startsWith('foundations-')) return 'config.py';
      return 'main.py';
    }
    if (tab === 'cli') return 'terminal.sh';
    return 'request.sh';
  };

  return (
    <aside className="w-full h-full flex flex-col bg-[#1e1e1e] border-l border-zinc-800 font-mono text-xs select-text shadow-2xl">
      {/* VS Code Tab Bar */}
      <div className="flex items-center justify-between bg-[#252526] border-b border-[#191919] px-2 shrink-0 select-none">
        <div className="flex items-center">
          {section.snippets.python && (
            <button
              onClick={() => setActiveTab('python')}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs border-r border-[#191919] transition-colors relative ${
                activeTab === 'python'
                  ? 'bg-[#1e1e1e] text-zinc-100 font-medium'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-[#2a2d2e]'
              }`}
            >
              {activeTab === 'python' && (
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-[#007acc]" />
              )}
              <FileCode size={13} className="text-[#3572A5]" />
              <span className="font-mono text-[11px]">{getTabFileName('python')}</span>
            </button>
          )}

          {(section.snippets.cli || section.id.startsWith('cli-')) && (
            <button
              onClick={() => setActiveTab('cli')}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs border-r border-[#191919] transition-colors relative ${
                activeTab === 'cli'
                  ? 'bg-[#1e1e1e] text-zinc-100 font-medium'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-[#2a2d2e]'
              }`}
            >
              {activeTab === 'cli' && (
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-[#007acc]" />
              )}
              <Terminal size={13} className="text-[#89e051]" />
              <span className="font-mono text-[11px]">{getTabFileName('cli')}</span>
            </button>
          )}

          {section.snippets.curl && (
            <button
              onClick={() => setActiveTab('curl')}
              className={`flex items-center gap-1.5 px-3 py-2 text-xs border-r border-[#191919] transition-colors relative ${
                activeTab === 'curl'
                  ? 'bg-[#1e1e1e] text-zinc-100 font-medium'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-[#2a2d2e]'
              }`}
            >
              {activeTab === 'curl' && (
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-[#007acc]" />
              )}
              <Terminal size={13} className="text-[#e34c26]" />
              <span className="font-mono text-[11px]">{getTabFileName('curl')}</span>
            </button>
          )}
        </div>

        {/* Copy Button */}
        <button
          onClick={() => copyToClipboard(currentSnippet)}
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px] text-zinc-400 hover:text-zinc-200 hover:bg-[#333333] transition-colors"
          title="Copy Code"
        >
          {hasCopiedCode ? (
            <>
              <Check size={12} className="text-[#4ec9b0]" />
              <span className="text-[#4ec9b0]">Copied</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy</span>
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
          <span>UTF-8</span>
        </div>
        <div className="flex items-center gap-2">
          <span>Ln {lineCount}, Col 1</span>
          <span>{activeTab === 'python' ? 'Python' : 'Shell'}</span>
        </div>
      </div>
    </aside>
  );
}
