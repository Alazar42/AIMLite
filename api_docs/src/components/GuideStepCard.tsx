import { useState, useMemo } from 'react';
import {
  Copy,
  Check,
  FileCode,
  Terminal,
  ExternalLink,
  Code2,
  FileText,
  Braces,
} from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markdown';
import { type GuideStep } from '../data/paradigmGuides';

interface GuideStepCardProps {
  step: GuideStep;
}

export default function GuideStepCard({ step }: GuideStepCardProps) {
  const [hasCopied, setHasCopied] = useState<boolean>(false);

  const copyCode = () => {
    navigator.clipboard.writeText(step.code);
    setHasCopied(true);
    setTimeout(() => setHasCopied(false), 1800);
  };

  const highlightedHtml = useMemo(() => {
    const raw = step.code;
    let grammar = Prism.languages.python;
    let lang = 'python';

    if (step.language === 'bash' || step.filename?.endsWith('.sh')) {
      grammar = Prism.languages.bash;
      lang = 'bash';
    } else if (step.language === 'json' || step.filename?.endsWith('.json')) {
      grammar = Prism.languages.json;
      lang = 'json';
    } else if (step.language === 'markdown' || step.filename?.endsWith('.md')) {
      grammar = Prism.languages.markdown;
      lang = 'markdown';
    }

    if (grammar) {
      try {
        return Prism.highlight(raw, grammar, lang);
      } catch {
        return raw;
      }
    }
    return raw;
  }, [step.code, step.language, step.filename]);

  const lineCount = step.code.split('\n').length;
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1);

  const renderFileIcon = () => {
    const fn = step.filename || '';
    if (fn.endsWith('.sh') || step.language === 'bash') {
      return <Terminal size={14} className="text-[#89e051]" />;
    }
    if (fn.endsWith('.json') || step.language === 'json') {
      return <Braces size={14} className="text-[#cbcb41]" />;
    }
    if (fn.endsWith('.md') || step.language === 'markdown') {
      return <FileText size={14} className="text-[#4ec9b0]" />;
    }
    return <FileCode size={14} className="text-[#3572A5]" />;
  };

  return (
    <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-4 sm:p-5 space-y-4 hover:border-zinc-700/70 transition-all shadow-md">
      {/* Step Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-800/60 pb-3">
        <div className="flex items-center gap-2.5">
          <span className="flex items-center justify-center w-6 h-6 rounded-full bg-sky-500/20 text-sky-400 font-mono text-xs font-bold border border-sky-500/30 shrink-0">
            {step.stepNumber}
          </span>
          <h3 className="text-sm sm:text-base font-bold text-zinc-100 tracking-tight">
            {step.title}
          </h3>
        </div>

        {step.badge && (
          <span className="self-start sm:self-auto text-[10px] font-mono uppercase px-2 py-0.5 rounded font-semibold bg-zinc-800 text-zinc-300 border border-zinc-700">
            {step.badge}
          </span>
        )}
      </div>

      {/* Description */}
      <p className="text-xs text-zinc-300 leading-relaxed font-sans">
        {step.description}
      </p>

      {/* External Link (e.g. Kaggle Download) */}
      {step.externalLink && (
        <a
          href={step.externalLink.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 border border-sky-500/30 text-sky-300 text-xs font-medium transition-colors"
        >
          <span>{step.externalLink.label}</span>
          <ExternalLink size={13} className="shrink-0" />
        </a>
      )}

      {/* Code Card */}
      <div className="rounded-lg overflow-hidden border border-zinc-800 bg-[#1e1e1e] font-mono text-xs shadow-inner">
        {/* Code Header Bar */}
        <div className="flex items-center justify-between bg-[#252526] border-b border-[#191919] px-3 py-1.5 select-none">
          <div className="flex items-center gap-2">
            {renderFileIcon()}
            <span className="font-mono text-[11px] text-zinc-200 font-medium">
              {step.filename || 'code'}
            </span>
            <span className="text-[10px] text-zinc-500 hidden sm:inline">
              ({lineCount} lines)
            </span>
          </div>

          <button
            onClick={copyCode}
            className="flex items-center gap-1 px-2 py-1 rounded text-[11px] text-zinc-400 hover:text-zinc-100 hover:bg-[#333333] transition-colors"
            title={`Copy ${step.filename || 'code'}`}
          >
            {hasCopied ? (
              <>
                <Check size={12} className="text-[#4ec9b0]" />
                <span className="text-[#4ec9b0] font-medium">Copied!</span>
              </>
            ) : (
              <>
                <Copy size={12} />
                <span>Copy {step.filename || 'code'}</span>
              </>
            )}
          </button>
        </div>

        {/* Code Body */}
        <div className="p-3 flex font-mono text-[11px] leading-[1.6] max-h-96 overflow-y-auto overflow-x-auto">
          {/* Line Numbers Gutter */}
          <div className="select-none text-right pr-3 text-[#858585] border-r border-[#2d2d2d] mr-3 shrink-0">
            {lineNumbers.map((num) => (
              <div key={num}>{num}</div>
            ))}
          </div>

          {/* Code Text */}
          <div className="flex-1 min-w-0 text-[#d4d4d4]">
            <pre className="m-0 p-0 font-mono whitespace-pre">
              <code dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
            </pre>
          </div>
        </div>
      </div>

      {/* Why This Code Callout */}
      {step.whyCode && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-sky-500/5 border border-sky-500/20 text-xs">
          <Code2 size={14} className="text-sky-400 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <strong className="text-sky-300 font-medium">Why this code: </strong>
            <span className="text-zinc-300">{step.whyCode}</span>
          </div>
        </div>
      )}
    </div>
  );
}
