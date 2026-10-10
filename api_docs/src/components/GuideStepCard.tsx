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
  Box,
  Layers,
} from 'lucide-react';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import 'prismjs/components/prism-bash';
import 'prismjs/components/prism-json';
import 'prismjs/components/prism-markdown';
import 'prismjs/components/prism-docker';
import 'prismjs/components/prism-yaml';
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

    if (step.language === 'docker' || step.filename === 'Dockerfile' || step.filename?.endsWith('.dockerfile')) {
      grammar = Prism.languages.docker || Prism.languages.bash;
      lang = 'docker';
    } else if (step.language === 'yaml' || step.filename?.endsWith('.yml') || step.filename?.endsWith('.yaml')) {
      grammar = Prism.languages.yaml || Prism.languages.bash;
      lang = 'yaml';
    } else if (step.language === 'bash' || step.filename?.endsWith('.sh')) {
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
    const iconClass = "text-zinc-500 dark:text-zinc-400";
    if (fn === 'Dockerfile' || step.language === 'docker') {
      return <Box size={14} className={iconClass} />;
    }
    if (fn.endsWith('.yml') || fn.endsWith('.yaml') || step.language === 'yaml') {
      return <Layers size={14} className={iconClass} />;
    }
    if (fn.endsWith('.sh') || step.language === 'bash') {
      return <Terminal size={14} className={iconClass} />;
    }
    if (fn.endsWith('.json') || step.language === 'json') {
      return <Braces size={14} className={iconClass} />;
    }
    if (fn.endsWith('.md') || step.language === 'markdown') {
      return <FileText size={14} className={iconClass} />;
    }
    return <FileCode size={14} className={iconClass} />;
  };

    return (
    <div className="rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md p-5 sm:p-6 space-y-4 hover:border-zinc-300 dark:hover:border-zinc-700 transition-all shadow-xs">
      {/* Step Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-zinc-200/80 dark:border-zinc-800/60 pb-3.5">
        <div className="flex items-center gap-3">
          <span className="flex items-center justify-center w-7 h-7 rounded-xl bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-mono text-xs font-bold shadow-xs shrink-0">
            {step.stepNumber}
          </span>
          <h3 className="text-sm sm:text-base font-bold text-zinc-900 dark:text-white tracking-tight font-heading">
            {step.title}
          </h3>
        </div>

        {step.badge && (
          <span className="self-start sm:self-auto text-[10px] font-mono uppercase px-2.5 py-1 rounded-lg font-bold bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-700">
            {step.badge}
          </span>
        )}
      </div>

      {/* Description */}
      <p className="text-xs text-zinc-600 dark:text-zinc-300 leading-relaxed font-sans">
        {step.description}
      </p>

      {/* External Link (e.g. Kaggle Download) */}
      {step.externalLink && (
        <a
          href={step.externalLink.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-zinc-100 hover:bg-zinc-200 dark:bg-zinc-800/80 dark:hover:bg-zinc-700 border border-zinc-200 dark:border-zinc-700 text-zinc-900 dark:text-white text-xs font-medium transition-colors"
        >
          <span>{step.externalLink.label}</span>
          <ExternalLink size={13} className="shrink-0 text-zinc-400" />
        </a>
      )}

      {/* Code Card with macOS Window Controls */}
      <div className="rounded-xl overflow-hidden border border-zinc-200/80 dark:border-zinc-800 bg-[#0d0e13] font-mono text-xs shadow-inner">
        {/* Code Header Bar with macOS Window Dots */}
        <div className="flex items-center justify-between bg-[#15171e] border-b border-zinc-800/80 px-3 py-2 select-none">
          <div className="flex items-center gap-2.5">
            {/* macOS traffic light dots */}
            <div className="flex items-center gap-1.5 mr-1">
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700 inline-block" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700 inline-block" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700 inline-block" />
            </div>

            <div className="flex items-center gap-1.5">
              {renderFileIcon()}
              <span className="font-mono text-[11px] text-zinc-200 font-medium">
                {step.filename || 'code'}
              </span>
            </div>
            <span className="text-[10px] text-zinc-500 hidden sm:inline">
              ({lineCount} lines)
            </span>
          </div>

          <button
            onClick={copyCode}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] bg-zinc-800 hover:bg-zinc-700 text-zinc-300 hover:text-white transition-colors border border-zinc-700/60"
            title={`Copy ${step.filename || 'code'}`}
          >
            {hasCopied ? (
              <>
                <Check size={12} className="text-zinc-100" />
                <span className="text-zinc-100 font-medium">Copied!</span>
              </>
            ) : (
              <>
                <Copy size={12} />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>

        {/* Code Body */}
        <div className="p-3.5 flex font-mono text-[11px] leading-[1.65] max-h-96 overflow-y-auto overflow-x-auto text-zinc-200">
          {/* Line Numbers Gutter */}
          <div className="select-none text-right pr-3.5 text-zinc-600 border-r border-zinc-800 mr-3.5 shrink-0">
            {lineNumbers.map((num) => (
              <div key={num}>{num}</div>
            ))}
          </div>

          {/* Code Text */}
          <div className="flex-1 min-w-0">
            <pre className="m-0 p-0 font-mono whitespace-pre">
              <code dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
            </pre>
          </div>
        </div>
      </div>

      {/* Why This Code Callout */}
      {step.whyCode && (
        <div className="flex items-start gap-2.5 p-3.5 rounded-xl bg-zinc-100/70 dark:bg-zinc-900/60 border border-zinc-200/80 dark:border-zinc-800 text-xs">
          <Code2 size={15} className="text-zinc-500 shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <strong className="text-zinc-900 dark:text-zinc-100 font-semibold">Why this code: </strong>
            <span className="text-zinc-600 dark:text-zinc-300">{step.whyCode}</span>
          </div>
        </div>
      )}
    </div>
  );
}

