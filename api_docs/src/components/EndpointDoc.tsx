import React, { useState } from 'react';
import {
  Copy,
  Check,
  ChevronRight,
  ChevronDown,
  BookOpen,
  Database,
  Cpu,
  RefreshCw,
  FolderTree,
  AlertCircle,
  Code2,
  Sparkles,
  Layers,
} from 'lucide-react';
import { type DocSection, type DocParameter } from '../data/aimliteDocs';
import GuideStepCard from './GuideStepCard';

interface EndpointDocProps {
  section: DocSection;
  onSelectSection: (id: string) => void;
  onCopySignature: (text: string) => void;
  hasCopiedSignature: boolean;
}

export default function EndpointDoc({
  section,
  onSelectSection,
  onCopySignature,
  hasCopiedSignature,
}: EndpointDocProps) {
  const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});

  const isGuideSection = Boolean(section.guideSteps && section.guideSteps.length > 0);

  const toggleRow = (name: string) => {
    setExpandedRows((prev) => ({
      ...prev,
      [name]: !prev[name],
    }));
  };

  const renderParameterRow = (param: DocParameter, depth: number = 0) => {
    const hasChildren = Boolean(param.children && param.children.length > 0);
    const isExpanded = expandedRows[param.name];

    return (
      <React.Fragment key={param.name}>
        <tr className="border-b border-zinc-800/60 hover:bg-zinc-900/30 text-xs">
          <td className="py-2 px-3 align-top">
            <div className="flex items-center gap-1" style={{ paddingLeft: `${depth * 12}px` }}>
              {hasChildren && (
                <button
                  onClick={() => toggleRow(param.name)}
                  className="p-0.5 rounded text-zinc-500 hover:text-zinc-300"
                >
                  {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                </button>
              )}
              <span className="font-mono font-medium text-zinc-200">{param.name}</span>
            </div>
          </td>

          <td className="py-2 px-3 align-top font-mono text-[11px] text-zinc-400">
            <code className="bg-zinc-900 px-1.5 py-0.5 rounded text-zinc-300 border border-zinc-800">
              {param.type}
            </code>
          </td>

          <td className="py-2 px-3 align-top">
            {param.required ? (
              <span className="text-[10px] font-mono uppercase px-1.5 py-0.2 rounded text-rose-400 bg-rose-500/10 border border-rose-500/20">
                Required
              </span>
            ) : (
              <span className="text-[10px] font-mono text-zinc-500">Optional</span>
            )}
          </td>

          <td className="py-2 px-3 align-top text-zinc-400 text-[11px] leading-relaxed">
            <div>{param.description}</div>
            {param.defaultValue && (
              <div className="mt-0.5 font-mono text-zinc-500">
                Default: <code className="text-sky-400">{param.defaultValue}</code>
              </div>
            )}
          </td>
        </tr>

        {hasChildren &&
          isExpanded &&
          param.children!.map((child) => renderParameterRow(child, depth + 1))}
      </React.Fragment>
    );
  };

  return (
    <div className="w-full max-w-3xl py-6 px-4 lg:px-6 space-y-6 select-text">
      {/* Top Switcher: 3 Paradigms for Guides, 3 Pillars for Core API */}
      {isGuideSection ? (
        <div className="flex items-center gap-1.5 p-1 bg-zinc-900/80 border border-zinc-800 rounded-xl text-xs overflow-x-auto">
          <button
            onClick={() => onSelectSection('paradigm-scratch')}
            className={`flex-1 min-w-[140px] flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg font-medium transition-all ${
              section.id === 'paradigm-scratch'
                ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
            }`}
          >
            <Layers size={13} className="text-sky-400 shrink-0" />
            <span>1. Scratch (Churn)</span>
          </button>

          <span className="text-zinc-600 text-xs shrink-0">→</span>

          <button
            onClick={() => onSelectSection('paradigm-rag')}
            className={`flex-1 min-w-[140px] flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg font-medium transition-all ${
              section.id === 'paradigm-rag'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
            }`}
          >
            <Sparkles size={13} className="text-purple-400 shrink-0" />
            <span>2. RAG (Knowledge QA)</span>
          </button>

          <span className="text-zinc-600 text-xs shrink-0">→</span>

          <button
            onClick={() => onSelectSection('paradigm-adapters')}
            className={`flex-1 min-w-[140px] flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg font-medium transition-all ${
              section.id === 'paradigm-adapters'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
            }`}
          >
            <Cpu size={13} className="text-emerald-400 shrink-0" />
            <span>3. Adapters (LoRA)</span>
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-1.5 p-1 bg-zinc-900/80 border border-zinc-800 rounded-xl text-xs">
          <button
            onClick={() => onSelectSection('pillar-data')}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg font-medium transition-all ${
              section.id === 'pillar-data'
                ? 'bg-sky-500/20 text-sky-300 border border-sky-500/30 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
            }`}
          >
            <Database size={13} className="text-sky-400 shrink-0" />
            <span>1. Data</span>
          </button>

          <span className="text-zinc-600 text-xs">→</span>

          <button
            onClick={() => onSelectSection('pillar-model')}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg font-medium transition-all ${
              section.id === 'pillar-model'
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
            }`}
          >
            <Cpu size={13} className="text-purple-400 shrink-0" />
            <span>2. Model</span>
          </button>

          <span className="text-zinc-600 text-xs">→</span>

          <button
            onClick={() => onSelectSection('pillar-lifecycle')}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-lg font-medium transition-all ${
              section.id === 'pillar-lifecycle'
                ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 shadow-sm'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60'
            }`}
          >
            <RefreshCw size={13} className="text-emerald-400 shrink-0" />
            <span>3. Lifecycle</span>
          </button>
        </div>
      )}

      {/* Header & Path */}
      <div className="space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded font-semibold bg-zinc-800 text-zinc-300 border border-zinc-700">
              {section.badge.label}
            </span>
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-zinc-100">
              {section.title}
            </h1>
          </div>
        </div>

        <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed">
          {section.overview}
        </p>

        {/* Minimal Signature / Quick Copy */}
        <div className="flex items-center justify-between gap-2 px-3 py-2 bg-zinc-900/90 border border-zinc-800 rounded-lg font-mono text-xs">
          <span className="text-zinc-300 truncate select-all">{section.signatureOrPath}</span>
          <button
            onClick={() => onCopySignature(section.signatureOrPath)}
            className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-zinc-200 shrink-0 transition-colors"
          >
            {hasCopiedSignature ? (
              <Check size={12} className="text-emerald-400" />
            ) : (
              <Copy size={12} />
            )}
            <span>{hasCopiedSignature ? 'Copied' : 'Copy'}</span>
          </button>
        </div>
      </div>

      {/* ================================================================= */}
      {/* IN-PAGE CODE WORKFLOW (For Paradigm Guides)                       */}
      {/* ================================================================= */}
      {isGuideSection ? (
        <div className="space-y-5 pt-2">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <div className="flex items-center gap-2">
              <Sparkles size={15} className="text-sky-400" />
              <h2 className="text-xs sm:text-sm font-bold text-zinc-100 uppercase tracking-wider">
                In-Page Project Implementation
              </h2>
            </div>
            <span className="text-[11px] text-zinc-400 font-mono">
              {section.guideSteps!.length} Sequential Steps
            </span>
          </div>

          <div className="space-y-5">
            {section.guideSteps!.map((step) => (
              <GuideStepCard key={step.stepNumber} step={step} />
            ))}
          </div>

          {/* Quick Notice */}
          <div className="flex items-center gap-2 p-3 rounded-lg bg-zinc-900/50 border border-zinc-800/80 text-xs text-zinc-400">
            <AlertCircle size={14} className="text-sky-400 shrink-0" />
            <span>
              All scripts can also be selected and copied individually via the code panel tabs on the right.
            </span>
          </div>
        </div>
      ) : (
        /* ================================================================= */
        /* STANDARD DOCUMENTATION VIEW (For Pillars & API Reference)         */
        /* ================================================================= */
        <>
          {/* Django Analogy */}
          <div className="p-3.5 rounded-xl bg-zinc-900/50 border border-zinc-800/90 space-y-2">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400">
              <BookOpen size={13} />
              <span>Django Analogy</span>
            </div>
            <p className="text-xs text-zinc-300 leading-relaxed font-sans">
              {section.djangoAnalogy}
            </p>
          </div>

          {/* Why Code / Design Rationale */}
          {section.whyCode && section.whyCode.length > 0 && (
            <div className="space-y-2.5">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-sky-400">
                <Code2 size={13} />
                <span>Why This Code (Component Rationale)</span>
              </div>

              <div className="grid grid-cols-1 gap-2 text-xs">
                {section.whyCode.map((item) => (
                  <div
                    key={item.component}
                    className="p-3 rounded-xl bg-zinc-900/60 border border-zinc-800/80 space-y-1 hover:border-zinc-700/80 transition-colors"
                  >
                    <div className="font-mono text-xs font-semibold text-sky-300">
                      {item.component}
                    </div>
                    <div className="text-xs text-zinc-300 leading-relaxed font-sans">
                      {item.reason}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Conventions */}
          {section.conventions && section.conventions.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-zinc-300">
                <FolderTree size={13} className="text-sky-400" />
                <span>Key Conventions</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                {section.conventions.map((conv) => (
                  <div
                    key={conv.title}
                    className="p-2.5 rounded-lg bg-zinc-900/40 border border-zinc-800/80 space-y-1"
                  >
                    <div className="font-semibold text-zinc-200 text-[11px]">{conv.title}</div>
                    <div className="text-zinc-400 text-[11px] leading-snug">{conv.description}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Parameters Table */}
          {section.parameters && section.parameters.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-zinc-300">
                {section.parametersTitle || 'Attributes & Options'}
              </div>

              <div className="border border-zinc-800 rounded-lg overflow-hidden bg-zinc-950/40">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-zinc-800 bg-zinc-900/60 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
                      <th className="py-2 px-3">Name</th>
                      <th className="py-2 px-3">Type</th>
                      <th className="py-2 px-3">Req</th>
                      <th className="py-2 px-3">Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {section.parameters.map((param) => renderParameterRow(param))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Strict Contract Tip */}
          <div className="flex items-center gap-2 p-2.5 rounded-lg bg-zinc-900/30 border border-zinc-800/60 text-[11px] text-zinc-400">
            <AlertCircle size={13} className="text-amber-400 shrink-0" />
            <span>
              <strong className="text-zinc-300">Convention rule:</strong> AIMLite strictly halts (exit code 1) if starter directories or files are empty to prevent invalid training cycles.
            </span>
          </div>
        </>
      )}
    </div>
  );
}
