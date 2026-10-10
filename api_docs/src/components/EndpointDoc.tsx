import React, { useState, useMemo } from 'react';
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
  Layers,
  History,
  Tag,
  ArrowLeft,
  Terminal,
  Sparkles,
  CheckCircle2,
} from 'lucide-react';
import { type DocSection, type DocParameter, NAVIGATION_CATEGORIES, DOC_SECTIONS } from '../data/aimliteDocs';
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
  const isChangelog = section.id === 'changelog';

  const toggleRow = (name: string) => {
    setExpandedRows((prev) => ({
      ...prev,
      [name]: !prev[name],
    }));
  };

  // Compute Next and Previous sections for bottom pagination
  const { prevSection, nextSection } = useMemo(() => {
    const allIds = NAVIGATION_CATEGORIES.flatMap((c) => c.items.map((i) => i.id));
    const currentIndex = allIds.indexOf(section.id);
    const prevId = currentIndex > 0 ? allIds[currentIndex - 1] : null;
    const nextId = currentIndex >= 0 && currentIndex < allIds.length - 1 ? allIds[currentIndex + 1] : null;

    return {
      prevSection: prevId ? DOC_SECTIONS[prevId] : null,
      nextSection: nextId ? DOC_SECTIONS[nextId] : null,
    };
  }, [section.id]);

  // Unified Badge Style (Single cohesive set of colors)
  const getBadgeStyle = () => {
    return 'bg-zinc-100 dark:bg-zinc-800/90 text-zinc-800 dark:text-zinc-200 border-zinc-200 dark:border-zinc-700';
  };

  // Unified Type Badge Style for parameters
  const getTypeBadgeStyle = () => {
    return 'bg-zinc-100 dark:bg-zinc-800/80 text-zinc-700 dark:text-zinc-300 border-zinc-200 dark:border-zinc-700';
  };

  const isParadigmGuide = section.id.startsWith('paradigm-');
  const isPillarSection = section.id.startsWith('pillar-');

  const renderParameterRow = (param: DocParameter, depth: number = 0) => {
    const hasChildren = Boolean(param.children && param.children.length > 0);
    const isExpanded = expandedRows[param.name];

    return (
      <React.Fragment key={param.name}>
        <tr className="border-b border-zinc-200/70 dark:border-zinc-800/60 hover:bg-zinc-100/60 dark:hover:bg-zinc-900/40 text-xs transition-colors group">
          <td className="py-3 px-3.5 align-top">
            <div className="flex items-center gap-1.5" style={{ paddingLeft: `${depth * 14}px` }}>
              {hasChildren && (
                <button
                  onClick={() => toggleRow(param.name)}
                  className="p-1 rounded-md text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-200/60 dark:hover:bg-zinc-800 transition-colors"
                >
                  {isExpanded ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                </button>
              )}
              <span className="font-mono font-semibold text-zinc-900 dark:text-zinc-100 tracking-tight">
                {param.name}
              </span>
            </div>
          </td>

          <td className="py-3 px-3.5 align-top font-mono text-[11px]">
            <span className={`inline-block px-2 py-0.5 rounded-md border font-medium ${getTypeBadgeStyle()}`}>
              {param.type}
            </span>
          </td>

          <td className="py-3 px-3.5 align-top">
            {param.required ? (
              <span className="inline-flex items-center gap-1 text-[10px] font-mono uppercase px-2 py-0.5 rounded-md text-zinc-900 dark:text-zinc-100 bg-zinc-200 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 font-bold tracking-wider">
                Required
              </span>
            ) : (
              <span className="text-[10px] font-mono text-zinc-400 dark:text-zinc-500">
                Optional
              </span>
            )}
          </td>

          <td className="py-3 px-3.5 align-top text-zinc-600 dark:text-zinc-300 text-[11px] leading-relaxed">
            <div>{param.description}</div>
            {param.defaultValue && (
              <div className="mt-1 font-mono text-zinc-500 dark:text-zinc-400 flex items-center gap-1.5">
                <span className="text-[10px] uppercase tracking-wider text-zinc-400">Default:</span>
                <code className="text-zinc-900 dark:text-zinc-200 font-medium px-1.5 py-0.2 bg-zinc-200/60 dark:bg-zinc-800 rounded text-[10px]">
                  {param.defaultValue}
                </code>
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
    <div className="w-full py-6 space-y-8 select-text">
      {/* ================================================================= */}
      {/* TOP PARADIGM SWITCHER (3 Paradigms Only - Deploy not included)    */}
      {/* ================================================================= */}
      {isParadigmGuide && (
        <div className="flex items-center gap-1.5 p-1.5 bg-white/70 dark:bg-zinc-900/70 backdrop-blur-xl border border-zinc-200/80 dark:border-zinc-800/80 rounded-2xl text-xs overflow-x-auto no-scrollbar shadow-xs">
          <button
            onClick={() => onSelectSection('paradigm-scratch')}
            className={`shrink-0 flex-1 min-w-[130px] flex items-center justify-center gap-2 py-2 px-3 rounded-xl font-medium transition-all ${
              section.id === 'paradigm-scratch'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50'
            }`}
          >
            <Layers size={14} className="shrink-0" />
            <span>1. Scratch (Churn)</span>
          </button>

          <span className="text-zinc-300 dark:text-zinc-700 text-xs shrink-0 font-mono">→</span>

          <button
            onClick={() => onSelectSection('paradigm-rag')}
            className={`shrink-0 flex-1 min-w-[130px] flex items-center justify-center gap-2 py-2 px-3 rounded-xl font-medium transition-all ${
              section.id === 'paradigm-rag'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50'
            }`}
          >
            <BookOpen size={14} className="shrink-0" />
            <span>2. RAG (Knowledge)</span>
          </button>

          <span className="text-zinc-300 dark:text-zinc-700 text-xs shrink-0 font-mono">→</span>

          <button
            onClick={() => onSelectSection('paradigm-adapters')}
            className={`shrink-0 flex-1 min-w-[130px] flex items-center justify-center gap-2 py-2 px-3 rounded-xl font-medium transition-all ${
              section.id === 'paradigm-adapters'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50'
            }`}
          >
            <Cpu size={14} className="shrink-0" />
            <span>3. Adapters (LoRA)</span>
          </button>
        </div>
      )}

      {/* ================================================================= */}
      {/* TOP PILLAR SWITCHER (3 Pillars Only)                              */}
      {/* ================================================================= */}
      {isPillarSection && (
        <div className="flex items-center gap-1.5 p-1.5 bg-white/70 dark:bg-zinc-900/70 backdrop-blur-xl border border-zinc-200/80 dark:border-zinc-800/80 rounded-2xl text-xs overflow-x-auto no-scrollbar shadow-xs">
          <button
            onClick={() => onSelectSection('pillar-data')}
            className={`shrink-0 flex-1 min-w-[95px] flex items-center justify-center gap-2 py-2 px-3 rounded-xl font-medium transition-all ${
              section.id === 'pillar-data'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50'
            }`}
          >
            <Database size={13} className="shrink-0 text-zinc-400 dark:text-zinc-500" />
            <span>1. Data</span>
          </button>

          <span className="text-zinc-300 dark:text-zinc-700 text-xs shrink-0 font-mono">→</span>

          <button
            onClick={() => onSelectSection('pillar-model')}
            className={`shrink-0 flex-1 min-w-[95px] flex items-center justify-center gap-2 py-2 px-3 rounded-xl font-medium transition-all ${
              section.id === 'pillar-model'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50'
            }`}
          >
            <Cpu size={13} className="shrink-0 text-zinc-400 dark:text-zinc-500" />
            <span>2. Model</span>
          </button>

          <span className="text-zinc-300 dark:text-zinc-700 text-xs shrink-0 font-mono">→</span>

          <button
            onClick={() => onSelectSection('pillar-lifecycle')}
            className={`shrink-0 flex-1 min-w-[95px] flex items-center justify-center gap-2 py-2 px-3 rounded-xl font-medium transition-all ${
              section.id === 'pillar-lifecycle'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100 hover:bg-zinc-200/50 dark:hover:bg-zinc-800/50'
            }`}
          >
            <RefreshCw size={13} className="shrink-0 text-zinc-400 dark:text-zinc-500" />
            <span>3. Lifecycle</span>
          </button>
        </div>
      )}

      {/* ================================================================= */}
      {/* ELEVATED HERO HEADER CARD                                         */}
      {/* ================================================================= */}
      <div className="relative rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 bg-white/70 dark:bg-zinc-900/50 backdrop-blur-xl p-5 sm:p-7 space-y-4 shadow-sm overflow-hidden">
        <div className="flex flex-wrap items-center gap-2">
          <span className={`text-[10px] font-mono uppercase px-2.5 py-1 rounded-lg font-bold border ${getBadgeStyle()}`}>
            {section.badge.label}
          </span>
          <span className="text-xs text-zinc-400 dark:text-zinc-500 font-mono flex items-center gap-1">
            <span>•</span>
            <span>AIMLite</span>
          </span>
          <span className="text-xs text-zinc-400 dark:text-zinc-500 font-mono hidden sm:inline-flex items-center gap-1">
            <span>•</span>
            <span className="text-zinc-600 dark:text-zinc-300 font-medium">Zero-Path Standard</span>
          </span>
        </div>

        <div className="space-y-2">
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-zinc-900 dark:text-white font-heading">
            {section.title}
          </h1>
          {section.subtitle && (
            <p className="text-xs sm:text-sm font-medium text-zinc-600 dark:text-zinc-400">
              {section.subtitle}
            </p>
          )}
          <p className="text-xs sm:text-sm text-zinc-600 dark:text-zinc-300 leading-relaxed font-sans max-w-4xl">
            {section.overview}
          </p>
        </div>

        {/* Interactive Signature / Endpoint Bar */}
        <div className="flex items-center justify-between gap-3 px-3.5 py-2.5 bg-zinc-900 dark:bg-black/90 text-zinc-100 border border-zinc-800 rounded-xl font-mono text-xs shadow-inner">
          <div className="flex items-center gap-2 min-w-0">
            <Terminal size={14} className="text-zinc-400 shrink-0" />
            <span className="truncate select-all text-zinc-200">{section.signatureOrPath}</span>
          </div>

          <button
            onClick={() => onCopySignature(section.signatureOrPath)}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 hover:text-white text-[11px] font-sans font-medium shrink-0 transition-colors border border-zinc-700/80"
          >
            {hasCopiedSignature ? (
              <>
                <Check size={12} className="text-zinc-100" />
                <span className="text-zinc-100">Copied</span>
              </>
            ) : (
              <>
                <Copy size={12} />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* ================================================================= */}
      {/* IN-PAGE CODE WORKFLOW (For Paradigm Guides)                       */}
      {/* ================================================================= */}
      {isGuideSection ? (
        <div className="space-y-6 pt-2">
          <div className="flex items-center justify-between border-b border-zinc-200/80 dark:border-zinc-800 pb-3">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center text-zinc-800 dark:text-zinc-200">
                <BookOpen size={16} />
              </div>
              <div>
                <h2 className="text-sm sm:text-base font-bold text-zinc-900 dark:text-white tracking-tight font-heading">
                  In-Page Project Implementation
                </h2>
                <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                  Complete file-by-file walkthrough with production calibration
                </div>
              </div>
            </div>
            <span className="text-xs text-zinc-600 dark:text-zinc-300 font-mono px-2.5 py-1 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 font-medium">
              {section.guideSteps!.length} Steps
            </span>
          </div>

          {/* Sequential Step Cards */}
          <div className="space-y-6">
            {section.guideSteps!.map((step) => (
              <GuideStepCard key={step.stepNumber} step={step} />
            ))}
          </div>

          {/* Console Sync Tip */}
          <div className="flex items-center gap-3 p-3.5 rounded-xl bg-zinc-100/70 dark:bg-zinc-900/40 border border-zinc-200/80 dark:border-zinc-800 text-xs text-zinc-600 dark:text-zinc-400">
            <Sparkles size={16} className="text-zinc-400 shrink-0" />
            <span>
              All scripts can also be selected, tested, and inspected via the synchronized <strong>Code Console</strong> on the right.
            </span>
          </div>

          {/* Parameters Table for Guide Sections (e.g. Docker Environment Variables) */}
          {section.parameters && section.parameters.length > 0 && (
            <div className="space-y-3 pt-4">
              <div className="flex items-center gap-2">
                <Tag size={15} className="text-zinc-500" />
                <h3 className="text-sm font-bold text-zinc-900 dark:text-white font-heading">
                  {section.parametersTitle || 'Environment Variables & Configuration'}
                </h3>
              </div>

              <div className="border border-zinc-200/80 dark:border-zinc-800 rounded-2xl overflow-hidden bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md shadow-xs">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[540px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-zinc-100/70 dark:bg-zinc-900/80 text-[10px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider font-mono">
                        <th className="py-3 px-3.5">Variable / Param</th>
                        <th className="py-3 px-3.5">Type</th>
                        <th className="py-3 px-3.5">Required</th>
                        <th className="py-3 px-3.5">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {section.parameters.map((param) => renderParameterRow(param))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Conventions for Guide Sections */}
          {section.conventions && section.conventions.length > 0 && (
            <div className="space-y-3 pt-4">
              <div className="flex items-center gap-2">
                <FolderTree size={15} className="text-zinc-500" />
                <h3 className="text-sm font-bold text-zinc-900 dark:text-white font-heading">
                  Production Architecture & Key Conventions
                </h3>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {section.conventions.map((conv) => (
                  <div
                    key={conv.title}
                    className="p-3.5 rounded-xl bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md border border-zinc-200/80 dark:border-zinc-800/80 space-y-1.5 hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors shadow-xs"
                  >
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 text-xs">
                      {conv.title}
                    </div>
                    <div className="text-zinc-600 dark:text-zinc-300 text-xs leading-relaxed">
                      {conv.description}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : isChangelog ? (
        /* ================================================================= */
        /* DEDICATED CHANGELOG VIEW                                          */
        /* ================================================================= */
        <div className="space-y-6 pt-2">
          <div className="flex items-center justify-between border-b border-zinc-200/80 dark:border-zinc-800 pb-3">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 flex items-center justify-center text-zinc-800 dark:text-zinc-200">
                <History size={16} />
              </div>
              <div>
                <h2 className="text-sm sm:text-base font-bold text-zinc-900 dark:text-white font-heading">
                  Release History & Migration Notes
                </h2>
                <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
                  Version-by-version architectural evolution of AIMLite
                </div>
              </div>
            </div>
            <span className="text-xs text-zinc-900 dark:text-zinc-100 font-mono bg-zinc-200/80 dark:bg-zinc-800 px-2.5 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 font-semibold">
              Latest: v2.1.2
            </span>
          </div>

          {section.conventions && (
            <div className="space-y-4">
              {section.conventions.map((release) => {
                const isLatest = release.title.includes('2.1.2');
                return (
                  <div
                    key={release.title}
                    className={`p-4 sm:p-5 rounded-2xl border transition-all ${
                      isLatest
                        ? 'bg-zinc-100 dark:bg-zinc-900 border-zinc-400 dark:border-zinc-600 shadow-sm'
                        : 'bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md border-zinc-200/80 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-2">
                      <div className="flex items-center gap-2">
                        <Tag size={14} className="text-zinc-500" />
                        <span className="font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100">
                          {release.title}
                        </span>
                      </div>
                      {isLatest && (
                        <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-bold bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 shadow-xs">
                          CURRENT
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-zinc-600 dark:text-zinc-300 leading-relaxed font-sans">
                      {release.description}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      ) : (
        /* ================================================================= */
        /* STANDARD DOCUMENTATION VIEW (For Pillars & API Reference)         */
        /* ================================================================= */
        <div className="space-y-6">
          {/* Django Analogy Featured Card */}
          {section.djangoAnalogy && (
            <div className="p-4 sm:p-5 rounded-2xl bg-zinc-100/70 dark:bg-zinc-900/40 border border-zinc-200/80 dark:border-zinc-800 space-y-2">
              <div className="flex items-center gap-2 text-xs font-bold text-zinc-900 dark:text-zinc-100 font-heading">
                <CheckCircle2 size={15} className="text-zinc-500" />
                <span>Django Analogy</span>
              </div>
              <p className="text-xs text-zinc-700 dark:text-zinc-300 leading-relaxed font-sans">
                {section.djangoAnalogy}
              </p>
            </div>
          )}

          {/* Why Code / Design Rationale */}
          {section.whyCode && section.whyCode.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-zinc-900 dark:text-zinc-100 font-heading">
                <Code2 size={15} className="text-zinc-500" />
                <span>Why This Code (Component Rationale)</span>
              </div>

              <div className="grid grid-cols-1 gap-2.5">
                {section.whyCode.map((item) => (
                  <div
                    key={item.component}
                    className="p-3.5 rounded-xl bg-white/70 dark:bg-zinc-900/50 backdrop-blur-md border border-zinc-200/80 dark:border-zinc-800/80 space-y-1 hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors shadow-xs"
                  >
                    <div className="font-mono text-xs font-bold text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 dark:bg-zinc-500 inline-block" />
                      <span>{item.component}</span>
                    </div>
                    <div className="text-xs text-zinc-600 dark:text-zinc-300 leading-relaxed font-sans pl-3.5">
                      {item.reason}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Key Conventions */}
          {section.conventions && section.conventions.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-zinc-900 dark:text-zinc-100 font-heading">
                <FolderTree size={15} className="text-zinc-500" />
                <span>Key Conventions</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {section.conventions.map((conv) => (
                  <div
                    key={conv.title}
                    className="p-3.5 rounded-xl bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md border border-zinc-200/80 dark:border-zinc-800/80 space-y-1 hover:border-zinc-300 dark:hover:border-zinc-700 transition-colors shadow-xs"
                  >
                    <div className="font-semibold text-zinc-900 dark:text-zinc-100 text-xs">
                      {conv.title}
                    </div>
                    <div className="text-zinc-600 dark:text-zinc-300 text-xs leading-relaxed">
                      {conv.description}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Parameters Table */}
          {section.parameters && section.parameters.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 text-xs font-bold text-zinc-900 dark:text-zinc-100 font-heading">
                <Tag size={15} className="text-zinc-500" />
                <span>{section.parametersTitle || 'Attributes & Options'}</span>
              </div>

              <div className="border border-zinc-200/80 dark:border-zinc-800 rounded-2xl overflow-hidden bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md shadow-xs">
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[540px] text-left border-collapse">
                    <thead>
                      <tr className="border-b border-zinc-200 dark:border-zinc-800 bg-zinc-100/70 dark:bg-zinc-900/80 text-[10px] font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-wider font-mono">
                        <th className="py-3 px-3.5">Name</th>
                        <th className="py-3 px-3.5">Type</th>
                        <th className="py-3 px-3.5">Required</th>
                        <th className="py-3 px-3.5">Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {section.parameters.map((param) => renderParameterRow(param))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Strict Contract Tip */}
          <div className="flex items-center gap-3 p-3.5 rounded-xl bg-zinc-100/80 dark:bg-zinc-900/40 border border-zinc-200/80 dark:border-zinc-800/80 text-xs text-zinc-600 dark:text-zinc-400">
            <AlertCircle size={15} className="text-zinc-500 shrink-0" />
            <span>
              <strong className="text-zinc-900 dark:text-zinc-200">Convention rule:</strong> AIMLite strictly halts (exit code 1) if starter directories or files are empty to prevent invalid training cycles.
            </span>
          </div>
        </div>
      )}

      {/* ================================================================= */}
      {/* BOTTOM PAGINATION: PREVIOUS / NEXT                                */}
      {/* ================================================================= */}
      <div className="pt-8 border-t border-zinc-200/80 dark:border-zinc-800/80 flex flex-col sm:flex-row items-center justify-between gap-3">
        {prevSection ? (
          <button
            onClick={() => onSelectSection(prevSection.id)}
            className="w-full sm:w-auto flex-1 flex items-center gap-3 p-3.5 rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700 bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md hover:bg-zinc-100/70 dark:hover:bg-zinc-800/60 text-left transition-all group shadow-xs"
          >
            <div className="w-8 h-8 rounded-xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center group-hover:bg-zinc-200 dark:group-hover:bg-zinc-700 transition-colors shrink-0">
              <ArrowLeft size={16} className="text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-white transition-colors" />
            </div>
            <div className="truncate">
              <div className="text-[10px] font-mono text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
                Previous
              </div>
              <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 group-hover:text-zinc-950 dark:group-hover:text-white transition-colors truncate">
                {prevSection.title}
              </div>
            </div>
          </button>
        ) : (
          <div className="hidden sm:block flex-1" />
        )}

        {nextSection && (
          <button
            onClick={() => onSelectSection(nextSection.id)}
            className="w-full sm:w-auto flex-1 flex items-center justify-end gap-3 p-3.5 rounded-2xl border border-zinc-200/80 dark:border-zinc-800/80 hover:border-zinc-300 dark:hover:border-zinc-700 bg-white/70 dark:bg-zinc-900/40 backdrop-blur-md hover:bg-zinc-100/70 dark:hover:bg-zinc-800/60 text-right transition-all group shadow-xs"
          >
            <div className="truncate">
              <div className="text-[10px] font-mono text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
                Next
              </div>
              <div className="text-xs font-bold text-zinc-800 dark:text-zinc-200 group-hover:text-zinc-950 dark:group-hover:text-white transition-colors truncate">
                {nextSection.title}
              </div>
            </div>
            <div className="w-8 h-8 rounded-xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center group-hover:bg-zinc-200 dark:group-hover:bg-zinc-700 transition-colors shrink-0">
              <ChevronRight size={16} className="text-zinc-400 group-hover:text-zinc-900 dark:group-hover:text-white transition-colors" />
            </div>
          </button>
        )}
      </div>
    </div>
  );
}

