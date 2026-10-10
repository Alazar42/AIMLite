import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Database,
  Cpu,
  RefreshCw,
  Terminal,
  Server,
  Settings,
  BookOpen,
  Layers,
  History,
  Home,
  FileCode,
  Box,
} from 'lucide-react';
import { NAVIGATION_CATEGORIES, type NavCategory } from '../data/aimliteDocs';

interface SidebarProps {
  activeSectionId: string;
  onSelectSection: (id: string) => void;
  onOpenSearch: () => void;
  onNavigateToOverview?: () => void;
  version: string;
  onChangeVersion: (ver: string) => void;
}

export default function Sidebar({
  activeSectionId,
  onSelectSection,
  onNavigateToOverview,
}: SidebarProps) {
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({
    'Supporting Foundations': false,
    'Zero-Path CLI Commands': false,
    'HTTP Inference Server': false,
    'Production & Deployment': false,
  });

  const toggleCategory = (catName: string) => {
    setCollapsedCategories((prev) => ({
      ...prev,
      [catName]: !prev[catName],
    }));
  };

  const getCategoryIcon = (name: string) => {
    const iconClass = "text-zinc-500 dark:text-zinc-400";
    switch (name) {
      case 'The 3 AI Paradigms':
        return <Layers size={13} className={iconClass} />;
      case 'The 3 Pillars of AIMLite':
        return <Database size={13} className={iconClass} />;
      case 'AIMLite RAG Deep Dive':
        return <BookOpen size={13} className={iconClass} />;
      case 'Supporting Foundations':
        return <Settings size={13} className={iconClass} />;
      case 'Zero-Path CLI Commands':
        return <Terminal size={13} className={iconClass} />;
      case 'HTTP Inference Server':
        return <Server size={13} className={iconClass} />;
      case 'Production & Deployment':
        return <Box size={13} className={iconClass} />;
      case 'Releases & Changelog':
        return <History size={13} className={iconClass} />;
      default:
        return null;
    }
  };

  const getItemIcon = (id: string) => {
    const iconClass = "text-zinc-400 dark:text-zinc-500 shrink-0";
    if (id === 'pillar-data') return <Database size={11} className={iconClass} />;
    if (id === 'pillar-model') return <Cpu size={11} className={iconClass} />;
    if (id === 'pillar-lifecycle') return <RefreshCw size={11} className={iconClass} />;
    if (id === 'paradigm-rag') return <BookOpen size={11} className={iconClass} />;
    if (id === 'paradigm-adapters') return <Cpu size={11} className={iconClass} />;
    if (id === 'paradigm-scratch') return <Layers size={11} className={iconClass} />;
    if (id === 'deployment-docker') return <Box size={11} className={iconClass} />;
    if (id === 'rag-hooks') return <FileCode size={11} className={iconClass} />;
    if (id === 'changelog') return <History size={11} className={iconClass} />;
    return null;
  };

  return (
    <aside className="w-full h-full flex flex-col bg-white dark:bg-[#09090b] border-r border-zinc-200 dark:border-zinc-800 select-none text-xs transition-colors">
      {/* Top Overview Quick Link */}
      {onNavigateToOverview && (
        <div className="p-2 border-b border-zinc-200 dark:border-zinc-800/80">
          <button
            onClick={onNavigateToOverview}
            className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-850 transition-colors"
          >
            <Home size={14} className="text-zinc-600 dark:text-zinc-400" />
            <span>Framework Overview</span>
          </button>
        </div>
      )}

      {/* Navigation List */}
      <div className="flex-1 overflow-y-auto px-2 py-3 space-y-3">
        {NAVIGATION_CATEGORIES.map((category: NavCategory) => {
          const isCollapsed = collapsedCategories[category.name];
          return (
            <div key={category.name} className="space-y-0.5">
              <button
                onClick={() => toggleCategory(category.name)}
                className="w-full flex items-center justify-between px-2 py-1 text-[10px] font-semibold text-zinc-500 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors uppercase tracking-wider"
              >
                <div className="flex items-center gap-1.5 truncate">
                  {getCategoryIcon(category.name)}
                  <span className="truncate">{category.name}</span>
                </div>
                {isCollapsed ? <ChevronRight size={11} /> : <ChevronDown size={11} />}
              </button>

              {!isCollapsed && (
                <div className="space-y-0.5 pl-1">
                  {category.items.map((item) => {
                    const isActive = activeSectionId === item.id;
                    const itemIcon = getItemIcon(item.id);
                    return (
                      <button
                        key={item.id}
                        onClick={() => onSelectSection(item.id)}
                        className={`w-full flex items-center justify-between px-2 py-1.5 rounded-lg text-[11px] transition-all ${isActive
                            ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-950 font-semibold shadow-sm'
                            : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-zinc-900'
                          }`}
                      >
                        <div className="flex items-center gap-1.5 truncate">
                          {itemIcon}
                          <span className="truncate">{item.label}</span>
                        </div>
                        {item.badge && (
                          <span
                            className={`text-[8px] font-mono uppercase px-1 rounded border transition-colors ${isActive
                                ? 'bg-zinc-800 text-zinc-200 border-zinc-700 dark:bg-zinc-200 dark:text-zinc-800 dark:border-zinc-300 font-semibold'
                                : item.badge === 'DOCKER'
                                  ? 'bg-cyan-500/10 text-cyan-600 border-cyan-500/30 dark:bg-cyan-500/20 dark:text-cyan-300 dark:border-cyan-500/40 font-bold'
                                  : item.badge.includes('2.1.2') || item.badge.includes('2.1.1') || item.badge.includes('2.1.0') || item.badge.includes('2.0.0') || item.badge.includes('1.0.8') || item.badge.includes('1.0.7') || item.badge.includes('1.0.6') || item.badge === 'NEW'
                                    ? 'bg-zinc-200 text-zinc-900 border-zinc-300 dark:bg-zinc-800 dark:text-zinc-200 dark:border-zinc-700 font-semibold'
                                    : 'bg-zinc-100 text-zinc-500 border-zinc-200 dark:bg-zinc-900 dark:text-zinc-500 dark:border-zinc-800'
                              }`}
                          >
                            {item.badge}
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
