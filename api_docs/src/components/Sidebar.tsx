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
  Sparkles,
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
    const iconClass = "text-zinc-400 dark:text-zinc-500";
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
    if (id === 'pillar-data') return <Database size={12} className={iconClass} />;
    if (id === 'pillar-model') return <Cpu size={12} className={iconClass} />;
    if (id === 'pillar-lifecycle') return <RefreshCw size={12} className={iconClass} />;
    if (id === 'paradigm-rag') return <BookOpen size={12} className={iconClass} />;
    if (id === 'paradigm-adapters') return <Cpu size={12} className={iconClass} />;
    if (id === 'paradigm-scratch') return <Layers size={12} className={iconClass} />;
    if (id === 'deployment-docker') return <Box size={12} className={iconClass} />;
    if (id === 'rag-hooks') return <FileCode size={12} className={iconClass} />;
    if (id === 'changelog') return <History size={12} className={iconClass} />;
    return null;
  };

  const getBadgeStyle = (_badge: string, isActive: boolean) => {
    if (isActive) {
      return 'bg-zinc-800 text-zinc-100 dark:bg-zinc-200 dark:text-zinc-900 border-zinc-700 dark:border-zinc-300 font-bold';
    }
    return 'bg-zinc-100 text-zinc-600 border-zinc-200/80 dark:bg-zinc-850 dark:text-zinc-400 dark:border-zinc-800 font-mono text-[8px] font-semibold';
  };

  return (
    <aside className="w-full h-full flex flex-col select-none text-xs">
      {/* Top Overview Quick Link */}
      {onNavigateToOverview && (
        <div className="p-3 border-b border-zinc-200/80 dark:border-zinc-800/80">
          <button
            onClick={onNavigateToOverview}
            className="w-full flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium text-zinc-700 dark:text-zinc-300 hover:text-zinc-950 dark:hover:text-white bg-zinc-100/70 hover:bg-zinc-200/70 dark:bg-zinc-900/60 dark:hover:bg-zinc-800/70 border border-zinc-200/60 dark:border-zinc-800/60 transition-all shadow-2xs group"
          >
            <div className="flex items-center gap-2">
              <Home size={14} className="text-zinc-500 group-hover:text-zinc-950 dark:text-zinc-400 dark:group-hover:text-white transition-colors" />
              <span className="font-semibold">Framework Overview</span>
            </div>
            <Sparkles size={12} className="text-zinc-400 group-hover:text-zinc-600 dark:text-zinc-600 dark:group-hover:text-zinc-300 transition-colors" />
          </button>
        </div>
      )}

      {/* Navigation Tree List */}
      <div className="flex-1 overflow-y-auto px-2.5 py-3 space-y-3.5 scrollbar-thin">
        {NAVIGATION_CATEGORIES.map((category: NavCategory) => {
          const isCollapsed = collapsedCategories[category.name];
          return (
            <div key={category.name} className="space-y-0.5">
              <button
                onClick={() => toggleCategory(category.name)}
                className="w-full flex items-center justify-between px-2.5 py-1 text-[10px] font-bold text-zinc-400 dark:text-zinc-500 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors uppercase tracking-wider font-mono"
              >
                <div className="flex items-center gap-2 truncate">
                  {getCategoryIcon(category.name)}
                  <span className="truncate">{category.name}</span>
                </div>
                {isCollapsed ? <ChevronRight size={11} /> : <ChevronDown size={11} />}
              </button>

              {!isCollapsed && (
                <div className="space-y-0.5 pt-0.5">
                  {category.items.map((item) => {
                    const isActive = activeSectionId === item.id;
                    const itemIcon = getItemIcon(item.id);
                    return (
                      <button
                        key={item.id}
                        onClick={() => onSelectSection(item.id)}
                        className={`w-full flex items-center justify-between px-2.5 py-1.5 rounded-xl text-[11px] transition-all relative ${
                          isActive
                            ? 'bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-950 font-semibold shadow-xs'
                            : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-zinc-100 hover:bg-zinc-100/90 dark:hover:bg-zinc-900/90'
                        }`}
                      >
                        <div className="flex items-center gap-2 truncate">
                          {itemIcon}
                          <span className="truncate">{item.label}</span>
                        </div>
                        {item.badge && (
                          <span
                            className={`text-[8px] font-mono uppercase px-1.5 py-0.2 rounded-md border transition-colors ${getBadgeStyle(
                              item.badge,
                              isActive
                            )}`}
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

      {/* Modern Status Footer */}
      <div className="p-3 border-t border-zinc-200/80 dark:border-zinc-800/80 bg-zinc-50/50 dark:bg-zinc-950/40">
        <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500 dark:text-zinc-400">
          <span>Python &ge; 3.10</span>
          <span>Apache 2.0</span>
        </div>
      </div>
    </aside>
  );
}
