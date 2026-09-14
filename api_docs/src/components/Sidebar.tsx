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
} from 'lucide-react';
import { NAVIGATION_CATEGORIES, type NavCategory } from '../data/modelkitDocs';

interface SidebarProps {
  activeSectionId: string;
  onSelectSection: (id: string) => void;
  onOpenSearch: () => void;
  version: string;
  onChangeVersion: (ver: string) => void;
}

export default function Sidebar({
  activeSectionId,
  onSelectSection,
}: SidebarProps) {
  const [collapsedCategories, setCollapsedCategories] = useState<Record<string, boolean>>({
    'Supporting Foundations': false,
    'Zero-Path CLI Commands': false,
    'HTTP Inference Server': false,
  });

  const toggleCategory = (catName: string) => {
    setCollapsedCategories((prev) => ({
      ...prev,
      [catName]: !prev[catName],
    }));
  };

  const getCategoryIcon = (name: string) => {
    switch (name) {
      case 'The 3 Pillars of ModelKit':
        return <Database size={12} className="text-sky-400" />;
      case 'Supporting Foundations':
        return <Settings size={12} className="text-purple-400" />;
      case 'Zero-Path CLI Commands':
        return <Terminal size={12} className="text-amber-400" />;
      case 'HTTP Inference Server':
        return <Server size={12} className="text-emerald-400" />;
      default:
        return null;
    }
  };

  const getItemIcon = (id: string) => {
    if (id === 'pillar-data') return <Database size={11} className="text-sky-400 shrink-0" />;
    if (id === 'pillar-model') return <Cpu size={11} className="text-purple-400 shrink-0" />;
    if (id === 'pillar-lifecycle') return <RefreshCw size={11} className="text-emerald-400 shrink-0" />;
    return null;
  };

  return (
    <aside className="w-full h-full flex flex-col bg-[#090a0f] border-r border-zinc-800/80 select-none text-xs">
      <div className="flex-1 overflow-y-auto px-2 py-3 space-y-3">
        {NAVIGATION_CATEGORIES.map((category: NavCategory) => {
          const isCollapsed = collapsedCategories[category.name];
          return (
            <div key={category.name} className="space-y-0.5">
              <button
                onClick={() => toggleCategory(category.name)}
                className="w-full flex items-center justify-between px-2 py-1 text-[10px] font-semibold text-zinc-500 hover:text-zinc-300 transition-colors uppercase tracking-wider"
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
                        className={`w-full flex items-center justify-between px-2 py-1 rounded text-[11px] transition-colors ${
                          isActive
                            ? 'bg-zinc-800 text-sky-400 font-medium'
                            : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
                        }`}
                      >
                        <div className="flex items-center gap-1.5 truncate">
                          {itemIcon}
                          <span className="truncate">{item.label}</span>
                        </div>
                        {item.badge && (
                          <span className="text-[8px] font-mono uppercase px-1 rounded bg-zinc-900 text-zinc-500 border border-zinc-800">
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
