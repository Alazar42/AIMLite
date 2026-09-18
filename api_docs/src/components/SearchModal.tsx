import { useEffect, useState, useRef } from 'react';
import { Search, ArrowRight, CornerDownLeft, X, Zap, Terminal, Server, Settings, Database } from 'lucide-react';
import { NAVIGATION_CATEGORIES, DOC_SECTIONS, type NavItem } from '../data/aimliteDocs';

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSection: (id: string) => void;
}

export default function SearchModal({ isOpen, onClose, onSelectSection }: SearchModalProps) {
  const [query, setQuery] = useState<string>('');
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Flatten all items with category names and section details for search
  const allItems: { item: NavItem; category: string; subtitle?: string; djangoAnalogy?: string }[] = [];
  NAVIGATION_CATEGORIES.forEach((cat) => {
    cat.items.forEach((item) => {
      const doc = DOC_SECTIONS[item.id];
      allItems.push({
        item,
        category: cat.name,
        subtitle: doc?.subtitle,
        djangoAnalogy: doc?.djangoAnalogy,
      });
    });
  });

  const q = query.toLowerCase().trim();
  const filtered = allItems.filter(({ item, category, subtitle, djangoAnalogy }) => {
    if (!q) return true;
    return (
      item.label.toLowerCase().includes(q) ||
      category.toLowerCase().includes(q) ||
      (subtitle && subtitle.toLowerCase().includes(q)) ||
      (djangoAnalogy && djangoAnalogy.toLowerCase().includes(q))
    );
  });

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        onClose();
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev + 1) % (filtered.length || 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => (prev - 1 + filtered.length) % (filtered.length || 1));
      } else if (e.key === 'Enter') {
        if (filtered[selectedIndex]) {
          onSelectSection(filtered[selectedIndex].item.id);
          onClose();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, filtered, selectedIndex, onClose, onSelectSection]);

  if (!isOpen) return null;

  const getItemIcon = (category: string) => {
    if (category.includes('Pillar')) return <Zap size={13} className="text-sky-400 shrink-0" />;
    if (category.includes('CLI')) return <Terminal size={13} className="text-amber-400 shrink-0" />;
    if (category.includes('HTTP')) return <Server size={13} className="text-emerald-400 shrink-0" />;
    if (category.includes('Foundations')) return <Settings size={13} className="text-purple-400 shrink-0" />;
    return <Database size={13} className="text-zinc-400 shrink-0" />;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 px-4 bg-zinc-950/75 backdrop-blur-md">
      <div
        className="w-full max-w-xl bg-[#0f1017] border border-zinc-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search Input Bar */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-zinc-800 bg-zinc-900/50">
          <Search size={18} className="text-zinc-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Search AIMLite (e.g. Data, Model, CSV, Django analogy, CLI)..."
            className="w-full bg-transparent text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none font-sans"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="text-zinc-500 hover:text-zinc-300 p-1 rounded"
            >
              <X size={14} />
            </button>
          )}
          <kbd className="px-1.5 py-0.5 text-[10px] font-mono bg-zinc-800 border border-zinc-700 text-zinc-400 rounded">
            ESC
          </kbd>
        </div>

        {/* Filtered Results List */}
        <div className="max-h-84 overflow-y-auto p-2 space-y-1">
          {filtered.length === 0 ? (
            <div className="py-8 text-center text-xs text-zinc-500 font-mono">
              No matching AIMLite documentation found for "{query}".
            </div>
          ) : (
            filtered.map(({ item, category, subtitle }, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <button
                  key={`${category}-${item.id}`}
                  onClick={() => {
                    onSelectSection(item.id);
                    onClose();
                  }}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-left text-xs transition-colors ${
                    isSelected ? 'bg-zinc-800/90 text-sky-400' : 'text-zinc-300 hover:bg-zinc-900'
                  }`}
                >
                  <div className="flex items-center gap-2.5 overflow-hidden flex-1 min-w-0 pr-2">
                    {getItemIcon(category)}
                    <span className="text-[10px] font-mono text-zinc-500 uppercase shrink-0">
                      {category.split(' ')[0]}
                    </span>
                    <ArrowRight size={11} className="text-zinc-600 shrink-0" />
                    <div className="truncate">
                      <span className="font-semibold text-zinc-100">{item.label}</span>
                      {subtitle && (
                        <span className="text-zinc-500 text-[11px] ml-2 truncate hidden sm:inline">
                          — {subtitle}
                        </span>
                      )}
                    </div>
                  </div>

                  {item.badge && (
                    <span
                      className="shrink-0 px-1.5 py-0.5 text-[9px] font-mono font-bold uppercase rounded border border-zinc-700 bg-zinc-900 text-zinc-400"
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })
          )}
        </div>

        {/* Modal Footer Hotkeys Bar */}
        <div className="px-4 py-2.5 bg-zinc-950/60 border-t border-zinc-800/80 flex items-center justify-between text-[11px] text-zinc-500">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-zinc-900 border border-zinc-800 rounded font-mono text-[9px]">
                ↑
              </kbd>
              <kbd className="px-1 py-0.5 bg-zinc-900 border border-zinc-800 rounded font-mono text-[9px]">
                ↓
              </kbd>
              <span>Navigate</span>
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1 py-0.5 bg-zinc-900 border border-zinc-800 rounded font-mono text-[9px]">
                <CornerDownLeft size={9} />
              </kbd>
              <span>Select</span>
            </span>
          </div>

          <span className="font-mono text-[10px] text-zinc-500">AIMLite Quick Index</span>
        </div>
      </div>
    </div>
  );
}

