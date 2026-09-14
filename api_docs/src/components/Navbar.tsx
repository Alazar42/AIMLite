import { Menu, Search, Terminal, Zap } from 'lucide-react';

interface NavbarProps {
  onOpenSearch: () => void;
  onToggleMobileNav: () => void;
  onToggleMobileConsole: () => void;
  onSelectSection: (id: string) => void;
  activeSectionId: string;
}

export default function Navbar({
  onOpenSearch,
  onToggleMobileNav,
  onToggleMobileConsole,
  onSelectSection,
  activeSectionId,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-30 w-full h-12 bg-[#090a0f]/95 backdrop-blur-md border-b border-zinc-800/80 flex items-center justify-between px-3 lg:px-5 select-none text-xs">
      {/* Brand & 3 Pillars Quick Jump */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleMobileNav}
          className="lg:hidden p-1 rounded bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200"
          title="Toggle Navigation Menu"
        >
          <Menu size={16} />
        </button>

        <div
          className="flex items-center gap-1.5 cursor-pointer"
          onClick={() => onSelectSection('pillar-data')}
        >
          <div className="w-5 h-5 rounded bg-gradient-to-br from-sky-400 to-indigo-600 flex items-center justify-center text-zinc-950 shadow-sm">
            <Zap size={12} />
          </div>
          <span className="font-bold text-zinc-100 tracking-tight">ModelKit</span>
          <span className="text-[10px] text-zinc-500 font-mono hidden sm:inline ml-1 px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800">
            Django for AI
          </span>
        </div>

        {/* Minimal Pillars Navigation */}
        <nav className="hidden md:flex items-center gap-1 pl-3 border-l border-zinc-800/80">
          <button
            onClick={() => onSelectSection('pillar-data')}
            className={`px-2 py-0.5 rounded transition-colors text-[11px] ${
              activeSectionId === 'pillar-data'
                ? 'bg-zinc-800 text-sky-400 font-medium'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Data
          </button>
          <button
            onClick={() => onSelectSection('pillar-model')}
            className={`px-2 py-0.5 rounded transition-colors text-[11px] ${
              activeSectionId === 'pillar-model'
                ? 'bg-zinc-800 text-purple-400 font-medium'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Model
          </button>
          <button
            onClick={() => onSelectSection('pillar-lifecycle')}
            className={`px-2 py-0.5 rounded transition-colors text-[11px] ${
              activeSectionId === 'pillar-lifecycle'
                ? 'bg-zinc-800 text-emerald-400 font-medium'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Lifecycle
          </button>
        </nav>
      </div>

      {/* Quick Search & Console */}
      <div className="flex items-center gap-2">
        <button
          onClick={onOpenSearch}
          className="flex items-center gap-2 px-2.5 py-1 bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-800 text-zinc-400 rounded-md transition-colors"
        >
          <Search size={12} className="text-zinc-500" />
          <span className="hidden sm:inline text-[11px]">Search</span>
          <kbd className="px-1 py-0.2 bg-zinc-800 border border-zinc-700 rounded text-[9px] font-mono text-zinc-400">
            ⌘K
          </kbd>
        </button>

        <button
          onClick={onToggleMobileConsole}
          className="xl:hidden p-1.5 rounded bg-zinc-900 border border-zinc-800 text-sky-400"
          title="Toggle Code Console"
        >
          <Terminal size={13} />
        </button>

        <div className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-[10px] text-emerald-400">
          <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse"></span>
          <span>v0.1.0</span>
        </div>
      </div>
    </header>
  );
}
