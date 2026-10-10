import { Menu, Search, Terminal, ExternalLink, BookOpen, Code2, Bot, Box } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

const GithubIcon = ({ size = 16, className = '' }: { size?: number; className?: string }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path
      fillRule="evenodd"
      clipRule="evenodd"
      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
    />
  </svg>
);

interface NavbarProps {
  onOpenSearch: () => void;
  onToggleMobileNav: () => void;
  onToggleMobileConsole: () => void;
  onToggleConsoleVisibility?: () => void;
  isConsoleVisible?: boolean;
  onSelectSection: (id: string) => void;
  onSelectViewMode: (mode: 'landing' | 'docs') => void;
  viewMode: 'landing' | 'docs';
  activeSectionId: string;
}

export default function Navbar({
  onOpenSearch,
  onToggleMobileNav,
  onToggleMobileConsole,
  onToggleConsoleVisibility,
  isConsoleVisible = true,
  onSelectSection,
  onSelectViewMode,
  viewMode,
  activeSectionId,
}: NavbarProps) {
  const isDocs = viewMode === 'docs';

  return (
    <header className="sticky top-0 z-40 w-full h-14 backdrop-blur-xl bg-white/75 dark:bg-[#07080a]/80 border-b border-zinc-200/80 dark:border-zinc-800/80 flex items-center justify-between px-3 sm:px-6 select-none text-xs transition-colors shadow-xs">
      {/* Brand & Main Navigation */}
      <div className="flex items-center gap-3 sm:gap-4 min-w-0">
        {/* Mobile Navigation Toggle */}
        <button
          onClick={onToggleMobileNav}
          className={`${isDocs ? 'lg:hidden' : 'md:hidden'} p-2 rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white transition-colors shrink-0`}
          title="Toggle Navigation Menu"
        >
          <Menu size={16} />
        </button>

        {/* Brand Logo with Ambient Hover Aura */}
        <div
          className="flex items-center gap-2.5 sm:gap-3 cursor-pointer group shrink-0"
          onClick={() => onSelectViewMode('landing')}
          title="AIMLite Home & Overview"
        >
          <div className="relative flex items-center justify-center w-8.5 h-8.5 sm:w-9.5 sm:h-9.5 rounded-xl bg-gradient-to-br from-zinc-100 to-zinc-200 dark:from-zinc-800 dark:to-zinc-900 border border-zinc-200/90 dark:border-zinc-700/80 shadow-xs group-hover:scale-105 group-hover:border-zinc-400 dark:group-hover:border-zinc-600 transition-all">
            {/* Light Mode Logo */}
            <img
              src="/logo.png"
              alt="AIMLite Logo"
              className="w-5.5 h-5.5 sm:w-6.5 sm:h-6.5 object-contain block dark:hidden transition-transform"
            />
            {/* Dark Mode Logo - High-contrast crisp white */}
            <img
              src="/logo_white.png"
              alt="AIMLite Logo"
              className="w-5.5 h-5.5 sm:w-6.5 sm:h-6.5 object-contain hidden dark:block drop-shadow-[0_0_10px_rgba(255,255,255,0.3)] transition-transform"
            />
          </div>

          <div className="flex flex-col">
            <span className="font-heading font-extrabold text-sm sm:text-base text-zinc-950 dark:text-white tracking-tight leading-none">
              AIMLite
            </span>
            <span className="hidden lg:inline text-[9px] text-zinc-400 dark:text-zinc-500 font-mono tracking-tight mt-0.5">
              The Django for AI
            </span>
          </div>
        </div>

        {/* Desktop Navigation Switcher */}
        <nav className="hidden md:flex items-center gap-1 pl-4 border-l border-zinc-200 dark:border-zinc-800/80">
          <button
            onClick={() => onSelectViewMode('landing')}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              viewMode === 'landing'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
          >
            Overview
          </button>

          <button
            onClick={() => {
              onSelectViewMode('docs');
              if (
                activeSectionId.startsWith('paradigm-') ||
                activeSectionId.startsWith('cli-') ||
                activeSectionId.startsWith('endpoint-') ||
                activeSectionId === 'deployment-docker' ||
                activeSectionId === 'changelog'
              ) {
                onSelectSection('pillar-data');
              }
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              isDocs && activeSectionId.startsWith('pillar-')
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
          >
            Core Pillars
          </button>

          <button
            onClick={() => {
              onSelectViewMode('docs');
              onSelectSection('paradigm-rag');
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              isDocs && activeSectionId.startsWith('paradigm-')
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
          >
            <BookOpen size={13} className="text-zinc-400 dark:text-zinc-500" />
            <span>3 Paradigms</span>
          </button>

          <button
            onClick={() => {
              onSelectViewMode('docs');
              onSelectSection('cli-init');
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              isDocs && activeSectionId.startsWith('cli-')
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
          >
            CLI
          </button>

          <button
            onClick={() => {
              onSelectViewMode('docs');
              onSelectSection('endpoint-predict');
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              isDocs && activeSectionId.startsWith('endpoint-')
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
          >
            HTTP Server
          </button>

          <button
            onClick={() => {
              onSelectViewMode('docs');
              onSelectSection('deployment-docker');
            }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              isDocs && activeSectionId === 'deployment-docker'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
          >
            <Box size={13} className="text-zinc-400 dark:text-zinc-500" />
            <span>Deploy</span>
          </button>

          <button
            onClick={() => {
              onSelectViewMode('docs');
              onSelectSection('changelog');
            }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              isDocs && activeSectionId === 'changelog'
                ? 'bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 font-semibold shadow-xs'
                : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-900'
            }`}
          >
            Changelog
          </button>
        </nav>
      </div>

      {/* Right Controls: Command Search, Console, ThemeToggle, llms.txt, GitHub */}
      <div className="flex items-center gap-2 sm:gap-2.5">
        {/* Modern Command Search Bar */}
        <button
          onClick={onOpenSearch}
          className="flex items-center gap-2 px-3 py-1.5 bg-zinc-100/90 hover:bg-zinc-200/80 dark:bg-zinc-900/90 dark:hover:bg-zinc-800/80 border border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 text-zinc-500 dark:text-zinc-400 rounded-xl transition-all shadow-xs group"
          title="Search Documentation (⌘K)"
        >
          <Search size={13} className="text-zinc-400 group-hover:text-zinc-700 dark:group-hover:text-zinc-200 transition-colors" />
          <span className="hidden sm:inline text-xs font-medium text-zinc-600 dark:text-zinc-400 group-hover:text-zinc-950 dark:group-hover:text-white transition-colors">
            Search docs...
          </span>
          <kbd className="hidden sm:inline-block px-1.5 py-0.5 bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-md text-[9px] font-mono text-zinc-500 dark:text-zinc-400 shadow-2xs">
            ⌘K
          </kbd>
        </button>

        {/* Desktop Code Console Visibility Toggle */}
        {isDocs && onToggleConsoleVisibility && (
          <button
            onClick={onToggleConsoleVisibility}
            className={`hidden xl:flex items-center gap-1.5 px-3 py-1.5 rounded-xl border text-xs font-mono transition-all ${
              isConsoleVisible
                ? 'bg-zinc-200/90 dark:bg-zinc-800 text-zinc-900 dark:text-white border-zinc-300 dark:border-zinc-700 font-semibold shadow-xs'
                : 'bg-zinc-100 dark:bg-zinc-900 text-zinc-600 dark:text-zinc-400 border-zinc-200 dark:border-zinc-800 hover:text-zinc-950 dark:hover:text-white'
            }`}
            title={isConsoleVisible ? 'Hide Code Console' : 'Show Code Console'}
          >
            <Code2 size={13} />
            <span>{isConsoleVisible ? 'Console' : 'Show Code'}</span>
          </button>
        )}

        {/* Mobile Console Toggle (in docs mode) */}
        {isDocs && (
          <button
            onClick={onToggleMobileConsole}
            className="xl:hidden p-2 rounded-xl bg-zinc-100 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-700 dark:text-zinc-300 hover:text-zinc-950 dark:hover:text-white transition-colors shrink-0"
            title="Toggle Code Console"
          >
            <Terminal size={14} />
          </button>
        )}

        {/* Theme Toggle (Light / System / Dark) */}
        <ThemeToggle />

        {/* Live Version Status Pill */}
        <button
          onClick={() => {
            onSelectViewMode('docs');
            onSelectSection('changelog');
          }}
          className="hidden min-[520px]:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-zinc-100/90 dark:bg-zinc-900/90 hover:bg-zinc-200/80 dark:hover:bg-zinc-800/80 border border-zinc-200 dark:border-zinc-800 text-[11px] text-zinc-800 dark:text-zinc-200 font-mono transition-all shrink-0"
          title="Release Changelog (v2.1.2 Latest Stable)"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 dark:bg-zinc-500 animate-pulse" />
          <span>v2.1.2</span>
        </button>

        {/* llms.txt Agent Specification Link */}
        <a
          href="/llms.txt"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl text-zinc-600 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-850 text-[11px] font-mono border border-zinc-200 dark:border-zinc-800 transition-colors"
          title="Open llms.txt standard guide for coding agents"
        >
          <Bot size={13} />
          <span>llms.txt</span>
        </a>

        {/* GitHub Link */}
        <a
          href="https://github.com/Alazar42/AIMLite"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-xl text-zinc-500 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-850 border border-zinc-200 dark:border-zinc-800 transition-colors hidden sm:flex items-center"
          title="GitHub Repository"
        >
          <GithubIcon size={15} />
        </a>

        {/* PyPI Link */}
        <a
          href="https://pypi.org/project/aimlite/"
          target="_blank"
          rel="noopener noreferrer"
          className="p-2 rounded-xl text-zinc-500 dark:text-zinc-400 hover:text-zinc-950 dark:hover:text-white hover:bg-zinc-100 dark:hover:bg-zinc-850 border border-zinc-200 dark:border-zinc-800 transition-colors hidden sm:flex items-center"
          title="PyPI Package"
        >
          <ExternalLink size={14} />
        </a>
      </div>
    </header>
  );
}
