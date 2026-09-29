import { useEffect, useState, useRef } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import EndpointDoc from './components/EndpointDoc';
import CodePlayground from './components/CodePlayground';
import SearchModal from './components/SearchModal';
import LandingPage from './components/LandingPage';
import { DOC_SECTIONS } from './data/aimliteDocs';
import { X, Code2 } from 'lucide-react';

export default function App() {
  const [viewMode, setViewMode] = useState<'landing' | 'docs'>('landing');
  const [activeSectionId, setActiveSectionId] = useState<string>('pillar-data');
  const [version, setVersion] = useState<string>('v2.0.0');
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);
  const [isMobileNavOpen, setIsMobileNavOpen] = useState<boolean>(false);
  const [isMobileConsoleOpen, setIsMobileConsoleOpen] = useState<boolean>(false);
  const [isConsoleVisible, setIsConsoleVisible] = useState<boolean>(true);
  const [hasCopiedSignature, setHasCopiedSignature] = useState<boolean>(false);

  const mainScrollRef = useRef<HTMLDivElement>(null);

  // Global Keyboard Listener for Cmd+K / Ctrl+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Scroll to top when changing section or view mode
  useEffect(() => {
    if (mainScrollRef.current) {
      mainScrollRef.current.scrollTo({ top: 0, behavior: 'smooth' });
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [activeSectionId, viewMode]);

  const currentSection = DOC_SECTIONS[activeSectionId] || DOC_SECTIONS['pillar-data'];

  const handleCopySignature = (text: string) => {
    navigator.clipboard.writeText(text);
    setHasCopiedSignature(true);
    setTimeout(() => setHasCopiedSignature(false), 2000);
  };

  const handleNavigateToDocs = (sectionId?: string) => {
    setViewMode('docs');
    if (sectionId && DOC_SECTIONS[sectionId]) {
      setActiveSectionId(sectionId);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-[#09090b] text-zinc-900 dark:text-zinc-100 flex flex-col font-sans antialiased selection:bg-zinc-200 dark:selection:bg-zinc-800 selection:text-zinc-950 dark:selection:text-zinc-100 transition-colors">
      {/* Top Sticky Header */}
      <Navbar
        onOpenSearch={() => setIsSearchOpen(true)}
        onToggleMobileNav={() => setIsMobileNavOpen(true)}
        onToggleMobileConsole={() => setIsMobileConsoleOpen(true)}
        onToggleConsoleVisibility={() => setIsConsoleVisible((prev) => !prev)}
        isConsoleVisible={isConsoleVisible}
        onSelectSection={(id) => {
          setViewMode('docs');
          setActiveSectionId(id);
        }}
        onSelectViewMode={setViewMode}
        viewMode={viewMode}
        activeSectionId={activeSectionId}
      />

      {/* Main View: Landing Page or 3-Column Documentation */}
      {viewMode === 'landing' ? (
        <main className="flex-1 w-full">
          <LandingPage onNavigateToDocs={handleNavigateToDocs} />
        </main>
      ) : (
        /* Main 3-Column Documentation Layout Container */
        <div className="flex-1 flex w-full max-w-[1440px] mx-auto overflow-hidden">
          {/* Left Column: Sticky Sidebar (~224px) */}
          <div className="hidden lg:block w-56 shrink-0 sticky top-13 h-[calc(100vh-3.25rem)]">
            <Sidebar
              activeSectionId={activeSectionId}
              onSelectSection={(id) => {
                setActiveSectionId(id);
                setIsMobileNavOpen(false);
              }}
              onOpenSearch={() => setIsSearchOpen(true)}
              onNavigateToOverview={() => setViewMode('landing')}
              version={version}
              onChangeVersion={setVersion}
            />
          </div>

          {/* Center Column: Main Documentation (~640px) */}
          <main
            ref={mainScrollRef}
            className="flex-1 min-w-0 overflow-y-auto h-[calc(100vh-3.25rem)] flex justify-center pb-16"
          >
            <EndpointDoc
              section={currentSection}
              onSelectSection={(id) => setActiveSectionId(id)}
              onCopySignature={handleCopySignature}
              hasCopiedSignature={hasCopiedSignature}
            />
          </main>

          {/* Right Column: Code Playground (~380px, sticky, hidable) */}
          {isConsoleVisible && (
            <div className="hidden xl:block w-[380px] shrink-0 sticky top-13 h-[calc(100vh-3.25rem)] transition-all">
              <CodePlayground
                section={currentSection}
                onClose={() => setIsConsoleVisible(false)}
              />
            </div>
          )}
        </div>
      )}

      {/* Floating Re-Open Button when Code Console is Hidden */}
      {viewMode === 'docs' && !isConsoleVisible && (
        <button
          onClick={() => setIsConsoleVisible(true)}
          className="hidden xl:flex fixed bottom-6 right-6 z-40 items-center gap-2 px-3.5 py-2 rounded-xl bg-zinc-900 text-white dark:bg-white dark:text-zinc-950 shadow-2xl border border-zinc-700 dark:border-zinc-300 hover:scale-105 transition-all text-xs font-mono font-semibold"
          title="Show Interactive Code Console"
        >
          <Code2 size={14} />
          <span>Show Code Console</span>
        </button>
      )}

      {/* Mobile Slide-Over Drawer: Navigation Menu */}
      {isMobileNavOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-zinc-950/70 backdrop-blur-sm"
            onClick={() => setIsMobileNavOpen(false)}
          />
          <div className="relative w-80 max-w-full bg-white dark:bg-[#09090b] h-full shadow-2xl z-10 flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-zinc-200 dark:border-zinc-800">
              <span className="text-xs font-semibold text-zinc-700 dark:text-zinc-400 uppercase tracking-wider">
                AIMLite Navigation
              </span>
              <button
                onClick={() => setIsMobileNavOpen(false)}
                className="p-1 rounded-md text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-200"
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <Sidebar
                activeSectionId={activeSectionId}
                onSelectSection={(id) => {
                  setActiveSectionId(id);
                  setIsMobileNavOpen(false);
                }}
                onOpenSearch={() => {
                  setIsMobileNavOpen(false);
                  setIsSearchOpen(true);
                }}
                onNavigateToOverview={() => {
                  setIsMobileNavOpen(false);
                  setViewMode('landing');
                }}
                version={version}
                onChangeVersion={setVersion}
              />
            </div>
          </div>
        </div>
      )}

      {/* Mobile Slide-Over Drawer: Interactive Console */}
      {isMobileConsoleOpen && (
        <div className="fixed inset-0 z-50 xl:hidden flex justify-end">
          <div
            className="fixed inset-0 bg-zinc-950/70 backdrop-blur-sm"
            onClick={() => setIsMobileConsoleOpen(false)}
          />
          <div className="relative w-full max-w-lg bg-zinc-900 dark:bg-[#0b0c13] h-full shadow-2xl z-10 flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-zinc-700 dark:border-zinc-800 bg-zinc-800 dark:bg-zinc-900">
              <span className="text-xs font-semibold text-zinc-200 dark:text-zinc-300 uppercase tracking-wider">
                Interactive Code Console
              </span>
              <button
                onClick={() => setIsMobileConsoleOpen(false)}
                className="p-1 rounded-md text-zinc-400 hover:text-white"
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <CodePlayground section={currentSection} />
            </div>
          </div>
        </div>
      )}

      {/* Cmd+K Search Palette Modal */}
      <SearchModal
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onSelectSection={(id) => {
          setViewMode('docs');
          setActiveSectionId(id);
        }}
      />
    </div>
  );
}
