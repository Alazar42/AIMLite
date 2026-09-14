import { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import EndpointDoc from './components/EndpointDoc';
import CodePlayground from './components/CodePlayground';
import SearchModal from './components/SearchModal';
import { DOC_SECTIONS } from './data/modelkitDocs';
import { X } from 'lucide-react';

export default function App() {
  const [activeSectionId, setActiveSectionId] = useState<string>('pillar-data');
  const [version, setVersion] = useState<string>('v0.1.0');
  const [isSearchOpen, setIsSearchOpen] = useState<boolean>(false);
  const [isMobileNavOpen, setIsMobileNavOpen] = useState<boolean>(false);
  const [isMobileConsoleOpen, setIsMobileConsoleOpen] = useState<boolean>(false);
  const [hasCopiedSignature, setHasCopiedSignature] = useState<boolean>(false);

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

  const currentSection = DOC_SECTIONS[activeSectionId] || DOC_SECTIONS['pillar-data'];

  const handleCopySignature = (text: string) => {
    navigator.clipboard.writeText(text);
    setHasCopiedSignature(true);
    setTimeout(() => setHasCopiedSignature(false), 2000);
  };

  return (
    <div className="min-h-screen bg-[#090a0f] text-zinc-100 flex flex-col font-sans antialiased selection:bg-sky-500/20 selection:text-sky-300">
      {/* Top Sticky Header */}
      <Navbar
        onOpenSearch={() => setIsSearchOpen(true)}
        onToggleMobileNav={() => setIsMobileNavOpen(true)}
        onToggleMobileConsole={() => setIsMobileConsoleOpen(true)}
        onSelectSection={(id) => setActiveSectionId(id)}
        activeSectionId={activeSectionId}
      />

      {/* Main 3-Column Layout Container */}
      <div className="flex-1 flex w-full max-w-[1440px] mx-auto overflow-hidden">
        {/* Left Column: Sticky Sidebar (~224px) */}
        <div className="hidden lg:block w-56 shrink-0 sticky top-12 h-[calc(100vh-3rem)]">
          <Sidebar
            activeSectionId={activeSectionId}
            onSelectSection={(id) => {
              setActiveSectionId(id);
              setIsMobileNavOpen(false);
            }}
            onOpenSearch={() => setIsSearchOpen(true)}
            version={version}
            onChangeVersion={setVersion}
          />
        </div>

        {/* Center Column: Main Documentation (~600px) */}
        <main className="flex-1 min-w-0 overflow-y-auto h-[calc(100vh-3rem)] flex justify-center">
          <EndpointDoc
            section={currentSection}
            onSelectSection={(id) => setActiveSectionId(id)}
            onCopySignature={handleCopySignature}
            hasCopiedSignature={hasCopiedSignature}
          />
        </main>

        {/* Right Column: Code Playground (~380px, sticky) */}
        <div className="hidden xl:block w-[380px] shrink-0 sticky top-12 h-[calc(100vh-3rem)]">
          <CodePlayground section={currentSection} />
        </div>
      </div>

      {/* Mobile Slide-Over Drawer: Navigation Menu */}
      {isMobileNavOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-zinc-950/80 backdrop-blur-sm"
            onClick={() => setIsMobileNavOpen(false)}
          />
          <div className="relative w-80 max-w-full bg-[#090a0f] h-full shadow-2xl z-10 flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-zinc-800">
              <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                ModelKit Navigation
              </span>
              <button
                onClick={() => setIsMobileNavOpen(false)}
                className="p-1 rounded-md text-zinc-400 hover:text-zinc-200"
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
            className="fixed inset-0 bg-zinc-950/80 backdrop-blur-sm"
            onClick={() => setIsMobileConsoleOpen(false)}
          />
          <div className="relative w-full max-w-lg bg-[#0b0c13] h-full shadow-2xl z-10 flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-zinc-800 bg-zinc-900">
              <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                Interactive Code Console
              </span>
              <button
                onClick={() => setIsMobileConsoleOpen(false)}
                className="p-1 rounded-md text-zinc-400 hover:text-zinc-200"
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
          setActiveSectionId(id);
        }}
      />
    </div>
  );
}

