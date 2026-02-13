import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";
import ReactMarkdown from 'react-markdown';

// --- Configuration ---
// We use gemini-3-pro-preview for complex reasoning and synthesis tasks.
const MODEL_NAME = 'gemini-3-pro-preview';

// --- Types ---
interface Source {
  title: string;
  uri: string;
}

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  sources?: Source[];
  isStreaming?: boolean;
}

// --- API Client ---
// Initialize outside component to avoid recreation, but key is checked inside.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- Components ---

const ClearDialog: React.FC<{ isOpen: boolean; onClose: () => void; onConfirm: () => void }> = ({ isOpen, onClose, onConfirm }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animation-fade-in">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-sm w-full p-6 shadow-2xl relative transform transition-all scale-100">
        <h3 className="text-lg font-bold text-white mb-2 brand-font">Clear Conversation</h3>
        <p className="text-slate-400 mb-6 text-sm">Are you sure you want to clear the current conversation? This action cannot be undone.</p>
        <div className="flex gap-3 justify-end">
          <button 
            onClick={onClose} 
            className="px-4 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors text-sm font-medium"
          >
            Cancel
          </button>
          <button 
            onClick={onConfirm} 
            className="px-4 py-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/50 transition-colors text-sm font-medium"
          >
            Clear Chat
          </button>
        </div>
      </div>
    </div>
  );
};

const Header: React.FC<{ onClear: () => void }> = ({ onClear }) => {
  return (
    <header className="fixed top-0 left-0 right-0 h-16 glass-panel z-50 flex items-center justify-between px-6 shadow-lg">
      <div className="flex items-center gap-2">
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-cyan-400 to-blue-600 flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <span className="text-2xl font-bold brand-font bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 borai-glow">
          BorAI
        </span>
      </div>
      <div className="flex items-center gap-4">
        <div className="hidden md:flex items-center gap-2 text-xs text-slate-400 bg-slate-800/50 px-3 py-1 rounded-full border border-slate-700">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
          <span>Online</span>
        </div>
        
        <button 
          onClick={onClear}
          className="text-slate-400 hover:text-red-400 transition-colors p-2 rounded-lg hover:bg-slate-800/50" 
          title="Clear History"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>

        <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-slate-400 hover:text-white transition-colors">
          <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
            <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
          </svg>
        </a>
      </div>
    </header>
  );
};

function TypingIndicator() {
  return (
    <div className="flex space-x-1.5 p-2">
      <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
      <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
      <div className="w-2 h-2 bg-slate-400 rounded-full typing-dot"></div>
    </div>
  );
}

const SourceChip: React.FC<{ source: Source }> = ({ source }) => {
  const getDomain = (url: string) => {
    try {
      return new URL(url).hostname.replace('www.', '');
    } catch {
      return 'Unknown Source';
    }
  };

  return (
    <a 
      href={source.uri} 
      target="_blank" 
      rel="noopener noreferrer"
      className="inline-flex items-center gap-2 max-w-full bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg px-3 py-2 transition-colors group text-left"
    >
      <div className="w-6 h-6 rounded-md bg-slate-600 flex-shrink-0 flex items-center justify-center text-xs font-bold text-slate-300">
        {getDomain(source.uri).charAt(0).toUpperCase()}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-xs font-medium text-slate-200 truncate">{source.title || getDomain(source.uri)}</div>
        <div className="text-[10px] text-slate-400 truncate">{getDomain(source.uri)}</div>
      </div>
      <svg className="w-4 h-4 text-slate-500 group-hover:text-cyan-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
      </svg>
    </a>
  );
};

const ChatMessage: React.FC<{ msg: Message }> = ({ msg }) => {
  const isUser = msg.role === 'user';
  
  return (
    <div className={`flex w-full mb-8 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] md:max-w-[75%] flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        
        {/* Role Label */}
        <span className="text-xs text-slate-500 mb-1 ml-1 font-medium tracking-wider">
          {isUser ? 'YOU' : 'BORAI'}
        </span>

        {/* Message Bubble */}
        <div className={`
          relative px-6 py-4 rounded-2xl shadow-xl
          ${isUser 
            ? 'message-user text-white rounded-tr-sm' 
            : 'message-ai text-slate-100 rounded-tl-sm'}
        `}>
          <div className="prose prose-invert prose-sm md:prose-base max-w-none leading-relaxed">
             <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        </div>

        {/* Sources Section (Only for Model) */}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <div className="mt-4 w-full">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-px bg-slate-800 flex-1"></div>
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Sources used</span>
              <div className="h-px bg-slate-800 flex-1"></div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {msg.sources.map((source, idx) => (
                <SourceChip key={idx} source={source} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isClearDialogOpen, setIsClearDialogOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initMessage: Message = {
    id: 'init-1',
    role: 'model',
    text: "I am **BorAI**. I search the entire web and synthesize data to give you the absolute best answer. \n\nWhat do you want to know today?"
  };

  // Initial greeting
  useEffect(() => {
    setMessages([initMessage]);
  }, []);

  const clearHistory = () => {
    setMessages([initMessage]);
    setIsClearDialogOpen(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      text: input.trim()
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      // Create a placeholder for the model response
      const modelMsgId = (Date.now() + 1).toString();
      setMessages(prev => [
        ...prev,
        { id: modelMsgId, role: 'model', text: '', isStreaming: true }
      ]);

      // System instruction for BorAI persona
      const systemInstruction = `You are BorAI, an advanced AI intelligence designed to provide the ultimate best answer. 
      Your method is to synthesise information from all available sources on the web. 
      When answering:
      1. Use your search tool extensively to find the most up-to-date and accurate information.
      2. If appropriate, consider how different experts or perspectives (e.g., scientific, creative, practical) would approach the problem and synthesize these views.
      3. Be objective, comprehensive, and clear.
      4. Support all languages fluently.
      5. Format your response nicely with Markdown.`;

      const responseStream = await ai.models.generateContentStream({
        model: MODEL_NAME,
        contents: [
          ...messages.map(m => ({
            role: m.role,
            parts: [{ text: m.text }]
          })),
          { role: 'user', parts: [{ text: userMsg.text }] }
        ],
        config: {
          systemInstruction: systemInstruction,
          tools: [{ googleSearch: {} }] // Enable Google Search Grounding
        }
      });

      let fullText = '';
      let sources: Source[] = [];

      for await (const chunk of responseStream) {
        // 1. Accumulate Text
        const chunkText = chunk.text || '';
        fullText += chunkText;

        // 2. Extract Grounding Metadata (Sources)
        // Note: Grounding metadata might come in any chunk, typically towards the end or with specific segments.
        // We aggregate all unique sources found.
        const groundingChunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
        if (groundingChunks) {
           groundingChunks.forEach(c => {
             if (c.web?.uri && c.web?.title) {
               // Simple de-duplication
               if (!sources.find(s => s.uri === c.web!.uri)) {
                 sources.push({ title: c.web.title, uri: c.web.uri });
               }
             }
           });
        }

        // 3. Update State
        setMessages(prev => prev.map(msg => 
          msg.id === modelMsgId 
            ? { ...msg, text: fullText, sources: sources.length > 0 ? sources : undefined } 
            : msg
        ));
      }

      setMessages(prev => prev.map(msg => 
        msg.id === modelMsgId ? { ...msg, isStreaming: false } : msg
      ));

    } catch (error) {
      console.error("Error generating content:", error);
      setMessages(prev => [
        ...prev,
        { 
          id: Date.now().toString(), 
          role: 'model', 
          text: "I encountered a connection error while trying to synthesize the answer. Please try again." 
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-950 text-slate-200">
      <Header onClear={() => setIsClearDialogOpen(true)} />

      <ClearDialog 
        isOpen={isClearDialogOpen} 
        onClose={() => setIsClearDialogOpen(false)} 
        onConfirm={clearHistory} 
      />

      {/* Main Chat Area */}
      <main className="flex-1 overflow-y-auto pt-24 pb-32 px-4 md:px-0">
        <div className="max-w-4xl mx-auto flex flex-col">
          {messages.map(msg => (
            <ChatMessage key={msg.id} msg={msg} />
          ))}
          {isLoading && messages[messages.length - 1]?.role === 'user' && (
             <div className="flex w-full mb-8 justify-start">
               <div className="max-w-[75%]">
                 <span className="text-xs text-slate-500 mb-1 ml-1 font-medium tracking-wider">BORAI</span>
                 <div className="message-ai rounded-2xl rounded-tl-sm px-4 py-3">
                   <TypingIndicator />
                 </div>
               </div>
             </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <div className="fixed bottom-0 left-0 right-0 p-4 md:p-6 bg-gradient-to-t from-slate-950 via-slate-950 to-transparent z-40">
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSubmit} className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl opacity-30 group-hover:opacity-60 transition duration-500 blur"></div>
            <div className="relative flex items-end gap-2 bg-slate-900 rounded-xl p-2 border border-slate-800 focus-within:border-slate-600 focus-within:ring-1 focus-within:ring-slate-600 transition-all shadow-2xl">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
                placeholder="Ask BorAI anything..."
                className="w-full bg-transparent border-0 text-slate-200 placeholder-slate-500 focus:ring-0 resize-none py-3 px-3 min-h-[56px] max-h-32"
                rows={1}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="mb-1 p-3 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 flex-shrink-0"
              >
                {isLoading ? (
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                  </svg>
                )}
              </button>
            </div>
            <div className="text-center mt-2">
              <span className="text-[10px] text-slate-500 uppercase tracking-widest">
                Powered by Gemini • Real-time Web Synthesis
              </span>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);