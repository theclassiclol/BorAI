import React, { useState, useEffect, useRef, createContext, useContext } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";
import ReactMarkdown from 'react-markdown';

// --- Configuration ---
const MODEL_NAME = 'gemini-3-flash-preview';

const LANGUAGES = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Español' },
  { code: 'fr', name: 'Français' },
  { code: 'de', name: 'Deutsch' },
  { code: 'zh', name: '中文 (Chinese)' },
  { code: 'ja', name: '日本語 (Japanese)' },
  { code: 'ko', name: '한국어 (Korean)' },
  { code: 'hi', name: 'हिन्दी (Hindi)' },
  { code: 'ar', name: 'العربية (Arabic)' },
  { code: 'pt', name: 'Português' },
  { code: 'ru', name: 'Русский (Russian)' },
  { code: 'it', name: 'Italiano' },
  { code: 'nl', name: 'Nederlands (Dutch)' },
  { code: 'tr', name: 'Türkçe (Turkish)' },
  { code: 'pl', name: 'Polski (Polish)' },
  { code: 'id', name: 'Bahasa Indonesia' },
  { code: 'vi', name: 'Tiếng Việt (Vietnamese)' },
  { code: 'th', name: 'ไทย (Thai)' },
];

type AppMode = 'standard' | 'tutor' | 'research';

// --- Types ---
interface Source {
  title: string;
  uri: string;
}

interface ImageAttachment {
  data: string; // Base64 string (raw)
  mimeType: string;
  previewUri: string; // Full data URI for preview
}

interface Message {
  id: string;
  role: 'user' | 'model';
  text: string;
  sources?: Source[];
  images?: ImageAttachment[];
  isStreaming?: boolean;
  isError?: boolean;
}

interface ChatSession {
  id: string;
  userId: string; // Foreign key to User
  title: string;
  messages: Message[];
  appMode: AppMode;
  createdAt: number;
  updatedAt: number;
}

interface UserProfile {
  uid: string;
  displayName: string;
  email: string;
  passwordHash?: string; // Simple hash for local auth
  photoURL: string;
  provider: 'local' | 'google' | 'microsoft';
  createdAt: number;
}

// --- IndexedDB Service ---
// Robust storage for large chat history and images
class DBService {
  private dbName = 'BorAI_DB';
  private dbVersion = 1;
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Users store
        if (!db.objectStoreNames.contains('users')) {
          const userStore = db.createObjectStore('users', { keyPath: 'uid' });
          userStore.createIndex('email', 'email', { unique: true });
        }

        // Sessions store
        if (!db.objectStoreNames.contains('sessions')) {
          const sessionStore = db.createObjectStore('sessions', { keyPath: 'id' });
          sessionStore.createIndex('userId', 'userId', { unique: false });
          sessionStore.createIndex('updatedAt', 'updatedAt', { unique: false });
        }
      };
    });
  }

  // --- User Operations ---

  async createUser(user: UserProfile): Promise<void> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction('users', 'readwrite');
      const store = tx.objectStore('users');
      const request = store.add(user);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getUser(uid: string): Promise<UserProfile | undefined> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction('users', 'readonly');
      const store = tx.objectStore('users');
      const request = store.get(uid);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async getUserByEmail(email: string): Promise<UserProfile | undefined> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction('users', 'readonly');
      const store = tx.objectStore('users');
      const index = store.index('email');
      const request = index.get(email);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  // --- Session Operations ---

  async saveSession(session: ChatSession): Promise<void> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction('sessions', 'readwrite');
      const store = tx.objectStore('sessions');
      const request = store.put(session);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  async getSessions(userId: string): Promise<ChatSession[]> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction('sessions', 'readonly');
      const store = tx.objectStore('sessions');
      const index = store.index('userId');
      const request = index.getAll(IDBKeyRange.only(userId));
      
      request.onsuccess = () => {
        const sessions = request.result as ChatSession[];
        // Sort by update time desc
        sessions.sort((a, b) => b.updatedAt - a.updatedAt);
        resolve(sessions);
      };
      request.onerror = () => reject(request.error);
    });
  }

  async deleteSession(id: string): Promise<void> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const tx = this.db!.transaction('sessions', 'readwrite');
      const store = tx.objectStore('sessions');
      const request = store.delete(id);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }
}

const db = new DBService();

// --- API Client ---
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- Helper Functions ---
const getErrorMessage = (error: any): string => {
  if (!error) return "An unknown error occurred.";
  const msg = error.message || error.toString();
  console.error("Raw API Error:", error);

  if (msg.includes("SAFETY") || msg.includes("BLOCKED")) return "I cannot answer this query because it triggers my safety filters.";
  if (msg.includes("401") || msg.includes("API key")) return "Authentication failed. Please verify your API key.";
  if (msg.includes("429") || msg.includes("403")) return "I'm receiving too many requests right now. Please wait.";
  if (msg.includes("500") || msg.includes("503")) return "My servers are currently overloaded. Please try again later.";
  if (msg.includes("fetch") || msg.includes("Failed to fetch")) return "Network error. Please check your connection.";

  return "I encountered an unexpected error.";
};

const fileToBase64 = (file: File): Promise<ImageAttachment> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64Data = result.split(',')[1];
      const mimeType = result.split(':')[1].split(';')[0];
      resolve({ data: base64Data, mimeType, previewUri: result });
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
};

const formatTimeAgo = (timestamp: number) => {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return 'Just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
};

// --- Auth Context ---
interface AuthContextType {
  user: UserProfile | null;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<void>;
  register: (email: string, pass: string, name: string) => Promise<void>;
  loginAsGuest: () => void;
  socialLoginMock: (provider: 'google' | 'microsoft') => Promise<void>;
  logout: () => void;
  showLoginModal: boolean;
  setShowLoginModal: (show: boolean) => void;
  authError: string | null;
  setAuthError: (err: string | null) => void;
}

const AuthContext = createContext<AuthContextType>({} as AuthContextType);

const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  // Initialize DB and load session
  useEffect(() => {
    const initAuth = async () => {
      try {
        await db.init();
        const storedUid = localStorage.getItem('borai_uid');
        if (storedUid) {
          const u = await db.getUser(storedUid);
          if (u) {
            setUser(u);
          } else {
            localStorage.removeItem('borai_uid'); // Invalid ID
          }
        }
      } catch (e) {
        console.error("Auth Init Error", e);
      } finally {
        setIsLoading(false);
      }
    };
    initAuth();
  }, []);

  const login = async (email: string, pass: string) => {
    setAuthError(null);
    try {
      const u = await db.getUserByEmail(email);
      // Simple hash check (In a real app, use bcrypt on server. Here we just simple compare for local privacy)
      // We'll base64 encode the password just to obscure it slightly in DB, but this is NOT secure for real threats, only local privacy.
      const passHash = btoa(pass); 
      
      if (!u || u.passwordHash !== passHash) {
        throw new Error("Invalid email or password");
      }
      
      setUser(u);
      localStorage.setItem('borai_uid', u.uid);
      setShowLoginModal(false);
    } catch (e: any) {
      setAuthError(e.message);
      throw e;
    }
  };

  const register = async (email: string, pass: string, name: string) => {
    setAuthError(null);
    try {
      const existing = await db.getUserByEmail(email);
      if (existing) throw new Error("Email already registered on this device");

      const newUser: UserProfile = {
        uid: 'user_' + Date.now() + Math.random().toString(36).substr(2, 9),
        email,
        displayName: name,
        passwordHash: btoa(pass),
        photoURL: `https://api.dicebear.com/7.x/initials/svg?seed=${name}`,
        provider: 'local',
        createdAt: Date.now()
      };

      await db.createUser(newUser);
      setUser(newUser);
      localStorage.setItem('borai_uid', newUser.uid);
      setShowLoginModal(false);
    } catch (e: any) {
      setAuthError(e.message);
      throw e;
    }
  };

  const loginAsGuest = async () => {
    const guestId = 'guest_user';
    let guest = await db.getUser(guestId);
    if (!guest) {
        guest = {
            uid: guestId,
            email: 'guest@local',
            displayName: 'Guest',
            photoURL: '',
            provider: 'local',
            createdAt: Date.now()
        };
        await db.createUser(guest);
    }
    setUser(guest);
    localStorage.setItem('borai_uid', guestId);
    setShowLoginModal(false);
  };

  // Mocks social login for local-only functionality
  const socialLoginMock = async (provider: 'google' | 'microsoft') => {
    setAuthError(null);
    // Simulate network delay
    await new Promise(r => setTimeout(r, 800));

    const name = provider === 'google' ? 'Google User' : 'Microsoft User';
    const email = provider === 'google' ? 'user@gmail.com' : 'user@outlook.com';
    
    // Check if we have a "social" user locally
    const uid = `local_${provider}_user`;
    let u = await db.getUser(uid);

    if (!u) {
       u = {
         uid,
         email,
         displayName: name,
         photoURL: `https://api.dicebear.com/7.x/avataaars/svg?seed=${provider}`,
         provider,
         createdAt: Date.now()
       };
       await db.createUser(u);
    }

    setUser(u);
    localStorage.setItem('borai_uid', u.uid);
    setShowLoginModal(false);
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('borai_uid');
    setShowLoginModal(true);
  };

  return (
    <AuthContext.Provider value={{ 
      user, isLoading, 
      login, register, loginAsGuest, socialLoginMock, logout, 
      showLoginModal, setShowLoginModal, 
      authError, setAuthError 
    }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => useContext(AuthContext);

// --- Components ---

const SignInDialog: React.FC = () => {
  const { showLoginModal, setShowLoginModal, login, register, loginAsGuest, socialLoginMock, authError, setAuthError } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (showLoginModal) {
      setAuthError(null);
      setEmail('');
      setPassword('');
      setName('');
      setIsRegistering(false);
    }
  }, [showLoginModal]);

  if (!showLoginModal) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) return;
    if (isRegistering && !name) return;

    setIsSubmitting(true);
    try {
      if (isRegistering) {
        await register(email, password, name);
      } else {
        await login(email, password);
      }
    } catch (e) {
      // Error handled in context
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animation-fade-in">
      <div 
        className="bg-slate-900 border border-slate-700 rounded-2xl w-full max-w-md p-8 shadow-2xl relative overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {!isRegistering && (
             <button 
              onClick={() => loginAsGuest()}
              className="absolute top-4 right-4 text-xs font-medium text-slate-400 hover:text-white transition-colors bg-slate-800 px-3 py-1 rounded-full border border-slate-700"
            >
              Skip as Guest
            </button>
        )}

        <div className="text-center mb-6">
          <h2 className="text-2xl font-bold text-white mb-2 brand-font">
            {isRegistering ? "Create Profile" : "Welcome Back"}
          </h2>
          <p className="text-slate-400 text-sm">
            {isRegistering ? "Your data is saved securely on this device." : "Sign in to access your local history."}
          </p>
        </div>

        {authError && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/50 rounded-lg text-red-200 text-sm text-center">
            {authError}
          </div>
        )}

        <div className="space-y-3 mb-6">
          <button 
            onClick={() => socialLoginMock('google')}
            className="w-full flex items-center justify-center gap-3 bg-white text-slate-900 font-medium py-2.5 rounded-xl hover:bg-slate-100 transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4" />
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05" />
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
            </svg>
            Continue with Google
          </button>

          <button 
            onClick={() => socialLoginMock('microsoft')}
            className="w-full flex items-center justify-center gap-3 bg-[#2F2F2F] text-white font-medium py-2.5 rounded-xl border border-slate-700 hover:bg-[#3F3F3F] transition-colors"
          >
            <svg className="w-5 h-5" viewBox="0 0 23 23">
              <path fill="#F25022" d="M1 1h10v10H1z"/>
              <path fill="#00A4EF" d="M1 12h10v10H1z"/>
              <path fill="#7FBA00" d="M12 1h10v10H12z"/>
              <path fill="#FFB900" d="M12 12h10v10H12z"/>
            </svg>
            Continue with Microsoft
          </button>
        </div>

        <div className="flex items-center gap-4 mb-6">
          <div className="h-px bg-slate-800 flex-1"></div>
          <span className="text-xs text-slate-500 font-medium">OR USE PROFILE</span>
          <div className="h-px bg-slate-800 flex-1"></div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {isRegistering && (
            <div className="animate-fade-in-up">
              <label className="block text-xs font-medium text-slate-400 mb-1.5 ml-1">Profile Name</label>
              <input 
                type="text" 
                required={isRegistering}
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
                placeholder="John Doe"
              />
            </div>
          )}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5 ml-1">Email / ID</label>
            <input 
              type="text" 
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
              placeholder="name@example.com"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5 ml-1">Passcode</label>
            <input 
              type="password" 
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-white focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
              placeholder="••••••••"
            />
          </div>
          <button 
            type="submit"
            disabled={isSubmitting}
            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-900/20 flex justify-center items-center"
          >
            {isSubmitting ? (
              <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              isRegistering ? "Create Profile" : "Sign In"
            )}
          </button>
        </form>
        
        <div className="mt-4 text-center">
          <button 
            onClick={() => setIsRegistering(!isRegistering)}
            className="text-sm text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            {isRegistering ? "Have a profile? Sign in" : "New user? Create profile"}
          </button>
        </div>
      </div>
    </div>
  );
};

const ClearDialog: React.FC<{ isOpen: boolean; onClose: () => void; onConfirm: () => void }> = ({ isOpen, onClose, onConfirm }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animation-fade-in">
      <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-sm w-full p-6 shadow-2xl relative">
        <h3 className="text-lg font-bold text-white mb-2 brand-font">Clear Conversation</h3>
        <p className="text-slate-400 mb-6 text-sm">Are you sure you want to clear the current conversation? This cannot be undone.</p>
        <div className="flex gap-3 justify-end">
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors text-sm font-medium">Cancel</button>
          <button onClick={onConfirm} className="px-4 py-2 rounded-lg bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/50 transition-colors text-sm font-medium">Clear Chat</button>
        </div>
      </div>
    </div>
  );
};

const Sidebar: React.FC<{ 
  isOpen: boolean; 
  onClose: () => void;
  sessions: ChatSession[];
  currentId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string, e: React.MouseEvent) => void;
}> = ({ isOpen, onClose, sessions, currentId, onSelect, onNew, onDelete }) => {
  const { user, setShowLoginModal, logout } = useAuth();

  return (
    <>
      {/* Backdrop */}
      <div 
        className={`fixed inset-0 bg-black/50 backdrop-blur-sm z-[55] transition-opacity duration-300 md:hidden ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        onClick={onClose}
      />
      
      {/* Sidebar Panel */}
      <div className={`fixed inset-y-0 left-0 z-[60] w-72 bg-slate-950/95 backdrop-blur-xl border-r border-slate-800 transform transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'} flex flex-col`}>
        {/* Header */}
        <div className="p-4 border-b border-slate-800">
          <button 
            onClick={onNew}
            className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white py-3 rounded-xl transition-all font-medium shadow-lg shadow-indigo-900/20"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" /></svg>
            New Chat
          </button>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {sessions.map(session => {
            const isActive = session.id === currentId;
            const firstUserMsg = session.messages.find(m => m.role === 'user');
            const hasImage = firstUserMsg?.images && firstUserMsg.images.length > 0;
            const previewImage = hasImage ? firstUserMsg!.images![0].previewUri : null;
            
            let ModeIcon;
            let modeColor;
            switch(session.appMode) {
              case 'tutor': 
                modeColor = 'text-violet-400';
                ModeIcon = (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path d="M12 14l9-5-9-5-9 5 9 5z" />
                    <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
                  </svg>
                );
                break;
              case 'research':
                modeColor = 'text-emerald-400';
                ModeIcon = (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                );
                break;
              default:
                modeColor = 'text-cyan-400';
                ModeIcon = (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                );
            }

            return (
              <div 
                key={session.id}
                onClick={() => onSelect(session.id)}
                className={`group relative flex items-center gap-3 p-3 rounded-xl cursor-pointer transition-all border ${isActive ? 'bg-slate-800 border-slate-700 shadow-md' : 'border-transparent hover:bg-slate-800/50 hover:border-slate-800'}`}
              >
                {/* Visual Preview */}
                <div className={`w-10 h-10 rounded-lg flex-shrink-0 flex items-center justify-center overflow-hidden border border-slate-700/50 ${hasImage ? 'bg-black' : 'bg-slate-900'}`}>
                  {hasImage ? (
                    <img src={previewImage!} alt="Chat Preview" className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" />
                  ) : (
                    <div className={`${modeColor} opacity-80 group-hover:opacity-100 transition-opacity`}>
                      {ModeIcon}
                    </div>
                  )}
                </div>

                {/* Text Info */}
                <div className="flex-1 min-w-0">
                  <h4 className={`text-sm font-medium truncate mb-0.5 ${isActive ? 'text-white' : 'text-slate-300 group-hover:text-white'}`}>
                    {session.title}
                  </h4>
                  <span className="text-[10px] text-slate-500 font-medium uppercase tracking-wider block">
                    {formatTimeAgo(session.updatedAt)}
                  </span>
                </div>

                {/* Delete Button */}
                <button
                  onClick={(e) => onDelete(session.id, e)}
                  className="absolute right-2 opacity-0 group-hover:opacity-100 p-1.5 text-slate-400 hover:text-red-400 hover:bg-slate-700 rounded-lg transition-all"
                  title="Delete Chat"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>
                </button>
              </div>
            );
          })}
        </div>
        
        {/* User Profile Footer */}
        <div className="p-4 border-t border-slate-800">
           {user ? (
             <div className="flex items-center gap-3">
               <div className="w-10 h-10 rounded-full bg-indigo-500 flex items-center justify-center overflow-hidden ring-2 ring-indigo-500/30">
                 {user.photoURL ? (
                   <img src={user.photoURL} alt={user.displayName} className="w-full h-full object-cover" />
                 ) : (
                   <span className="text-white font-bold">{user.displayName.charAt(0)}</span>
                 )}
               </div>
               <div className="flex-1 min-w-0">
                 <p className="text-sm font-medium text-white truncate">{user.displayName}</p>
                 <button 
                   onClick={logout}
                   className="text-xs text-slate-400 hover:text-red-400 transition-colors"
                 >
                   Sign Out (Local)
                 </button>
               </div>
             </div>
           ) : (
             <button
                onClick={() => setShowLoginModal(true)}
                className="w-full py-2.5 rounded-lg border border-slate-700 hover:bg-slate-800 text-slate-300 hover:text-white text-sm font-medium transition-colors"
             >
               Sign In / Create Profile
             </button>
           )}
           <div className="mt-4 text-center">
             <div className="text-[10px] text-slate-600 uppercase tracking-widest font-semibold">BorAI v2.0</div>
           </div>
        </div>
      </div>
    </>
  );
};

interface HeaderProps { 
  onMenuClick: () => void;
  onClear: () => void;
  language: string;
  setLanguage: (lang: string) => void;
  appMode: AppMode;
  setAppMode: (mode: AppMode) => void;
}

const Header: React.FC<HeaderProps> = ({ onMenuClick, onClear, language, setLanguage, appMode, setAppMode }) => {
  
  const getLogoGradient = () => {
    switch(appMode) {
      case 'tutor': return 'from-violet-500 to-fuchsia-600';
      case 'research': return 'from-emerald-400 to-teal-600';
      default: return 'from-cyan-400 to-blue-600';
    }
  };

  const getTextGradient = () => {
    switch(appMode) {
      case 'tutor': return 'from-violet-400 to-fuchsia-500';
      case 'research': return 'from-emerald-400 to-teal-500';
      default: return 'from-cyan-400 to-blue-500';
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 h-16 glass-panel z-50 flex items-center justify-between px-4 md:px-6 shadow-lg">
      <div className="flex items-center gap-3">
        <button 
          onClick={onMenuClick}
          className="p-2 -ml-2 text-slate-400 hover:text-white hover:bg-slate-800/50 rounded-lg transition-colors"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
        </button>
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors duration-500 bg-gradient-to-tr ${getLogoGradient()}`}>
            {appMode === 'tutor' && (
               <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path d="M12 14l9-5-9-5-9 5 9 5z" />
                  <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 14l9-5-9-5-9 5 9 5zm0 0l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14zm-4 6v-7.5l4-2.222" />
               </svg>
            )}
            {appMode === 'research' && (
               <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
               </svg>
            )}
            {appMode === 'standard' && (
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            )}
          </div>
          <span className={`text-xl md:text-2xl font-bold brand-font bg-clip-text text-transparent borai-glow transition-colors duration-500 bg-gradient-to-r ${getTextGradient()}`}>
            BorAI 
            {appMode === 'tutor' && <span className="text-sm font-light tracking-widest text-white/80 ml-1 hidden sm:inline">TUTOR</span>}
            {appMode === 'research' && <span className="text-sm font-light tracking-widest text-white/80 ml-1 hidden sm:inline">RESEARCH</span>}
          </span>
        </div>
      </div>
      <div className="flex items-center gap-2 md:gap-4">
        
        {/* Mode Toggles */}
        <div className="flex items-center gap-1 bg-slate-800/50 p-1 rounded-xl border border-slate-700/50">
          <button
            onClick={() => setAppMode(appMode === 'tutor' ? 'standard' : 'tutor')}
            className={`p-1.5 rounded-lg transition-all duration-300 ${
              appMode === 'tutor'
                ? 'bg-violet-500/20 text-violet-200 shadow-sm' 
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
            }`}
            title="Homework/Tutor Mode"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
               <path d="M12 14l9-5-9-5-9 5 9 5z" />
               <path d="M12 14l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14z" />
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 14l9-5-9-5-9 5 9 5zm0 0l6.16-3.422a12.083 12.083 0 01.665 6.479A11.952 11.952 0 0012 20.055a11.952 11.952 0 00-6.824-2.998 12.078 12.078 0 01.665-6.479L12 14zm-4 6v-7.5l4-2.222" />
            </svg>
          </button>
          
          <button
            onClick={() => setAppMode(appMode === 'research' ? 'standard' : 'research')}
            className={`p-1.5 rounded-lg transition-all duration-300 ${
              appMode === 'research'
                ? 'bg-emerald-500/20 text-emerald-200 shadow-sm' 
                : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
            }`}
            title="Deep Research Mode"
          >
             <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
             </svg>
          </button>
        </div>

        <div className="relative hidden md:block">
             <select 
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="appearance-none bg-slate-800 text-slate-300 text-xs font-medium border border-slate-700 rounded-lg pl-3 pr-8 py-1.5 focus:outline-none focus:ring-1 focus:ring-cyan-500 focus:border-cyan-500 hover:bg-slate-750 transition-colors cursor-pointer"
             >
                {LANGUAGES.map(lang => (
                  <option key={lang.code} value={lang.code}>{lang.name}</option>
                ))}
             </select>
             <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-400">
                <svg className="h-3 w-3 fill-current" viewBox="0 0 20 20">
                    <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" fillRule="evenodd" />
                </svg>
             </div>
        </div>

        <button 
          onClick={onClear}
          className="text-slate-400 hover:text-red-400 transition-colors p-2 rounded-lg hover:bg-slate-800/50" 
          title="Clear Conversation (Ctrl+K)"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
          </svg>
        </button>
      </div>
    </header>
  );
};

const SourceChip: React.FC<{ source: Source; index: number }> = ({ source, index }) => {
  return (
    <a 
      href={source.uri} 
      target="_blank" 
      rel="noopener noreferrer"
      className="flex items-center gap-2 p-2 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 rounded-lg transition-colors group text-left"
    >
      <div className="w-5 h-5 flex-shrink-0 flex items-center justify-center bg-slate-800 rounded text-xs text-slate-400 group-hover:text-cyan-400 font-mono">
        {index + 1}
      </div>
      <span className="text-xs text-slate-300 truncate group-hover:text-white">
        {source.title}
      </span>
    </a>
  );
};

const TypingIndicator: React.FC = () => {
  return (
    <div className="flex items-center gap-1 py-1">
      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
      <div className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce"></div>
    </div>
  );
};

const ChatMessage: React.FC<{ msg: Message }> = ({ msg }) => {
  const isUser = msg.role === 'user';
  const isError = msg.isError;
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    if (!msg.text) return;
    navigator.clipboard.writeText(msg.text).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }).catch(err => {
      console.error('Failed to copy:', err);
    });
  };
  
  return (
    <div className={`flex w-full mb-8 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[85%] md:max-w-[75%] flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        <span className="text-xs text-slate-500 mb-1 ml-1 font-medium tracking-wider">
          {isUser ? 'YOU' : 'BORAI'}
        </span>
        <div className={`relative px-6 py-4 rounded-2xl shadow-xl group ${isUser ? 'message-user text-white rounded-tr-sm' : isError ? 'bg-red-500/10 border border-red-500/50 text-red-200 rounded-tl-sm' : 'message-ai text-slate-100 rounded-tl-sm'}`}>
          {!isUser && !isError && (
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 p-1.5 rounded-lg bg-slate-900/50 backdrop-blur-sm border border-slate-700/50 text-slate-400 hover:text-white hover:bg-slate-800 transition-all opacity-0 group-hover:opacity-100 z-10"
              title="Copy to clipboard"
            >
              {isCopied ? (
                <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
              ) : (
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
              )}
            </button>
          )}

          {msg.images && msg.images.length > 0 && (
            <div className="mb-4 flex flex-wrap gap-2">
              {msg.images.map((img, idx) => (
                <div key={idx} className="relative rounded-lg overflow-hidden border border-white/20">
                  <img src={img.previewUri} alt="User uploaded attachment" className="max-h-64 object-cover" />
                </div>
              ))}
            </div>
          )}

          <div className="prose prose-invert prose-sm md:prose-base max-w-none leading-relaxed">
             <ReactMarkdown>{msg.text}</ReactMarkdown>
          </div>
        </div>

        {!isUser && !isError && msg.sources && msg.sources.length > 0 && (
          <div className="mt-4 w-full">
            <div className="flex items-center gap-2 mb-2">
              <div className="h-px bg-slate-800 flex-1"></div>
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-widest">Sources used</span>
              <div className="h-px bg-slate-800 flex-1"></div>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {msg.sources.map((source, idx) => (
                <SourceChip key={`${source.uri}-${idx}`} source={source} index={idx} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const createNewSession = (userId: string): ChatSession => ({
  id: Date.now().toString() + Math.random().toString(36).substr(2, 5),
  userId,
  title: 'New Conversation',
  messages: [{
    id: 'init-1',
    role: 'model',
    text: "I am **BorAI**. I search the entire web and synthesize data to give you the absolute best answer. \n\nWhat do you want to know today?"
  }],
  appMode: 'standard',
  createdAt: Date.now(),
  updatedAt: Date.now()
});

const ChatApp: React.FC = () => {
  const { user, setShowLoginModal, showLoginModal } = useAuth();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isClearDialogOpen, setIsClearDialogOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [language, setLanguage] = useState('en');
  const [pendingImages, setPendingImages] = useState<ImageAttachment[]>([]);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Force login if no user
  useEffect(() => {
    if (!user) {
        setShowLoginModal(true);
        setSessions([]);
    } else {
        // Load sessions from DB
        const loadSessions = async () => {
           try {
             const loadedSessions = await db.getSessions(user.uid);
             if (loadedSessions.length > 0) {
                setSessions(loadedSessions);
                setCurrentSessionId(loadedSessions[0].id);
             } else {
                const newSession = createNewSession(user.uid);
                await db.saveSession(newSession);
                setSessions([newSession]);
                setCurrentSessionId(newSession.id);
             }
           } catch (e) {
             console.error("Error loading sessions", e);
           }
        };
        loadSessions();
    }
  }, [user]);

  // Persist Current Session to DB on changes
  useEffect(() => {
    if (!user || !currentSessionId) return;
    
    const sessionToSave = sessions.find(s => s.id === currentSessionId);
    if (sessionToSave) {
        // Debounce saving slightly or just save on every change
        db.saveSession(sessionToSave).catch(e => console.error("Save failed", e));
    }
  }, [sessions, currentSessionId, user]);

  const currentSession = sessions.find(s => s.id === currentSessionId);
  // Safe access
  const messages = currentSession ? currentSession.messages : [];
  const appMode = currentSession ? currentSession.appMode : 'standard';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, pendingImages, currentSessionId]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsClearDialogOpen(true);
      }
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'i') {
        e.preventDefault();
        fileInputRef.current?.click();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, []);

  const updateCurrentSession = (updates: Partial<ChatSession>) => {
    setSessions(prev => prev.map(s => s.id === currentSessionId ? { ...s, ...updates, updatedAt: Date.now() } : s));
  };

  const handleNewChat = async () => {
    if (!user) return;
    const newSession = createNewSession(user.uid);
    await db.saveSession(newSession);
    setSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    setIsSidebarOpen(false);
    setInput('');
    setPendingImages([]);
  };

  const handleDeleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!user) return;

    await db.deleteSession(id);
    const newSessions = sessions.filter(s => s.id !== id);
    
    if (newSessions.length === 0) {
      await handleNewChat();
    } else {
      setSessions(newSessions);
      if (currentSessionId === id) {
        setCurrentSessionId(newSessions[0].id);
      }
    }
  };

  const clearHistory = async () => {
    if (!user) return;
    const newSession = createNewSession(user.uid);
    // Keep the ID if we just want to clear messages, or replace? replacing is easier for "Clear" usually means empty chat
    // But let's keep ID to not break references, just reset content
    const cleanedSession = { 
        ...currentSession!, 
        messages: newSession.messages, 
        updatedAt: Date.now() 
    };
    
    setSessions(prev => prev.map(s => s.id === currentSessionId ? cleanedSession : s));
    await db.saveSession(cleanedSession);
    
    setIsClearDialogOpen(false);
    setPendingImages([]);
  };

  const setAppMode = (mode: AppMode) => {
    updateCurrentSession({ appMode: mode });
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      try {
        const file = e.target.files[0];
        if (!file.type.startsWith('image/')) { alert('Please upload an image file.'); return; }
        // IndexedDB can handle larger files, but let's keep a reasonable limit for performance
        if (file.size > 10 * 1024 * 1024) { alert('File too large. Please select an image under 10MB.'); return; }
        const imageAttachment = await fileToBase64(file);
        setPendingImages(prev => [...prev, imageAttachment]);
        if (fileInputRef.current) fileInputRef.current.value = '';
      } catch (error) {
        console.error("Error processing file:", error);
        alert("Failed to process image.");
      }
    }
  };

  const removePendingImage = (index: number) => {
    setPendingImages(prev => prev.filter((_, i) => i !== index));
  };

  const generateTitle = (text: string) => {
    return text.length > 40 ? text.substring(0, 40) + '...' : text;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentSession) return;
    if ((!input.trim() && pendingImages.length === 0) || isLoading) return;

    const currentImages = [...pendingImages];
    const currentInput = input.trim();

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      text: currentInput,
      images: currentImages.length > 0 ? currentImages : undefined
    };
    
    const modelMsgId = (Date.now() + 1).toString();

    let newTitle = currentSession.title;
    if (currentSession.messages.length === 1 && currentSession.title === 'New Conversation') {
      newTitle = generateTitle(currentInput || (currentImages.length > 0 ? 'Image Query' : 'Conversation'));
    }

    // Optimistic Update
    const updatedSession = {
        ...currentSession,
        title: newTitle,
        messages: [...currentSession.messages, userMsg],
        updatedAt: Date.now()
    };
    
    setSessions(prev => prev.map(s => s.id === currentSessionId ? updatedSession : s));
    
    setInput('');
    setPendingImages([]);
    setIsLoading(true);

    let fullText = '';

    try {
      // Add placeholder for AI
      setSessions(prev => prev.map(s => 
        s.id === currentSessionId 
        ? { ...s, messages: [...s.messages, { id: modelMsgId, role: 'model', text: '', isStreaming: true }] }
        : s
      ));

      const selectedLangName = LANGUAGES.find(l => l.code === language)?.name || 'English';

      const standardInstruction = `You are BorAI. Synthesise information from all available sources on the web to provide the ultimate best answer.
      Instructions:
      1. SEARCH: Use search extensively.
      2. MULTIMODAL: Analyze provided images.
      3. KEY FINDINGS: For research queries, start with "## Key Findings" (translated).
      4. OUTPUT: Answer in ${selectedLangName}.`;

      const homeworkInstruction = `You are BorAI Tutor.
      Instructions:
      1. IDENTIFY: Determine subject and concept.
      2. STEP-BY-STEP: Break down solutions. Explain *why*.
      3. ENCOURAGE: Maintain academic tone.
      4. OUTPUT: Answer in ${selectedLangName}.`;

      const researchInstruction = `You are BorAI Researcher.
      Instructions:
      1. DEEP SEARCH: Granular, recent data.
      2. CROSS-REFERENCE: Verify facts.
      3. REPORT FORMAT: Executive Summary, Detailed Analysis, Data/Stats.
      4. CITATIONS: Required.
      5. OUTPUT: Answer in ${selectedLangName}.`;

      let systemInstruction;
      switch (appMode) {
        case 'tutor': systemInstruction = homeworkInstruction; break;
        case 'research': systemInstruction = researchInstruction; break;
        default: systemInstruction = standardInstruction; break;
      }

      const historyContents = currentSession.messages.map(m => ({
        role: m.role,
        parts: [
          ...(m.images?.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.data } })) || []),
          ...(m.text ? [{ text: m.text }] : [])
        ]
      }));

      const currentParts: any[] = [
        ...(currentImages.map(img => ({ inlineData: { mimeType: img.mimeType, data: img.data } }))),
        ...(currentInput ? [{ text: currentInput }] : (currentImages.length > 0 ? [{ text: "" }] : []))
      ];

      const requestConfig: any = {
        systemInstruction: systemInstruction,
        tools: [{ googleSearch: {} }]
      };

      if (appMode === 'research') {
        requestConfig.thinkingConfig = { thinkingBudget: 2048 };
      }

      // RETRY LOGIC for Rate Limits (429)
      const makeRequest = async () => {
         return await ai.models.generateContentStream({
            model: MODEL_NAME,
            contents: [...historyContents, { role: 'user', parts: currentParts }],
            config: requestConfig
          });
      };

      let responseStream;
      let attempt = 0;
      const maxRetries = 3;
      
      while (attempt <= maxRetries) {
        try {
          responseStream = await makeRequest();
          break; // Success
        } catch (e: any) {
          const msg = e.message || e.toString();
          // Check for Rate Limit (429), Forbidden (403), or Server Error (503)
          if ((msg.includes('429') || msg.includes('Too Many Requests') || msg.includes('403') || msg.includes('503')) && attempt < maxRetries) {
             attempt++;
             // Exponential backoff: 1s, 2s, 4s
             const waitTime = Math.pow(2, attempt - 1) * 1000; 
             console.warn(`BorAI: Rate limit hit. Retrying in ${waitTime}ms...`);
             await new Promise(r => setTimeout(r, waitTime));
             continue;
          }
          throw e; // Fatal error or max retries reached
        }
      }

      let sources: Source[] = [];

      for await (const chunk of responseStream) {
        const chunkText = chunk.text || '';
        fullText += chunkText;

        const groundingChunks = chunk.candidates?.[0]?.groundingMetadata?.groundingChunks;
        if (groundingChunks) {
           groundingChunks.forEach(c => {
             if (c.web?.uri && c.web?.title) {
               if (!sources.find(s => s.uri === c.web!.uri)) {
                 sources.push({ title: c.web.title, uri: c.web.uri });
               }
             }
           });
        }

        const currentSources = [...sources];
        
        setSessions(prev => prev.map(s => 
          s.id === currentSessionId 
          ? { 
              ...s, 
              messages: s.messages.map(m => m.id === modelMsgId ? { ...m, text: fullText, sources: currentSources.length > 0 ? currentSources : undefined } : m)
            }
          : s
        ));
      }

      setSessions(prev => prev.map(s => 
        s.id === currentSessionId 
        ? { 
            ...s, 
            messages: s.messages.map(m => m.id === modelMsgId ? { ...m, isStreaming: false } : m)
          }
        : s
      ));

    } catch (error: any) {
      const errorText = getErrorMessage(error);
      setSessions(prev => prev.map(s => 
        s.id === currentSessionId 
        ? { 
            ...s, 
            messages: s.messages.map(m => m.id === modelMsgId ? { ...m, text: fullText ? `${fullText}\n\n---\n\n⚠️ **${errorText}**` : errorText, isError: true, isStreaming: false } : m)
          }
        : s
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const getThemeColor = () => {
    switch(appMode) {
      case 'tutor': return 'text-violet-400';
      case 'research': return 'text-emerald-400';
      default: return 'text-slate-500';
    }
  };

  const getMessageBg = () => {
    switch(appMode) {
      case 'tutor': return 'bg-slate-800 border border-violet-500/30';
      case 'research': return 'bg-slate-800 border border-emerald-500/30';
      default: return 'message-ai';
    }
  };

  const getButtonClass = () => {
    switch(appMode) {
      case 'tutor': return 'bg-violet-600 hover:bg-violet-500';
      case 'research': return 'bg-emerald-600 hover:bg-emerald-500';
      default: return 'bg-indigo-600 hover:bg-indigo-500';
    }
  };

  const getGlowClass = () => {
    switch(appMode) {
      case 'tutor': return 'bg-gradient-to-r from-violet-500 to-fuchsia-600';
      case 'research': return 'bg-gradient-to-r from-emerald-500 to-teal-600';
      default: return 'bg-gradient-to-r from-cyan-500 to-blue-600';
    }
  };

  if (!user && !showLoginModal) {
      // Fallback if modal is closed but no user (shouldn't happen with current logic but for safety)
      return <div className="flex h-screen items-center justify-center bg-slate-950 text-white">Loading...</div>;
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-950 text-slate-200">
      <SignInDialog />
      <Sidebar 
        isOpen={isSidebarOpen} 
        onClose={() => setIsSidebarOpen(false)}
        sessions={sessions}
        currentId={currentSessionId}
        onSelect={(id) => { setCurrentSessionId(id); setIsSidebarOpen(false); }}
        onNew={handleNewChat}
        onDelete={handleDeleteSession}
      />

      <Header 
        onMenuClick={() => setIsSidebarOpen(!isSidebarOpen)}
        onClear={() => setIsClearDialogOpen(true)} 
        language={language}
        setLanguage={setLanguage}
        appMode={appMode}
        setAppMode={setAppMode}
      />

      <ClearDialog 
        isOpen={isClearDialogOpen} 
        onClose={() => setIsClearDialogOpen(false)} 
        onConfirm={clearHistory} 
      />

      <main className="flex-1 overflow-y-auto pt-24 pb-32 px-4 md:px-0">
        <div className="max-w-4xl mx-auto flex flex-col">
          {messages.map(msg => (
            <ChatMessage key={msg.id} msg={msg} />
          ))}
          {isLoading && messages[messages.length - 1]?.role === 'user' && (
             <div className="flex w-full mb-8 justify-start">
               <div className="max-w-[75%]">
                 <span className={`text-xs mb-1 ml-1 font-medium tracking-wider ${getThemeColor()}`}>
                    {appMode === 'tutor' ? 'BORAI TUTOR' : appMode === 'research' ? 'BORAI RESEARCHER' : 'BORAI'}
                 </span>
                 <div className={`rounded-2xl rounded-tl-sm px-4 py-3 ${getMessageBg()}`}>
                   <TypingIndicator />
                 </div>
               </div>
             </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <div className="fixed bottom-0 left-0 right-0 p-4 md:p-6 bg-gradient-to-t from-slate-950 via-slate-950 to-transparent z-40">
        <div className="max-w-4xl mx-auto">
          {appMode !== 'standard' && (
            <div className="flex items-center gap-2 mb-2 ml-1 animate-fade-in-up">
              <span className="flex h-2 w-2 relative">
                 <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${appMode === 'tutor' ? 'bg-violet-400' : 'bg-emerald-400'}`}></span>
                 <span className={`relative inline-flex rounded-full h-2 w-2 ${appMode === 'tutor' ? 'bg-violet-500' : 'bg-emerald-500'}`}></span>
              </span>
              <span className={`text-xs font-medium tracking-wider ${appMode === 'tutor' ? 'text-violet-300' : 'text-emerald-300'}`}>
                {appMode === 'tutor' ? 'HOMEWORK MODE ACTIVE' : 'DEEP RESEARCH MODE ACTIVE'}
              </span>
            </div>
          )}

          {pendingImages.length > 0 && (
            <div className="flex gap-3 mb-3 overflow-x-auto pb-2 px-1">
              {pendingImages.map((img, idx) => (
                <div key={idx} className="relative group flex-shrink-0">
                  <div className="relative w-20 h-20 rounded-xl overflow-hidden border border-slate-700 shadow-lg">
                    <img src={img.previewUri} alt="Preview" className="w-full h-full object-cover" />
                  </div>
                  <button 
                    onClick={() => removePendingImage(idx)}
                    className="absolute -top-2 -right-2 bg-slate-800 text-slate-400 hover:text-red-400 rounded-full p-1 border border-slate-700 shadow-md transition-colors"
                  >
                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          )}

          <form onSubmit={handleSubmit} className="relative group">
            <div className={`absolute -inset-0.5 rounded-2xl opacity-30 group-hover:opacity-60 transition duration-500 blur ${getGlowClass()}`}></div>
            <div className="relative flex items-end gap-2 bg-slate-900 rounded-xl p-2 border border-slate-800 focus-within:border-slate-600 focus-within:ring-1 focus-within:ring-slate-600 transition-all shadow-2xl">
              
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileSelect}
                accept="image/png, image/jpeg, image/webp"
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="mb-1 p-3 rounded-lg text-slate-400 hover:text-cyan-400 hover:bg-slate-800 transition-all flex-shrink-0"
                title="Upload image (Ctrl+Shift+I)"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </button>

              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    if (!e.shiftKey || e.ctrlKey || e.metaKey) {
                      e.preventDefault();
                      handleSubmit(e as unknown as React.FormEvent);
                    }
                  }
                }}
                placeholder={
                  appMode === 'tutor' 
                    ? "Paste a problem or upload homework..." 
                    : appMode === 'research'
                      ? "Enter a research topic..."
                      : (pendingImages.length > 0 ? "Ask about this image..." : "Ask BorAI anything...")
                }
                className="w-full bg-transparent border-0 text-slate-200 placeholder-slate-500 focus:ring-0 resize-none py-3 px-3 min-h-[56px] max-h-32"
                rows={1}
              />
              <button
                type="submit"
                disabled={isLoading || (!input.trim() && pendingImages.length === 0)}
                className={`mb-1 p-3 rounded-lg text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 active:scale-95 flex-shrink-0 ${getButtonClass()}`}
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

function App() {
  return (
    <AuthProvider>
      <ChatApp />
    </AuthProvider>
  );
}

const root = createRoot(document.getElementById('root')!);
root.render(<App />);