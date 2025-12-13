import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import {
  Send,
  Database,
  Upload,
  RefreshCw,
  Trash2,
  Image as ImageIcon,
  ChevronRight,
  AlertCircle,
  CheckCircle,
  Loader2,
  Menu,
  X,
  FlaskConical,
  BarChart3,
  Settings,
  MessageSquare,
  Table,
  Beaker,
  Thermometer,
  DollarSign,
  AlertTriangle
} from 'lucide-react';

// API Configuration
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ============================================================
// API Functions
// ============================================================

const api = {
  async getStatus() {
    const res = await fetch(`${API_BASE}/api/status`);
    return res.json();
  },
  
  async chat(message, sessionId) {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: sessionId })
    });
    return res.json();
  },
  
  async getTables() {
    const res = await fetch(`${API_BASE}/api/tables`);
    return res.json();
  },
  
  async reindex() {
    const res = await fetch(`${API_BASE}/api/reindex`, { method: 'POST' });
    return res.json();
  },
  
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: formData
    });
    return res.json();
  },
  
  async getPlots() {
    const res = await fetch(`${API_BASE}/api/plots`);
    return res.json();
  },
  
  async clearPlots() {
    const res = await fetch(`${API_BASE}/api/plots`, { method: 'DELETE' });
    return res.json();
  },
  
  async clearSession(sessionId) {
    const res = await fetch(`${API_BASE}/api/session/${sessionId}`, { method: 'DELETE' });
    return res.json();
  }
};

// ============================================================
// Components
// ============================================================

// Status Badge Component
function StatusBadge({ status }) {
  const isReady = status === 'ready';
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
      isReady ? 'bg-emerald-500/20 text-emerald-400' : 'bg-amber-500/20 text-amber-400'
    }`}>
      {isReady ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
      {isReady ? 'Ready' : 'Limited'}
    </div>
  );
}

// Message Component
function Message({ message, isUser }) {
  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
        isUser ? 'bg-primary-600' : 'bg-slate-700'
      }`}>
        {isUser ? (
          <span className="text-sm font-medium">You</span>
        ) : (
          <FlaskConical size={16} />
        )}
      </div>
      <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
        <div className={`inline-block rounded-2xl px-4 py-3 ${
          isUser 
            ? 'bg-primary-600 text-white' 
            : 'bg-slate-800 text-slate-100'
        }`}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        {message.images && message.images.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {message.images.map((img, i) => (
              <a 
                key={i}
                href={`${API_BASE}/plots/${img}`}
                target="_blank"
                rel="noopener noreferrer"
                className="block"
              >
                <img 
                  src={`${API_BASE}/plots/${img}`}
                  alt={`Plot ${i + 1}`}
                  className="rounded-lg max-w-xs border border-slate-700 hover:border-primary-500 transition-colors"
                />
              </a>
            ))}
          </div>
        )}
        {message.elapsed && (
          <p className="text-xs text-slate-500 mt-1">
            {message.elapsed.toFixed(1)}s • {message.iterations} iterations
          </p>
        )}
      </div>
    </div>
  );
}

// Typing Indicator
function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center">
        <FlaskConical size={16} />
      </div>
      <div className="bg-slate-800 rounded-2xl px-4 py-3">
        <div className="typing-indicator flex gap-1">
          <span className="w-2 h-2 bg-slate-500 rounded-full"></span>
          <span className="w-2 h-2 bg-slate-500 rounded-full"></span>
          <span className="w-2 h-2 bg-slate-500 rounded-full"></span>
        </div>
      </div>
    </div>
  );
}

// Quick Action Button
function QuickAction({ icon: Icon, label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-sm text-slate-300 hover:text-white transition-colors"
    >
      <Icon size={16} />
      {label}
    </button>
  );
}

// Sidebar Component
function Sidebar({ isOpen, onClose, status, onReindex, onUpload, onClearPlots }) {
  const [tables, setTables] = useState([]);
  const [plots, setPlots] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  const loadData = async () => {
    try {
      const [tablesRes, plotsRes] = await Promise.all([
        api.getTables(),
        api.getPlots()
      ]);
      setTables(tablesRes.tables || []);
      setPlots(plotsRes.plots || []);
    } catch (e) {
      console.error('Failed to load sidebar data:', e);
    }
  };

  const handleReindex = async () => {
    setIsLoading(true);
    try {
      await onReindex();
      await loadData();
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setIsLoading(true);
      try {
        await onUpload(file);
        await loadData();
      } finally {
        setIsLoading(false);
      }
    }
  };

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <div className={`fixed top-0 right-0 h-full w-80 bg-slate-800 border-l border-slate-700 z-50 transform transition-transform duration-300 ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-slate-700">
            <h2 className="font-semibold">Data Management</h2>
            <button onClick={onClose} className="p-1 hover:bg-slate-700 rounded">
              <X size={20} />
            </button>
          </div>
          
          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* System Status */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">System Status</h3>
              <div className="bg-slate-900 rounded-lg p-3 space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Status</span>
                  <StatusBadge status={status?.status} />
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Tables</span>
                  <span>{status?.tables_loaded || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Tools</span>
                  <span>{status?.tools_available || 0}</span>
                </div>
              </div>
              {status?.missing_files?.length > 0 && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                  <div className="flex gap-2 text-amber-400 text-sm">
                    <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium">Missing files:</p>
                      <ul className="mt-1 text-amber-300/80">
                        {status.missing_files.map(f => (
                          <li key={f}>• {f}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Actions */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Actions</h3>
              <div className="space-y-2">
                <button
                  onClick={handleReindex}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-700 rounded-lg transition-colors"
                >
                  {isLoading ? <Loader2 size={16} className="animate-spin" /> : <RefreshCw size={16} />}
                  Reindex Data
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 rounded-lg transition-colors"
                >
                  <Upload size={16} />
                  Upload CSV
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
                <button
                  onClick={onClearPlots}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 rounded-lg transition-colors"
                >
                  <Trash2 size={16} />
                  Clear Plots
                </button>
              </div>
            </div>
            
            {/* Tables */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Loaded Tables</h3>
              {tables.length === 0 ? (
                <p className="text-slate-500 text-sm">No tables loaded</p>
              ) : (
                <div className="space-y-2">
                  {tables.map(table => (
                    <div key={table.name} className="bg-slate-900 rounded-lg p-3">
                      <div className="flex items-center gap-2 text-primary-400">
                        <Table size={14} />
                        <span className="font-medium text-sm">{table.name}</span>
                      </div>
                      <p className="text-xs text-slate-500 mt-1">
                        {table.rows.toLocaleString()} rows • {table.columns.length} columns
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Recent Plots */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-slate-400 uppercase tracking-wider">Recent Plots</h3>
              {plots.length === 0 ? (
                <p className="text-slate-500 text-sm">No plots generated</p>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  {plots.slice(0, 6).map(plot => (
                    <a
                      key={plot.filename}
                      href={`${API_BASE}${plot.url}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block"
                    >
                      <img 
                        src={`${API_BASE}${plot.url}`}
                        alt={plot.filename}
                        className="rounded-lg border border-slate-700 hover:border-primary-500 transition-colors"
                      />
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

// ============================================================
// Main App Component
// ============================================================

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [status, setStatus] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notification, setNotification] = useState(null);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Load status on mount
  useEffect(() => {
    loadStatus();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input after loading
  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus();
    }
  }, [isLoading]);

  const loadStatus = async () => {
    try {
      const data = await api.getStatus();
      setStatus(data);
    } catch (e) {
      console.error('Failed to load status:', e);
    }
  };

  const showNotification = (message, type = 'info') => {
    setNotification({ message, type });
    setTimeout(() => setNotification(null), 3000);
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await api.chat(userMessage.content, sessionId);
      
      if (!sessionId && response.session_id) {
        setSessionId(response.session_id);
      }

      const assistantMessage = {
        role: 'assistant',
        content: response.response,
        images: response.images,
        elapsed: response.elapsed_time,
        iterations: response.iterations,
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (e) {
      console.error('Chat error:', e);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${e.message}. Please try again.`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleQuickAction = (prompt) => {
    setInput(prompt);
    inputRef.current?.focus();
  };

  const handleReindex = async () => {
    try {
      await api.reindex();
      await loadStatus();
      showNotification('Data reindexed successfully', 'success');
    } catch (e) {
      showNotification('Failed to reindex data', 'error');
    }
  };

  const handleUpload = async (file) => {
    try {
      await api.uploadFile(file);
      await loadStatus();
      showNotification(`Uploaded ${file.name} successfully`, 'success');
    } catch (e) {
      showNotification('Failed to upload file', 'error');
    }
  };

  const handleClearPlots = async () => {
    try {
      await api.clearPlots();
      showNotification('Plots cleared', 'success');
    } catch (e) {
      showNotification('Failed to clear plots', 'error');
    }
  };

  const handleClearChat = async () => {
    if (sessionId) {
      await api.clearSession(sessionId);
    }
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className="h-screen flex flex-col bg-slate-900">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center">
              <FlaskConical size={22} />
            </div>
            <div>
              <h1 className="font-semibold text-lg">Polymer Solubility Analyzer</h1>
              <p className="text-xs text-slate-500">AI-powered solvent selection & separation analysis</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status={status?.status} />
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
            >
              <Menu size={20} />
            </button>
          </div>
        </div>
      </header>

      {/* Notification */}
      {notification && (
        <div className={`fixed top-4 right-4 z-50 px-4 py-2 rounded-lg shadow-lg ${
          notification.type === 'success' ? 'bg-emerald-600' : 
          notification.type === 'error' ? 'bg-red-600' : 'bg-slate-700'
        }`}>
          {notification.message}
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 overflow-hidden flex flex-col max-w-5xl mx-auto w-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center mb-4">
                <FlaskConical size={32} />
              </div>
              <h2 className="text-2xl font-semibold mb-2">Welcome to Polymer Solubility Analyzer</h2>
              <p className="text-slate-400 max-w-md mb-8">
                AI-powered analysis for polymer-solvent systems. Ask questions about solubility, 
                separation strategies, and solvent properties.
              </p>
              
              {/* Quick Actions */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
                <QuickAction 
                  icon={Database} 
                  label="What tables are available?"
                  onClick={() => handleQuickAction("What tables are available?")}
                />
                <QuickAction 
                  icon={Beaker} 
                  label="List available solvents"
                  onClick={() => handleQuickAction("List the available polymers in the database")}
                />
                <QuickAction 
                  icon={Thermometer} 
                  label="Separation analysis"
                  onClick={() => handleQuickAction("Find solvents to separate LDPE from PET at 25°C")}
                />
                <QuickAction 
                  icon={DollarSign} 
                  label="Rank by cost"
                  onClick={() => handleQuickAction("Rank solvents by energy cost (cheapest first)")}
                />
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <Message key={i} message={msg} isUser={msg.role === 'user'} />
              ))}
              {isLoading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="flex-shrink-0 border-t border-slate-800 bg-slate-900/80 backdrop-blur-sm p-4">
          <div className="flex gap-3 items-end">
            <button
              onClick={handleClearChat}
              className="p-2.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
              title="Clear chat"
            >
              <Trash2 size={20} />
            </button>
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about polymer solubility, separation strategies, solvent properties..."
                rows={1}
                className="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 pr-12 resize-none focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent placeholder-slate-500"
                style={{ minHeight: '48px', maxHeight: '200px' }}
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="absolute right-2 bottom-2 p-2 bg-primary-600 hover:bg-primary-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg transition-colors"
              >
                {isLoading ? (
                  <Loader2 size={18} className="animate-spin" />
                ) : (
                  <Send size={18} />
                )}
              </button>
            </div>
          </div>
          <p className="text-xs text-slate-600 mt-2 text-center">
            Press Enter to send • Shift+Enter for new line
          </p>
        </div>
      </main>

      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        status={status}
        onReindex={handleReindex}
        onUpload={handleUpload}
        onClearPlots={handleClearPlots}
      />
    </div>
  );
}

export default App;
