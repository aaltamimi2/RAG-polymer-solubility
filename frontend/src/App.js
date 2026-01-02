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
  AlertTriangle,
  Download,
  Moon,
  Sun,
  Layers,
  Shield,
  Activity,
  Droplet,
  Brain
} from 'lucide-react';
import api, { API_BASE } from './api';

// ============================================================
// Utility Functions
// ============================================================

// Extract export ID from message content
function extractExportId(content) {
  const match = content.match(/\/api\/export\/([a-f0-9]{8})/);
  return match ? match[1] : null;
}

// Download CSV file
async function downloadCSV(exportId, showNotification) {
  try {
    const blob = await api.downloadExport(exportId);
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `export_${exportId}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);

    if (showNotification) {
      showNotification('CSV downloaded successfully', 'success');
    }
  } catch (error) {
    console.error('CSV download error:', error);
    if (showNotification) {
      showNotification(error.message || 'Failed to download CSV', 'error');
    }
  }
}

// ============================================================
// Components
// ============================================================

// Status Badge Component
function StatusBadge({ status }) {
  const isReady = status === 'ready';
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-headline" style={{
      backgroundColor: isReady ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
      color: isReady ? 'var(--success)' : 'var(--warning)'
    }}>
      {isReady ? <CheckCircle size={14} /> : <AlertCircle size={14} />}
      {isReady ? 'Ready' : 'Limited'}
    </div>
  );
}

// Message Component
function Message({ message, isUser, onDownloadCSV }) {
  const exportId = !isUser ? extractExportId(message.content) : null;

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 font-headline" style={{
        backgroundColor: isUser ? 'var(--primary)' : 'var(--bg-tertiary)',
        color: isUser ? 'white' : 'var(--text-primary)'
      }}>
        {isUser ? (
          <span className="text-sm font-medium">You</span>
        ) : (
          <FlaskConical size={16} />
        )}
      </div>
      <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : ''}`}>
        <div className="inline-block rounded-2xl px-4 py-3 font-body" style={{
          backgroundColor: isUser ? 'var(--primary)' : 'var(--bg-secondary)',
          color: isUser ? 'white' : 'var(--text-primary)',
          border: isUser ? 'none' : '1px solid var(--border-color)'
        }}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="markdown-content">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          )}
        </div>
        {exportId && (
          <button
            onClick={() => onDownloadCSV(exportId)}
            className="mt-2 flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors font-headline"
            style={{
              backgroundColor: 'var(--success)',
              color: 'white'
            }}
            aria-label={`Download CSV export ${exportId}`}
          >
            <Download size={16} aria-hidden="true" />
            Download CSV
          </button>
        )}
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
                  className="rounded-lg max-w-xs border transition-colors"
                  style={{ borderColor: 'var(--border-color)' }}
                />
              </a>
            ))}
          </div>
        )}
        {message.elapsed && (
          <p className="text-xs mt-1 font-mono" style={{ color: 'var(--text-tertiary)' }}>
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
      <div className="w-8 h-8 rounded-full flex items-center justify-center" style={{
        backgroundColor: 'var(--bg-tertiary)',
        color: 'var(--text-primary)'
      }}>
        <FlaskConical size={16} />
      </div>
      <div className="rounded-2xl px-4 py-3" style={{
        backgroundColor: 'var(--bg-secondary)',
        border: '1px solid var(--border-color)'
      }}>
        <div className="typing-indicator flex gap-1">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--text-tertiary)' }}></span>
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--text-tertiary)' }}></span>
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--text-tertiary)' }}></span>
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
      className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors font-headline"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        color: 'var(--text-secondary)',
        border: '1px solid var(--border-color)'
      }}
      onMouseOver={(e) => {
        e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
        e.currentTarget.style.color = 'var(--text-primary)';
      }}
      onMouseOut={(e) => {
        e.currentTarget.style.backgroundColor = 'var(--bg-secondary)';
        e.currentTarget.style.color = 'var(--text-secondary)';
      }}
      aria-label={`Quick action: ${label}`}
    >
      <Icon size={16} aria-hidden="true" />
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
          className="fixed inset-0 z-40 lg:hidden"
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed top-0 right-0 h-full w-80 z-50 transform transition-transform duration-300 ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`} style={{
        backgroundColor: 'var(--bg-secondary)',
        borderLeft: '1px solid var(--border-color)'
      }}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 font-headline" style={{
            borderBottom: '1px solid var(--border-color)'
          }}>
            <h2 className="font-semibold" style={{ color: 'var(--text-primary)' }}>Data Management</h2>
            <button
              onClick={onClose}
              className="p-1 rounded transition-colors"
              style={{ color: 'var(--text-primary)' }}
              onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
              onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              aria-label="Close sidebar"
            >
              <X size={20} aria-hidden="true" />
            </button>
          </div>
          
          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6">
            {/* System Status */}
            <div className="space-y-3">
              <h3 className="text-sm font-medium uppercase tracking-wider font-headline" style={{ color: 'var(--text-tertiary)' }}>
                System Status
              </h3>
              <div className="rounded-lg p-3 space-y-2 font-body" style={{
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border-color)'
              }}>
                <div className="flex justify-between items-center">
                  <span style={{ color: 'var(--text-secondary)' }}>Status</span>
                  <StatusBadge status={status?.status} />
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--text-secondary)' }}>Tables</span>
                  <span style={{ color: 'var(--text-primary)' }}>{status?.tables_loaded || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span style={{ color: 'var(--text-secondary)' }}>Tools</span>
                  <span style={{ color: 'var(--text-primary)' }}>{status?.tools_available || 0}</span>
                </div>
              </div>
              {status?.missing_files?.length > 0 && (
                <div className="rounded-lg p-3" style={{
                  backgroundColor: 'rgba(245, 158, 11, 0.1)',
                  border: '1px solid rgba(245, 158, 11, 0.3)'
                }}>
                  <div className="flex gap-2 text-sm" style={{ color: 'var(--warning)' }}>
                    <AlertTriangle size={16} className="flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium font-headline">Missing files:</p>
                      <ul className="mt-1 font-body">
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
              <h3 className="text-sm font-medium uppercase tracking-wider font-headline" style={{ color: 'var(--text-tertiary)' }}>Actions</h3>
              <div className="space-y-2">
                <button
                  onClick={handleReindex}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg transition-colors font-headline"
                  style={{
                    backgroundColor: isLoading ? 'var(--bg-tertiary)' : 'var(--primary)',
                    color: 'white',
                    opacity: isLoading ? 0.6 : 1,
                    cursor: isLoading ? 'not-allowed' : 'pointer'
                  }}
                  aria-label="Reindex database from CSV files"
                >
                  {isLoading ? <Loader2 size={16} className="animate-spin" aria-hidden="true" /> : <RefreshCw size={16} aria-hidden="true" />}
                  Reindex Data
                </button>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg transition-colors font-headline"
                  style={{
                    backgroundColor: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)',
                    opacity: isLoading ? 0.6 : 1,
                    cursor: isLoading ? 'not-allowed' : 'pointer'
                  }}
                  aria-label="Upload new CSV file"
                >
                  <Upload size={16} aria-hidden="true" />
                  Upload CSV
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  aria-label="CSV file upload input"
                />
                <button
                  onClick={onClearPlots}
                  disabled={isLoading}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg transition-colors font-headline"
                  style={{
                    backgroundColor: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)',
                    opacity: isLoading ? 0.6 : 1,
                    cursor: isLoading ? 'not-allowed' : 'pointer'
                  }}
                  aria-label="Clear all generated plots"
                >
                  <Trash2 size={16} aria-hidden="true" />
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
  const [theme, setTheme] = useState('light'); // Light mode as default
  const [selectedModel, setSelectedModel] = useState('gemini-2.5-flash-lite'); // Gemini model selection
  const [mlPolymerTypes, setMlPolymerTypes] = useState(null); // ML polymer types data
  const [showMlTypes, setShowMlTypes] = useState(false); // Show ML types view

  // ML Workflow state
  const [mlStep, setMlStep] = useState('types'); // 'types' | 'polymers' | 'solvents' | 'results'
  const [selectedType, setSelectedType] = useState(null);
  const [polymersInType, setPolymersInType] = useState(null);
  const [selectedPolymers, setSelectedPolymers] = useState([]);
  const [solventInput, setSolventInput] = useState('');

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Initialize theme from localStorage or default to light
  useEffect(() => {
    const savedTheme = localStorage.getItem('dissolve-theme') || 'light';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  // Initialize model from localStorage or default to flash-lite
  useEffect(() => {
    const savedModel = localStorage.getItem('dissolve-model') || 'gemini-2.5-flash-lite';
    setSelectedModel(savedModel);
  }, []);

  // Save model to localStorage when changed
  const handleModelChange = (model) => {
    setSelectedModel(model);
    localStorage.setItem('dissolve-model', model);
    showNotification(`Switched to ${model}`, 'info');
  };

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

  const loadMlPolymerTypes = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/ml/polymer-types`);
      const data = await response.json();
      setMlPolymerTypes(data);
      setShowMlTypes(true);
      setMlStep('types');
    } catch (e) {
      console.error('Failed to load ML polymer types:', e);
      showNotification('Failed to load polymer types', 'error');
    }
  };

  const selectPolymerType = async (polymerType) => {
    try {
      const response = await fetch(`${API_BASE}/api/ml/polymers-by-type/${encodeURIComponent(polymerType)}`);
      const data = await response.json();
      setSelectedType(polymerType);
      setPolymersInType(data);
      setSelectedPolymers([]);
      setMlStep('polymers');
    } catch (e) {
      console.error('Failed to load polymers:', e);
      showNotification('Failed to load polymers', 'error');
    }
  };

  const togglePolymerSelection = (polymer) => {
    setSelectedPolymers(prev => {
      const exists = prev.find(p => p.polymer === polymer.polymer);
      if (exists) {
        return prev.filter(p => p.polymer !== polymer.polymer);
      } else {
        return [...prev, polymer];
      }
    });
  };

  const selectAllPolymers = () => {
    if (polymersInType && polymersInType.polymers) {
      setSelectedPolymers(polymersInType.polymers);
    }
  };

  const proceedToSolventSelection = () => {
    if (selectedPolymers.length === 0) {
      showNotification('Please select at least one polymer', 'error');
      return;
    }
    setMlStep('solvents');
  };

  const runMlPrediction = async () => {
    if (!solventInput.trim()) {
      showNotification('Please enter at least one solvent', 'error');
      return;
    }

    // Close ML tool and return to chat
    setShowMlTypes(false);
    setMlStep('types');

    // Create query for ML prediction
    const solvents = solventInput.split(',').map(s => s.trim()).filter(s => s);

    if (selectedPolymers.length === 1) {
      // Single polymer prediction
      const query = `Predict solubility of ${selectedPolymers[0].polymer} in ${solvents.join(', ')} using machine learning with Hansen parameters`;
      handleQuickAction(query);
    } else {
      // Multiple polymers
      const polymerNames = selectedPolymers.map(p => p.polymer).join(', ');
      const query = `Predict solubility for these polymers: ${polymerNames} in ${solvents.join(', ')} using machine learning with Hansen parameters`;
      handleQuickAction(query);
    }

    // Reset state
    setSelectedPolymers([]);
    setSolventInput('');
    setPolymersInType(null);
    setSelectedType(null);
  };

  const backToMlTypes = () => {
    setMlStep('types');
    setSelectedPolymers([]);
    setPolymersInType(null);
    setSelectedType(null);
  };

  const backToPolymerSelection = () => {
    setMlStep('polymers');
    setSolventInput('');
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
      const response = await api.chat(userMessage.content, sessionId, selectedModel);
      
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

  const handleDownloadCSV = (exportId) => {
    downloadCSV(exportId, showNotification);
  };

  const handleExportConversation = () => {
    if (messages.length === 0) {
      showNotification('No conversation to export', 'error');
      return;
    }

    try {
      const conversationData = {
        session_id: sessionId,
        exported_at: new Date().toISOString(),
        message_count: messages.length,
        messages: messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp || new Date().toISOString(),
          elapsed_time: msg.elapsed,
          iterations: msg.iterations,
          images: msg.images || [],
        })),
      };

      const blob = new Blob([JSON.stringify(conversationData, null, 2)], {
        type: 'application/json',
      });

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `dissolve_conversation_${sessionId || 'unknown'}_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      showNotification('Conversation exported successfully', 'success');
    } catch (error) {
      console.error('Export error:', error);
      showNotification('Failed to export conversation', 'error');
    }
  };

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('dissolve-theme', newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  return (
    <div className="h-screen flex flex-col" style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* Header */}
      <header className="flex-shrink-0 backdrop-blur-sm" style={{
        borderBottom: '1px solid var(--border-color)',
        backgroundColor: 'var(--bg-secondary)'
      }}>
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{
              background: 'linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%)'
            }}>
              <FlaskConical size={22} style={{ color: 'white' }} />
            </div>
            <div>
              <h1 className="font-semibold text-lg font-headline" style={{ color: 'var(--text-primary)' }}>
                DISSOLVE Agent
              </h1>
              <p className="text-xs font-body" style={{ color: 'var(--text-secondary)' }}>
                AI-powered solvent selection & separation analysis
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status={status?.status} />
            {/* Model Selector */}
            <select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              className="px-3 py-1.5 text-sm rounded-lg transition-colors font-body cursor-pointer"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)'
              }}
              title="Select Gemini model"
            >
              <option value="gemini-2.5-flash-lite">Flash Lite (Cheapest)</option>
              <option value="gemini-2.5-flash">Flash</option>
              <option value="gemini-2.5-pro">Pro (Most Capable)</option>
            </select>
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg transition-colors"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                color: 'var(--text-primary)'
              }}
              aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
              title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            >
              {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
            </button>
            {messages.length > 0 && (
              <button
                onClick={handleExportConversation}
                className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg transition-colors font-headline"
                style={{
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)'
                }}
                title="Export conversation as JSON"
              >
                <Download size={16} />
                <span className="hidden sm:inline">Export</span>
              </button>
            )}
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 rounded-lg transition-colors"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                color: 'var(--text-primary)'
              }}
              aria-label="Open data management sidebar"
            >
              <Menu size={20} aria-hidden="true" />
            </button>
          </div>
        </div>
      </header>

      {/* Notification */}
      {notification && (
        <div className="fixed top-4 right-4 z-50 px-4 py-2 rounded-lg font-headline" style={{
          backgroundColor: notification.type === 'success' ? 'var(--success)' :
                          notification.type === 'error' ? 'var(--error)' : 'var(--bg-tertiary)',
          color: 'white',
          boxShadow: 'var(--shadow-lg)'
        }}>
          {notification.message}
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 overflow-hidden flex flex-col max-w-5xl mx-auto w-full">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
              {showMlTypes && mlPolymerTypes ? (
                // ML Polymer Types View
                <div className="w-full max-w-6xl">
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-2xl font-semibold font-headline" style={{ color: 'var(--text-primary)' }}>
                        Select Polymer Type
                      </h2>
                      <p className="text-sm font-body mt-1" style={{ color: 'var(--text-secondary)' }}>
                        {mlPolymerTypes.total_types} types • {mlPolymerTypes.total_polymers} polymers available
                      </p>
                    </div>
                    <button
                      onClick={() => setShowMlTypes(false)}
                      className="px-4 py-2 rounded-lg font-medium transition-colors"
                      style={{
                        backgroundColor: 'var(--bg-secondary)',
                        color: 'var(--text-secondary)'
                      }}
                    >
                      Back
                    </button>
                  </div>
                  {mlStep === 'types' && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-h-96 overflow-y-auto">
                      {mlPolymerTypes.polymer_types.map((polymerType, idx) => (
                        <button
                          key={idx}
                          onClick={() => selectPolymerType(polymerType.type)}
                          className="p-4 rounded-lg text-left transition-all hover:shadow-md"
                          style={{
                            backgroundColor: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)'
                          }}
                          onMouseOver={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
                            e.currentTarget.style.borderColor = 'var(--primary)';
                          }}
                          onMouseOut={(e) => {
                            e.currentTarget.style.backgroundColor = 'var(--bg-secondary)';
                            e.currentTarget.style.borderColor = 'var(--border-color)';
                          }}
                        >
                          <div className="font-semibold text-sm mb-1 font-headline" style={{ color: 'var(--text-primary)' }}>
                            {polymerType.type}
                          </div>
                          <div className="text-xs font-body" style={{ color: 'var(--text-secondary)' }}>
                            {polymerType.count} {polymerType.count === 1 ? 'polymer' : 'polymers'}
                          </div>
                        </button>
                      ))}
                    </div>
                  )}

                  {mlStep === 'polymers' && polymersInType && (
                    <div className="w-full max-w-4xl">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="text-xl font-semibold font-headline" style={{ color: 'var(--text-primary)' }}>
                            Select Polymer(s) - {selectedType}
                          </h3>
                          <p className="text-sm font-body mt-1" style={{ color: 'var(--text-secondary)' }}>
                            {polymersInType.count} {polymersInType.count === 1 ? 'polymer' : 'polymers'} • {selectedPolymers.length} selected
                          </p>
                        </div>
                        <button
                          onClick={backToMlTypes}
                          className="px-4 py-2 rounded-lg font-medium transition-colors"
                          style={{
                            backgroundColor: 'var(--bg-secondary)',
                            color: 'var(--text-secondary)'
                          }}
                        >
                          Back
                        </button>
                      </div>

                      <div className="mb-4">
                        <button
                          onClick={selectAllPolymers}
                          className="px-4 py-2 rounded-lg font-medium transition-colors mr-2"
                          style={{
                            backgroundColor: 'var(--primary)',
                            color: 'white'
                          }}
                        >
                          Select All
                        </button>
                        <button
                          onClick={() => setSelectedPolymers([])}
                          className="px-4 py-2 rounded-lg font-medium transition-colors"
                          style={{
                            backgroundColor: 'var(--bg-secondary)',
                            color: 'var(--text-secondary)'
                          }}
                        >
                          Clear Selection
                        </button>
                      </div>

                      <div className="max-h-80 overflow-y-auto mb-4 space-y-2">
                        {polymersInType.polymers.map((polymer, idx) => {
                          const isSelected = selectedPolymers.find(p => p.polymer === polymer.polymer);
                          return (
                            <div
                              key={idx}
                              onClick={() => togglePolymerSelection(polymer)}
                              className="p-3 rounded-lg cursor-pointer transition-all"
                              style={{
                                backgroundColor: isSelected ? 'var(--primary-light)' : 'var(--bg-secondary)',
                                border: `2px solid ${isSelected ? 'var(--primary)' : 'var(--border-color)'}`,
                              }}
                            >
                              <div className="flex items-center">
                                <input
                                  type="checkbox"
                                  checked={!!isSelected}
                                  onChange={() => {}}
                                  className="mr-3"
                                  style={{ accentColor: 'var(--primary)' }}
                                />
                                <div className="flex-1">
                                  <div className="font-semibold text-sm font-headline" style={{ color: 'var(--text-primary)' }}>
                                    {polymer.polymer}
                                  </div>
                                  <div className="text-xs font-body mt-1" style={{ color: 'var(--text-secondary)' }}>
                                    δD: {polymer.dispersion.toFixed(1)} • δP: {polymer.polar.toFixed(1)} • δH: {polymer.hydrogen_bonding.toFixed(1)} • R₀: {polymer.interaction_radius.toFixed(1)}
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      <button
                        onClick={proceedToSolventSelection}
                        disabled={selectedPolymers.length === 0}
                        className="w-full px-6 py-3 rounded-lg font-medium transition-colors"
                        style={{
                          backgroundColor: selectedPolymers.length > 0 ? 'var(--primary)' : 'var(--bg-tertiary)',
                          color: selectedPolymers.length > 0 ? 'white' : 'var(--text-tertiary)',
                          cursor: selectedPolymers.length > 0 ? 'pointer' : 'not-allowed'
                        }}
                      >
                        Next: Select Solvents →
                      </button>
                    </div>
                  )}

                  {mlStep === 'solvents' && (
                    <div className="w-full max-w-2xl">
                      <div className="flex items-center justify-between mb-4">
                        <div>
                          <h3 className="text-xl font-semibold font-headline" style={{ color: 'var(--text-primary)' }}>
                            Enter Solvent(s)
                          </h3>
                          <p className="text-sm font-body mt-1" style={{ color: 'var(--text-secondary)' }}>
                            {selectedPolymers.length} {selectedPolymers.length === 1 ? 'polymer' : 'polymers'} selected
                          </p>
                        </div>
                        <button
                          onClick={backToPolymerSelection}
                          className="px-4 py-2 rounded-lg font-medium transition-colors"
                          style={{
                            backgroundColor: 'var(--bg-secondary)',
                            color: 'var(--text-secondary)'
                          }}
                        >
                          Back
                        </button>
                      </div>

                      <div className="mb-4 p-4 rounded-lg" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                        <p className="text-sm font-body mb-2" style={{ color: 'var(--text-primary)' }}>
                          Selected Polymers:
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {selectedPolymers.map((polymer, idx) => (
                            <span
                              key={idx}
                              className="px-3 py-1 rounded-full text-xs font-medium"
                              style={{
                                backgroundColor: 'var(--primary)',
                                color: 'white'
                              }}
                            >
                              {polymer.polymer}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="mb-4">
                        <label className="block text-sm font-medium font-body mb-2" style={{ color: 'var(--text-primary)' }}>
                          Solvent Name(s)
                        </label>
                        <input
                          type="text"
                          value={solventInput}
                          onChange={(e) => setSolventInput(e.target.value)}
                          placeholder="e.g., Toluene, Acetone, Water (comma-separated for multiple)"
                          className="w-full px-4 py-3 rounded-lg font-body"
                          style={{
                            backgroundColor: 'var(--bg-secondary)',
                            border: '1px solid var(--border-color)',
                            color: 'var(--text-primary)'
                          }}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                              runMlPrediction();
                            }
                          }}
                        />
                        <p className="text-xs font-body mt-1" style={{ color: 'var(--text-secondary)' }}>
                          Enter one or more solvent names separated by commas
                        </p>
                      </div>

                      <button
                        onClick={runMlPrediction}
                        disabled={!solventInput.trim()}
                        className="w-full px-6 py-3 rounded-lg font-medium transition-colors"
                        style={{
                          backgroundColor: solventInput.trim() ? 'var(--primary)' : 'var(--bg-tertiary)',
                          color: solventInput.trim() ? 'white' : 'var(--text-tertiary)',
                          cursor: solventInput.trim() ? 'pointer' : 'not-allowed'
                        }}
                      >
                        Run ML Prediction
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                // Welcome Screen
                <>
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4" style={{
                    background: 'linear-gradient(135deg, var(--primary) 0%, var(--primary-hover) 100%)'
                  }}>
                    <FlaskConical size={32} style={{ color: 'white' }} />
                  </div>
                  <h2 className="text-2xl font-semibold mb-2 font-headline" style={{ color: 'var(--text-primary)' }}>
                    Welcome to DISSOLVE Agent
                  </h2>
                  <p className="max-w-md mb-8 font-body" style={{ color: 'var(--text-secondary)' }}>
                    AI-powered analysis for polymer-solvent systems. Ask questions about solubility,
                    separation strategies, and solvent properties.
                  </p>

                  {/* Quick Actions */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 w-full max-w-4xl">
                <QuickAction
                  icon={Beaker}
                  label="List Polymers"
                  onClick={() => handleQuickAction("List all available polymers in the database")}
                />
                <QuickAction
                  icon={Droplet}
                  label="List Solvents"
                  onClick={() => handleQuickAction("List all available solvents in the database")}
                />
                <QuickAction
                  icon={Layers}
                  label="Three-Layer Film"
                  onClick={() => handleQuickAction("Analyze separation for a three-layer film: PVDF/LDPE/PET at 25°C")}
                />
                <QuickAction
                  icon={Shield}
                  label="Safety Ranking"
                  onClick={() => handleQuickAction("Rank common solvents by safety (G-score and LogP) for PVDF")}
                />
                <QuickAction
                  icon={DollarSign}
                  label="Cost Ranking"
                  onClick={() => handleQuickAction("Rank solvents by energy cost (cheapest first)")}
                />
                <QuickAction
                  icon={Thermometer}
                  label="Boiling Point"
                  onClick={() => handleQuickAction("Show common solvents ranked by boiling point")}
                />
                <QuickAction
                  icon={Activity}
                  label="Integrated Analysis"
                  onClick={() => handleQuickAction("Perform integrated analysis across selectivity, safety, cost, and boiling point for PVDF separation")}
                />
                <QuickAction
                  icon={Brain}
                  label="ML Prediction"
                  onClick={loadMlPolymerTypes}
                />
              </div>
                </>
              )}
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <Message key={i} message={msg} isUser={msg.role === 'user'} onDownloadCSV={handleDownloadCSV} />
              ))}
              {isLoading && <TypingIndicator />}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="flex-shrink-0 backdrop-blur-sm p-4" style={{
          borderTop: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-secondary)'
        }}>
          <div className="flex gap-3 items-end">
            <button
              onClick={handleClearChat}
              className="p-2.5 rounded-lg transition-colors"
              style={{
                color: 'var(--text-secondary)',
                backgroundColor: 'transparent'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.color = 'var(--text-primary)';
                e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.color = 'var(--text-secondary)';
                e.currentTarget.style.backgroundColor = 'transparent';
              }}
              title="Clear chat"
              aria-label="Clear chat history"
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
                className="w-full rounded-xl px-4 py-3 pr-12 resize-none focus:outline-none font-body"
                style={{
                  minHeight: '48px',
                  maxHeight: '200px',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)'
                }}
                disabled={isLoading}
                aria-label="Chat message input"
                aria-describedby="input-help-text"
              />
              <span id="input-help-text" className="sr-only">
                Press Enter to send message, Shift+Enter to add a new line
              </span>
              <button
                onClick={handleSend}
                disabled={!input.trim() || isLoading}
                className="absolute right-2 bottom-2 p-2 rounded-lg transition-colors"
                style={{
                  backgroundColor: (!input.trim() || isLoading) ? 'var(--bg-tertiary)' : 'var(--primary)',
                  color: (!input.trim() || isLoading) ? 'var(--text-tertiary)' : 'white',
                  cursor: (!input.trim() || isLoading) ? 'not-allowed' : 'pointer'
                }}
                aria-label={isLoading ? 'Sending message...' : 'Send message'}
              >
                {isLoading ? (
                  <Loader2 size={18} className="animate-spin" aria-hidden="true" />
                ) : (
                  <Send size={18} aria-hidden="true" />
                )}
              </button>
            </div>
          </div>
          <p className="text-xs mt-2 text-center font-mono" style={{ color: 'var(--text-tertiary)' }} aria-hidden="true">
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
