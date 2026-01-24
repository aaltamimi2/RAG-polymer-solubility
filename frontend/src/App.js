import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Send,
  Upload,
  RefreshCw,
  Trash2,
  AlertCircle,
  CheckCircle,
  Loader2,
  Menu,
  X,
  FlaskConical,
  Table,
  Beaker,
  AlertTriangle,
  Download,
  Moon,
  Sun,
  Layers,
  Brain,
  Search,
  Calculator,
  Rocket,
  BookOpen,
  Library,
  FileText
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
function Message({ message, isUser, onDownloadCSV, onReportIssue }) {
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
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  a: ({ node, children, ...props }) => (
                    <a
                      {...props}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: 'var(--primary)', textDecoration: 'underline' }}
                    >{children}</a>
                  ),
                  table: ({ node, ...props }) => (
                    <div style={{ overflowX: 'auto', margin: '1rem 0' }}>
                      <table {...props} />
                    </div>
                  )
                }}
              >
                {message.content}
              </ReactMarkdown>
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
          <div className="flex items-center gap-2 mt-1">
            <p className="text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
              {message.elapsed.toFixed(1)}s • {message.iterations} iterations
            </p>
            {!isUser && onReportIssue && (
              <button
                onClick={() => onReportIssue(message)}
                className="flex items-center gap-1 text-xs transition-colors"
                style={{ color: 'var(--text-tertiary)' }}
                onMouseOver={(e) => e.currentTarget.style.color = 'var(--warning)'}
                onMouseOut={(e) => e.currentTarget.style.color = 'var(--text-tertiary)'}
                title="Report an issue with this response"
              >
                <AlertTriangle size={12} />
                Report
              </button>
            )}
          </div>
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

// Issue Report Modal
function IssueReportModal({ isOpen, onClose, message, userQuestion, onSubmit }) {
  const [description, setDescription] = useState('');
  const [issueType, setIssueType] = useState('incorrect_response');
  const [severity, setSeverity] = useState('medium');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [result, setResult] = useState(null);

  const issueTypes = [
    { value: 'incorrect_response', label: 'Incorrect Response' },
    { value: 'ui_bug', label: 'UI Bug' },
    { value: 'api_error', label: 'API Error' },
    { value: 'performance', label: 'Performance Issue' },
    { value: 'data_issue', label: 'Data Issue' },
    { value: 'other', label: 'Other' },
  ];

  const severities = [
    { value: 'low', label: 'Low' },
    { value: 'medium', label: 'Medium' },
    { value: 'high', label: 'High' },
    { value: 'critical', label: 'Critical' },
  ];

  const handleSubmit = async () => {
    if (!description.trim()) return;

    setIsSubmitting(true);
    setResult(null);

    try {
      const report = {
        user_question: userQuestion || '',
        assistant_response: message?.content || '',
        elapsed_time: message?.elapsed || 0,
        iterations: message?.iterations || 0,
        images: (message?.images || []).map(img => ({
          filename: img,
          base64: ''
        })),
        user_description: description,
        issue_type: issueType,
        severity: severity,
      };

      const response = await onSubmit(report);
      setResult(response);
    } catch (error) {
      setResult({ success: false, error: error.message });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    setDescription('');
    setIssueType('incorrect_response');
    setSeverity('medium');
    setResult(null);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 p-4" style={{ backgroundColor: 'rgba(0, 0, 0, 0.6)' }}>
      <div className="rounded-xl max-w-lg w-full max-h-[90vh] overflow-y-auto" style={{ backgroundColor: 'var(--bg-secondary)' }}>
        <div className="flex items-center justify-between p-4" style={{ borderBottom: '1px solid var(--border-color)' }}>
          <h2 className="font-semibold text-lg flex items-center gap-2 font-headline" style={{ color: 'var(--text-primary)' }}>
            <AlertTriangle size={20} style={{ color: 'var(--warning)' }} />
            Report Issue
          </h2>
          <button
            onClick={handleClose}
            className="p-1 rounded transition-colors"
            style={{ color: 'var(--text-primary)' }}
            onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
            onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
          >
            <X size={20} />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {!result ? (
            <>
              <div>
                <label className="block text-sm font-medium mb-2 font-body" style={{ color: 'var(--text-secondary)' }}>
                  Issue Type
                </label>
                <select
                  value={issueType}
                  onChange={(e) => setIssueType(e.target.value)}
                  className="w-full rounded-lg px-3 py-2 font-body focus:outline-none"
                  style={{
                    backgroundColor: 'var(--bg-primary)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-primary)'
                  }}
                >
                  {issueTypes.map(type => (
                    <option key={type.value} value={type.value}>{type.label}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 font-body" style={{ color: 'var(--text-secondary)' }}>
                  Severity
                </label>
                <div className="flex gap-2">
                  {severities.map(sev => (
                    <button
                      key={sev.value}
                      onClick={() => setSeverity(sev.value)}
                      className="px-3 py-1.5 rounded-lg text-sm transition-colors font-headline"
                      style={{
                        backgroundColor: severity === sev.value ? 'var(--primary)' : 'var(--bg-tertiary)',
                        color: severity === sev.value ? 'white' : 'var(--text-secondary)'
                      }}
                    >
                      {sev.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 font-body" style={{ color: 'var(--text-secondary)' }}>
                  Describe the Issue *
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What went wrong? What did you expect to happen?"
                  rows={4}
                  className="w-full rounded-lg px-3 py-2 font-body resize-none focus:outline-none"
                  style={{
                    backgroundColor: 'var(--bg-primary)',
                    border: '1px solid var(--border-color)',
                    color: 'var(--text-primary)'
                  }}
                />
              </div>

              <div className="rounded-lg p-3 text-sm font-body" style={{ backgroundColor: 'var(--bg-primary)' }}>
                <p className="font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>What happens when you submit:</p>
                <ul className="list-disc list-inside space-y-1" style={{ color: 'var(--text-tertiary)' }}>
                  <li>AI analyzes the issue against the codebase</li>
                  <li>A diagnosis with root cause is generated</li>
                  <li>If fixable, a GitHub PR is automatically created</li>
                </ul>
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  onClick={handleClose}
                  className="px-4 py-2 transition-colors font-headline"
                  style={{ color: 'var(--text-secondary)' }}
                  onMouseOver={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
                  onMouseOut={(e) => e.currentTarget.style.color = 'var(--text-secondary)'}
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={!description.trim() || isSubmitting}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg transition-colors font-headline"
                  style={{
                    backgroundColor: (!description.trim() || isSubmitting) ? 'var(--bg-tertiary)' : 'var(--primary)',
                    color: (!description.trim() || isSubmitting) ? 'var(--text-tertiary)' : 'white',
                    cursor: (!description.trim() || isSubmitting) ? 'not-allowed' : 'pointer'
                  }}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 size={16} className="animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send size={16} />
                      Submit Report
                    </>
                  )}
                </button>
              </div>
            </>
          ) : (
            <div className="space-y-4">
              <div className="flex items-start gap-3 p-4 rounded-lg" style={{
                backgroundColor: result.success ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)'
              }}>
                {result.success ? (
                  <CheckCircle className="flex-shrink-0 mt-0.5" size={20} style={{ color: 'var(--success)' }} />
                ) : (
                  <AlertCircle className="flex-shrink-0 mt-0.5" size={20} style={{ color: 'var(--error)' }} />
                )}
                <div>
                  <p className="font-medium font-headline" style={{ color: result.success ? 'var(--success)' : 'var(--error)' }}>
                    {result.success ? 'Issue Reported Successfully' : 'Failed to Submit Report'}
                  </p>
                  {result.success && result.issue_url && (
                    <a
                      href={result.issue_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm underline mt-1 block"
                      style={{ color: 'var(--primary)' }}
                    >
                      View GitHub Issue
                    </a>
                  )}
                  {result.success && result.pr_url && (
                    <a
                      href={result.pr_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm underline mt-1 block"
                      style={{ color: 'var(--primary)' }}
                    >
                      View Pull Request
                    </a>
                  )}
                  {!result.success && result.error && (
                    <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>{result.error}</p>
                  )}
                </div>
              </div>
              <button
                onClick={handleClose}
                className="w-full px-4 py-2 rounded-lg font-headline"
                style={{ backgroundColor: 'var(--primary)', color: 'white' }}
              >
                Close
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Quick Action Button (simple version)
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

// Enhanced Quick Action with Multiple Examples (persists counter in localStorage)
function QuickActionWithExamples({ icon: Icon, label, examples, onSelectExample, currentInput }) {
  const storageKey = `example-index-${label.replace(/\s+/g, '-').toLowerCase()}`;

  const [exampleIndex, setExampleIndex] = useState(() => {
    // Initialize from localStorage
    const saved = localStorage.getItem(storageKey);
    return saved ? Math.min(parseInt(saved, 10), examples.length - 1) : 0;
  });
  const [isHovered, setIsHovered] = useState(false);

  // examples[exampleIndex] available if needed for display

  const handleClick = () => {
    // If current input matches an example, cycle to next
    const matchIndex = examples.findIndex(ex => ex === currentInput);
    let nextIndex;
    if (matchIndex !== -1) {
      nextIndex = (matchIndex + 1) % examples.length;
    } else {
      nextIndex = (exampleIndex + 1) % examples.length;
    }
    setExampleIndex(nextIndex);
    localStorage.setItem(storageKey, nextIndex.toString());
    onSelectExample(examples[nextIndex]);
  };

  return (
    <button
      onClick={handleClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className="flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm transition-all font-headline"
      style={{
        backgroundColor: isHovered ? 'var(--bg-tertiary)' : 'var(--bg-secondary)',
        color: isHovered ? 'var(--text-primary)' : 'var(--text-secondary)',
        border: '1px solid var(--border-color)'
      }}
      aria-label={`Quick action: ${label}`}
    >
      <Icon size={16} aria-hidden="true" style={{ color: 'var(--primary)' }} />
      <span className="font-semibold">{label}</span>
      <span
        className="ml-auto text-xs px-1.5 py-0.5 rounded"
        style={{
          backgroundColor: 'var(--primary)',
          color: 'white',
          opacity: 0.9
        }}
      >
        {exampleIndex + 1}/{examples.length}
      </span>
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

  // RAG Knowledgebase state
  const [ragStatus, setRagStatus] = useState(null);

  // Issue Report Modal state
  const [issueModalOpen, setIssueModalOpen] = useState(false);
  const [issueMessage, setIssueMessage] = useState(null);
  const [issueContext, setIssueContext] = useState({ userQuestion: '' });

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
    loadRagStatus();
  }, []);

  const loadRagStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/rag/status`);
      const data = await response.json();
      setRagStatus(data);
    } catch (e) {
      console.error('Failed to load RAG status:', e);
    }
  };

  const handleKbChange = async (kbName) => {
    try {
      const response = await fetch(`${API_BASE}/api/rag/switch-kb`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kb_name: kbName })
      });
      const data = await response.json();
      if (response.ok) {
        showNotification(`Switched to ${kbName}`, 'success');
        loadRagStatus(); // Refresh KB status
      } else {
        showNotification(data.detail || 'Failed to switch KB', 'error');
      }
    } catch (e) {
      console.error('Failed to switch KB:', e);
      showNotification('Failed to switch KB', 'error');
    }
  };

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

  // Auto-resize textarea when input changes (e.g., from quick actions)
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

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

  const handleReportIssue = (message) => {
    // Find the preceding user message for context
    const msgIndex = messages.findIndex(m => m === message);
    let userQuestion = '';
    if (msgIndex > 0) {
      for (let i = msgIndex - 1; i >= 0; i--) {
        if (messages[i].role === 'user') {
          userQuestion = messages[i].content;
          break;
        }
      }
    }
    setIssueMessage(message);
    setIssueContext({ userQuestion });
    setIssueModalOpen(true);
  };

  const handleSubmitIssue = async (report) => {
    try {
      const response = await api.reportIssue({
        ...report,
        session_id: sessionId,
      });
      return response;
    } catch (error) {
      throw error;
    }
  };

  const handleExportConversation = async () => {
    if (messages.length === 0) {
      showNotification('No conversation to export', 'error');
      return;
    }

    if (!sessionId) {
      showNotification('No active session to export', 'error');
      return;
    }

    try {
      showNotification('Generating CSV export...', 'info');

      // Call backend API to export session as CSV
      const blob = await api.exportSessionAsCSV(sessionId);

      // Download the CSV file
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation_${sessionId}_${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      showNotification('Conversation exported as CSV successfully', 'success');
    } catch (error) {
      console.error('Export error:', error);
      showNotification(error.message || 'Failed to export conversation', 'error');
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
              <h1 className="font-semibold text-xl font-headline" style={{ color: 'var(--text-primary)' }}>
                DISSOLVE Agent
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge status={status?.status} />
            {/* Knowledge Base Selector */}
            {ragStatus?.available && ragStatus?.all_kbs?.length > 0 && (
              <select
                value={ragStatus.active_kb || ''}
                onChange={(e) => handleKbChange(e.target.value)}
                className="px-3 py-1.5 text-sm rounded-lg transition-colors font-body cursor-pointer"
                style={{
                  backgroundColor: 'var(--bg-tertiary)',
                  color: 'var(--text-primary)',
                  border: '1px solid var(--border-color)'
                }}
                title={`RAG Knowledge Base: ${ragStatus.paper_count} papers, ${ragStatus.chunk_count} chunks`}
              >
                {ragStatus.all_kbs.map(kb => (
                  <option key={kb.name} value={kb.name}>
                    {kb.name} ({kb.papers} papers)
                  </option>
                ))}
              </select>
            )}
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
                title="Export conversation as CSV"
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
                    DISSOLVE Agent
                  </h2>
                  <p className="max-w-md mb-8 font-body" style={{ color: 'var(--text-secondary)' }}>
                    Ask questions about polymer solubility, separation strategies, and solvent properties.
                  </p>

                  {/* Quick Actions - 8 Tools in 2 Rows */}
                  <div className="w-full max-w-5xl space-y-3">
                    {/* Top Row */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <QuickActionWithExamples
                        icon={Beaker}
                        label="Polymer Dissolution"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "What solvents dissolve PS at 120°C? Rank by safety and show boiling points",
                          "Find the best solvents for LDPE dissolution at 100°C, prioritize low toxicity (LogP)",
                          "Which solvents dissolve PET at 140°C? Include G-scores and energy costs"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={Search}
                        label="Solvent Properties"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Compare safety profiles (G-score, LogP, toxicity) of chloroform, toluene, and acetone",
                          "List all solvents with boiling point below 100°C and G-score above 5",
                          "What are the full properties of xylene? Include cost, safety, and boiling point",
                          "Find the cheapest solvents with G-score above 5 and BP between 80-120°C",
                          "Show me solvents with low toxicity (LogP < 3) and their boiling points",
                          "Compare cost per kg for acetone, toluene, ethanol, and DMF",
                          "Which solvents have BP > 150°C and are considered safe (G-score > 6)?",
                          "Create a heatmap of solubility for PS at 120°C",
                          "Find solvents with high boiling points (>180°C) and low environmental impact",
                          "What's the cost-benefit analysis of using toluene vs xylene for industrial processes?"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={Layers}
                        label="Multilayer Separation"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Design separation strategy for 3-layer packaging film: LDPE/EVOH/PET at 120°C",
                          "Find optimal temperatures to separate HDPE, PP, and PS from mixed plastic waste",
                          "Analyze separation process for PVC/LDPE/HDPE multilayer film"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={Rocket}
                        label="Advanced Analysis"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Full integrated analysis for LDPE/EVOH/PET separation: find optimal temp per layer, rank by safety",
                          "Find the best overall solvent for separating HDPE from PP considering selectivity, safety, cost, and BP",
                          "Compare complete profiles of top 5 solvents for PVC/LDPE separation at 100°C",
                          "I have a mixed plastic waste stream containing PE, PET, PS, and EVOH. Plan an optimal separation sequence that prioritizes safety (G-score), then analyze the techno-economics for the top solvent choices at 5,000 kg/hr throughput. Include LCA comparison to virgin polymer production and show me the GWP breakdown. Explain your reasoning and which tools you are using at each step.",
                          "Compare the full separation workflows for two scenarios: (1) LDPE/HDPE/PP mixed polyolefins and (2) PET/PVC/PS mixed engineering plastics. For each, find the optimal separation sequence, identify the best solvents ranked by safety, calculate TEA at 1000 kg/hr, and generate visualizations. Explain your tool selection and reasoning throughout.",
                          "Perform a comprehensive solvent screening for EVOH dissolution: first find all solvents that dissolve EVOH above 80% at 120°C, then rank them by G-score safety, get PubChem toxicity data for the top 5, run TEA/LCA comparison, and create a summary visualization. Walk me through your analysis step by step.",
                          "Design a complete recycling process for multilayer food packaging (LDPE outer/EVOH barrier/PET inner): determine the optimal dissolution sequence and temperatures, identify selective solvents for each layer with safety data, calculate energy requirements and costs at industrial scale (5000 kg/hr), and compare environmental impact to virgin production. Explain each tool call and your reasoning.",
                          "I need to separate a 4-polymer mixture of HDPE, LDPE, PP, and PS. First analyze selectivity between all pairs, then plan the optimal separation sequence considering both selectivity and safety. For each step, provide the recommended solvent with full properties (BP, G-score, LogP, energy), run TEA at 2000 kg/hr throughput, and generate comparison visualizations. Document your reasoning and tool usage throughout the analysis."
                        ]}
                      />
                    </div>
                    {/* Bottom Row */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <QuickActionWithExamples
                        icon={Calculator}
                        label="TEA/LCA Analysis"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Run TEA for toluene recovery at 100 kg/hr polymer throughput",
                          "What's the capital cost and payback period for solvent recovery with acetone?",
                          "LCA analysis: What's the carbon footprint of using DMF for polymer separation?",
                          "Compare toluene, acetone, and ethanol on cost and environmental impact",
                          "Which solvent has the lowest operating cost for LDPE separation at 95% recovery?",
                          "Calculate CO2 emissions for cyclohexane solvent recovery",
                          "TEA/LCA comparison: DMF vs DMSO vs NMP - which is cheapest and greenest?",
                          "What's the energy consumption for recovering xylene at 80°C?",
                          "Full techno-economic analysis for ethanol at 200 kg/hr with 98% recovery",
                          "Compare the environmental impact of chloroform vs dichloromethane recovery"
                        ]}
                      />
                      <QuickAction
                        icon={Brain}
                        label="HSP ML Prediction"
                        onClick={loadMlPolymerTypes}
                      />
                      <QuickActionWithExamples
                        icon={FlaskConical}
                        label="PubChem Safety"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Get PubChem safety data for toluene - show GHS hazards and molecular properties",
                          "What are the safety hazards for dichloromethane (DCM)?",
                          "Compare safety profiles of benzene, toluene, and xylene using PubChem data",
                          "Which is safer according to PubChem: acetone or MEK?",
                          "Create a PubChem safety chart comparing ethanol, methanol, and isopropanol",
                          "What's the LD50 and environmental toxicity of toluene, benzene, and acetone?",
                          "Is acetone biodegradable? Compare with DCM and chloroform",
                          "Get aquatic toxicity data for hexane, heptane, and cyclohexane",
                          "PubChem safety comparison: DMF vs DMSO vs NMP",
                          "Is benzene carcinogenic? Get full PubChem safety profile"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={BookOpen}
                        label="Literature Search"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Search Web of Science for peer-reviewed articles on PET dissolution",
                          "Find Google Scholar papers on Hansen solubility parameters",
                          "What are recent WoS publications on selective polymer separation?",
                          "Search for articles about polyethylene solubility in the last 5 years",
                          "Find research on green solvents for polymer recycling"
                        ]}
                      />
                    </div>
                    {/* Third Row - Patents + RAG */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
                      <QuickActionWithExamples
                        icon={FileText}
                        label="Patent Search"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Search patents for polymer dissolution solvent recovery",
                          "Find patents on PET recycling technologies from Eastman",
                          "Look up patent US10457803",
                          "What patents exist for selective dissolution of mixed plastics?",
                          "Search Google Patents for solvent-based recycling after 2020",
                          "Save patent US10457803 to RAG for searching",
                          "Find Dow patents on polymer separation and save to RAG"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={Library}
                        label="RAG Literature"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Search the indexed literature for polystyrene dissolution",
                          "Ask literature: What solvents are best for PET recycling?",
                          "Run full RAG diagnostics",
                          "Visualize my document embeddings using t-SNE",
                          "Analyze search scores for 'polymer dissolution'",
                          "Which documents are most similar in my collection?",
                          "Compare dense vs sparse retrieval performance",
                          "How much does reranking improve my results?",
                          "Analyze query expansion effectiveness",
                          "Download open-access papers on polymer recycling to RAG"
                        ]}
                      />
                    </div>
                  </div>
                  <p className="text-xs mt-3 font-body" style={{ color: 'var(--text-tertiary)' }}>
                    Click buttons to cycle through examples. Number shows current example (e.g., 3/10).
                  </p>
                </>
              )}
            </div>
          ) : (
            <>
              {messages.map((msg, i) => (
                <Message
                  key={i}
                  message={msg}
                  isUser={msg.role === 'user'}
                  onDownloadCSV={handleDownloadCSV}
                  onReportIssue={msg.role === 'assistant' ? handleReportIssue : undefined}
                />
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
                onChange={(e) => {
                  setInput(e.target.value);
                  // Auto-resize textarea
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
                }}
                onKeyDown={handleKeyDown}
                placeholder="Ask about polymer solubility, separation strategies, solvent properties..."
                rows={1}
                className="w-full rounded-xl px-4 py-3 pr-12 resize-none focus:outline-none font-body overflow-y-auto"
                style={{
                  minHeight: '48px',
                  maxHeight: '200px',
                  backgroundColor: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  color: 'var(--text-primary)',
                  transition: 'height 0.1s ease'
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

      {/* Issue Report Modal */}
      <IssueReportModal
        isOpen={issueModalOpen}
        onClose={() => setIssueModalOpen(false)}
        message={issueMessage}
        userQuestion={issueContext.userQuestion}
        onSubmit={handleSubmitIssue}
      />
    </div>
  );
}

export default App;
