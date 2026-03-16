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
  GitBranch,
  BarChart3,
  ShieldAlert
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
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-headline transition-colors border"
                style={{
                  color: 'var(--warning)',
                  backgroundColor: 'rgba(245, 158, 11, 0.12)',
                  borderColor: 'rgba(245, 158, 11, 0.28)',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(245, 158, 11, 0.18)';
                  e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.42)';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(245, 158, 11, 0.12)';
                  e.currentTarget.style.borderColor = 'rgba(245, 158, 11, 0.28)';
                }}
                title="Report an issue with this response"
              >
                <AlertTriangle size={13} />
                Report Issue
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

  const issueUrl = result?.issue_url || result?.issue_result?.html_url || result?.issue_result?.issue_url;
  const prUrl = result?.pr_url || result?.pr_result?.pr_url || result?.pr_result?.html_url;
  const diagnosis = result?.diagnosis || null;
  const confidencePct = typeof diagnosis?.confidence === 'number' ? Math.round(diagnosis.confidence * 100) : null;

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
                  <li>The report is saved with the response context and metadata</li>
                  <li>AI analyzes the issue against the codebase and generates a diagnosis</li>
                  <li>If the issue is fixable, backend automation can open a GitHub PR; otherwise it files a diagnostic issue</li>
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
                  {result?.message && (
                    <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>{result.message}</p>
                  )}
                  {diagnosis && (
                    <div
                      className="mt-3 rounded-lg p-3"
                      style={{
                        backgroundColor: 'var(--bg-primary)',
                        border: '1px solid var(--border-color)',
                      }}
                    >
                      <p className="text-xs uppercase tracking-wide font-semibold" style={{ color: 'var(--text-tertiary)' }}>
                        AI Diagnosis
                      </p>
                      {diagnosis.summary && (
                        <p className="text-sm mt-2 font-semibold" style={{ color: 'var(--text-primary)' }}>
                          {diagnosis.summary}
                        </p>
                      )}
                      {diagnosis.root_cause && (
                        <p className="text-sm mt-2" style={{ color: 'var(--text-secondary)' }}>
                          {diagnosis.root_cause}
                        </p>
                      )}
                      <div className="mt-2 flex flex-wrap gap-2 text-xs font-medium">
                        {diagnosis.fix_category && (
                          <span
                            className="px-2 py-1 rounded-full"
                            style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                          >
                            {diagnosis.fix_category.replace(/_/g, ' ')}
                          </span>
                        )}
                        {confidencePct !== null && (
                          <span
                            className="px-2 py-1 rounded-full"
                            style={{ backgroundColor: 'rgba(37, 99, 235, 0.12)', color: '#2563eb' }}
                          >
                            confidence {confidencePct}%
                          </span>
                        )}
                        {Array.isArray(diagnosis.affected_files) && diagnosis.affected_files.length > 0 && (
                          <span
                            className="px-2 py-1 rounded-full"
                            style={{ backgroundColor: 'rgba(245, 158, 11, 0.12)', color: '#b45309' }}
                          >
                            {diagnosis.affected_files.length} file{diagnosis.affected_files.length === 1 ? '' : 's'}
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                  {result.success && issueUrl && (
                    <a
                      href={issueUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm underline mt-1 block"
                      style={{ color: 'var(--primary)' }}
                    >
                      View GitHub Issue
                    </a>
                  )}
                  {result.success && prUrl && (
                    <a
                      href={prUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm underline mt-1 block"
                      style={{ color: 'var(--primary)' }}
                    >
                      View Pull Request
                    </a>
                  )}
                  {result.success && result.local_report_path && (
                    <p className="text-xs mt-2 font-mono" style={{ color: 'var(--text-tertiary)' }}>
                      Saved locally: {result.local_report_path}
                    </p>
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

const WORKFLOW_ICON_MAP = {
  layers: Layers,
  'alert-triangle': AlertTriangle,
  calculator: Calculator,
  'book-open': BookOpen,
  search: Search,
  brain: Brain,
  'bar-chart-3': BarChart3,
  'shield-alert': ShieldAlert,
  'flask-conical': FlaskConical,
};

const WORKFLOW_STATUS_META = {
  planned: { label: 'Planned', color: '#64748b', bg: 'rgba(100, 116, 139, 0.12)' },
  running: { label: 'Running', color: '#2563eb', bg: 'rgba(37, 99, 235, 0.12)' },
  waiting_on_dependencies: { label: 'Waiting on deps', color: '#475569', bg: 'rgba(71, 85, 105, 0.12)' },
  waiting_on_handoff: { label: 'Waiting on handoff', color: '#d97706', bg: 'rgba(217, 119, 6, 0.12)' },
  completed: { label: 'Completed', color: '#059669', bg: 'rgba(5, 150, 105, 0.12)' },
  failed: { label: 'Failed', color: '#dc2626', bg: 'rgba(220, 38, 38, 0.12)' },
};

const WORKFLOW_ACCENT_META = {
  blue: { border: '#60a5fa', bg: 'rgba(96, 165, 250, 0.10)', iconBg: 'rgba(59, 130, 246, 0.16)' },
  amber: { border: '#f59e0b', bg: 'rgba(245, 158, 11, 0.10)', iconBg: 'rgba(245, 158, 11, 0.16)' },
  green: { border: '#34d399', bg: 'rgba(52, 211, 153, 0.10)', iconBg: 'rgba(16, 185, 129, 0.16)' },
  violet: { border: '#a78bfa', bg: 'rgba(167, 139, 250, 0.10)', iconBg: 'rgba(139, 92, 246, 0.16)' },
  orange: { border: '#fb923c', bg: 'rgba(251, 146, 60, 0.10)', iconBg: 'rgba(249, 115, 22, 0.16)' },
  pink: { border: '#f472b6', bg: 'rgba(244, 114, 182, 0.10)', iconBg: 'rgba(236, 72, 153, 0.16)' },
  cyan: { border: '#22d3ee', bg: 'rgba(34, 211, 238, 0.10)', iconBg: 'rgba(6, 182, 212, 0.16)' },
  indigo: { border: '#818cf8', bg: 'rgba(129, 140, 248, 0.10)', iconBg: 'rgba(99, 102, 241, 0.16)' },
  rose: { border: '#fb7185', bg: 'rgba(251, 113, 133, 0.10)', iconBg: 'rgba(244, 63, 94, 0.16)' },
  slate: { border: '#94a3b8', bg: 'rgba(148, 163, 184, 0.10)', iconBg: 'rgba(148, 163, 184, 0.16)' },
};

const EDGE_STATUS_META = {
  planned: { color: 'rgba(148, 163, 184, 0.6)', dash: '6 6' },
  waiting_on_dependency: { color: 'rgba(100, 116, 139, 0.75)', dash: '6 6' },
  in_progress: { color: 'rgba(59, 130, 246, 0.8)', dash: '8 6' },
  handoff_pending: { color: 'rgba(245, 158, 11, 0.85)', dash: '6 4' },
  handoff_ready: { color: 'rgba(16, 185, 129, 0.9)', dash: '' },
  blocked: { color: 'rgba(239, 68, 68, 0.9)', dash: '4 5' },
};

function formatToolDuration(durationMs) {
  if (typeof durationMs !== 'number') return null;
  if (durationMs < 1000) return `${durationMs} ms`;
  return `${(durationMs / 1000).toFixed(durationMs >= 10000 ? 0 : 1)} s`;
}

function WorkflowGraphPanel({ graph, isLoading, error, hasSubmittedQuery }) {
  const levels = graph?.levels || [];
  const nodes = graph?.nodes || [];
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const maxRows = Math.max(1, ...levels.map(level => level.nodes.length || 0), nodes.length ? 0 : 1);
  const nodeWidth = 176;
  const nodeHeight = 84;
  const columnGap = 214;
  const rowGap = 108;
  const paddingX = 32;
  const paddingY = 26;
  const graphWidth = Math.max(720, paddingX * 2 + Math.max(1, levels.length) * columnGap);
  const graphHeight = Math.max(160, paddingY * 2 + maxRows * rowGap);

  const positions = {};
  levels.forEach((level, columnIndex) => {
    level.nodes.forEach((nodeId, rowIndex) => {
      positions[nodeId] = {
        x: paddingX + columnIndex * columnGap,
        y: paddingY + rowIndex * rowGap,
      };
    });
  });

  useEffect(() => {
    const nextNodes = graph?.nodes || [];
    if (!nextNodes.length) {
      setSelectedNodeId(null);
      return;
    }
    if (!selectedNodeId || !nextNodes.some(node => node.id === selectedNodeId)) {
      setSelectedNodeId(nextNodes[0].id);
    }
  }, [graph, selectedNodeId]);

  const selectedNode = nodes.find(node => node.id === selectedNodeId) || null;

  return (
    <section
      className="rounded-2xl border px-4 py-4"
      style={{
        backgroundColor: 'var(--bg-secondary)',
        borderColor: 'var(--border-color)',
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <GitBranch size={16} style={{ color: 'var(--primary)' }} />
            <h3 className="text-sm font-semibold font-headline" style={{ color: 'var(--text-primary)' }}>
              Agent Workflow
            </h3>
            {(graph?.mode || hasSubmittedQuery) && (
              <span
                className="text-xs px-2 py-0.5 rounded-full font-medium"
                style={{
                  color: 'var(--text-secondary)',
                  backgroundColor: 'var(--bg-tertiary)',
                }}
              >
                {graph?.mode === 'live' ? 'Live execution' : 'Planned workflow'}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm font-body" style={{ color: 'var(--text-secondary)' }}>
            {graph?.mode === 'live'
              ? 'Current topological execution state for the active routed workflow.'
              : hasSubmittedQuery
                ? 'Planned topological workflow for the submitted query.'
                : 'Send a query to view the routed subagent graph.'}
          </p>
        </div>
        <div className="flex flex-col items-start gap-2 sm:items-end">
          {graph?.summary && (
            <div className="flex flex-wrap gap-2 text-xs font-medium">
              <span className="px-2 py-1 rounded-full" style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}>
                {graph.summary.total_nodes} steps
              </span>
              {graph.summary.completed_nodes > 0 && (
                <span className="px-2 py-1 rounded-full" style={{ backgroundColor: 'rgba(5, 150, 105, 0.12)', color: '#059669' }}>
                  {graph.summary.completed_nodes} complete
                </span>
              )}
              {graph.summary.running_nodes > 0 && (
                <span className="px-2 py-1 rounded-full" style={{ backgroundColor: 'rgba(37, 99, 235, 0.12)', color: '#2563eb' }}>
                  {graph.summary.running_nodes} running
                </span>
              )}
              {graph.summary.failed_nodes > 0 && (
                <span className="px-2 py-1 rounded-full" style={{ backgroundColor: 'rgba(220, 38, 38, 0.12)', color: '#dc2626' }}>
                  {graph.summary.failed_nodes} failed
                </span>
              )}
            </div>
          )}
          {graph?.langsmith?.trace_url && (
            <div className="flex flex-wrap gap-2">
              <a
                href={graph.langsmith.trace_url}
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors"
                style={{
                  backgroundColor: 'var(--primary)',
                  color: 'white',
                }}
              >
                Open trace
              </a>
              {graph.langsmith.shared_trace_url && (
                <a
                  href={graph.langsmith.shared_trace_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors"
                  style={{
                    backgroundColor: 'var(--bg-tertiary)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                  }}
                >
                  Public trace
                </a>
              )}
            </div>
          )}
        </div>
      </div>

      {graph?.next_action?.label && (
        <div
          className="mt-3 rounded-xl px-3 py-2 text-sm font-body"
          style={{
            backgroundColor: 'var(--bg-tertiary)',
            color: 'var(--text-secondary)',
            border: '1px solid var(--border-color)',
          }}
        >
          <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>Next:</span> {graph.next_action.label}
        </div>
      )}

      {graph?.langsmith?.enabled && (
        <div className="mt-3 text-xs font-mono" style={{ color: 'var(--text-tertiary)' }}>
          LangSmith tracing enabled
          {graph.langsmith.project ? ` • project: ${graph.langsmith.project}` : ''}
          {graph.langsmith.thread_id ? ` • thread: ${graph.langsmith.thread_id}` : ''}
        </div>
      )}

      {isLoading ? (
        <div className="mt-4 flex items-center gap-2 text-sm font-body" style={{ color: 'var(--text-secondary)' }}>
          <Loader2 size={16} className="animate-spin" />
          Planning routed subagent graph…
        </div>
      ) : error ? (
        <div className="mt-4 flex items-center gap-2 text-sm font-body" style={{ color: 'var(--error)' }}>
          <AlertCircle size={16} />
          {error}
        </div>
      ) : !graph || nodes.length === 0 ? (
        <div className="mt-4 text-sm font-body" style={{ color: 'var(--text-secondary)' }}>
          {hasSubmittedQuery
            ? 'No workflow graph is available yet for this run.'
            : 'Send a query to view the routed subagent graph.'}
        </div>
      ) : (
        <>
          <div className="mt-4 overflow-x-auto">
            <div style={{ width: graphWidth, height: graphHeight, position: 'relative' }}>
              <svg width={graphWidth} height={graphHeight} style={{ position: 'absolute', inset: 0 }}>
                {graph.edges.map(edge => {
                  const source = positions[edge.source];
                  const target = positions[edge.target];
                  if (!source || !target) return null;
                  const startX = source.x + nodeWidth;
                  const startY = source.y + nodeHeight / 2;
                  const endX = target.x;
                  const endY = target.y + nodeHeight / 2;
                  const midX = startX + (endX - startX) / 2;
                  const edgeStyle = EDGE_STATUS_META[edge.status] || EDGE_STATUS_META.planned;
                  return (
                    <path
                      key={edge.id}
                      d={`M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`}
                      fill="none"
                      stroke={edgeStyle.color}
                      strokeWidth="2.5"
                      strokeDasharray={edgeStyle.dash}
                      strokeLinecap="round"
                    />
                  );
                })}
              </svg>
              {nodes.map(node => {
                const pos = positions[node.id];
                if (!pos) return null;
                const Icon = WORKFLOW_ICON_MAP[node.icon] || FlaskConical;
                const statusMeta = WORKFLOW_STATUS_META[node.status] || WORKFLOW_STATUS_META.planned;
                const accentMeta = WORKFLOW_ACCENT_META[node.accent] || WORKFLOW_ACCENT_META.slate;
                const isSelected = node.id === selectedNodeId;
                return (
                  <div
                    key={node.id}
                    className="absolute rounded-2xl border p-3 cursor-pointer"
                    style={{
                      left: pos.x,
                      top: pos.y,
                      width: nodeWidth,
                      minHeight: nodeHeight,
                      backgroundColor: accentMeta.bg,
                      borderColor: isSelected ? 'var(--primary)' : accentMeta.border,
                      boxShadow: isSelected ? 'var(--shadow-md)' : 'var(--shadow-sm)',
                      backdropFilter: 'blur(10px)',
                    }}
                    onClick={() => setSelectedNodeId(node.id)}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                        style={{ backgroundColor: accentMeta.iconBg, color: accentMeta.border }}
                      >
                        <Icon size={18} />
                      </div>
                      <div className="min-w-0">
                        <div className="text-sm font-semibold font-headline leading-tight" style={{ color: 'var(--text-primary)' }}>
                          {node.label}
                        </div>
                        <div className="text-[11px] font-mono mt-1 break-words" style={{ color: 'var(--text-tertiary)' }}>
                          {node.subagent}
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-2">
                      <span
                        className="text-[11px] px-2 py-1 rounded-full font-semibold"
                        style={{ color: statusMeta.color, backgroundColor: statusMeta.bg }}
                      >
                        {statusMeta.label}
                      </span>
                      <div className="flex items-center gap-2">
                        {node?.langsmith?.tool_count > 0 && (
                          <span className="text-[11px] font-medium" style={{ color: accentMeta.border }}>
                            {node.langsmith.tool_count} tool{node.langsmith.tool_count > 1 ? 's' : ''}
                          </span>
                        )}
                        {node.depends_on.length > 0 && (
                          <span className="text-[11px] font-medium" style={{ color: 'var(--text-tertiary)' }}>
                            {node.depends_on.length} dep{node.depends_on.length > 1 ? 's' : ''}
                          </span>
                        )}
                        {node?.langsmith?.trace_url && (
                          <a
                            href={node.langsmith.trace_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[11px] font-semibold underline"
                            style={{ color: accentMeta.border }}
                          >
                            Trace
                          </a>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {selectedNode && (
            <div
              className="mt-4 rounded-2xl border p-4"
              style={{
                backgroundColor: 'var(--bg-tertiary)',
                borderColor: 'var(--border-color)',
              }}
            >
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <div className="text-sm font-semibold font-headline" style={{ color: 'var(--text-primary)' }}>
                    {selectedNode.label}
                  </div>
                  <div className="text-xs font-mono mt-1" style={{ color: 'var(--text-tertiary)' }}>
                    {selectedNode.subagent}
                  </div>
                  <div className="mt-2 text-sm font-body" style={{ color: 'var(--text-secondary)' }}>
                    {selectedNode.description || 'No additional task description captured.'}
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  {selectedNode?.langsmith?.trace_url && (
                    <a
                      href={selectedNode.langsmith.trace_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors"
                      style={{
                        backgroundColor: 'var(--primary)',
                        color: 'white',
                      }}
                    >
                      Open node trace
                    </a>
                  )}
                </div>
              </div>

              <div className="mt-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-xs uppercase tracking-wider font-semibold" style={{ color: 'var(--text-tertiary)' }}>
                    Tool calls
                  </div>
                  {selectedNode?.langsmith?.tool_count > 0 && (
                    <div className="text-xs font-medium" style={{ color: 'var(--text-tertiary)' }}>
                      {selectedNode.langsmith.tool_count} traced
                      {selectedNode?.langsmith?.total_duration_ms
                        ? ` • ${formatToolDuration(selectedNode.langsmith.total_duration_ms)} total`
                        : ''}
                    </div>
                  )}
                </div>
                {selectedNode?.langsmith?.tools_error ? (
                  <div className="mt-2 text-sm font-body" style={{ color: 'var(--error)' }}>
                    {selectedNode.langsmith.tools_error}
                  </div>
                ) : selectedNode?.langsmith?.tools?.length > 0 ? (
                  <div className="mt-3 space-y-2">
                    {selectedNode.langsmith.tools.map((tool, index) => {
                      const statusColor = tool.error ? 'var(--error)' : 'var(--text-secondary)';
                      return (
                        <div
                          key={`${tool.run_id || tool.name || 'tool'}-${index}`}
                          className="rounded-xl border px-3 py-3"
                          style={{
                            backgroundColor: 'var(--bg-secondary)',
                            borderColor: 'var(--border-color)',
                            marginLeft: `${Math.max(0, tool.depth || 0) * 14}px`,
                          }}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-start gap-3 min-w-0">
                              <div
                                className="w-6 h-6 rounded-full flex items-center justify-center text-[11px] font-bold flex-shrink-0"
                                style={{
                                  backgroundColor: 'var(--bg-tertiary)',
                                  color: 'var(--text-primary)',
                                }}
                              >
                                {tool.order || index + 1}
                              </div>
                              <div className="min-w-0">
                                <div className="text-sm font-semibold font-headline break-words" style={{ color: 'var(--text-primary)' }}>
                                  {tool.name}
                                </div>
                                <div className="mt-1 text-xs font-mono" style={{ color: statusColor }}>
                                  {tool.status || 'unknown'}
                                  {typeof tool.depth === 'number' && tool.depth > 0 ? ` • depth ${tool.depth}` : ''}
                                  {tool.duration_ms ? ` • ${formatToolDuration(tool.duration_ms)}` : ''}
                                </div>
                                {tool.error && (
                                  <div className="mt-1 text-xs font-body" style={{ color: 'var(--error)' }}>
                                    {tool.error}
                                  </div>
                                )}
                              </div>
                            </div>
                            {tool.trace_url && (
                              <a
                                href={tool.trace_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs font-semibold underline flex-shrink-0"
                                style={{ color: 'var(--primary)' }}
                              >
                                Trace
                              </a>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="mt-2 text-sm font-body" style={{ color: 'var(--text-secondary)' }}>
                    {selectedNode?.langsmith?.trace_url
                      ? 'No traced tool calls were recorded for this subagent run.'
                      : 'Tool-level trace details are available once this node has a LangSmith trace.'}
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="mt-4 flex flex-wrap gap-2 text-xs font-medium">
            {Object.entries(WORKFLOW_STATUS_META).map(([key, meta]) => (
              <span
                key={key}
                className="px-2 py-1 rounded-full"
                style={{ color: meta.color, backgroundColor: meta.bg }}
              >
                {meta.label}
              </span>
            ))}
          </div>
        </>
      )}
    </section>
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
  const [selectedModel, setSelectedModel] = useState('gemini-3.1-flash-lite-preview'); // Gemini model selection
  const [showWorkflowGraph, setShowWorkflowGraph] = useState(false);
  const [workflowGraph, setWorkflowGraph] = useState(null);
  const [workflowLoading, setWorkflowLoading] = useState(false);
  const [workflowError, setWorkflowError] = useState(null);
  const [workflowQuery, setWorkflowQuery] = useState(null);
  const [mlPolymerTypes, setMlPolymerTypes] = useState(null); // ML polymer types data
  const [showMlTypes, setShowMlTypes] = useState(false); // Show ML types view

  // ML Workflow state
  const [mlStep, setMlStep] = useState('types'); // 'types' | 'polymers' | 'solvents' | 'results'
  const [selectedType, setSelectedType] = useState(null);
  const [polymersInType, setPolymersInType] = useState(null);
  const [selectedPolymers, setSelectedPolymers] = useState([]);
  const [solventInput, setSolventInput] = useState('');

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
    const savedModel = localStorage.getItem('dissolve-model') || 'gemini-3.1-flash-lite-preview';
    const modelAliases = {
      'gemini-2.5-flash-lite': 'gemini-3.1-flash-lite-preview',
      'gemini-2.5-flash': 'gemini-3-flash-preview',
      'gemini-2.5-pro': 'gemini-3.1-pro-preview',
    };
    setSelectedModel(modelAliases[savedModel] || savedModel);
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem('dissolve-show-workflow-graph');
    setShowWorkflowGraph(saved === 'true');
  }, []);

  useEffect(() => {
    localStorage.setItem('dissolve-show-workflow-graph', showWorkflowGraph ? 'true' : 'false');
  }, [showWorkflowGraph]);

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

  // Auto-resize textarea when input changes (e.g., from quick actions)
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  useEffect(() => {
    if (!showWorkflowGraph) {
      setWorkflowLoading(false);
      setWorkflowError(null);
      return;
    }

    const shouldLoadPreview = Boolean(workflowQuery && (isLoading || !sessionId));
    const shouldLoadLive = Boolean(!isLoading && sessionId);

    if (!shouldLoadLive && !shouldLoadPreview) {
      setWorkflowLoading(false);
      setWorkflowError(null);
      return;
    }

    let cancelled = false;
    const loadWorkflow = async () => {
      setWorkflowLoading(true);
      try {
        const data = shouldLoadLive
          ? await api.getSessionWorkflow(sessionId)
          : await api.previewWorkflow(workflowQuery);
        if (!cancelled) {
          setWorkflowGraph(data);
          setWorkflowError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setWorkflowGraph(null);
          setWorkflowError(e.message || 'Failed to load workflow graph');
        }
      } finally {
        if (!cancelled) {
          setWorkflowLoading(false);
        }
      }
    };

    const timeoutId = window.setTimeout(loadWorkflow, shouldLoadPreview ? 120 : 0);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [showWorkflowGraph, workflowQuery, sessionId, isLoading]);

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
      const data = await api.getMlPolymerTypes();
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
      const data = await api.getMlPolymersByType(polymerType);
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
    const trimmedInput = input.trim();

    const userMessage = {
      role: 'user',
      content: trimmedInput,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setWorkflowQuery(trimmedInput);
    setWorkflowError(null);
    if (showWorkflowGraph) {
      setWorkflowGraph(null);
      setWorkflowLoading(true);
    }
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
    setWorkflowQuery(null);
    setWorkflowGraph(null);
    setWorkflowLoading(false);
    setWorkflowError(null);
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

  const hasWorkflowPanel = showWorkflowGraph && (workflowQuery || sessionId || workflowGraph || workflowLoading || workflowError);

  return (
    <div className="h-screen flex flex-col" style={{ backgroundColor: 'var(--bg-primary)' }}>
      {/* Header */}
      <header className="flex-shrink-0 backdrop-blur-sm" style={{
        borderBottom: '1px solid var(--border-color)',
        backgroundColor: 'var(--bg-secondary)'
      }}>
        <div className="max-w-[1380px] mx-auto px-4 py-3 flex items-center justify-between">
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
            <button
              onClick={() => setShowWorkflowGraph(prev => !prev)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm rounded-lg transition-colors font-headline"
              style={{
                backgroundColor: showWorkflowGraph ? 'var(--primary)' : 'var(--bg-tertiary)',
                color: showWorkflowGraph ? 'white' : 'var(--text-primary)',
              }}
              title={showWorkflowGraph ? 'Hide agent workflow graph' : 'Show agent workflow graph'}
            >
              <GitBranch size={16} />
              <span className="hidden sm:inline">Agent Graph</span>
            </button>
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
              <option value="gemini-3.1-flash-lite-preview">3.1 Flash Lite Preview (Default)</option>
              <option value="gemini-3.1-pro-preview">3.1 Pro Preview</option>
              <option value="gemini-3-flash-preview">3 Flash Preview</option>
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
      <main className="flex-1 overflow-hidden w-full">
        <div className="h-full max-w-[1380px] mx-auto w-full flex flex-col lg:flex-row">
          <div className="min-w-0 flex-1 flex flex-col overflow-hidden">
            {hasWorkflowPanel && (
              <div className="flex-shrink-0 px-4 pt-4 lg:hidden">
                <WorkflowGraphPanel
                  graph={workflowGraph}
                  isLoading={workflowLoading}
                  error={workflowError}
                  hasSubmittedQuery={Boolean(workflowQuery || sessionId)}
                />
              </div>
            )}
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6 min-h-0">
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
                    Ask questions about polymer solubility, contaminant removal, separation planning, techno-economics, and research synthesis.
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
                          "Find solvents that dissolve EVOH at 120°C while keeping LDPE insoluble, then rank candidates by safety and boiling point",
                          "Screen PET dissolution solvents below 140°C and summarize the top candidates by G-score, price, and atmospheric-pressure margin",
                          "Which solvents dissolve PS near 100°C? Include solubility, boiling point, and solvent recovery considerations",
                          "Find low-hazard solvents for LDPE dissolution at 100°C and compare them on cost and operating window"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={Search}
                        label="Solvent Screening"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Compare acetone, ethyl acetate, THF, and methyl acetate on G-score, price, boiling point, and GWP",
                          "List solvents with boiling point below 100°C, G-score above 5, and TEA price data available",
                          "Find solvents for EVOH processing with lower hazard and lower price than DMF",
                          "Summarize safety, cost, and operating-window tradeoffs for toluene, xylene, and cyclohexanone",
                          "Which solvents have boiling point above 150°C and still look reasonable on G-score and cost?",
                          "Create a property comparison for acetone, methanol, ethyl acetate, and methyl acetate"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={Layers}
                        label="Separation Planning"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Design an atmospheric-pressure separation sequence for LDPE/EVOH/PET multilayer film and justify each solvent choice",
                          "Find the optimal separation order and recommended temperatures for HDPE, PP, and PS mixed waste",
                          "Plan selective dissolution for EVOH/LDPE packaging and call out any narrow boiling-point margins at 1 atm",
                          "Compare two feasible separation routes for PET/PVC/PS and recommend the best one"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={Rocket}
                        label="Integrated Workflow"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Do a literature search and patent search for solvent-based delamination of HDPE/EVOH food packaging, answer with RAG, then design an atmospheric-pressure separation sequence and create a summary chart",
                          "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream using selective dissolution, propose up to 1 additional wash step for phthalate removal, then run TEA on solvent recovery for the best option",
                          "For multilayer LDPE/EVOH/PET packaging, combine literature search, separation planning, contaminant wash planning, solvent recovery TEA, and a final visualization",
                          "Research green solvents for PET/EVOH separation, compare with patent coverage, screen PFAS wash options, then visualize the recommended process and tradeoffs",
                          "Compare two end-to-end recycling workflows for EVOH barrier films: one minimizing wash steps and one minimizing solvent hazard, then plot the tradeoffs",
                          "I have a mixed plastic waste stream containing PE, PET, PS, and EVOH. Plan an optimal separation sequence, analyze contaminant wash options, run techno-economics for the top solvent choices at 5,000 kg/hr, and generate a comparison visualization"
                        ]}
                      />
                    </div>
                    {/* Bottom Row */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                      <QuickActionWithExamples
                        icon={Calculator}
                        label="TEA + Recovery"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Run TEA for ethyl acetate recovery after contaminant washing at 500 kg/hr polymer throughput",
                          "Compare solvent recovery economics for methyl acetate versus acetone for PFAS and phthalate wash removal",
                          "Estimate solvent recovery cost and GWP for the best HDPE/EVOH separation solvent at 1000 kg/hr",
                          "Compare TEA/LCA for two candidate separation solvents and recommend the better recovery option",
                          "What is the payback impact of 95% versus 99% solvent recovery for acetone in LDPE separation?",
                          "Analyze energy use and operating cost for recovering toluene after PS dissolution at atmospheric pressure"
                        ]}
                      />
                      <QuickAction
                        icon={Brain}
                        label="HSP ML Prediction"
                        onClick={loadMlPolymerTypes}
                      />
                      <QuickActionWithExamples
                        icon={AlertTriangle}
                        label="Contaminant Removal"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "For an LDPE/EVOH mixed waste stream, find wash solvents to remove PFAS and phthalates while keeping the polymers insoluble",
                          "Screen contaminant-removal options for HDPE with phthalates and rank the best wash solvents by safety and cost",
                          "Compare a single wash step versus a two-step wash plan for removing PFAS and phthalates from LDPE packaging",
                          "Which solvents can leach PFAS from EVOH at atmospheric pressure without dissolving the polymer?",
                          "Plan up to 2 wash steps to remove contaminant families from an EVOH/LDPE multilayer film",
                          "Find the best contaminant-removal solvent for PET contaminated with phthalates and summarize the tradeoffs",
                          "For HDPE/EVOH food-packaging waste, propose a wash sequence for phthalate removal before techno-economic analysis",
                          "Identify shared wash solvents that can remove both PFAS and phthalates from LDPE without forcing separate wash steps"
                        ]}
                      />
                      <QuickActionWithExamples
                        icon={BookOpen}
                        label="Research + RAG"
                        currentInput={input}
                        onSelectExample={handleQuickAction}
                        examples={[
                          "Search the literature and patents on EVOH delamination, then synthesize the main solvent trends with RAG",
                          "Find recent papers and patents on selective dissolution of HDPE/EVOH food packaging and summarize the consensus",
                          "What does the literature say about PFAS removal from polyolefin packaging during solvent washing?",
                          "Search for green solvents for PET recycling and compare research trends against patent activity",
                          "Find prior art and academic work on phthalate removal from recycled plastics and summarize the main approaches"
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
          </div>

          {hasWorkflowPanel && (
            <aside
              className="hidden lg:block lg:w-[360px] xl:w-[430px] flex-shrink-0 overflow-y-auto"
              style={{
                borderLeft: '1px solid var(--border-color)',
                backgroundColor: 'var(--bg-primary)',
              }}
            >
              <div className="p-4">
                <WorkflowGraphPanel
                  graph={workflowGraph}
                  isLoading={workflowLoading}
                  error={workflowError}
                  hasSubmittedQuery={Boolean(workflowQuery || sessionId)}
                />
              </div>
            </aside>
          )}
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
