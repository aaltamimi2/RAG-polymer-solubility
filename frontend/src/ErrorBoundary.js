import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

/**
 * Error Boundary Component
 * Catches JavaScript errors anywhere in the component tree and displays a fallback UI
 */
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log error details for debugging
    console.error('Error boundary caught an error:', error, errorInfo);

    this.setState({
      error: error,
      errorInfo: errorInfo
    });

    // You could also log the error to an error reporting service here
    // Example: logErrorToService(error, errorInfo);
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-slate-800 rounded-xl border border-slate-700 p-6 space-y-4">
            <div className="flex items-center gap-3 text-red-400">
              <AlertTriangle size={24} />
              <h1 className="text-xl font-semibold">Something went wrong</h1>
            </div>

            <p className="text-slate-300">
              The application encountered an unexpected error. This has been logged for investigation.
            </p>

            {this.state.error && (
              <div className="bg-slate-900 rounded-lg p-3 border border-slate-700">
                <p className="text-sm font-mono text-red-400">
                  {this.state.error.toString()}
                </p>
              </div>
            )}

            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <details className="text-xs text-slate-400">
                <summary className="cursor-pointer hover:text-slate-300">
                  Error details (development only)
                </summary>
                <pre className="mt-2 overflow-auto bg-slate-900 p-2 rounded border border-slate-700">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            <div className="flex gap-3 pt-2">
              <button
                onClick={this.handleReload}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-600 hover:bg-primary-700 rounded-lg transition-colors"
              >
                <RefreshCw size={16} />
                Reload Application
              </button>
              <button
                onClick={this.handleReset}
                className="px-4 py-2.5 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors"
              >
                Try Again
              </button>
            </div>

            <p className="text-xs text-slate-500 text-center">
              If this problem persists, please contact support
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
