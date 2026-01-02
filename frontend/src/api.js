/**
 * API Client with Retry Logic and Timeout Handling
 *
 * Features:
 * - Exponential backoff retry strategy
 * - Configurable timeouts
 * - Automatic error handling
 * - Request deduplication
 */

const API_BASE = process.env.REACT_APP_API_URL || window.location.origin;

// Default configuration
const DEFAULT_CONFIG = {
  maxRetries: 3,
  timeout: 60000, // 60 seconds
  retryDelay: 1000, // Initial delay: 1 second
};

/**
 * Fetch with retry and timeout support
 * @param {string} url - Request URL
 * @param {object} options - Fetch options
 * @param {object} config - Retry configuration
 * @returns {Promise<Response>} - Fetch response
 */
async function fetchWithRetry(url, options = {}, config = {}) {
  const { maxRetries, timeout, retryDelay } = { ...DEFAULT_CONFIG, ...config };

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      // Create abort controller for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout);

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Don't retry on client errors (4xx)
      if (response.status >= 400 && response.status < 500) {
        return response;
      }

      // Return successful responses
      if (response.ok) {
        return response;
      }

      // Retry on server errors (5xx) if attempts remain
      if (attempt < maxRetries) {
        const delay = retryDelay * Math.pow(2, attempt); // Exponential backoff
        console.warn(`Request failed (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }

      return response;

    } catch (error) {
      const isLastAttempt = attempt === maxRetries;

      // Don't retry on abort (user-initiated cancellation)
      if (error.name === 'AbortError' && isLastAttempt) {
        throw new Error('Request timeout - the server is taking too long to respond');
      }

      // Don't retry network errors on last attempt
      if (isLastAttempt) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        throw new Error(`Network error: ${error.message}`);
      }

      // Retry with exponential backoff
      const delay = retryDelay * Math.pow(2, attempt);
      console.warn(`Request error (attempt ${attempt + 1}/${maxRetries + 1}): ${error.message}, retrying in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

/**
 * API Client
 */
const api = {
  /**
   * Get system status
   */
  async getStatus() {
    const response = await fetchWithRetry(`${API_BASE}/api/status`, {}, { timeout: 5000 });
    if (!response.ok) {
      throw new Error(`Failed to get status: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Send chat message to agent
   * @param {string} message - User message
   * @param {string} sessionId - Session ID (optional)
   * @param {string} model - Gemini model to use (optional, defaults to gemini-2.5-flash-lite)
   */
  async chat(message, sessionId, model = 'gemini-2.5-flash-lite') {
    const response = await fetchWithRetry(
      `${API_BASE}/api/chat`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId, model }),
      },
      { timeout: 120000 } // 2 minutes for agent responses
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Chat request failed');
    }

    return response.json();
  },

  /**
   * Get loaded tables information
   */
  async getTables() {
    const response = await fetchWithRetry(`${API_BASE}/api/tables`);
    if (!response.ok) {
      throw new Error(`Failed to get tables: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Reindex all CSV files
   */
  async reindex() {
    const response = await fetchWithRetry(
      `${API_BASE}/api/reindex`,
      { method: 'POST' },
      { timeout: 30000 } // 30 seconds
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Reindex failed');
    }

    return response.json();
  },

  /**
   * Upload CSV file
   * @param {File} file - File to upload
   */
  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetchWithRetry(
      `${API_BASE}/api/upload`,
      {
        method: 'POST',
        body: formData,
      },
      { timeout: 30000 } // 30 seconds
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  },

  /**
   * Get list of generated plots
   */
  async getPlots() {
    const response = await fetchWithRetry(`${API_BASE}/api/plots`);
    if (!response.ok) {
      throw new Error(`Failed to get plots: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Clear all plots
   */
  async clearPlots() {
    const response = await fetchWithRetry(`${API_BASE}/api/plots`, { method: 'DELETE' });
    if (!response.ok) {
      throw new Error(`Failed to clear plots: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Clear chat session
   * @param {string} sessionId - Session ID to clear
   */
  async clearSession(sessionId) {
    const response = await fetchWithRetry(`${API_BASE}/api/session/${sessionId}`, { method: 'DELETE' });
    if (!response.ok) {
      throw new Error(`Failed to clear session: ${response.statusText}`);
    }
    return response.json();
  },

  /**
   * Download CSV export by ID
   * @param {string} exportId - Export ID
   * @returns {Promise<Blob>} - CSV file blob
   */
  async downloadExport(exportId) {
    const response = await fetchWithRetry(`${API_BASE}/api/export/${exportId}`);

    if (response.status === 404) {
      throw new Error('Export not found or expired (exports are available for 30 minutes)');
    }

    if (!response.ok) {
      throw new Error(`Failed to download export: ${response.statusText}`);
    }

    return response.blob();
  },
};

export default api;
export { API_BASE };
