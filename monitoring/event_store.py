"""
Event Store for Multi-Agent Trace Storage

Provides in-memory storage for execution traces with:
- Maximum capacity (1000 traces by default)
- TTL-based expiration (24 hours by default)
- Thread-safe access for async operations
- Efficient retrieval by trace_id or session_id
"""

import threading
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Deque

from .models import StoredTrace, TraceQuery

logger = logging.getLogger(__name__)


class EventStore:
    """
    Thread-safe in-memory event store for execution traces.

    Uses a deque for O(1) appends with automatic size limiting,
    plus a dict for O(1) lookups by trace_id.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_hours: int = 24,
    ):
        """
        Initialize the event store.

        Args:
            max_size: Maximum number of traces to store (oldest evicted first)
            ttl_hours: Time-to-live for traces in hours
        """
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)

        # Storage structures
        self._traces: Deque[StoredTrace] = deque(maxlen=max_size)
        self._by_id: Dict[str, StoredTrace] = {}
        self._by_session: Dict[str, List[str]] = {}  # session_id -> [trace_ids]

        # Thread safety
        self._lock = threading.RLock()

        logger.info(f"EventStore initialized: max_size={max_size}, ttl={ttl_hours}h")

    def store(self, trace: StoredTrace) -> bool:
        """
        Store a trace in the event store.

        Args:
            trace: The trace to store

        Returns:
            True if stored successfully
        """
        with self._lock:
            # Set expiration time
            trace.stored_at = datetime.now()
            trace.expires_at = trace.stored_at + self.ttl

            # If we're at capacity, remove the oldest trace from indices
            if len(self._traces) >= self.max_size:
                oldest = self._traces[0]
                self._remove_from_indices(oldest)

            # Add to storage
            self._traces.append(trace)
            self._by_id[trace.trace_id] = trace

            # Index by session
            if trace.session_id not in self._by_session:
                self._by_session[trace.session_id] = []
            self._by_session[trace.session_id].append(trace.trace_id)

            logger.debug(f"Stored trace {trace.trace_id} for session {trace.session_id}")
            return True

    def get(self, trace_id: str) -> Optional[StoredTrace]:
        """
        Get a trace by ID.

        Args:
            trace_id: The trace identifier

        Returns:
            The trace if found and not expired, None otherwise
        """
        with self._lock:
            trace = self._by_id.get(trace_id)
            if trace and not trace.is_expired():
                return trace
            return None

    def get_by_session(self, session_id: str) -> List[StoredTrace]:
        """
        Get all traces for a session.

        Args:
            session_id: The session identifier

        Returns:
            List of non-expired traces for the session
        """
        with self._lock:
            trace_ids = self._by_session.get(session_id, [])
            traces = []
            for tid in trace_ids:
                trace = self._by_id.get(tid)
                if trace and not trace.is_expired():
                    traces.append(trace)
            return sorted(traces, key=lambda t: t.stored_at, reverse=True)

    def get_recent(self, limit: int = 50) -> List[StoredTrace]:
        """
        Get the most recent traces.

        Args:
            limit: Maximum number of traces to return

        Returns:
            List of recent non-expired traces (newest first)
        """
        with self._lock:
            traces = []
            # Iterate from newest to oldest
            for trace in reversed(self._traces):
                if not trace.is_expired():
                    traces.append(trace)
                    if len(traces) >= limit:
                        break
            return traces

    def query(self, q: TraceQuery) -> List[StoredTrace]:
        """
        Query traces with filters.

        Args:
            q: Query parameters

        Returns:
            List of matching traces
        """
        with self._lock:
            results = []

            for trace in reversed(self._traces):
                # Skip expired
                if trace.is_expired():
                    continue

                # Apply filters
                if q.session_id and trace.session_id != q.session_id:
                    continue
                if q.path and trace.path != q.path:
                    continue
                if q.min_complexity and trace.complexity < q.min_complexity:
                    continue
                if q.max_complexity and trace.complexity > q.max_complexity:
                    continue
                if q.success_only and not trace.success:
                    continue
                if q.since and trace.start_time < q.since:
                    continue
                if q.until and trace.start_time > q.until:
                    continue

                results.append(trace)

            # Apply pagination
            total = len(results)
            start = q.offset
            end = q.offset + q.limit
            return results[start:end]

    def get_all_active(self) -> List[StoredTrace]:
        """
        Get all non-expired traces.

        Returns:
            List of all active traces
        """
        with self._lock:
            return [t for t in self._traces if not t.is_expired()]

    def cleanup_expired(self) -> int:
        """
        Remove expired traces from the store.

        Returns:
            Number of traces removed
        """
        with self._lock:
            expired_ids = []
            active_traces = []

            for trace in self._traces:
                if trace.is_expired():
                    expired_ids.append(trace.trace_id)
                else:
                    active_traces.append(trace)

            # Remove from indices
            for trace_id in expired_ids:
                trace = self._by_id.pop(trace_id, None)
                if trace:
                    self._remove_from_session_index(trace)

            # Rebuild deque
            self._traces = deque(active_traces, maxlen=self.max_size)

            if expired_ids:
                logger.info(f"Cleaned up {len(expired_ids)} expired traces")

            return len(expired_ids)

    def count(self) -> int:
        """Get total trace count."""
        with self._lock:
            return len(self._traces)

    def count_active(self) -> int:
        """Get count of non-expired traces."""
        with self._lock:
            return sum(1 for t in self._traces if not t.is_expired())

    def clear(self) -> None:
        """Clear all traces from the store."""
        with self._lock:
            self._traces.clear()
            self._by_id.clear()
            self._by_session.clear()
            logger.info("Event store cleared")

    def _remove_from_indices(self, trace: StoredTrace) -> None:
        """Remove a trace from all indices."""
        self._by_id.pop(trace.trace_id, None)
        self._remove_from_session_index(trace)

    def _remove_from_session_index(self, trace: StoredTrace) -> None:
        """Remove a trace from the session index."""
        if trace.session_id in self._by_session:
            try:
                self._by_session[trace.session_id].remove(trace.trace_id)
                if not self._by_session[trace.session_id]:
                    del self._by_session[trace.session_id]
            except ValueError:
                pass  # Already removed


# Global singleton instance
event_store = EventStore()
