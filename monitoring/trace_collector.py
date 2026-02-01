"""
Trace Collector for Multi-Agent Execution Traces

Provides non-blocking trace collection that:
- Validates and enriches trace data
- Converts execution_trace from MultiAgentState to StoredTrace
- Stores traces asynchronously without blocking the main request
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from .models import StoredTrace, HandoffDetail, PathType
from .event_store import event_store

logger = logging.getLogger(__name__)


class TraceCollector:
    """
    Collects and stores execution traces from the multi-agent system.

    Designed for non-blocking operation in async request handlers.
    """

    def __init__(self):
        """Initialize the trace collector."""
        self._pending_count = 0
        self._stored_count = 0
        self._error_count = 0

    async def store_async(
        self,
        execution_trace: Dict[str, Any],
        session_id: str,
        query: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store an execution trace asynchronously.

        This method is designed to be called with asyncio.create_task()
        to avoid blocking the main request handler.

        Args:
            execution_trace: The execution_trace dict from MultiAgentState
            session_id: The session identifier
            query: The original user query
            result: Optional full result dict with additional context

        Returns:
            The trace_id if stored successfully, None otherwise
        """
        self._pending_count += 1

        try:
            # Validate and convert to StoredTrace
            stored_trace = self._convert_to_stored_trace(
                execution_trace, session_id, query, result
            )

            if stored_trace:
                # Store in event store (thread-safe)
                event_store.store(stored_trace)
                self._stored_count += 1
                logger.debug(f"Trace {stored_trace.trace_id} stored successfully")
                return stored_trace.trace_id
            else:
                logger.warning("Failed to convert execution trace")
                self._error_count += 1
                return None

        except Exception as e:
            logger.error(f"Error storing trace: {e}")
            self._error_count += 1
            return None

        finally:
            self._pending_count -= 1

    def _convert_to_stored_trace(
        self,
        execution_trace: Dict[str, Any],
        session_id: str,
        query: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> Optional[StoredTrace]:
        """
        Convert execution_trace dict to StoredTrace model.

        Args:
            execution_trace: Raw execution trace from state
            session_id: Session identifier
            query: Original query
            result: Full result dict for additional context

        Returns:
            StoredTrace model if conversion succeeds
        """
        if not execution_trace:
            return None

        result = result or {}

        try:
            # Extract trace_id (required)
            trace_id = execution_trace.get("trace_id")
            if not trace_id:
                trace_id = f"trace-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"

            # Extract path type
            path_str = execution_trace.get("path") or result.get("path", "standard")
            try:
                path = PathType(path_str)
            except ValueError:
                path = PathType.STANDARD

            # Extract complexity
            complexity = execution_trace.get("complexity") or result.get("complexity", 3)
            if not isinstance(complexity, int):
                try:
                    complexity = int(complexity)
                except (ValueError, TypeError):
                    complexity = 3
            complexity = max(1, min(5, complexity))  # Clamp to 1-5

            # Extract timing
            start_time = execution_trace.get("start_time")
            if isinstance(start_time, str):
                try:
                    start_time = datetime.fromisoformat(start_time)
                except ValueError:
                    start_time = datetime.now()
            elif not isinstance(start_time, datetime):
                start_time = datetime.now()

            end_time = execution_trace.get("end_time")
            if isinstance(end_time, str):
                try:
                    end_time = datetime.fromisoformat(end_time)
                except ValueError:
                    end_time = None
            elif not isinstance(end_time, datetime):
                end_time = None

            # Calculate duration
            total_duration_ms = execution_trace.get("total_elapsed_s")
            if total_duration_ms:
                total_duration_ms = float(total_duration_ms) * 1000
            elif end_time and start_time:
                total_duration_ms = (end_time - start_time).total_seconds() * 1000

            # Extract agent timings
            agent_timings = execution_trace.get("agent_timings", {})
            if not isinstance(agent_timings, dict):
                agent_timings = {}

            # Extract agents visited
            agents_visited = execution_trace.get("agents_visited", [])
            if not agents_visited and agent_timings:
                agents_visited = list(agent_timings.keys())

            # Convert handoff metrics
            handoff_metrics = self._convert_handoff_metrics(
                execution_trace.get("handoff_metrics")
                or result.get("handoff_metrics", [])
            )

            # Extract specialist info
            specialist = result.get("specialist") or execution_trace.get("specialist")
            collaboration_specialists = (
                result.get("collaboration_specialists")
                or execution_trace.get("collaboration_specialists", [])
            )

            # Extract results summary
            separation_results = result.get("separation_results", {})
            tea_results = result.get("tea_results", {})

            solvents_found = []
            if separation_results:
                solvents_found = separation_results.get("solvents", [])

            cost_per_kg = None
            if tea_results:
                cost_per_kg = tea_results.get("cost_per_kg")

            # Determine success
            success = not bool(execution_trace.get("errors"))
            error = None
            if execution_trace.get("errors"):
                error = "; ".join(execution_trace["errors"][:3])

            return StoredTrace(
                trace_id=trace_id,
                session_id=session_id,
                query=query[:500],  # Truncate long queries
                complexity=complexity,
                path=path,
                start_time=start_time,
                end_time=end_time,
                total_duration_ms=total_duration_ms,
                agents_visited=agents_visited,
                agent_timings=agent_timings,
                handoff_metrics=handoff_metrics,
                specialist=specialist,
                collaboration_specialists=collaboration_specialists,
                solvents_found=solvents_found,
                cost_per_kg=cost_per_kg,
                success=success,
                error=error,
            )

        except Exception as e:
            logger.error(f"Error converting trace: {e}")
            return None

    def _convert_handoff_metrics(
        self, metrics: Any
    ) -> List[HandoffDetail]:
        """Convert raw handoff metrics to HandoffDetail models."""
        if not metrics or not isinstance(metrics, list):
            return []

        result = []
        for m in metrics:
            if not isinstance(m, dict):
                continue

            try:
                # Parse timestamp
                timestamp = m.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp = None

                detail = HandoffDetail(
                    handoff_id=m.get("handoff_id", ""),
                    from_agent=m.get("from_agent", "unknown"),
                    to_agent=m.get("to_agent", "unknown"),
                    duration_ms=m.get("duration_ms"),
                    tools_called=m.get("tools_called", []),
                    success=m.get("success", True),
                    error_message=m.get("error_message"),
                    timestamp=timestamp,
                )
                result.append(detail)
            except Exception as e:
                logger.warning(f"Error converting handoff metric: {e}")
                continue

        return result

    def get_stats(self) -> Dict[str, int]:
        """Get collector statistics."""
        return {
            "pending": self._pending_count,
            "stored": self._stored_count,
            "errors": self._error_count,
        }


# Global singleton instance
trace_collector = TraceCollector()
