"""
Metrics Aggregator for Multi-Agent Monitoring

Computes summary statistics from stored traces including:
- Success rates overall and by specialist
- Duration percentiles (p50, p95, p99)
- Path and complexity distributions
- Agent usage statistics
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict

from .models import StoredTrace, MetricsSummary
from .event_store import event_store

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Aggregates metrics from stored traces.

    Computes statistics lazily on demand rather than maintaining
    running statistics, since trace volume is expected to be moderate.
    """

    def __init__(self):
        """Initialize the metrics aggregator."""
        pass

    def compute_summary(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> MetricsSummary:
        """
        Compute a metrics summary for the specified time window.

        Args:
            since: Start of time window (None = all time)
            until: End of time window (None = now)

        Returns:
            MetricsSummary with computed statistics
        """
        # Get all active traces
        all_traces = event_store.get_all_active()

        # Apply time window filter
        traces = []
        for t in all_traces:
            if since and t.start_time < since:
                continue
            if until and t.start_time > until:
                continue
            traces.append(t)

        if not traces:
            return MetricsSummary(
                total_traces=len(all_traces),
                active_traces=len(traces),
                window_start=since,
                window_end=until,
            )

        # Compute statistics
        return MetricsSummary(
            total_traces=len(all_traces),
            active_traces=len(traces),
            success_rate=self._compute_success_rate(traces),
            specialist_success_rates=self._compute_specialist_success_rates(traces),
            avg_duration_ms=self._compute_avg_duration(traces),
            p50_duration_ms=self._compute_percentile(traces, 50),
            p95_duration_ms=self._compute_percentile(traces, 95),
            p99_duration_ms=self._compute_percentile(traces, 99),
            path_distribution=self._compute_path_distribution(traces),
            complexity_distribution=self._compute_complexity_distribution(traces),
            agent_usage=self._compute_agent_usage(traces),
            avg_handoffs_per_trace=self._compute_avg_handoffs(traces),
            window_start=since,
            window_end=until,
            computed_at=datetime.now(),
        )

    def _compute_success_rate(self, traces: List[StoredTrace]) -> float:
        """Compute overall success rate."""
        if not traces:
            return 0.0
        successful = sum(1 for t in traces if t.success)
        return successful / len(traces)

    def _compute_specialist_success_rates(
        self, traces: List[StoredTrace]
    ) -> Dict[str, float]:
        """Compute success rate by specialist."""
        specialist_counts: Dict[str, Dict[str, int]] = {}

        for t in traces:
            specialist = t.specialist or "none"
            if specialist not in specialist_counts:
                specialist_counts[specialist] = {"total": 0, "success": 0}
            specialist_counts[specialist]["total"] += 1
            if t.success:
                specialist_counts[specialist]["success"] += 1

        return {
            s: c["success"] / c["total"] if c["total"] > 0 else 0.0
            for s, c in specialist_counts.items()
        }

    def _compute_avg_duration(self, traces: List[StoredTrace]) -> float:
        """Compute average duration."""
        durations = [t.total_duration_ms for t in traces if t.total_duration_ms]
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    def _compute_percentile(
        self, traces: List[StoredTrace], percentile: float
    ) -> float:
        """Compute duration percentile."""
        durations = sorted(
            [t.total_duration_ms for t in traces if t.total_duration_ms]
        )
        if not durations:
            return 0.0

        # Calculate index
        idx = (percentile / 100) * (len(durations) - 1)
        lower_idx = int(idx)
        upper_idx = min(lower_idx + 1, len(durations) - 1)

        # Linear interpolation
        if lower_idx == upper_idx:
            return durations[lower_idx]

        fraction = idx - lower_idx
        return durations[lower_idx] + fraction * (
            durations[upper_idx] - durations[lower_idx]
        )

    def _compute_path_distribution(
        self, traces: List[StoredTrace]
    ) -> Dict[str, int]:
        """Compute distribution of routing paths."""
        distribution: Dict[str, int] = {}
        for t in traces:
            path = t.path.value if hasattr(t.path, "value") else str(t.path)
            distribution[path] = distribution.get(path, 0) + 1
        return distribution

    def _compute_complexity_distribution(
        self, traces: List[StoredTrace]
    ) -> Dict[int, int]:
        """Compute distribution of complexity scores."""
        distribution: Dict[int, int] = {}
        for t in traces:
            distribution[t.complexity] = distribution.get(t.complexity, 0) + 1
        return distribution

    def _compute_agent_usage(self, traces: List[StoredTrace]) -> Dict[str, int]:
        """Compute agent usage counts."""
        usage: Dict[str, int] = {}
        for t in traces:
            for agent in t.agents_visited:
                usage[agent] = usage.get(agent, 0) + 1
        return usage

    def _compute_avg_handoffs(self, traces: List[StoredTrace]) -> float:
        """Compute average handoffs per trace."""
        if not traces:
            return 0.0
        total_handoffs = sum(len(t.handoff_metrics) for t in traces)
        return total_handoffs / len(traces)


# Global singleton instance
_aggregator = MetricsAggregator()


def get_metrics_summary(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> MetricsSummary:
    """
    Get a metrics summary for the specified time window.

    Convenience function wrapping the MetricsAggregator singleton.

    Args:
        since: Start of time window (None = all time)
        until: End of time window (None = now)

    Returns:
        MetricsSummary with computed statistics
    """
    return _aggregator.compute_summary(since=since, until=until)
