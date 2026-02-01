"""
Monitoring Infrastructure for DISSOLVE Multi-Agent System

This module provides observability and trace collection for the multi-agent
workflow, leveraging the existing P3 tracing infrastructure.

Components:
    - models: Pydantic models for traces and metrics
    - event_store: In-memory trace storage with TTL
    - trace_collector: Non-blocking trace collection
    - metrics_aggregator: Summary statistics computation

Usage:
    from monitoring import trace_collector, event_store, get_metrics_summary

    # Store a trace (non-blocking)
    await trace_collector.store_async(execution_trace, session_id, query)

    # Get recent traces
    traces = event_store.get_recent(limit=10)

    # Get metrics summary
    summary = get_metrics_summary()
"""

from .models import (
    StoredTrace,
    MetricsSummary,
    AgentTiming,
    HandoffDetail,
    TraceQuery,
)
from .event_store import EventStore, event_store
from .trace_collector import TraceCollector, trace_collector
from .metrics_aggregator import MetricsAggregator, get_metrics_summary

__all__ = [
    # Models
    "StoredTrace",
    "MetricsSummary",
    "AgentTiming",
    "HandoffDetail",
    "TraceQuery",
    # Event Store
    "EventStore",
    "event_store",
    # Trace Collector
    "TraceCollector",
    "trace_collector",
    # Metrics
    "MetricsAggregator",
    "get_metrics_summary",
]
