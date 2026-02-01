"""
Pydantic Models for Monitoring and Observability

These models define the structure for stored traces, metrics summaries,
and query parameters for the monitoring API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


class PathType(str, Enum):
    """Routing path types in the multi-agent system."""
    FAST = "fast"
    STANDARD = "standard"
    SPECIALIST = "specialist"
    INTEGRATED = "integrated"


class HandoffDetail(BaseModel):
    """Details of a single handoff between agents."""
    handoff_id: str = Field(description="Unique handoff identifier")
    from_agent: str = Field(description="Source agent name")
    to_agent: str = Field(description="Target agent name")
    duration_ms: Optional[float] = Field(default=None, description="Time in source agent (ms)")
    tools_called: List[str] = Field(default_factory=list, description="Tools invoked")
    success: bool = Field(default=True, description="Whether handoff succeeded")
    error_message: Optional[str] = Field(default=None)
    timestamp: Optional[datetime] = Field(default=None)

    class Config:
        extra = "allow"


class AgentTiming(BaseModel):
    """Timing information for a single agent's execution."""
    agent_name: str = Field(description="Name of the agent")
    start_time: Optional[float] = Field(default=None, description="Start timestamp (epoch)")
    end_time: Optional[float] = Field(default=None, description="End timestamp (epoch)")
    duration_ms: Optional[float] = Field(default=None, description="Execution duration (ms)")
    tools_called: int = Field(default=0, description="Number of tools invoked")

    class Config:
        extra = "allow"


class StoredTrace(BaseModel):
    """
    Complete trace record stored in the event store.

    This consolidates execution_trace data from MultiAgentState with
    additional metadata for storage and retrieval.
    """
    # Core identifiers
    trace_id: str = Field(description="Unique trace identifier")
    session_id: str = Field(description="Associated session ID")

    # Query context
    query: str = Field(description="Original user query")
    complexity: int = Field(default=3, ge=1, le=5, description="Complexity score (1-5)")
    path: PathType = Field(default=PathType.STANDARD, description="Routing path taken")

    # Timing
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(default=None)
    total_duration_ms: Optional[float] = Field(default=None)

    # Agent execution details
    agents_visited: List[str] = Field(default_factory=list)
    agent_timings: Dict[str, float] = Field(default_factory=dict)
    handoff_metrics: List[HandoffDetail] = Field(default_factory=list)

    # Results summary
    specialist: Optional[str] = Field(default=None, description="Primary specialist used")
    collaboration_specialists: List[str] = Field(default_factory=list)
    solvents_found: List[str] = Field(default_factory=list)
    cost_per_kg: Optional[float] = Field(default=None)
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)

    # Storage metadata
    stored_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(default=None)

    class Config:
        extra = "allow"

    def is_expired(self) -> bool:
        """Check if this trace has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class MetricsSummary(BaseModel):
    """
    Aggregated metrics summary for the monitoring dashboard.

    Computed from stored traces in the event store.
    """
    # Counts
    total_traces: int = Field(default=0, description="Total traces in store")
    active_traces: int = Field(default=0, description="Non-expired traces")

    # Success rates
    success_rate: float = Field(default=0.0, ge=0, le=1, description="Overall success rate")
    specialist_success_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="Success rate by specialist"
    )

    # Timing statistics
    avg_duration_ms: float = Field(default=0.0, description="Average trace duration")
    p50_duration_ms: float = Field(default=0.0, description="Median duration")
    p95_duration_ms: float = Field(default=0.0, description="95th percentile duration")
    p99_duration_ms: float = Field(default=0.0, description="99th percentile duration")

    # Path distribution
    path_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of traces by path type"
    )
    complexity_distribution: Dict[int, int] = Field(
        default_factory=dict,
        description="Count of traces by complexity score"
    )

    # Agent statistics
    agent_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of times each agent was used"
    )
    avg_handoffs_per_trace: float = Field(default=0.0)

    # Time window
    window_start: Optional[datetime] = Field(default=None)
    window_end: Optional[datetime] = Field(default=None)
    computed_at: datetime = Field(default_factory=datetime.now)

    class Config:
        extra = "allow"


class TraceQuery(BaseModel):
    """Query parameters for trace retrieval."""
    limit: int = Field(default=50, ge=1, le=500, description="Max traces to return")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    session_id: Optional[str] = Field(default=None, description="Filter by session")
    path: Optional[PathType] = Field(default=None, description="Filter by path type")
    min_complexity: Optional[int] = Field(default=None, ge=1, le=5)
    max_complexity: Optional[int] = Field(default=None, ge=1, le=5)
    success_only: bool = Field(default=False, description="Only successful traces")
    since: Optional[datetime] = Field(default=None, description="Traces after this time")
    until: Optional[datetime] = Field(default=None, description="Traces before this time")

    class Config:
        extra = "allow"


class TraceListResponse(BaseModel):
    """Response for trace list API."""
    traces: List[StoredTrace]
    total: int = Field(description="Total matching traces (before pagination)")
    limit: int
    offset: int
    has_more: bool

    class Config:
        extra = "allow"
