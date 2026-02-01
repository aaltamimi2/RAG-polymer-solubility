"""
Pydantic Schemas for Multi-Agent Communication

These schemas replace regex-based parsing with structured, validated data
for reliable inter-agent communication in the DISSOLVE system.

Usage:
    from agent_schemas import SeparationResult, TEAResult, HandoffPayload

    # Parse structured results
    sep_result = SeparationResult(
        sequences=[["LDPE", "PET", "PP"]],
        solvents=["xylene", "dmf"],
        polymers=["LDPE", "PET", "PP"],
        temperature=80.0
    )

    # Create handoff payload
    handoff = HandoffPayload(
        from_agent="separation",
        to_agent="tea_lca",
        query_summary="3-polymer separation planning",
        structured_results=sep_result.model_dump()
    )
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum


class AgentType(str, Enum):
    """Available agent types in the DISSOLVE system."""
    SEPARATION = "separation"
    TEA_LCA = "tea_lca"
    LITERATURE = "literature"
    AGGREGATOR = "smart_aggregator"
    ROUTER = "router"


class SeparationStep(BaseModel):
    """A single step in a separation sequence."""
    step_number: int = Field(ge=1, description="Step number in sequence")
    target_polymer: str = Field(description="Polymer being separated")
    remaining_polymers: List[str] = Field(default_factory=list, description="Polymers remaining after this step")
    solvent: str = Field(description="Solvent used for separation")
    selectivity: float = Field(default=0.0, description="Selectivity percentage")
    temperature: float = Field(default=80.0, ge=0, le=300, description="Temperature in Celsius")


class SeparationResult(BaseModel):
    """Structured output from the Separation Agent.

    This replaces regex-based parsing of separation tool outputs.
    """
    # Core separation data
    sequences: List[List[str]] = Field(
        default_factory=list,
        description="Polymer separation sequences (ordered lists)"
    )
    solvents: List[str] = Field(
        default_factory=list,
        description="Unique solvents identified for separation"
    )
    selectivities: List[float] = Field(
        default_factory=list,
        description="Selectivity values (0-100 scale)"
    )

    # Context
    polymers: List[str] = Field(
        default_factory=list,
        description="Input polymer list"
    )
    temperature: float = Field(
        default=80.0,
        ge=0, le=300,
        description="Processing temperature in Celsius"
    )

    # Best results (for quick access)
    best_sequence: Optional[List[str]] = Field(
        default=None,
        description="Optimal separation sequence"
    )
    best_solvent: Optional[str] = Field(
        default=None,
        description="Most recommended solvent"
    )

    # Greedy algorithm results (for >6 polymers)
    greedy_steps: Optional[List[SeparationStep]] = Field(
        default=None,
        description="Step-by-step greedy separation plan"
    )
    algorithm_used: Optional[str] = Field(
        default=None,
        description="Algorithm used: 'exhaustive' or 'greedy'"
    )

    # Metadata
    raw_response: Optional[str] = Field(
        default=None,
        description="Original tool output for fallback parsing"
    )

    class Config:
        extra = "allow"  # Allow extra fields for flexibility


class TEAResult(BaseModel):
    """Structured output from the TEA/LCA Agent.

    Contains techno-economic analysis results for solvent recovery.
    """
    # Core economic data
    msp_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Minimum Selling Price per solvent ($/kg)"
    )
    best_solvent: Optional[str] = Field(
        default=None,
        description="Most cost-effective solvent"
    )
    cost_per_kg: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total cost per kg polymer processed"
    )

    # CAPEX/OPEX
    total_capex: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total capital expenditure ($)"
    )
    total_opex: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual operating expenditure ($/yr)"
    )

    # Financial metrics
    payback_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="Payback period in years"
    )
    npv: Optional[float] = Field(
        default=None,
        description="Net Present Value ($)"
    )
    irr: Optional[float] = Field(
        default=None,
        description="Internal Rate of Return (%)"
    )

    # LCA data
    co2_emissions: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 emissions (kg CO2/kg polymer)"
    )
    energy_consumption: Optional[float] = Field(
        default=None,
        ge=0,
        description="Energy consumption (MJ/kg polymer)"
    )
    gwp: Optional[float] = Field(
        default=None,
        description="Global Warming Potential"
    )

    # Breakdown
    cost_breakdown: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed cost breakdown"
    )
    solvents_analyzed: List[str] = Field(
        default_factory=list,
        description="List of solvents evaluated"
    )

    # Process parameters
    throughput_kg_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Process throughput (kg/hr)"
    )
    recovery_rate: Optional[float] = Field(
        default=None,
        ge=0, le=1,
        description="Solvent recovery rate (0-1)"
    )

    # Metadata
    raw_response: Optional[str] = Field(
        default=None,
        description="Original tool output for fallback"
    )

    class Config:
        extra = "allow"


class LiteratureResult(BaseModel):
    """Structured output from the Literature Agent."""
    papers_found: int = Field(default=0, ge=0)
    key_findings: List[str] = Field(default_factory=list)
    citations: List[Dict[str, str]] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)
    raw_response: Optional[str] = Field(default=None)

    class Config:
        extra = "allow"


# ============================================================
# P2: Task-Oriented Handoff Schemas
# ============================================================

class TEATaskRequest(BaseModel):
    """
    Task-oriented request for TEA analysis.

    Instead of passing full message history, this schema defines
    exactly what the TEA agent needs to perform its analysis.
    This reduces context bloat significantly.
    """
    # Required inputs
    solvents: List[str] = Field(
        description="List of solvents to analyze (from separation results)"
    )
    throughput_kg_hr: float = Field(
        default=100.0,
        ge=0,
        description="Process throughput in kg/hr"
    )
    recovery_rate: float = Field(
        default=0.95,
        ge=0, le=1,
        description="Solvent recovery rate (0-1)"
    )

    # Optional parameters
    include_capex: bool = Field(default=True)
    include_lca: bool = Field(default=False)
    compare_solvents: bool = Field(default=True)

    # Context from separation (minimal)
    polymers: List[str] = Field(default_factory=list)
    temperature: float = Field(default=80.0)
    best_sequence: Optional[List[str]] = Field(default=None)

    def to_instruction(self) -> str:
        """Convert task request to agent instruction."""
        solvents_str = ", ".join(self.solvents[:5])
        return f"""
**TEA ANALYSIS TASK**

Analyze techno-economics for: {solvents_str}
- Throughput: {self.throughput_kg_hr} kg/hr
- Recovery rate: {self.recovery_rate * 100}%
- Include CAPEX: {self.include_capex}
- Include LCA: {self.include_lca}

For each solvent, call analyze_solvent_recovery_tea and report:
1. Cost per kg polymer processed
2. CAPEX and OPEX
3. Payback period

{f'Compare solvents using compare_solvents_tea_lca.' if self.compare_solvents else ''}
Recommend the most cost-effective option.
"""

    class Config:
        extra = "allow"


class SeparationTaskRequest(BaseModel):
    """
    Task-oriented request for separation planning.

    Defines exactly what the separation agent needs.
    """
    polymers: List[str] = Field(
        description="List of polymers to separate"
    )
    temperature: float = Field(
        default=80.0,
        ge=0, le=300,
        description="Processing temperature in Celsius"
    )
    top_k_solvents: int = Field(
        default=3,
        ge=1, le=10,
        description="Number of top solvents per step"
    )
    ranking_criterion: str = Field(
        default="selectivity",
        description="Criterion for ranking: selectivity, cost, safety"
    )

    def to_instruction(self) -> str:
        """Convert task request to agent instruction."""
        polymers_str = ", ".join(self.polymers)
        return f"""
**SEPARATION PLANNING TASK**

Plan separation sequence for: {polymers_str}
- Temperature: {self.temperature}°C
- Top solvents per step: {self.top_k_solvents}
- Ranking by: {self.ranking_criterion}

Call plan_sequential_separation with:
- polymers: "{','.join(self.polymers)}"
- temperature: {self.temperature}
- top_k_solvents: {self.top_k_solvents}

Report the optimal separation sequence and solvents.
"""

    class Config:
        extra = "allow"


class LiteratureTaskRequest(BaseModel):
    """Task-oriented request for literature search."""
    search_topic: str = Field(description="Topic to search for")
    polymers: List[str] = Field(default_factory=list)
    solvents: List[str] = Field(default_factory=list)
    max_results: int = Field(default=10, ge=1, le=50)
    search_rag_first: bool = Field(default=True)

    def to_instruction(self) -> str:
        """Convert task request to agent instruction."""
        context = []
        if self.polymers:
            context.append(f"Polymers: {', '.join(self.polymers)}")
        if self.solvents:
            context.append(f"Solvents: {', '.join(self.solvents)}")

        return f"""
**LITERATURE SEARCH TASK**

Search for: {self.search_topic}
{chr(10).join(context) if context else ''}

{'Search internal RAG first with search_literature_rag.' if self.search_rag_first else ''}
Max results: {self.max_results}

Report key findings with citations.
"""

    class Config:
        extra = "allow"


class AggregatorTaskRequest(BaseModel):
    """Task-oriented request for result aggregation."""
    separation_summary: Optional[str] = Field(default=None)
    tea_summary: Optional[str] = Field(default=None)
    literature_summary: Optional[str] = Field(default=None)

    # Key metrics to highlight
    best_solvent: Optional[str] = Field(default=None)
    best_sequence: Optional[List[str]] = Field(default=None)
    cost_per_kg: Optional[float] = Field(default=None)

    # Original query for context
    original_query: Optional[str] = Field(default=None)

    class Config:
        extra = "allow"


class HandoffPayload(BaseModel):
    """Payload for inter-agent handoffs.

    Used with LangGraph Command objects to pass structured context
    between agents during collaborative workflows.
    """
    from_agent: str = Field(description="Source agent name")
    to_agent: str = Field(description="Target agent name")
    query_summary: str = Field(description="Brief summary of the task")

    # Structured results from source agent
    structured_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validated results from source agent"
    )

    # Keys to include from shared_context
    context_keys: List[str] = Field(
        default_factory=list,
        description="Which shared_context keys are relevant"
    )

    # Task-specific parameters for target agent
    task_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Parameters specific to the target task"
    )

    # Timestamp for tracking
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When handoff was created"
    )

    class Config:
        extra = "allow"


class SharedContext(BaseModel):
    """Shared context passed between all agents in a collaboration."""
    original_query: str = Field(description="Original user query")
    polymers: List[str] = Field(default_factory=list)
    temperature: float = Field(default=80.0)
    throughput_kg_hr: float = Field(default=100.0)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Optional parameters
    recovery_rate: float = Field(default=0.95, ge=0, le=1)
    ranking_criterion: Optional[str] = Field(default="selectivity")

    class Config:
        extra = "allow"


class MultiAgentExecutionState(BaseModel):
    """Tracks execution state across multi-agent collaboration.

    This provides structured tracking instead of ad-hoc state fields.
    """
    # Execution tracking
    active_agent: Optional[str] = Field(default=None)
    handoff_history: List[HandoffPayload] = Field(default_factory=list)
    pending_handoff: Optional[HandoffPayload] = Field(default=None)

    # Iteration control
    iteration_count: int = Field(default=0, ge=0)
    max_iterations: int = Field(default=15, ge=1)

    # Timing
    start_time: Optional[datetime] = Field(default=None)
    agent_start_times: Dict[str, datetime] = Field(default_factory=dict)

    # Results storage
    separation_result: Optional[SeparationResult] = Field(default=None)
    tea_result: Optional[TEAResult] = Field(default=None)
    literature_result: Optional[LiteratureResult] = Field(default=None)

    class Config:
        extra = "allow"


# ============================================================
# P3: Enhanced Handoff Tracking and Supervisor Schemas
# ============================================================

class HandoffMetrics(BaseModel):
    """
    Enhanced metrics for a single handoff event.

    P3: Provides detailed tracking for debugging and performance analysis.
    """
    handoff_id: str = Field(description="Unique ID for this handoff")
    from_agent: str = Field(description="Source agent")
    to_agent: str = Field(description="Target agent")

    # Timing
    timestamp: datetime = Field(default_factory=datetime.now)
    duration_ms: Optional[float] = Field(default=None, description="Time spent in source agent (ms)")

    # Data transfer metrics
    input_tokens_estimate: Optional[int] = Field(default=None, description="Estimated input tokens")
    output_tokens_estimate: Optional[int] = Field(default=None, description="Estimated output tokens")
    context_size_bytes: Optional[int] = Field(default=None, description="Size of context passed")

    # Result quality indicators
    tools_called: List[str] = Field(default_factory=list)
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)

    # Task context (P2 task request summary)
    task_type: Optional[str] = Field(default=None, description="e.g., 'separation', 'tea', 'literature'")
    task_summary: Optional[str] = Field(default=None, description="Brief summary of the task")

    class Config:
        extra = "allow"


class ExecutionTrace(BaseModel):
    """
    Complete execution trace for a multi-agent collaboration.

    P3: Enables comprehensive debugging and performance analysis.
    """
    trace_id: str = Field(description="Unique ID for this execution")
    query: str = Field(description="Original user query")
    complexity: int = Field(default=0, ge=0, le=5)
    path: str = Field(default="standard")

    # Timing
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(default=None)
    total_duration_ms: Optional[float] = Field(default=None)

    # Agent sequence
    agents_visited: List[str] = Field(default_factory=list)
    handoffs: List[HandoffMetrics] = Field(default_factory=list)

    # Results
    separation_found: bool = Field(default=False)
    tea_completed: bool = Field(default=False)
    solvents_count: int = Field(default=0)
    cost_per_kg: Optional[float] = Field(default=None)

    # Quality metrics
    total_tool_calls: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)

    def add_handoff(self, metrics: HandoffMetrics):
        """Add a handoff to the trace."""
        self.handoffs.append(metrics)
        if metrics.to_agent not in self.agents_visited:
            self.agents_visited.append(metrics.to_agent)
        self.total_tool_calls += len(metrics.tools_called)
        if not metrics.success and metrics.error_message:
            self.errors.append(metrics.error_message)

    def finalize(self):
        """Finalize the trace with end time and duration."""
        self.end_time = datetime.now()
        if self.start_time:
            delta = self.end_time - self.start_time
            self.total_duration_ms = delta.total_seconds() * 1000

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        duration_str = f"{self.total_duration_ms:.0f}ms" if self.total_duration_ms else "N/A"
        return f"""
Execution Trace: {self.trace_id}
  Query: {self.query[:50]}...
  Complexity: {self.complexity}, Path: {self.path}
  Duration: {duration_str}
  Agents: {' → '.join(self.agents_visited)}
  Handoffs: {len(self.handoffs)}
  Tool calls: {self.total_tool_calls}
  Solvents: {self.solvents_count}, Cost: ${self.cost_per_kg or 'N/A'}/kg
  Errors: {len(self.errors)}
"""

    class Config:
        extra = "allow"


class ReviewerFeedback(BaseModel):
    """
    Feedback from a reviewer agent on separation or TEA results.

    P0 Enhancement: Enables review/revision loops for quality improvement.
    Based on GPT-Researcher pattern.
    """
    is_acceptable: bool = Field(
        default=True,
        description="Whether results meet quality thresholds"
    )
    quality_score: float = Field(
        default=1.0,
        ge=0, le=1,
        description="Overall quality score (0-1)"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Specific issues identified"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )
    requires_revision: bool = Field(
        default=False,
        description="True if results need revision before proceeding"
    )
    revision_instructions: Optional[str] = Field(
        default=None,
        description="Specific instructions for revision"
    )

    # Validation metrics for separation
    solvents_count: int = Field(default=0, ge=0)
    min_selectivity: Optional[float] = Field(default=None)
    max_selectivity: Optional[float] = Field(default=None)
    has_sequence: bool = Field(default=False)

    # Retry configuration
    retry_count: int = Field(default=0, ge=0)  # No upper limit - clamped in logic
    max_retries: int = Field(default=2, ge=0)
    retry_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Modified parameters for retry (e.g., wider temperature range)"
    )

    class Config:
        extra = "allow"


class SupervisorDecision(BaseModel):
    """
    Decision made by the supervisor for dynamic routing.

    P3: Enables supervisor to make intelligent routing decisions.
    """
    next_agent: str = Field(description="Next agent to route to")
    reason: str = Field(description="Reasoning for this decision")

    # Optional: modify the task
    modified_task: Optional[Dict[str, Any]] = Field(default=None)

    # Re-routing indicators
    is_reroute: bool = Field(default=False, description="True if changing from planned route")
    original_plan: Optional[List[str]] = Field(default=None)
    remaining_agents: Optional[List[str]] = Field(default=None)

    # Confidence
    confidence: float = Field(default=1.0, ge=0, le=1)

    class Config:
        extra = "allow"


# ============================================================
# Helper Functions for Schema Parsing
# ============================================================

def parse_to_separation_result(
    raw_data: Union[Dict[str, Any], str],
    polymers: Optional[List[str]] = None,
    temperature: float = 80.0
) -> SeparationResult:
    """
    Parse raw separation data into SeparationResult.

    Handles both dict and string inputs with fallback to regex parsing.
    """
    if isinstance(raw_data, str):
        # Import parse function from multi_agent_system for fallback
        from multi_agent_system import parse_separation_results
        parsed = parse_separation_results(raw_data)
        return SeparationResult(
            **parsed,
            polymers=polymers or [],
            temperature=temperature
        )
    elif isinstance(raw_data, dict):
        # Add defaults for missing fields
        raw_data.setdefault("polymers", polymers or [])
        raw_data.setdefault("temperature", temperature)
        return SeparationResult(**raw_data)
    else:
        return SeparationResult(polymers=polymers or [], temperature=temperature)


def parse_to_tea_result(
    raw_data: Union[Dict[str, Any], str],
    solvents_analyzed: Optional[List[str]] = None
) -> TEAResult:
    """
    Parse raw TEA data into TEAResult.

    Handles both dict and string inputs with fallback to regex parsing.
    """
    if isinstance(raw_data, str):
        from multi_agent_system import parse_tea_results
        parsed = parse_tea_results(raw_data)
        if solvents_analyzed:
            parsed["solvents_analyzed"] = solvents_analyzed
        return TEAResult(**parsed)
    elif isinstance(raw_data, dict):
        if solvents_analyzed:
            raw_data.setdefault("solvents_analyzed", solvents_analyzed)
        return TEAResult(**raw_data)
    else:
        return TEAResult(solvents_analyzed=solvents_analyzed or [])


# ============================================================
# Module Exports
# ============================================================

__all__ = [
    # Enums
    "AgentType",
    # Core schemas
    "SeparationStep",
    "SeparationResult",
    "TEAResult",
    "LiteratureResult",
    # P2: Task-oriented handoff schemas
    "TEATaskRequest",
    "SeparationTaskRequest",
    "LiteratureTaskRequest",
    "AggregatorTaskRequest",
    # Handoff schemas
    "HandoffPayload",
    "SharedContext",
    "MultiAgentExecutionState",
    # P3: Enhanced tracking and supervisor schemas
    "HandoffMetrics",
    "ExecutionTrace",
    "SupervisorDecision",
    # P0 Enhancement: Review/Revision loop
    "ReviewerFeedback",
    # Helper functions
    "parse_to_separation_result",
    "parse_to_tea_result",
]
