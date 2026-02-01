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
    """Structured output from the Literature Agent.

    Enhanced for multi-knowledgebase support and collaboration.
    """
    papers_found: int = Field(default=0, ge=0)
    key_findings: List[str] = Field(default_factory=list)
    citations: List[Dict[str, str]] = Field(default_factory=list)
    knowledge_gaps: List[str] = Field(default_factory=list)

    # Multi-KB support
    knowledgebases_searched: List[str] = Field(
        default_factory=list,
        description="List of knowledgebases queried"
    )
    external_sources_used: List[str] = Field(
        default_factory=list,
        description="External sources used (google_scholar, wos, patents)"
    )
    confidence_score: float = Field(
        default=0.0, ge=0, le=1,
        description="Overall confidence in results (0-1)"
    )
    suggested_refinements: List[str] = Field(
        default_factory=list,
        description="Suggested query refinements if low confidence"
    )

    # Context for collaboration with other agents
    polymers_mentioned: List[str] = Field(
        default_factory=list,
        description="Polymers mentioned in literature findings"
    )
    solvents_mentioned: List[str] = Field(
        default_factory=list,
        description="Solvents mentioned in literature findings"
    )
    temperatures_mentioned: List[float] = Field(
        default_factory=list,
        description="Temperatures mentioned in literature (°C)"
    )

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
        solvents_list = self.solvents[:5]
        solvents_str = ", ".join(solvents_list)

        # Build explicit tool call instructions
        tool_calls = "\n".join([
            f"  - analyze_solvent_recovery_tea(solvent='{s}', throughput_kg_hr={self.throughput_kg_hr})"
            for s in solvents_list[:3]  # Show first 3 as examples
        ])

        return f"""
**REQUIRED: TECHNO-ECONOMIC ANALYSIS**

You MUST call the analyze_solvent_recovery_tea tool for these solvents: {solvents_str}

**STEP 1 - REQUIRED TOOL CALLS:**
{tool_calls}

**Parameters:**
- Throughput: {self.throughput_kg_hr} kg/hr
- Recovery rate: {self.recovery_rate * 100}%

**STEP 2 - EXTRACT FROM EACH TOOL RESULT:**
- Cost per kg polymer (look for "Cost per kg polymer: $X.XX/kg")
- Total CAPEX and annual OPEX
- Payback period in years

{f'**STEP 3:** Call compare_solvents_tea_lca to compare all solvents.' if self.compare_solvents and len(solvents_list) > 1 else ''}

**IMPORTANT:** You must call the tools above. Do NOT just describe what you would do.
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
    """Task-oriented request for literature search.

    Supports multi-knowledgebase search with auto-selection.
    """
    search_topic: str = Field(description="Topic to search for")
    polymers: List[str] = Field(default_factory=list)
    solvents: List[str] = Field(default_factory=list)
    max_results: int = Field(default=10, ge=1, le=50)
    search_rag_first: bool = Field(default=True)

    # Multi-KB support
    knowledgebases: List[str] = Field(
        default_factory=list,
        description="Specific KBs to search (empty = auto-select)"
    )
    include_external: bool = Field(
        default=False,
        description="Include Google Scholar/WoS if RAG insufficient"
    )

    # Collaboration context
    upstream_agent: Optional[str] = Field(
        default=None,
        description="Which agent requested this search (separation, tea)"
    )

    def to_instruction(self) -> str:
        """Convert task request to agent instruction."""
        context = []
        if self.polymers:
            context.append(f"Polymers of interest: {', '.join(self.polymers)}")
        if self.solvents:
            context.append(f"Solvents of interest: {', '.join(self.solvents)}")

        kb_instruction = ""
        if self.knowledgebases:
            kb_instruction = f"Search these knowledgebases: {', '.join(self.knowledgebases)}"
        else:
            kb_instruction = "Auto-select relevant knowledgebases based on query content."

        external_instruction = ""
        if self.include_external:
            external_instruction = "If RAG results are insufficient, also search Google Scholar."

        return f"""
**LITERATURE SEARCH TASK**

**Topic:** {self.search_topic}

**Context:**
{chr(10).join(context) if context else 'No specific polymers/solvents specified.'}

**Search Strategy:**
1. {kb_instruction}
2. Use search_literature_rag or ask_literature tool for each KB.
3. {external_instruction if external_instruction else 'Focus on internal RAG sources.'}

**Requirements:**
- Find up to {self.max_results} relevant papers/sections
- Extract key findings related to the topic
- Include citations with source and page numbers
- Note any polymers, solvents, or temperatures mentioned

**Output Format:**
Provide a structured summary with:
1. Key findings (bullet points)
2. Relevant citations
3. Any knowledge gaps identified
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
# P3: TOOL OUTPUT SCHEMAS
# ============================================================

class ToolOutputBase(BaseModel):
    """Base class for all tool outputs with common metadata."""
    tool_name: str = Field(description="Name of the tool that generated this output")
    success: bool = Field(default=True, description="Whether tool execution succeeded")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_ms: Optional[float] = Field(default=None, description="Tool execution time")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Confidence in results")

    class Config:
        extra = "allow"


class SeparationToolOutput(ToolOutputBase):
    """Validated output from separation analysis tools."""
    tool_name: str = Field(default="separation_tool")

    # Core results
    solvents: List[str] = Field(default_factory=list, description="Identified solvents")
    selectivities: List[float] = Field(default_factory=list, description="Selectivity values")
    sequences: List[List[str]] = Field(default_factory=list, description="Separation sequences")
    best_sequence: Optional[List[str]] = Field(default=None)
    best_solvent: Optional[str] = Field(default=None)

    # Algorithm metadata
    algorithm_used: str = Field(default="unknown", description="Algorithm: greedy, dp, exhaustive")
    polymers_analyzed: List[str] = Field(default_factory=list)
    temperature: float = Field(default=80.0)

    # Quality indicators
    min_selectivity: Optional[float] = Field(default=None)
    max_selectivity: Optional[float] = Field(default=None)
    coverage_complete: bool = Field(default=False, description="All polymers can be separated")


class TEAToolOutput(ToolOutputBase):
    """Validated output from TEA analysis tools."""
    tool_name: str = Field(default="tea_tool")

    # Core economic results
    solvent: str = Field(description="Solvent analyzed")
    cost_per_kg: Optional[float] = Field(default=None, ge=0)
    total_capex: Optional[float] = Field(default=None, ge=0)
    annual_opex: Optional[float] = Field(default=None, ge=0)
    payback_years: Optional[float] = Field(default=None, ge=0)

    # Process parameters
    throughput_kg_hr: float = Field(default=100.0, ge=0)
    recovery_rate: float = Field(default=0.95, ge=0, le=1)

    # LCA data (optional)
    co2_kg_per_kg: Optional[float] = Field(default=None)
    energy_mj_per_kg: Optional[float] = Field(default=None)

    # Breakdown
    cost_breakdown: Dict[str, float] = Field(default_factory=dict)


class LiteratureToolOutput(ToolOutputBase):
    """Validated output from literature search tools."""
    tool_name: str = Field(default="literature_tool")

    # Search results
    papers_found: int = Field(default=0, ge=0)
    relevant_excerpts: List[str] = Field(default_factory=list)
    citations: List[Dict[str, str]] = Field(default_factory=list)

    # Source tracking
    knowledgebase_used: str = Field(default="unknown")
    query_used: str = Field(default="")

    # Extracted data
    polymers_mentioned: List[str] = Field(default_factory=list)
    solvents_mentioned: List[str] = Field(default_factory=list)
    temperatures_mentioned: List[float] = Field(default_factory=list)

    # Quality
    relevance_score: float = Field(default=0.0, ge=0, le=1)


class ComparisonToolOutput(ToolOutputBase):
    """Validated output from comparison tools."""
    tool_name: str = Field(default="comparison_tool")

    # Comparison results
    items_compared: List[str] = Field(default_factory=list)
    ranking: List[str] = Field(default_factory=list, description="Items in rank order")
    scores: Dict[str, float] = Field(default_factory=dict)
    best_item: Optional[str] = Field(default=None)

    # Comparison criteria
    criteria_used: List[str] = Field(default_factory=list)
    weights: Dict[str, float] = Field(default_factory=dict)


# ============================================================
# P3: HANDOFF VALIDATION
# ============================================================

class HandoffValidationResult(BaseModel):
    """Result of validating a handoff payload."""
    is_valid: bool = Field(default=True)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_payload: Optional[Dict[str, Any]] = Field(default=None)

    @classmethod
    def success(cls, payload: Dict[str, Any]) -> "HandoffValidationResult":
        return cls(is_valid=True, validated_payload=payload)

    @classmethod
    def failure(cls, errors: List[str]) -> "HandoffValidationResult":
        return cls(is_valid=False, errors=errors)


class HandoffContract(BaseModel):
    """Contract defining expected inputs/outputs for an agent handoff."""
    from_agent: str = Field(description="Source agent type")
    to_agent: str = Field(description="Target agent type")

    # Required fields in task_params
    required_fields: List[str] = Field(default_factory=list)
    optional_fields: List[str] = Field(default_factory=list)

    # Type constraints
    field_types: Dict[str, str] = Field(default_factory=dict, description="Field name -> type name")

    # Validation rules
    min_solvents: Optional[int] = Field(default=None)
    min_selectivity: Optional[float] = Field(default=None)

    def validate(self, task_params: Dict[str, Any]) -> HandoffValidationResult:
        """Validate task_params against this contract."""
        errors = []
        warnings = []

        # Check required fields
        for field in self.required_fields:
            if field not in task_params:
                errors.append(f"Missing required field: {field}")

        # Type checking
        for field, expected_type in self.field_types.items():
            if field in task_params:
                value = task_params[field]
                if expected_type == "list" and not isinstance(value, list):
                    errors.append(f"Field {field} should be list, got {type(value).__name__}")
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    errors.append(f"Field {field} should be float, got {type(value).__name__}")
                elif expected_type == "str" and not isinstance(value, str):
                    errors.append(f"Field {field} should be str, got {type(value).__name__}")

        # Domain-specific validation
        if self.min_solvents and "solvents" in task_params:
            if len(task_params["solvents"]) < self.min_solvents:
                errors.append(f"Need at least {self.min_solvents} solvents, got {len(task_params['solvents'])}")

        if errors:
            return HandoffValidationResult.failure(errors)
        return HandoffValidationResult.success(task_params)


# Pre-defined contracts for common handoffs
HANDOFF_CONTRACTS: Dict[str, HandoffContract] = {
    "separation_to_tea": HandoffContract(
        from_agent="separation",
        to_agent="tea_lca",
        required_fields=["solvents", "throughput_kg_hr"],
        optional_fields=["polymers", "temperature", "recovery_rate"],
        field_types={"solvents": "list", "throughput_kg_hr": "float"},
        min_solvents=1
    ),
    "separation_to_literature": HandoffContract(
        from_agent="separation",
        to_agent="literature",
        required_fields=["search_topic"],
        optional_fields=["polymers", "solvents", "knowledgebases"],
        field_types={"search_topic": "str", "polymers": "list"}
    ),
    "tea_to_aggregator": HandoffContract(
        from_agent="tea_lca",
        to_agent="aggregator",
        required_fields=[],
        optional_fields=["cost_per_kg", "best_solvent", "solvents_analyzed"],
        field_types={"cost_per_kg": "float"}
    ),
    "literature_to_tea": HandoffContract(
        from_agent="literature",
        to_agent="tea_lca",
        required_fields=["solvents"],
        optional_fields=["key_findings", "polymers"],
        field_types={"solvents": "list"}
    ),
}


def validate_handoff(
    from_agent: str,
    to_agent: str,
    task_params: Dict[str, Any]
) -> HandoffValidationResult:
    """Validate a handoff using the appropriate contract."""
    contract_key = f"{from_agent}_to_{to_agent}"

    if contract_key not in HANDOFF_CONTRACTS:
        # No contract defined - allow with warning
        return HandoffValidationResult(
            is_valid=True,
            warnings=[f"No contract defined for {contract_key}, skipping validation"],
            validated_payload=task_params
        )

    contract = HANDOFF_CONTRACTS[contract_key]
    return contract.validate(task_params)


# ============================================================
# P4: ERROR RECOVERY AND PARTIAL RESULTS
# ============================================================

class PartialResult(BaseModel):
    """Represents partial results from a failed or incomplete agent execution."""
    agent: str = Field(description="Agent that produced partial results")
    completion_percentage: float = Field(default=0.0, ge=0, le=100)

    # What was completed
    completed_steps: List[str] = Field(default_factory=list)
    partial_data: Dict[str, Any] = Field(default_factory=dict)

    # What failed
    failed_step: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    error_type: Optional[str] = Field(default=None)

    # Recovery suggestions
    can_continue: bool = Field(default=False, description="Whether downstream can use partial data")
    recovery_suggestions: List[str] = Field(default_factory=list)
    fallback_values: Dict[str, Any] = Field(default_factory=dict)

    def to_handoff_context(self) -> Dict[str, Any]:
        """Convert partial result to context for downstream agent."""
        return {
            "upstream_partial": True,
            "upstream_agent": self.agent,
            "completion_percentage": self.completion_percentage,
            "available_data": self.partial_data,
            "fallback_values": self.fallback_values,
            "error_context": self.error_message,
        }


class ErrorContext(BaseModel):
    """Detailed error context for debugging and recovery."""
    error_type: str = Field(description="Type of error: tool_failure, validation, timeout, etc.")
    error_message: str = Field(description="Human-readable error message")

    # Location
    agent: str = Field(description="Agent where error occurred")
    tool_name: Optional[str] = Field(default=None)
    step_name: Optional[str] = Field(default=None)

    # State at failure
    state_snapshot: Dict[str, Any] = Field(default_factory=dict)
    input_params: Dict[str, Any] = Field(default_factory=dict)

    # Recovery
    is_recoverable: bool = Field(default=False)
    recovery_action: Optional[str] = Field(default=None, description="skip, retry, fallback, abort")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=2)

    # Timestamps
    occurred_at: datetime = Field(default_factory=datetime.now)


class RecoveryStrategy(BaseModel):
    """Strategy for recovering from agent failures."""
    error_type: str = Field(description="Error type this strategy handles")

    # Actions
    action: str = Field(description="Action to take: retry, skip, fallback, escalate")
    max_retries: int = Field(default=2)
    fallback_agent: Optional[str] = Field(default=None)
    fallback_values: Dict[str, Any] = Field(default_factory=dict)

    # Conditions
    applies_to_agents: List[str] = Field(default_factory=list, description="Empty = all agents")
    min_completion_for_continue: float = Field(default=50.0, description="Min % completion to continue")


# Default recovery strategies
DEFAULT_RECOVERY_STRATEGIES: List[RecoveryStrategy] = [
    RecoveryStrategy(
        error_type="tool_failure",
        action="retry",
        max_retries=2,
        applies_to_agents=["separation", "tea_lca"]
    ),
    RecoveryStrategy(
        error_type="timeout",
        action="fallback",
        fallback_values={"solvents": ["xylene"], "cost_per_kg": 2.50},
        min_completion_for_continue=30.0
    ),
    RecoveryStrategy(
        error_type="validation",
        action="skip",
        min_completion_for_continue=0.0
    ),
]


# ============================================================
# P4: CONDITIONAL ROUTING
# ============================================================

class RoutingCondition(BaseModel):
    """A condition for routing decisions."""
    field: str = Field(description="State field to check")
    operator: str = Field(description="Comparison: eq, ne, gt, lt, gte, lte, in, contains")
    value: Any = Field(description="Value to compare against")

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate this condition against state."""
        if self.field not in state:
            return False

        actual = state[self.field]

        if self.operator == "eq":
            return actual == self.value
        elif self.operator == "ne":
            return actual != self.value
        elif self.operator == "gt":
            return actual > self.value
        elif self.operator == "lt":
            return actual < self.value
        elif self.operator == "gte":
            return actual >= self.value
        elif self.operator == "lte":
            return actual <= self.value
        elif self.operator == "in":
            return actual in self.value
        elif self.operator == "contains":
            return self.value in actual
        return False


class RoutingRule(BaseModel):
    """A routing rule with conditions and target."""
    name: str = Field(description="Rule name for debugging")
    conditions: List[RoutingCondition] = Field(default_factory=list)
    all_conditions: bool = Field(default=True, description="True=AND, False=OR")

    # Target
    target_agent: str = Field(description="Agent to route to if conditions met")
    priority: int = Field(default=0, description="Higher priority rules checked first")

    # Optional state modifications
    state_updates: Dict[str, Any] = Field(default_factory=dict)

    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate all conditions."""
        if not self.conditions:
            return True

        results = [c.evaluate(state) for c in self.conditions]

        if self.all_conditions:
            return all(results)
        return any(results)


class ConditionalRouter(BaseModel):
    """Router with multiple conditional rules."""
    name: str = Field(default="conditional_router")
    rules: List[RoutingRule] = Field(default_factory=list)
    default_target: str = Field(description="Target if no rules match")

    def route(self, state: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Determine routing based on state. Returns (target, state_updates)."""
        # Sort by priority (descending)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority, reverse=True)

        for rule in sorted_rules:
            if rule.evaluate(state):
                return rule.target_agent, rule.state_updates

        return self.default_target, {}


# Pre-configured routers
QUALITY_BASED_ROUTER = ConditionalRouter(
    name="quality_router",
    rules=[
        RoutingRule(
            name="low_quality_retry",
            conditions=[
                RoutingCondition(field="quality_score", operator="lt", value=0.5),
                RoutingCondition(field="retry_count", operator="lt", value=2),
            ],
            target_agent="collab_separation_agent",
            priority=10,
            state_updates={"retry_count": 1}  # Will be incremented
        ),
        RoutingRule(
            name="no_solvents_retry",
            conditions=[
                RoutingCondition(field="solvents_count", operator="eq", value=0),
                RoutingCondition(field="retry_count", operator="lt", value=2),
            ],
            target_agent="collab_separation_agent",
            priority=9
        ),
        RoutingRule(
            name="good_quality_proceed",
            conditions=[
                RoutingCondition(field="quality_score", operator="gte", value=0.7),
            ],
            target_agent="collab_tea_agent",
            priority=5
        ),
    ],
    default_target="collab_tea_agent"
)


# ============================================================
# P4: CONTEXT PRUNING
# ============================================================

class ContextBudget(BaseModel):
    """Budget constraints for context size."""
    max_tokens: int = Field(default=4000, description="Max tokens for context")
    max_messages: int = Field(default=20, description="Max messages to include")
    max_results_per_agent: int = Field(default=3, description="Max result items per agent")

    # Priorities (higher = keep)
    field_priorities: Dict[str, int] = Field(
        default_factory=lambda: {
            "solvents": 10,
            "cost_per_kg": 10,
            "best_sequence": 9,
            "selectivities": 8,
            "polymers": 7,
            "key_findings": 6,
            "citations": 3,
            "raw_response": 1,
        }
    )


class ContextSummary(BaseModel):
    """Summarized context for handoff."""
    original_size_tokens: int = Field(default=0)
    pruned_size_tokens: int = Field(default=0)
    compression_ratio: float = Field(default=1.0)

    # Summarized data
    summary_text: str = Field(default="")
    key_values: Dict[str, Any] = Field(default_factory=dict)
    dropped_fields: List[str] = Field(default_factory=list)

    @classmethod
    def from_results(
        cls,
        results: Dict[str, Any],
        budget: ContextBudget
    ) -> "ContextSummary":
        """Create a summarized context from full results."""
        import json

        original_text = json.dumps(results, default=str)
        original_tokens = len(original_text) // 4  # Rough estimate

        # Prioritize fields
        key_values = {}
        dropped = []

        sorted_fields = sorted(
            results.keys(),
            key=lambda k: budget.field_priorities.get(k, 5),
            reverse=True
        )

        current_tokens = 0
        for field in sorted_fields:
            value = results[field]
            field_text = json.dumps({field: value}, default=str)
            field_tokens = len(field_text) // 4

            if current_tokens + field_tokens <= budget.max_tokens:
                # Truncate lists if needed
                if isinstance(value, list) and len(value) > budget.max_results_per_agent:
                    value = value[:budget.max_results_per_agent]
                key_values[field] = value
                current_tokens += field_tokens
            else:
                dropped.append(field)

        # Generate summary text
        summary_parts = []
        if "solvents" in key_values:
            summary_parts.append(f"Solvents: {', '.join(key_values['solvents'][:3])}")
        if "cost_per_kg" in key_values:
            summary_parts.append(f"Cost: ${key_values['cost_per_kg']:.2f}/kg")
        if "best_sequence" in key_values:
            summary_parts.append(f"Sequence: {' → '.join(key_values['best_sequence'])}")

        return cls(
            original_size_tokens=original_tokens,
            pruned_size_tokens=current_tokens,
            compression_ratio=current_tokens / max(original_tokens, 1),
            summary_text=" | ".join(summary_parts),
            key_values=key_values,
            dropped_fields=dropped
        )


def prune_context(
    state: Dict[str, Any],
    budget: Optional[ContextBudget] = None
) -> Dict[str, Any]:
    """Prune state to fit within context budget."""
    if budget is None:
        budget = ContextBudget()

    pruned = {}

    # Always keep essential fields
    essential = ["collaboration_mode", "shared_context", "trace_id"]
    for field in essential:
        if field in state:
            pruned[field] = state[field]

    # Summarize results
    for result_key in ["separation_results", "tea_results", "literature_results"]:
        if result_key in state and state[result_key]:
            summary = ContextSummary.from_results(state[result_key], budget)
            pruned[result_key] = summary.key_values

    # Truncate messages
    if "messages" in state:
        pruned["messages"] = state["messages"][-budget.max_messages:]

    return pruned


# ============================================================
# P5: TOOL CHAINING
# ============================================================

class ToolCall(BaseModel):
    """A single tool call in a chain."""
    tool_name: str = Field(description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="IDs of tools this depends on")

    # Output handling
    output_key: str = Field(description="Key to store output under")
    extract_fields: List[str] = Field(default_factory=list, description="Fields to extract from output")


class ToolChain(BaseModel):
    """A chain of tool calls with dependencies."""
    name: str = Field(description="Chain name")
    description: str = Field(default="")

    # Tools in the chain
    tools: List[ToolCall] = Field(default_factory=list)

    # Execution settings
    parallel_execution: bool = Field(default=False, description="Run independent tools in parallel")
    stop_on_error: bool = Field(default=True)

    # Context management
    input_schema: Dict[str, str] = Field(default_factory=dict, description="Required inputs")
    output_schema: Dict[str, str] = Field(default_factory=dict, description="Expected outputs")

    def get_execution_order(self) -> List[List[str]]:
        """Get tools in execution order (grouped by dependency level)."""
        # Build dependency graph
        tool_ids = {t.output_key: t for t in self.tools}
        levels: List[List[str]] = []
        executed = set()

        while len(executed) < len(self.tools):
            # Find tools whose dependencies are all satisfied
            level = []
            for tool in self.tools:
                if tool.output_key in executed:
                    continue
                if all(d in executed for d in tool.depends_on):
                    level.append(tool.output_key)

            if not level:
                # Circular dependency or missing tool
                remaining = [t.output_key for t in self.tools if t.output_key not in executed]
                level = remaining  # Force execution

            levels.append(level)
            executed.update(level)

        return levels


# Pre-defined tool chains
SEPARATION_ANALYSIS_CHAIN = ToolChain(
    name="separation_analysis",
    description="Complete separation analysis workflow",
    tools=[
        ToolCall(
            tool_name="find_optimal_separation_sequence",
            parameters={"algorithm": "greedy"},
            output_key="sequence_result",
            extract_fields=["solvents", "best_sequence"]
        ),
        ToolCall(
            tool_name="rank_solvents_for_separation",
            parameters={"criterion": "selectivity"},
            depends_on=["sequence_result"],
            output_key="ranking_result",
            extract_fields=["ranking", "scores"]
        ),
        ToolCall(
            tool_name="calculate_selectivity_detailed",
            depends_on=["ranking_result"],
            output_key="selectivity_result",
            extract_fields=["min_selectivity", "max_selectivity"]
        ),
    ],
    parallel_execution=False,
    input_schema={"polymers": "list", "temperature": "float"},
    output_schema={"solvents": "list", "best_sequence": "list", "selectivities": "list"}
)

TEA_COMPARISON_CHAIN = ToolChain(
    name="tea_comparison",
    description="TEA analysis with solvent comparison",
    tools=[
        ToolCall(
            tool_name="analyze_solvent_recovery_tea",
            parameters={},
            output_key="tea_results",
            extract_fields=["cost_per_kg", "payback_years"]
        ),
        ToolCall(
            tool_name="compare_solvents_tea_lca",
            depends_on=["tea_results"],
            output_key="comparison_result",
            extract_fields=["best_solvent", "ranking"]
        ),
    ],
    parallel_execution=False,
    input_schema={"solvents": "list", "throughput_kg_hr": "float"},
    output_schema={"cost_per_kg": "float", "best_solvent": "str"}
)


# ============================================================
# P5: DEPENDENCY GRAPH
# ============================================================

class AgentDependency(BaseModel):
    """Dependency relationship between agents."""
    agent: str = Field(description="Agent name")
    depends_on: List[str] = Field(default_factory=list, description="Agents this depends on")

    # Dependency type
    dependency_type: str = Field(
        default="required",
        description="required, optional, conditional"
    )

    # Conditions for conditional dependencies
    condition: Optional[RoutingCondition] = Field(default=None)

    # Data requirements
    required_data: List[str] = Field(default_factory=list, description="Data fields needed from dependencies")


class AgentGraph(BaseModel):
    """Dependency graph of agents."""
    name: str = Field(default="agent_graph")
    agents: List[AgentDependency] = Field(default_factory=list)

    def get_execution_order(self) -> List[List[str]]:
        """Get agents in execution order (topological sort)."""
        agent_map = {a.agent: a for a in self.agents}
        levels: List[List[str]] = []
        executed = set()

        while len(executed) < len(self.agents):
            level = []
            for agent in self.agents:
                if agent.agent in executed:
                    continue
                # Check if all required dependencies are satisfied
                required_deps = [
                    d for d in agent.depends_on
                    if agent_map.get(d, AgentDependency(agent=d)).dependency_type == "required"
                ]
                if all(d in executed for d in required_deps):
                    level.append(agent.agent)

            if not level and len(executed) < len(self.agents):
                # Handle cycles by forcing execution
                remaining = [a.agent for a in self.agents if a.agent not in executed]
                level = remaining[:1]

            if level:
                levels.append(level)
                executed.update(level)

        return levels

    def get_parallel_groups(self) -> List[List[str]]:
        """Get groups of agents that can run in parallel."""
        return self.get_execution_order()

    def validate(self) -> List[str]:
        """Validate the graph for cycles and missing dependencies."""
        errors = []
        agent_names = {a.agent for a in self.agents}

        for agent in self.agents:
            for dep in agent.depends_on:
                if dep not in agent_names:
                    errors.append(f"Agent {agent.agent} depends on unknown agent {dep}")

        # Check for cycles (simple DFS)
        def has_cycle(agent: str, path: set) -> bool:
            if agent in path:
                return True
            path.add(agent)
            agent_def = next((a for a in self.agents if a.agent == agent), None)
            if agent_def:
                for dep in agent_def.depends_on:
                    if has_cycle(dep, path.copy()):
                        return True
            return False

        for agent in self.agents:
            if has_cycle(agent.agent, set()):
                errors.append(f"Cycle detected involving agent {agent.agent}")
                break

        return errors


# Default agent dependency graph
DEFAULT_AGENT_GRAPH = AgentGraph(
    name="polymer_separation_graph",
    agents=[
        AgentDependency(
            agent="router",
            depends_on=[],
            dependency_type="required"
        ),
        AgentDependency(
            agent="separation",
            depends_on=["router"],
            dependency_type="required",
            required_data=["polymers", "temperature"]
        ),
        AgentDependency(
            agent="separation_reviewer",
            depends_on=["separation"],
            dependency_type="required",
            required_data=["separation_results"]
        ),
        AgentDependency(
            agent="literature",
            depends_on=["separation"],
            dependency_type="optional",
            required_data=["solvents", "polymers"]
        ),
        AgentDependency(
            agent="tea_lca",
            depends_on=["separation_reviewer"],
            dependency_type="required",
            required_data=["solvents", "throughput_kg_hr"]
        ),
        AgentDependency(
            agent="aggregator",
            depends_on=["tea_lca"],
            dependency_type="required",
            required_data=["separation_results", "tea_results"]
        ),
    ]
)


# ============================================================
# P5: OBSERVABILITY / DECISION LOGGING
# ============================================================

class DecisionType(str, Enum):
    """Types of decisions agents can make."""
    ROUTING = "routing"
    TOOL_SELECTION = "tool_selection"
    PARAMETER_CHOICE = "parameter_choice"
    RETRY = "retry"
    FALLBACK = "fallback"
    TERMINATION = "termination"


class DecisionLog(BaseModel):
    """Log entry for an agent decision."""
    decision_id: str = Field(description="Unique ID")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Context
    agent: str = Field(description="Agent making the decision")
    decision_type: DecisionType = Field(description="Type of decision")

    # Decision details
    options_considered: List[str] = Field(default_factory=list)
    chosen_option: str = Field(description="Selected option")
    reasoning: str = Field(default="", description="Why this option was chosen")
    confidence: float = Field(default=1.0, ge=0, le=1)

    # State context
    relevant_state: Dict[str, Any] = Field(default_factory=dict)

    # Outcome (filled in later)
    outcome: Optional[str] = Field(default=None)
    outcome_success: Optional[bool] = Field(default=None)


class ObservabilityConfig(BaseModel):
    """Configuration for observability/logging."""
    enabled: bool = Field(default=True)
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR")

    # What to log
    log_decisions: bool = Field(default=True)
    log_tool_calls: bool = Field(default=True)
    log_handoffs: bool = Field(default=True)
    log_state_changes: bool = Field(default=False)

    # Storage
    store_in_state: bool = Field(default=True)
    max_logs_in_state: int = Field(default=100)

    # Callbacks
    callback_url: Optional[str] = Field(default=None)


class AgentObserver:
    """Observer for tracking agent decisions and actions."""

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()
        self.logs: List[DecisionLog] = []

    def log_decision(
        self,
        agent: str,
        decision_type: DecisionType,
        options: List[str],
        chosen: str,
        reasoning: str = "",
        confidence: float = 1.0,
        state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Log a decision and return the decision ID."""
        import uuid

        if not self.config.enabled or not self.config.log_decisions:
            return ""

        decision_id = str(uuid.uuid4())[:8]

        log = DecisionLog(
            decision_id=decision_id,
            agent=agent,
            decision_type=decision_type,
            options_considered=options,
            chosen_option=chosen,
            reasoning=reasoning,
            confidence=confidence,
            relevant_state=self._extract_relevant_state(state) if state else {}
        )

        self.logs.append(log)

        # Trim if needed
        if len(self.logs) > self.config.max_logs_in_state:
            self.logs = self.logs[-self.config.max_logs_in_state:]

        return decision_id

    def update_outcome(self, decision_id: str, outcome: str, success: bool):
        """Update the outcome of a logged decision."""
        for log in self.logs:
            if log.decision_id == decision_id:
                log.outcome = outcome
                log.outcome_success = success
                break

    def _extract_relevant_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields from state for logging."""
        relevant_keys = [
            "collaboration_mode", "current_specialist_index",
            "retry_count", "quality_score", "solvents_count"
        ]
        return {k: state.get(k) for k in relevant_keys if k in state}

    def get_logs_for_state(self) -> List[Dict[str, Any]]:
        """Get logs in a format suitable for state storage."""
        return [log.model_dump() for log in self.logs]

    def generate_report(self) -> str:
        """Generate a summary report of all decisions."""
        if not self.logs:
            return "No decisions logged."

        lines = [f"Decision Report ({len(self.logs)} decisions)", "=" * 40]

        by_agent = {}
        for log in self.logs:
            by_agent.setdefault(log.agent, []).append(log)

        for agent, agent_logs in by_agent.items():
            lines.append(f"\n{agent} ({len(agent_logs)} decisions):")
            for log in agent_logs[-5:]:  # Last 5 per agent
                outcome_str = f" → {log.outcome}" if log.outcome else ""
                lines.append(f"  - [{log.decision_type.value}] {log.chosen_option}{outcome_str}")

        return "\n".join(lines)


# Global observer instance
_global_observer: Optional[AgentObserver] = None

def get_observer() -> AgentObserver:
    """Get or create the global observer."""
    global _global_observer
    if _global_observer is None:
        _global_observer = AgentObserver()
    return _global_observer

def log_agent_decision(
    agent: str,
    decision_type: DecisionType,
    options: List[str],
    chosen: str,
    reasoning: str = "",
    confidence: float = 1.0,
    state: Optional[Dict[str, Any]] = None
) -> str:
    """Convenience function to log a decision."""
    return get_observer().log_decision(
        agent, decision_type, options, chosen, reasoning, confidence, state
    )


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
    "DecisionType",
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
    # P3: Tool Output Schemas
    "ToolOutputBase",
    "SeparationToolOutput",
    "TEAToolOutput",
    "LiteratureToolOutput",
    "ComparisonToolOutput",
    # P3: Handoff Validation
    "HandoffValidationResult",
    "HandoffContract",
    "HANDOFF_CONTRACTS",
    "validate_handoff",
    # P4: Error Recovery
    "PartialResult",
    "ErrorContext",
    "RecoveryStrategy",
    "DEFAULT_RECOVERY_STRATEGIES",
    # P4: Conditional Routing
    "RoutingCondition",
    "RoutingRule",
    "ConditionalRouter",
    "QUALITY_BASED_ROUTER",
    # P4: Context Pruning
    "ContextBudget",
    "ContextSummary",
    "prune_context",
    # P5: Tool Chaining
    "ToolCall",
    "ToolChain",
    "SEPARATION_ANALYSIS_CHAIN",
    "TEA_COMPARISON_CHAIN",
    # P5: Dependency Graph
    "AgentDependency",
    "AgentGraph",
    "DEFAULT_AGENT_GRAPH",
    # P5: Observability
    "DecisionLog",
    "ObservabilityConfig",
    "AgentObserver",
    "get_observer",
    "log_agent_decision",
    # Helper functions
    "parse_to_separation_result",
    "parse_to_tea_result",
]
