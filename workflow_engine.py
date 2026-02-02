"""
Declarative Workflow Engine for Multi-Agent System

This module provides a clean, declarative architecture for defining and executing
multi-agent workflows. It replaces manual enumeration of agents and routing logic
with a data-driven approach.

Key concepts:
- AgentConfig: Declarative agent definition (categories, tools, custom logic)
- Stage: Unified stage definition (single agent = sequential, multiple = parallel)
- Trigger: Declarative trigger conditions (replaces lambdas)
- Workflow: Complete workflow definition with stages and trigger
- WorkflowEngine: Generic executor that runs any workflow
- HybridOrchestrator: Combines predefined workflows with LLM planning

Telemetry:
- ToolCallTrace: Tracks individual tool executions
- AgentTrace: Tracks agent iterations and tool calls
- StageTrace: Tracks stage execution with filter info
- WorkflowTrace: Complete workflow execution trace

Usage:
    # Basic usage
    engine = WorkflowEngine(sql_agent_node, tool_node_class, all_tools)
    workflow = engine.select_workflow(context)
    result, trace = await engine.run_workflow(state, workflow)

    # With hybrid orchestrator
    orchestrator = create_hybrid_orchestrator(sql_agent_node, tool_class, tools)
    result, trace = await orchestrator.orchestrate(state, query, context, return_trace=True)

    # Analyze trace
    print(format_trace_report(trace))
    analysis = analyze_trace_performance(trace)
"""

import asyncio
import logging
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set, Union, TypeVar, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# RELIABILITY CONFIGURATION
# ============================================================

@dataclass
class ReliabilityConfig:
    """Configuration for agent execution reliability."""

    # Timeout settings
    agent_timeout_seconds: float = 120.0  # Max time for single agent execution
    tool_timeout_seconds: float = 60.0    # Max time for tool batch execution
    llm_call_timeout_seconds: float = 45.0  # Max time for single LLM call

    # Retry settings
    max_retries: int = 3                  # Maximum retry attempts
    retry_base_delay: float = 1.0         # Base delay between retries (seconds)
    retry_max_delay: float = 30.0         # Maximum delay between retries
    retry_exponential_base: float = 2.0   # Exponential backoff multiplier
    retry_jitter: float = 0.1             # Random jitter factor (0-1)

    # Retryable exceptions (will be filled with actual exception types)
    retryable_exceptions: tuple = (
        TimeoutError,
        ConnectionError,
        OSError,  # Network errors
    )


# Global default config (can be overridden per-engine)
DEFAULT_RELIABILITY_CONFIG = ReliabilityConfig()


class AgentTimeoutError(Exception):
    """Raised when an agent execution exceeds the timeout limit."""
    def __init__(self, agent_name: str, timeout_seconds: float, message: str = None):
        self.agent_name = agent_name
        self.timeout_seconds = timeout_seconds
        super().__init__(message or f"Agent '{agent_name}' timed out after {timeout_seconds}s")


class AgentRetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    def __init__(self, agent_name: str, attempts: int, last_error: Exception):
        self.agent_name = agent_name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Agent '{agent_name}' failed after {attempts} attempts. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        )


T = TypeVar('T')


async def retry_with_backoff(
    coro_func: Callable[[], Awaitable[T]],
    config: ReliabilityConfig = None,
    operation_name: str = "operation",
) -> T:
    """
    Execute an async operation with exponential backoff retry.

    Args:
        coro_func: Async function to execute (called fresh each retry)
        config: Reliability configuration
        operation_name: Name for logging

    Returns:
        Result of the coroutine

    Raises:
        AgentRetryExhaustedError: If all retries fail
    """
    config = config or DEFAULT_RELIABILITY_CONFIG
    last_error = None

    for attempt in range(config.max_retries):
        try:
            return await coro_func()

        except config.retryable_exceptions as e:
            last_error = e

            if attempt < config.max_retries - 1:
                # Calculate delay with exponential backoff + jitter
                delay = min(
                    config.retry_base_delay * (config.retry_exponential_base ** attempt),
                    config.retry_max_delay
                )
                # Add jitter
                jitter = delay * config.retry_jitter * random.random()
                delay += jitter

                logger.warning(
                    f"{operation_name}: Attempt {attempt + 1}/{config.max_retries} failed "
                    f"({type(e).__name__}: {str(e)[:100]}). Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"{operation_name}: All {config.max_retries} attempts failed. "
                    f"Last error: {type(e).__name__}: {e}"
                )

        except Exception as e:
            # Non-retryable exception - fail immediately
            logger.error(f"{operation_name}: Non-retryable error: {type(e).__name__}: {e}")
            raise

    # All retries exhausted
    raise AgentRetryExhaustedError(operation_name, config.max_retries, last_error)


async def execute_with_timeout(
    coro: Awaitable[T],
    timeout_seconds: float,
    operation_name: str = "operation",
) -> T:
    """
    Execute a coroutine with a timeout.

    Args:
        coro: Coroutine to execute
        timeout_seconds: Maximum execution time
        operation_name: Name for logging

    Returns:
        Result of the coroutine

    Raises:
        AgentTimeoutError: If execution exceeds timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"{operation_name}: Timed out after {timeout_seconds}s")
        raise AgentTimeoutError(operation_name, timeout_seconds)


# ============================================================
# AGENT CONFIGURATION
# ============================================================

@dataclass
class AgentConfig:
    """
    Declarative agent configuration.

    Attributes:
        name: Unique agent identifier
        categories: Tool categories this agent can access
        max_iterations: Maximum tool loop iterations
        custom_node: Optional custom async function for special agents
        description: Human-readable description for logging/debugging
        task_prompt: Stage-specific instruction to prepend to the query
    """
    name: str
    categories: List[str] = field(default_factory=list)
    max_iterations: int = 8
    custom_node: Optional[Callable] = None
    description: str = ""
    task_prompt: str = ""  # Stage-specific instruction for the agent

    def __post_init__(self):
        if not self.description:
            self.description = f"Agent: {self.name}"


# ============================================================
# TRIGGER DSL
# ============================================================

@dataclass
class Trigger:
    """
    Declarative trigger conditions for workflow selection.

    Replaces lambda functions with serializable, debuggable conditions.
    All conditions are AND'd together (all must be true).

    Attributes:
        min_polymers: Minimum number of polymers required
        max_polymers: Maximum number of polymers allowed
        requires_specialists: All listed specialists must be present
        excludes_specialists: None of listed specialists can be present
        has_constraints: Specific constraints that must be present
        custom: Escape hatch for complex logic (Callable[[dict], bool])
    """
    min_polymers: Optional[int] = None
    max_polymers: Optional[int] = None
    requires_specialists: Optional[List[str]] = None
    excludes_specialists: Optional[List[str]] = None
    has_constraints: Optional[List[str]] = None
    custom: Optional[Callable[[dict], bool]] = None

    def evaluate(self, ctx: dict) -> bool:
        """Evaluate all conditions against context."""
        polymers = ctx.get("polymers", [])
        specialists = set(ctx.get("specialists", []))
        constraints = set(ctx.get("constraints", []))

        # Polymer count conditions
        if self.min_polymers is not None and len(polymers) < self.min_polymers:
            return False
        if self.max_polymers is not None and len(polymers) > self.max_polymers:
            return False

        # Specialist requirements
        if self.requires_specialists:
            if not set(self.requires_specialists) <= specialists:
                return False

        # Specialist exclusions
        if self.excludes_specialists:
            if set(self.excludes_specialists) & specialists:
                return False

        # Constraint requirements
        if self.has_constraints:
            if not set(self.has_constraints) <= constraints:
                return False

        # Custom logic (escape hatch)
        if self.custom is not None:
            if not self.custom(ctx):
                return False

        return True

    def explain(self, ctx: dict) -> str:
        """Explain why trigger matched or didn't match."""
        reasons = []
        polymers = ctx.get("polymers", [])
        specialists = set(ctx.get("specialists", []))

        if self.min_polymers is not None:
            status = "✓" if len(polymers) >= self.min_polymers else "✗"
            reasons.append(f"{status} min_polymers={self.min_polymers} (got {len(polymers)})")

        if self.max_polymers is not None:
            status = "✓" if len(polymers) <= self.max_polymers else "✗"
            reasons.append(f"{status} max_polymers={self.max_polymers} (got {len(polymers)})")

        if self.requires_specialists:
            has_all = set(self.requires_specialists) <= specialists
            status = "✓" if has_all else "✗"
            reasons.append(f"{status} requires={self.requires_specialists}")

        if self.excludes_specialists:
            has_none = not (set(self.excludes_specialists) & specialists)
            status = "✓" if has_none else "✗"
            reasons.append(f"{status} excludes={self.excludes_specialists}")

        return "; ".join(reasons) if reasons else "always matches"


# Always-true trigger for default workflows
ALWAYS = Trigger()


# ============================================================
# STAGE & WORKFLOW DEFINITIONS
# ============================================================

@dataclass
class ContextFilter:
    """
    Declarative context filter applied before a stage runs.

    Attributes:
        top_n_polymers: Keep only top N polymers (by profitability)
        exclude_polymers: Remove specific polymers
        custom: Custom filter function
    """
    top_n_polymers: Optional[int] = None
    exclude_polymers: Optional[List[str]] = None
    custom: Optional[Callable[[dict], dict]] = None

    def apply(self, ctx: dict) -> dict:
        """Apply filter to context."""
        result = dict(ctx)

        if self.top_n_polymers is not None:
            # Use top_polymers if available (from profitability screening)
            top = ctx.get("top_polymers", ctx.get("polymers", []))
            result["polymers"] = top[:self.top_n_polymers]

        if self.exclude_polymers:
            current = result.get("polymers", [])
            result["polymers"] = [p for p in current if p not in self.exclude_polymers]

        if self.custom:
            result = self.custom(result)

        return result


@dataclass
class Stage:
    """
    Unified stage definition for workflows.

    Single agent = sequential execution
    Multiple agents = parallel execution

    Attributes:
        agents: List of agent names to run
        filter: Optional context filter applied before stage
        timeout_seconds: Maximum time for this stage
    """
    agents: List[str]
    filter: Optional[ContextFilter] = None
    timeout_seconds: Optional[float] = None

    @property
    def is_parallel(self) -> bool:
        """True if this stage runs multiple agents in parallel."""
        return len(self.agents) > 1

    def __repr__(self):
        if self.is_parallel:
            return f"Parallel({self.agents})"
        return f"Stage({self.agents[0]})"


@dataclass
class Workflow:
    """
    Complete workflow definition.

    Attributes:
        name: Unique workflow identifier
        stages: Ordered list of stages to execute
        trigger: Conditions that activate this workflow
        priority: Higher priority workflows are checked first
        description: Human-readable description
    """
    name: str
    stages: List[Stage]
    trigger: Trigger = field(default_factory=lambda: ALWAYS)
    priority: int = 0
    description: str = ""

    def __post_init__(self):
        if not self.description:
            stage_desc = " → ".join(str(s) for s in self.stages)
            self.description = f"{self.name}: {stage_desc}"


# ============================================================
# AGENT RESULT
# ============================================================

@dataclass
class AgentResult:
    """
    Standardized agent execution result.

    Provides consistent structure for merging results from
    parallel or sequential agent executions.
    """
    agent_name: str
    messages: List = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    trace: Optional["AgentTrace"] = None  # Telemetry trace

    def merge_into(self, target: dict) -> None:
        """Merge this result into a target state dict."""
        # Extend messages
        if self.messages:
            target.setdefault("messages", []).extend(self.messages)

        # Merge named results (separation_results, tea_results, etc.)
        for key, value in self.results.items():
            if value is not None:
                target[key] = value

        # Update timing info
        target.setdefault("agent_timings", {})[self.agent_name] = self.elapsed_seconds


# ============================================================
# WORKFLOW TELEMETRY
# ============================================================

@dataclass
class ToolCallTrace:
    """Trace of a single tool call within an agent."""
    tool_name: str
    arguments: Dict[str, Any]
    result_summary: str  # Truncated for storage
    duration_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentTrace:
    """Trace of a single agent execution."""
    agent_name: str
    iterations: int
    tool_calls: List[ToolCallTrace] = field(default_factory=list)
    duration_seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None
    input_context: Dict[str, Any] = field(default_factory=dict)
    output_keys: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def failed_tool_calls(self) -> int:
        return sum(1 for tc in self.tool_calls if not tc.success)


@dataclass
class StageTrace:
    """Trace of a workflow stage (may contain multiple parallel agents)."""
    stage_index: int
    stage_type: str  # "parallel" or "sequential"
    agents: List[str]
    agent_traces: List[AgentTrace] = field(default_factory=list)
    duration_seconds: float = 0.0
    filter_applied: Optional[str] = None  # Description of filter if any
    polymers_before_filter: Optional[List[str]] = None
    polymers_after_filter: Optional[List[str]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def all_successful(self) -> bool:
        return all(at.success for at in self.agent_traces)


@dataclass
class WorkflowTrace:
    """Complete trace of a workflow execution."""
    workflow_name: str
    query: str
    context: Dict[str, Any]
    stages: List[StageTrace] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    used_planner: bool = False
    planning_time_ms: float = 0.0
    planning_reasoning: Optional[str] = None
    predefined_confidence: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_agents_run(self) -> int:
        return sum(len(st.agent_traces) for st in self.stages)

    @property
    def total_tool_calls(self) -> int:
        return sum(at.total_tool_calls for st in self.stages for at in st.agent_traces)

    @property
    def failed_stages(self) -> int:
        return sum(1 for st in self.stages if not st.all_successful)

    def to_summary(self) -> Dict[str, Any]:
        """Generate a concise summary for logging/storage."""
        return {
            "workflow": self.workflow_name,
            "query_preview": self.query[:100] + "..." if len(self.query) > 100 else self.query,
            "used_planner": self.used_planner,
            "planning_time_ms": self.planning_time_ms,
            "predefined_confidence": self.predefined_confidence,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "stages_count": len(self.stages),
            "agents_run": self.total_agents_run,
            "tool_calls": self.total_tool_calls,
            "failed_stages": self.failed_stages,
            "success": self.success,
            "timestamp": self.timestamp,
        }

    def to_detailed(self) -> Dict[str, Any]:
        """Generate detailed trace for debugging."""
        return {
            "summary": self.to_summary(),
            "context": self.context,
            "planning": {
                "used_planner": self.used_planner,
                "planning_time_ms": self.planning_time_ms,
                "reasoning": self.planning_reasoning,
                "predefined_confidence": self.predefined_confidence,
            },
            "stages": [
                {
                    "index": st.stage_index,
                    "type": st.stage_type,
                    "agents": st.agents,
                    "duration_seconds": round(st.duration_seconds, 2),
                    "filter": st.filter_applied,
                    "polymers_before": st.polymers_before_filter,
                    "polymers_after": st.polymers_after_filter,
                    "agent_traces": [
                        {
                            "agent": at.agent_name,
                            "iterations": at.iterations,
                            "tool_calls": at.total_tool_calls,
                            "failed_tools": at.failed_tool_calls,
                            "duration_seconds": round(at.duration_seconds, 2),
                            "success": at.success,
                            "error": at.error,
                            "output_keys": at.output_keys,
                        }
                        for at in st.agent_traces
                    ],
                }
                for st in self.stages
            ],
            "error": self.error,
        }


# ============================================================
# WORKFLOW ENGINE
# ============================================================

class WorkflowEngine:
    """
    Generic workflow execution engine.

    Executes any workflow defined using the declarative Stage/Workflow
    classes. Handles tool loops, parallel execution, and result merging.

    Usage:
        engine = WorkflowEngine(sql_agent_node, tool_node_class, all_tools)

        # Select workflow based on context
        workflow = engine.select_workflow({
            "polymers": ["PE", "PP", "PS", "PET"],
            "specialists": ["separation", "tea_lca"],
        })

        # Execute workflow
        result = await engine.run_workflow(state, workflow)
    """

    def __init__(
        self,
        sql_agent_node: Callable,
        tool_node_class: Callable,
        all_tools: List,
        agents: Dict[str, AgentConfig] = None,
        workflows: List[Workflow] = None,
        reliability_config: ReliabilityConfig = None,
    ):
        """
        Initialize the workflow engine.

        Args:
            sql_agent_node: The base SQL agent node function
            tool_node_class: Class/function to create tool nodes
            all_tools: List of all available tools
            agents: Agent registry (uses default if None)
            workflows: Workflow registry (uses default if None)
            reliability_config: Configuration for timeouts and retries
        """
        self.sql_agent_node = sql_agent_node
        self.tool_node_class = tool_node_class
        self.all_tools = all_tools
        self.agents = agents or {}
        self.workflows = workflows or []
        self.reliability = reliability_config or DEFAULT_RELIABILITY_CONFIG
        self._tool_cache: Dict[str, Any] = {}

    def register_agent(self, config: AgentConfig) -> None:
        """Register an agent configuration."""
        self.agents[config.name] = config
        logger.debug(f"Registered agent: {config.name}")

    def register_workflow(self, workflow: Workflow) -> None:
        """Register a workflow and maintain priority order."""
        self.workflows.append(workflow)
        self.workflows.sort(key=lambda w: -w.priority)
        logger.debug(f"Registered workflow: {workflow.name} (priority={workflow.priority})")

    def _get_tools(self, agent_name: str):
        """Get or create tool node for an agent."""
        if agent_name not in self._tool_cache:
            config = self.agents.get(agent_name)
            if config and config.categories:
                # Filter tools by category (would need tool_categories mapping)
                # For now, use all tools
                tools = self.all_tools
            else:
                tools = self.all_tools
            self._tool_cache[agent_name] = self.tool_node_class(tools)
        return self._tool_cache[agent_name]

    def _extract_separation_from_messages(self, messages: List) -> Optional[Dict]:
        """Extract separation results from tool messages.

        Uses structured JSON extraction from tools that return {"display": ..., "data": ...}.
        Only processes ToolMessages to avoid matching LLM summaries.
        """
        import json
        from langchain_core.messages import ToolMessage

        for msg in reversed(messages):
            if not hasattr(msg, 'content'):
                continue

            # Only process ToolMessages (skip AIMessages which may contain summaries)
            is_tool_message = isinstance(msg, ToolMessage) or getattr(msg, 'type', None) == 'tool'
            if not is_tool_message:
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Structured JSON extraction
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "data" in parsed and "display" in parsed:
                    data = parsed["data"]
                    # Map structured data to expected format
                    return {
                        "solvents": data.get("solvents", []),
                        "selectivities": data.get("selectivities", []),
                        "polymers": data.get("polymers_analyzed", data.get("polymers", [])),
                        "temperature": data.get("temperature"),
                        "best_sequence": data.get("best_sequence"),
                        "best_solvent": data.get("best_solvent"),
                        "algorithm_used": data.get("algorithm_used"),
                        "tool_output": parsed.get("display", content),
                    }
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    def _extract_tea_from_messages(self, messages: List) -> Optional[Dict]:
        """Extract TEA results from tool messages.

        Uses structured JSON extraction from tools that return {"display": ..., "data": ...}.
        Only processes ToolMessages to avoid matching LLM summaries.
        """
        import json
        from langchain_core.messages import ToolMessage

        for msg in reversed(messages):
            if not hasattr(msg, 'content'):
                continue

            # Only process ToolMessages (skip AIMessages which may contain summaries)
            is_tool_message = isinstance(msg, ToolMessage) or getattr(msg, 'type', None) == 'tool'
            if not is_tool_message:
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Structured JSON extraction
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "data" in parsed and "display" in parsed:
                    data = parsed["data"]
                    # Map structured data to expected format
                    # Handle both TEAToolOutput and STRAPToolOutput schemas
                    tci = data.get("tci_millions")
                    opex = data.get("annual_operating_cost_millions")
                    return {
                        "cost_per_kg": data.get("unit_operating_cost", data.get("cost_per_kg")),
                        "total_capex": tci * 1e6 if tci else data.get("total_capex"),
                        "total_opex": opex * 1e6 if opex else data.get("annual_opex"),
                        "payback_years": data.get("simple_payback_years", data.get("payback_years")),
                        "roi_pct": data.get("roi_pct"),
                        "throughput_kg_hr": data.get("throughput_kg_hr"),
                        "capacity_mt_yr": data.get("capacity_mt_yr"),
                        "msp_values": data.get("msp_by_polymer", data.get("msp_values", {})),
                        "gwp_by_polymer": data.get("gwp_by_polymer", {}),
                        "gwp_reduction_pct": data.get("gwp_reduction_pct", {}),
                        "tool_output": parsed.get("display", content),
                    }
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    async def run_agent(self, state: dict, agent_name: str) -> AgentResult:
        """
        Run a single agent with full tool loop, timeout protection, and retry logic.

        Features:
        - Overall agent timeout (default: 120s)
        - LLM call timeout (default: 45s)
        - Tool execution timeout (default: 60s)
        - Automatic retry with exponential backoff for transient failures

        Args:
            state: Current workflow state
            agent_name: Name of agent to run

        Returns:
            AgentResult with messages, results, and trace
        """
        config = self.agents.get(agent_name)
        if not config:
            logger.error(f"Unknown agent: {agent_name}")
            trace = AgentTrace(
                agent_name=agent_name,
                iterations=0,
                success=False,
                error=f"Unknown agent: {agent_name}",
            )
            return AgentResult(
                agent_name=agent_name,
                success=False,
                error=f"Unknown agent: {agent_name}",
                trace=trace,
            )

        start_time = time.time()
        logger.info(f"  [{agent_name}] Starting (timeout: {self.reliability.agent_timeout_seconds}s)...")

        # Capture input context for trace (truncate large values)
        input_context = {}
        for key in ["polymers", "specialists", "constraints", "shared_context"]:
            if key in state:
                val = state[key]
                if isinstance(val, dict):
                    input_context[key] = {k: str(v)[:100] for k, v in val.items()}
                elif isinstance(val, list):
                    input_context[key] = val[:10] if len(val) <= 10 else val[:10] + ["..."]
                else:
                    input_context[key] = str(val)[:200]

        tool_call_traces: List[ToolCallTrace] = []
        iterations = 0
        retry_count = 0

        async def _run_agent_with_timeout() -> AgentResult:
            """Inner function wrapped with overall timeout."""
            nonlocal iterations, tool_call_traces, retry_count

            try:
                # Custom node takes precedence
                if config.custom_node:
                    # Wrap custom node execution with timeout
                    result = await execute_with_timeout(
                        config.custom_node(state, self.sql_agent_node),
                        timeout_seconds=self.reliability.agent_timeout_seconds,
                        operation_name=f"{agent_name}(custom)",
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"  [{agent_name}] Completed (custom) in {elapsed:.1f}s")

                    output_keys = [k for k in result.keys() if k != "messages"]
                    trace = AgentTrace(
                        agent_name=agent_name,
                        iterations=1,
                        tool_calls=[],  # Custom nodes don't track individual tools
                        duration_seconds=elapsed,
                        success=True,
                        input_context=input_context,
                        output_keys=output_keys,
                    )

                    return AgentResult(
                        agent_name=agent_name,
                        messages=result.get("messages", []),
                        results={k: v for k, v in result.items() if k != "messages"},
                        elapsed_seconds=elapsed,
                        trace=trace,
                    )

                # Standard agent with tool loop
                agent_state = dict(state)
                agent_state["selected_categories"] = config.categories
                tools = self._get_tools(agent_name)

                # Inject task-specific prompt if configured
                if config.task_prompt:
                    from langchain_core.messages import HumanMessage as WFHumanMessage
                    existing_messages = agent_state.get("messages", [])
                    # Find the original query
                    original_query = ""
                    for msg in existing_messages:
                        if hasattr(msg, 'content') and isinstance(msg, WFHumanMessage):
                            original_query = msg.content
                            break
                    # Create a focused task message
                    task_message = WFHumanMessage(
                        content=f"{config.task_prompt}\n\nOriginal request: {original_query}"
                    )
                    agent_state["messages"] = [task_message]
                    logger.info(f"  [{agent_name}] Injected task prompt: {config.task_prompt[:50]}...")

                for i in range(config.max_iterations):
                    iterations = i + 1

                    # LLM call with timeout
                    try:
                        result = await execute_with_timeout(
                            self.sql_agent_node(agent_state),
                            timeout_seconds=self.reliability.llm_call_timeout_seconds,
                            operation_name=f"{agent_name}:llm_call",
                        )
                        # Properly extend messages instead of replacing
                        if "messages" in result:
                            result_msgs = result["messages"]
                            # Extend existing messages with new ones
                            existing_msgs = agent_state.get("messages", [])
                            agent_state["messages"] = existing_msgs + result_msgs
                            # Update other state keys (exclude messages)
                            for k, v in result.items():
                                if k != "messages":
                                    agent_state[k] = v
                        else:
                            agent_state.update(result)
                    except AgentTimeoutError:
                        logger.warning(f"  [{agent_name}] LLM call timed out at iteration {i+1}")
                        raise

                    # Check for pending tool calls
                    msgs = agent_state.get("messages", [])
                    if msgs and hasattr(msgs[-1], 'tool_calls') and msgs[-1].tool_calls:
                        pending_calls = msgs[-1].tool_calls
                        n_calls = len(pending_calls)
                        logger.debug(f"  [{agent_name}] Executing {n_calls} tool calls (iter {i+1})")

                        # Track each tool call
                        for tc in pending_calls:
                            tool_name = tc.get("name", "unknown") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                            tool_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})

                            tool_call_traces.append(ToolCallTrace(
                                tool_name=tool_name,
                                arguments={k: str(v)[:100] for k, v in tool_args.items()} if tool_args else {},
                                result_summary="",  # Will be filled after execution
                                duration_ms=0,  # Will be updated
                                success=True,  # Assume success, update if failed
                            ))

                        # Execute all tool calls with timeout
                        tool_start = time.time()
                        try:
                            # Support both __call__ (AsyncToolNode) and ainvoke (LangGraph ToolNode)
                            if hasattr(tools, 'ainvoke'):
                                tool_coro = tools.ainvoke(agent_state)
                            else:
                                tool_coro = tools(agent_state)

                            tool_result = await execute_with_timeout(
                                tool_coro,
                                timeout_seconds=self.reliability.tool_timeout_seconds,
                                operation_name=f"{agent_name}:tools",
                            )
                            # Properly extend messages instead of replacing
                            if "messages" in tool_result:
                                tool_msgs = tool_result["messages"]
                                # Extend existing messages with tool results
                                existing_msgs = agent_state.get("messages", [])
                                agent_state["messages"] = existing_msgs + tool_msgs
                                # Update other state keys (exclude messages)
                                for k, v in tool_result.items():
                                    if k != "messages":
                                        agent_state[k] = v
                            else:
                                agent_state.update(tool_result)
                            tool_duration = (time.time() - tool_start) * 1000

                            # Update traces with results
                            new_msgs = agent_state.get("messages", [])
                            tool_msg_idx = len(new_msgs) - n_calls
                            for j, tc_trace in enumerate(tool_call_traces[-n_calls:]):
                                tc_trace.duration_ms = tool_duration / n_calls  # Approximate
                                if tool_msg_idx + j < len(new_msgs):
                                    result_msg = new_msgs[tool_msg_idx + j]
                                    content = getattr(result_msg, 'content', str(result_msg))
                                    tc_trace.result_summary = str(content)[:200]

                        except AgentTimeoutError as timeout_err:
                            # Mark recent tool calls as timed out
                            for tc_trace in tool_call_traces[-n_calls:]:
                                tc_trace.success = False
                                tc_trace.error = f"Timeout after {self.reliability.tool_timeout_seconds}s"
                            raise

                        except Exception as tool_err:
                            # Mark recent tool calls as failed
                            for tc_trace in tool_call_traces[-n_calls:]:
                                tc_trace.success = False
                                tc_trace.error = str(tool_err)[:200]
                            raise

                        continue

                    # No more tool calls - done
                    break

                elapsed = time.time() - start_time
                logger.info(f"  [{agent_name}] Completed in {elapsed:.1f}s ({iterations} iterations)")

                # Extract results from state keys first
                result_keys = [
                    "separation_results", "tea_results", "literature_results",
                    "profitability_results", "shared_context", "top_polymers",
                ]
                results = {k: agent_state.get(k) for k in result_keys if k in agent_state}

                # Parse tool messages to extract structured results if not already set
                # This handles tools that return results in ToolMessage content
                messages = agent_state.get("messages", [])
                if not results.get("separation_results") and agent_name == "separation":
                    sep_results = self._extract_separation_from_messages(messages)
                    if sep_results:
                        results["separation_results"] = sep_results
                        logger.info(f"  [{agent_name}] Extracted separation_results with {len(sep_results.get('solvents', []))} solvents")

                if not results.get("tea_results") and agent_name == "tea_lca":
                    tea_extracted = self._extract_tea_from_messages(messages)
                    if tea_extracted:
                        results["tea_results"] = tea_extracted
                        logger.info(f"  [{agent_name}] Extracted tea_results (cost=${tea_extracted.get('cost_per_kg')})")

                output_keys = list(results.keys())

                trace = AgentTrace(
                    agent_name=agent_name,
                    iterations=iterations,
                    tool_calls=tool_call_traces,
                    duration_seconds=elapsed,
                    success=True,
                    input_context=input_context,
                    output_keys=output_keys,
                )

                return AgentResult(
                    agent_name=agent_name,
                    messages=agent_state.get("messages", []),
                    results=results,
                    elapsed_seconds=elapsed,
                    trace=trace,
                )

            except (AgentTimeoutError, AgentRetryExhaustedError) as e:
                # These are terminal errors - don't retry
                elapsed = time.time() - start_time
                error_msg = str(e)
                logger.error(f"  [{agent_name}] Failed (terminal) after {elapsed:.1f}s: {error_msg}")

                trace = AgentTrace(
                    agent_name=agent_name,
                    iterations=iterations,
                    tool_calls=tool_call_traces,
                    duration_seconds=elapsed,
                    success=False,
                    error=error_msg,
                    input_context=input_context,
                    output_keys=[],
                )

                return AgentResult(
                    agent_name=agent_name,
                    elapsed_seconds=elapsed,
                    success=False,
                    error=error_msg,
                    trace=trace,
                )

        # Execute with overall timeout (catch-all)
        try:
            return await asyncio.wait_for(
                _run_agent_with_timeout(),
                timeout=self.reliability.agent_timeout_seconds
            )

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            error_msg = f"Agent execution timed out after {self.reliability.agent_timeout_seconds}s"
            logger.error(f"  [{agent_name}] {error_msg}")

            trace = AgentTrace(
                agent_name=agent_name,
                iterations=iterations,
                tool_calls=tool_call_traces,
                duration_seconds=elapsed,
                success=False,
                error=error_msg,
                input_context=input_context,
                output_keys=[],
            )

            return AgentResult(
                agent_name=agent_name,
                elapsed_seconds=elapsed,
                success=False,
                error=error_msg,
                trace=trace,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"  [{agent_name}] Failed after {elapsed:.1f}s: {e}")

            trace = AgentTrace(
                agent_name=agent_name,
                iterations=iterations,
                tool_calls=tool_call_traces,
                duration_seconds=elapsed,
                success=False,
                error=str(e),
                input_context=input_context,
                output_keys=[],
            )

            return AgentResult(
                agent_name=agent_name,
                elapsed_seconds=elapsed,
                success=False,
                error=str(e),
                trace=trace,
            )

    async def run_stage(
        self, state: dict, stage: Stage, stage_index: int = 0
    ) -> tuple[dict, StageTrace]:
        """
        Run a workflow stage (parallel or sequential).

        Args:
            state: Current workflow state
            stage: Stage definition
            stage_index: Index of this stage in the workflow

        Returns:
            Tuple of (updated state, StageTrace)
        """
        start_time = time.time()
        stage_type = "parallel" if stage.is_parallel else "sequential"
        agent_traces: List[AgentTrace] = []

        # Track filter application
        filter_applied = None
        polymers_before = None
        polymers_after = None

        # Apply context filter if specified
        if stage.filter:
            # Build context from both shared_context and root state
            ctx = dict(state.get("shared_context", {}))
            # Propagate top_polymers from root state if available
            if "top_polymers" in state and "top_polymers" not in ctx:
                ctx["top_polymers"] = state["top_polymers"]

            polymers_before = ctx.get("polymers", [])
            filter_applied = str(stage.filter)

            state["shared_context"] = stage.filter.apply(ctx)
            polymers_after = state["shared_context"].get("polymers", [])
            logger.info(f"  Applied filter: polymers={polymers_after}")

        if stage.is_parallel:
            # Parallel execution
            logger.info(f"Running parallel stage: {stage.agents}")
            tasks = [self.run_agent(dict(state), agent) for agent in stage.agents]

            if stage.timeout_seconds:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=stage.timeout_seconds
                )
            else:
                results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results and collect traces
            merged = dict(state)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Parallel agent failed: {result}")
                    # Create error trace for failed agent
                    agent_traces.append(AgentTrace(
                        agent_name="unknown",
                        iterations=0,
                        success=False,
                        error=str(result),
                        duration_seconds=0,
                    ))
                    continue
                result.merge_into(merged)
                if result.trace:
                    agent_traces.append(result.trace)

            elapsed = time.time() - start_time
            stage_trace = StageTrace(
                stage_index=stage_index,
                stage_type=stage_type,
                agents=stage.agents,
                agent_traces=agent_traces,
                duration_seconds=elapsed,
                filter_applied=filter_applied,
                polymers_before_filter=polymers_before,
                polymers_after_filter=polymers_after,
            )

            return merged, stage_trace

        else:
            # Sequential execution (single agent)
            agent_name = stage.agents[0]
            logger.info(f"Running sequential stage: {agent_name}")
            result = await self.run_agent(state, agent_name)

            merged = dict(state)
            result.merge_into(merged)

            if result.trace:
                agent_traces.append(result.trace)

            elapsed = time.time() - start_time
            stage_trace = StageTrace(
                stage_index=stage_index,
                stage_type=stage_type,
                agents=stage.agents,
                agent_traces=agent_traces,
                duration_seconds=elapsed,
                filter_applied=filter_applied,
                polymers_before_filter=polymers_before,
                polymers_after_filter=polymers_after,
            )

            return merged, stage_trace

    async def run_workflow(
        self,
        state: dict,
        workflow: Workflow,
        trace: Optional[WorkflowTrace] = None,
    ) -> tuple[dict, WorkflowTrace]:
        """
        Execute a complete workflow.

        Args:
            state: Initial workflow state
            workflow: Workflow to execute
            trace: Optional pre-initialized WorkflowTrace (for planning metadata)

        Returns:
            Tuple of (final state, WorkflowTrace)
        """
        logger.info(f"Executing workflow: {workflow.name}")
        logger.info(f"  Stages: {[str(s) for s in workflow.stages]}")

        start_time = time.time()
        current_state = dict(state)
        current_state["workflow_name"] = workflow.name
        current_state["workflow_start_time"] = datetime.now().isoformat()

        # Initialize or update trace
        if trace is None:
            trace = WorkflowTrace(
                workflow_name=workflow.name,
                query=state.get("query", ""),
                context={
                    "polymers": state.get("polymers", []),
                    "specialists": state.get("specialists", []),
                    "constraints": state.get("constraints", []),
                },
            )
        else:
            trace.workflow_name = workflow.name

        stage_traces: List[StageTrace] = []

        try:
            for i, stage in enumerate(workflow.stages):
                logger.info(f"Stage {i+1}/{len(workflow.stages)}: {stage}")

                current_state, stage_trace = await self.run_stage(
                    current_state, stage, stage_index=i
                )
                stage_traces.append(stage_trace)

                logger.info(f"Stage {i+1} completed in {stage_trace.duration_seconds:.1f}s")

            total_elapsed = time.time() - start_time
            current_state["workflow_elapsed_seconds"] = total_elapsed

            # Finalize trace
            trace.stages = stage_traces
            trace.total_duration_seconds = total_elapsed
            trace.success = all(st.all_successful for st in stage_traces)

            logger.info(f"Workflow '{workflow.name}' completed in {total_elapsed:.1f}s")

            # Attach trace summary to state
            current_state["workflow_trace"] = trace.to_summary()

            return current_state, trace

        except Exception as e:
            total_elapsed = time.time() - start_time
            trace.stages = stage_traces
            trace.total_duration_seconds = total_elapsed
            trace.success = False
            trace.error = str(e)

            logger.error(f"Workflow '{workflow.name}' failed after {total_elapsed:.1f}s: {e}")

            current_state["workflow_trace"] = trace.to_summary()
            current_state["workflow_error"] = str(e)

            return current_state, trace

    def select_workflow(self, context: dict) -> Workflow:
        """
        Select the appropriate workflow based on context.

        Workflows are checked in priority order (highest first).
        First matching workflow is returned.

        Args:
            context: Dict with polymers, specialists, constraints, etc.

        Returns:
            Matching workflow (or last workflow as fallback)
        """
        for workflow in self.workflows:
            if workflow.trigger.evaluate(context):
                logger.info(f"Selected workflow: {workflow.name}")
                logger.debug(f"  Trigger: {workflow.trigger.explain(context)}")
                return workflow

        # Fallback to last workflow (should have ALWAYS trigger)
        if self.workflows:
            logger.warning(f"No workflow matched, using fallback: {self.workflows[-1].name}")
            return self.workflows[-1]

        raise ValueError("No workflows registered")


# ============================================================
# LLM WORKFLOW PLANNER
# ============================================================

# Planner LLM instance (created lazily)
_planner_llm = None


def _get_planner_llm():
    """Get or create the planner LLM instance."""
    global _planner_llm
    if _planner_llm is None:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            _planner_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",  # Use stronger model for planning
                temperature=0,
                max_tokens=2048,
                timeout=20,
                max_retries=2,
            )
            logger.info("WorkflowPlanner: Initialized with gemini-2.0-flash")
        except Exception as e:
            logger.warning(f"WorkflowPlanner: Failed to initialize LLM: {e}")
            _planner_llm = None
    return _planner_llm


WORKFLOW_PLANNER_PROMPT = '''You are a workflow planner for a polymer solubility analysis system. Given a user query and available agents, synthesize an optimal execution workflow.

## Available Agents

{agent_descriptions}

## Agent Capabilities Summary
- **separation**: Finds optimal solvents and dissolution sequences for polymer mixtures
- **tea_lca**: Calculates costs, CAPEX/OPEX, payback period, environmental impact
- **literature**: Searches academic papers, patents, and knowledge base
- **profitability**: Screens polymers by economic value (use when >3 polymers)

## Dependency Rules
- `tea_lca` typically needs separation results first (solvents, sequences)
- `literature` can run in parallel with most agents (no dependencies)
- `profitability` should run first when filtering many polymers
- Multiple independent agents CAN run in parallel

## User Query
{query}

## Extracted Context
- Polymers: {polymers}
- Constraints: {constraints}

## Task
Design a workflow that efficiently answers this query. Return JSON:

{{
    "reasoning": "Brief explanation of workflow design decisions",
    "stages": [
        {{
            "agents": ["agent1", "agent2"],  // Multiple = parallel, single = sequential
            "rationale": "Why these agents at this stage"
        }}
    ],
    "estimated_complexity": 1-5,
    "fallback_workflow": "name of predefined workflow if planning fails"
}}

Return ONLY valid JSON.'''


@dataclass
class PlannedWorkflow:
    """Result of LLM workflow planning."""
    workflow: Workflow
    reasoning: str
    estimated_complexity: int
    planning_time_ms: float
    used_fallback: bool = False
    fallback_reason: str = ""


class WorkflowPlanner:
    """
    LLM-based workflow planner for dynamic workflow synthesis.

    Analyzes user queries to:
    1. Identify required agent capabilities
    2. Determine dependencies between agents
    3. Synthesize optimal parallel/sequential workflow

    Used as fallback when no predefined workflow matches.
    """

    def __init__(self, agents: Dict[str, AgentConfig]):
        """
        Initialize planner with available agents.

        Args:
            agents: Registry of available agent configurations
        """
        self.agents = agents

    def _format_agent_descriptions(self) -> str:
        """Format agent configs for the planner prompt."""
        lines = []
        for name, config in self.agents.items():
            desc = config.description or f"Agent: {name}"
            cats = ", ".join(config.categories) if config.categories else "custom"
            lines.append(f"- **{name}**: {desc} (tools: {cats})")
        return "\n".join(lines)

    def _parse_planner_response(self, response_text: str) -> Optional[Dict]:
        """Parse JSON from planner LLM response."""
        import json
        try:
            text = response_text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"WorkflowPlanner: Failed to parse JSON: {e}")
            return None

    def _validate_agents(self, stage_agents: List[str]) -> List[str]:
        """Filter to only valid agent names."""
        valid = [a for a in stage_agents if a in self.agents]
        invalid = [a for a in stage_agents if a not in self.agents]
        if invalid:
            logger.warning(f"WorkflowPlanner: Unknown agents ignored: {invalid}")
        return valid

    def _build_workflow_from_plan(self, plan: Dict) -> Workflow:
        """Convert planner output to Workflow object."""
        stages = []

        for stage_def in plan.get("stages", []):
            agents = stage_def.get("agents", [])
            valid_agents = self._validate_agents(agents)

            if valid_agents:
                stages.append(Stage(agents=valid_agents))

        if not stages:
            # Fallback to single separation stage
            stages = [Stage(["separation"])]

        return Workflow(
            name="llm_planned",
            stages=stages,
            trigger=ALWAYS,
            priority=-1,  # Lower than predefined
            description=plan.get("reasoning", "LLM-planned workflow"),
        )

    async def plan(
        self,
        query: str,
        context: dict,
        fallback_workflow: Optional[Workflow] = None,
    ) -> PlannedWorkflow:
        """
        Plan a workflow for the given query.

        Args:
            query: User query string
            context: Extracted context (polymers, constraints, etc.)
            fallback_workflow: Workflow to use if planning fails

        Returns:
            PlannedWorkflow with synthesized workflow
        """
        import time
        start_time = time.time()

        llm = _get_planner_llm()
        if llm is None:
            if fallback_workflow:
                return PlannedWorkflow(
                    workflow=fallback_workflow,
                    reasoning="LLM unavailable, using fallback",
                    estimated_complexity=3,
                    planning_time_ms=0,
                    used_fallback=True,
                    fallback_reason="LLM not initialized",
                )
            raise ValueError("Planner LLM not available and no fallback provided")

        # Build prompt
        prompt = WORKFLOW_PLANNER_PROMPT.format(
            agent_descriptions=self._format_agent_descriptions(),
            query=query,
            polymers=context.get("polymers", []),
            constraints=context.get("constraints", []),
        )

        try:
            response = await asyncio.to_thread(llm.invoke, prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            plan = self._parse_planner_response(response_text)
            planning_time = (time.time() - start_time) * 1000

            if plan is None:
                if fallback_workflow:
                    return PlannedWorkflow(
                        workflow=fallback_workflow,
                        reasoning="Failed to parse planner response",
                        estimated_complexity=3,
                        planning_time_ms=planning_time,
                        used_fallback=True,
                        fallback_reason="JSON parse error",
                    )
                raise ValueError("Failed to parse workflow plan")

            workflow = self._build_workflow_from_plan(plan)

            logger.info(f"WorkflowPlanner: Synthesized workflow in {planning_time:.0f}ms")
            logger.info(f"  Reasoning: {plan.get('reasoning', 'N/A')}")
            logger.info(f"  Stages: {[str(s) for s in workflow.stages]}")

            return PlannedWorkflow(
                workflow=workflow,
                reasoning=plan.get("reasoning", ""),
                estimated_complexity=plan.get("estimated_complexity", 3),
                planning_time_ms=planning_time,
            )

        except Exception as e:
            planning_time = (time.time() - start_time) * 1000
            logger.error(f"WorkflowPlanner: Planning failed: {e}")

            if fallback_workflow:
                return PlannedWorkflow(
                    workflow=fallback_workflow,
                    reasoning=f"Planning error: {str(e)[:100]}",
                    estimated_complexity=3,
                    planning_time_ms=planning_time,
                    used_fallback=True,
                    fallback_reason=str(e),
                )
            raise


# ============================================================
# HYBRID ORCHESTRATOR
# ============================================================

class HybridOrchestrator:
    """
    Hybrid workflow orchestrator combining predefined workflows with LLM planning.

    Strategy:
    1. Try to match a predefined workflow (fast, predictable)
    2. If no match or low confidence, use LLM planner (flexible, slower)
    3. Execute the selected/planned workflow

    This gives the best of both worlds:
    - Common patterns are fast and predictable
    - Novel queries get custom workflows
    """

    def __init__(
        self,
        engine: WorkflowEngine,
        planner: Optional[WorkflowPlanner] = None,
        planning_threshold: float = 0.7,
    ):
        """
        Initialize hybrid orchestrator.

        Args:
            engine: Workflow execution engine with predefined workflows
            planner: LLM workflow planner (created if not provided)
            planning_threshold: Confidence below which to use planner
        """
        self.engine = engine
        self.planner = planner or WorkflowPlanner(engine.agents)
        self.planning_threshold = planning_threshold

    def _calculate_match_confidence(self, workflow: Workflow, context: dict) -> float:
        """
        Calculate confidence that a predefined workflow matches the query.

        Factors:
        - Trigger specificity (more conditions = higher confidence)
        - Agent coverage (do workflow agents cover query needs?)
        - Priority (higher priority = more specific)
        """
        trigger = workflow.trigger

        # Base confidence from trigger specificity
        conditions_count = sum([
            trigger.min_polymers is not None,
            trigger.max_polymers is not None,
            trigger.requires_specialists is not None,
            trigger.excludes_specialists is not None,
            trigger.has_constraints is not None,
        ])

        if conditions_count == 0:
            # ALWAYS trigger - low confidence
            return 0.3

        # More conditions = higher confidence
        specificity = min(conditions_count / 3, 1.0)

        # Priority bonus
        priority_bonus = min(workflow.priority / 20, 0.2)

        return min(0.5 + specificity * 0.4 + priority_bonus, 1.0)

    async def orchestrate(
        self,
        state: dict,
        query: str,
        context: dict,
        force_planning: bool = False,
        return_trace: bool = False,
    ) -> Union[dict, tuple[dict, WorkflowTrace]]:
        """
        Orchestrate workflow execution with hybrid selection and telemetry.

        Args:
            state: Initial workflow state
            query: Original user query
            context: Extracted context (polymers, specialists, constraints)
            force_planning: If True, skip predefined and use planner
            return_trace: If True, return (state, trace) tuple

        Returns:
            Final state with workflow results, or (state, WorkflowTrace) if return_trace=True
        """
        import time
        start_time = time.time()

        # Initialize workflow trace
        trace = WorkflowTrace(
            workflow_name="pending",
            query=query,
            context=context,
        )

        workflow = None
        used_planner = False
        planning_result = None
        predefined_confidence = None

        if not force_planning:
            # Try predefined workflow first
            try:
                workflow = self.engine.select_workflow(context)
                predefined_confidence = self._calculate_match_confidence(workflow, context)

                logger.info(f"HybridOrchestrator: Predefined '{workflow.name}' "
                           f"(confidence={predefined_confidence:.2f})")

                if predefined_confidence < self.planning_threshold:
                    logger.info(f"  Confidence below threshold ({self.planning_threshold}), "
                               f"trying LLM planner...")
                    force_planning = True

            except ValueError:
                force_planning = True

        if force_planning:
            # Use LLM planner
            fallback = workflow or (self.engine.workflows[-1] if self.engine.workflows else None)

            planning_result = await self.planner.plan(
                query=query,
                context=context,
                fallback_workflow=fallback,
            )

            workflow = planning_result.workflow
            used_planner = not planning_result.used_fallback

            logger.info(f"HybridOrchestrator: {'Planned' if used_planner else 'Fallback'} "
                       f"'{workflow.name}' ({planning_result.planning_time_ms:.0f}ms)")

        # Populate trace with planning metadata
        trace.used_planner = used_planner
        trace.planning_time_ms = planning_result.planning_time_ms if planning_result else 0.0
        trace.planning_reasoning = planning_result.reasoning if planning_result else None
        trace.predefined_confidence = predefined_confidence

        # Execute workflow with trace
        result, trace = await self.engine.run_workflow(state, workflow, trace=trace)

        # Finalize timing
        trace.total_duration_seconds = time.time() - start_time

        # Add orchestration metadata (legacy compatibility)
        result["orchestration"] = {
            "workflow_name": workflow.name,
            "used_planner": used_planner,
            "planning_time_ms": planning_result.planning_time_ms if planning_result else 0,
            "planning_reasoning": planning_result.reasoning if planning_result else None,
            "total_time_seconds": trace.total_duration_seconds,
        }

        # Store detailed trace in state
        result["workflow_trace_detailed"] = trace.to_detailed()

        if return_trace:
            return result, trace
        return result

    async def explain_plan(self, query: str, context: dict) -> Dict[str, Any]:
        """
        Explain what workflow would be selected without executing.

        Useful for debugging and user transparency.

        Returns:
            Dict with workflow selection explanation
        """
        # Check predefined
        try:
            predefined = self.engine.select_workflow(context)
            predefined_confidence = self._calculate_match_confidence(predefined, context)
        except ValueError:
            predefined = None
            predefined_confidence = 0

        # Get LLM plan
        planning_result = await self.planner.plan(
            query=query,
            context=context,
            fallback_workflow=predefined,
        )

        would_use_planner = predefined_confidence < self.planning_threshold

        return {
            "query": query,
            "context": context,
            "predefined_workflow": {
                "name": predefined.name if predefined else None,
                "confidence": predefined_confidence,
                "stages": [str(s) for s in predefined.stages] if predefined else [],
            },
            "llm_planned_workflow": {
                "reasoning": planning_result.reasoning,
                "stages": [str(s) for s in planning_result.workflow.stages],
                "planning_time_ms": planning_result.planning_time_ms,
                "used_fallback": planning_result.used_fallback,
            },
            "decision": "planner" if would_use_planner else "predefined",
            "threshold": self.planning_threshold,
        }


# ============================================================
# DEFAULT AGENT & WORKFLOW REGISTRY
# ============================================================

def create_default_agents() -> Dict[str, AgentConfig]:
    """Create default agent configurations."""
    return {
        "separation": AgentConfig(
            name="separation",
            categories=["separation", "advanced_separation", "dissolution", "solvent_properties", "visualization"],
            description="Plans optimal polymer separation sequences",
            task_prompt=(
                "You are the SEPARATION SPECIALIST. Your task is to analyze polymer separation.\n"
                "Focus ONLY on finding solvents and dissolution conditions. Use the separation tools available.\n"
                "Call analyze_selective_solubility_enhanced or find_optimal_separation_conditions to find solvents.\n"
                "Do NOT perform economic analysis - that will be handled by another specialist."
            ),
        ),
        "tea_lca": AgentConfig(
            name="tea_lca",
            categories=["economics", "strap", "visualization", "solvent_properties"],
            description="Performs techno-economic and lifecycle analysis",
            task_prompt=(
                "You are the TEA/LCA SPECIALIST. Your task is to perform economic and environmental analysis.\n"
                "Focus ONLY on techno-economic analysis (TEA) and lifecycle assessment (LCA).\n"
                "Use analyze_solvent_recovery_tea for economic analysis and generate_lca_visualizations for LCA.\n"
                "The separation analysis has already been done - focus on costs, payback, and environmental impact."
            ),
        ),
        "literature": AgentConfig(
            name="literature",
            categories=["literature", "rag"],
            description="Searches literature and RAG knowledge base",
            task_prompt=(
                "You are the LITERATURE SPECIALIST. Your task is to search published research.\n"
                "Focus ONLY on finding relevant literature and extracting process parameters.\n"
                "Use search_strap_core or search_rag_literature to find relevant papers and data.\n"
                "Do NOT perform separation analysis or economic analysis - those will be handled by other specialists."
            ),
        ),
        # Profitability agent will be registered with custom_node by multi_agent_system
    }


def create_default_workflows() -> List[Workflow]:
    """Create default workflow configurations."""
    return [
        # TEA-first: >3 polymers, screen for profitability first
        Workflow(
            name="tea_first",
            trigger=Trigger(min_polymers=4, requires_specialists=["separation"]),
            stages=[
                Stage(["profitability", "literature"]),  # Parallel
                Stage(["separation"], filter=ContextFilter(top_n_polymers=3)),
                Stage(["tea_lca"]),
            ],
            priority=10,
            description="TEA-first screening for complex mixtures (>3 polymers)",
        ),

        # Parallel separation + literature
        Workflow(
            name="parallel_sep_lit",
            trigger=Trigger(requires_specialists=["separation", "literature"]),
            stages=[
                Stage(["separation", "literature"]),  # Parallel
                Stage(["tea_lca"]),
            ],
            priority=5,
            description="Parallel separation and literature search",
        ),

        # Standard separation → TEA
        Workflow(
            name="standard_sep_tea",
            trigger=Trigger(requires_specialists=["separation"]),
            stages=[
                Stage(["separation"]),
                Stage(["tea_lca"]),
            ],
            priority=2,
            description="Standard separation then TEA",
        ),

        # Literature only
        Workflow(
            name="literature_only",
            trigger=Trigger(requires_specialists=["literature"], excludes_specialists=["separation", "tea_lca"]),
            stages=[
                Stage(["literature"]),
            ],
            priority=1,
            description="Literature search only",
        ),

        # Fallback: just run whatever specialists are requested
        Workflow(
            name="fallback",
            trigger=ALWAYS,
            stages=[
                Stage(["separation"]),  # Will be skipped if not in specialists
            ],
            priority=0,
            description="Fallback workflow",
        ),
    ]


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_engine(
    sql_agent_node: Callable,
    tool_node_class: Callable,
    all_tools: List,
    profitability_node: Callable = None,
) -> WorkflowEngine:
    """
    Create a fully configured workflow engine.

    Args:
        sql_agent_node: The base SQL agent node
        tool_node_class: Tool node factory
        all_tools: All available tools
        profitability_node: Custom profitability screening node

    Returns:
        Configured WorkflowEngine
    """
    engine = WorkflowEngine(
        sql_agent_node=sql_agent_node,
        tool_node_class=tool_node_class,
        all_tools=all_tools,
        agents=create_default_agents(),
        workflows=create_default_workflows(),
    )

    # Register profitability agent with custom node
    if profitability_node:
        engine.register_agent(AgentConfig(
            name="profitability",
            categories=[],
            custom_node=profitability_node,
            description="TEA-first profitability screening",
        ))

    return engine


def create_hybrid_orchestrator(
    sql_agent_node: Callable,
    tool_node_class: Callable,
    all_tools: List,
    profitability_node: Callable = None,
    planning_threshold: float = 0.7,
) -> HybridOrchestrator:
    """
    Create a fully configured hybrid orchestrator.

    The hybrid orchestrator combines:
    - Predefined workflows for common patterns (fast, predictable)
    - LLM planner for novel queries (flexible, slower)

    Args:
        sql_agent_node: The base SQL agent node
        tool_node_class: Tool node factory
        all_tools: All available tools
        profitability_node: Custom profitability screening node
        planning_threshold: Confidence threshold for using LLM planner

    Returns:
        Configured HybridOrchestrator

    Usage:
        orchestrator = create_hybrid_orchestrator(sql_agent_node, tool_class, tools)

        # Execute with automatic workflow selection
        result = await orchestrator.orchestrate(
            state=initial_state,
            query="Find best solvent for PE/PP separation with cost analysis",
            context={"polymers": ["PE", "PP"], "specialists": ["separation", "tea_lca"]},
        )

        # Explain what workflow would be selected (without executing)
        explanation = await orchestrator.explain_plan(query, context)
    """
    engine = create_engine(
        sql_agent_node=sql_agent_node,
        tool_node_class=tool_node_class,
        all_tools=all_tools,
        profitability_node=profitability_node,
    )

    planner = WorkflowPlanner(engine.agents)

    return HybridOrchestrator(
        engine=engine,
        planner=planner,
        planning_threshold=planning_threshold,
    )


# ============================================================
# INTEGRATION HELPER
# ============================================================

async def hybrid_workflow_node(
    state: dict,
    orchestrator: HybridOrchestrator,
    query_extractor: Callable = None,
) -> dict:
    """
    LangGraph node that uses the hybrid orchestrator.

    Drop-in replacement for manual executor nodes.

    Args:
        state: LangGraph state dict
        orchestrator: Configured HybridOrchestrator
        query_extractor: Function to extract query from state (optional)

    Returns:
        Updated state with workflow results
    """
    # Extract query from state
    if query_extractor:
        query = query_extractor(state)
    else:
        # Default: look in messages for HumanMessage
        query = ""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, 'content') and hasattr(msg, '__class__'):
                if msg.__class__.__name__ == 'HumanMessage':
                    query = msg.content
                    break

    # Build context
    shared_context = state.get("shared_context", {})
    context = {
        "polymers": shared_context.get("polymers", []),
        "specialists": state.get("collaboration_specialists", []),
        "constraints": shared_context.get("constraints", []),
    }

    # Execute via hybrid orchestrator
    result = await orchestrator.orchestrate(
        state=state,
        query=query,
        context=context,
    )

    return result


# ============================================================
# TRACE FORMATTING & ANALYSIS
# ============================================================

def format_trace_report(trace: WorkflowTrace, verbose: bool = False) -> str:
    """
    Generate a human-readable trace report for debugging.

    Args:
        trace: WorkflowTrace to format
        verbose: Include tool call details

    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"WORKFLOW TRACE: {trace.workflow_name}")
    lines.append("=" * 60)

    # Summary
    lines.append(f"\nQuery: {trace.query[:100]}{'...' if len(trace.query) > 100 else ''}")
    lines.append(f"Timestamp: {trace.timestamp}")
    lines.append(f"Total Duration: {trace.total_duration_seconds:.2f}s")
    lines.append(f"Success: {'✓' if trace.success else '✗'}")
    if trace.error:
        lines.append(f"Error: {trace.error}")

    # Planning info
    lines.append(f"\n--- Planning ---")
    lines.append(f"Used LLM Planner: {trace.used_planner}")
    if trace.predefined_confidence is not None:
        lines.append(f"Predefined Confidence: {trace.predefined_confidence:.2f}")
    if trace.planning_time_ms > 0:
        lines.append(f"Planning Time: {trace.planning_time_ms:.0f}ms")
    if trace.planning_reasoning:
        lines.append(f"Reasoning: {trace.planning_reasoning[:200]}")

    # Context
    lines.append(f"\n--- Context ---")
    lines.append(f"Polymers: {trace.context.get('polymers', [])}")
    lines.append(f"Specialists: {trace.context.get('specialists', [])}")

    # Stages
    lines.append(f"\n--- Stages ({len(trace.stages)}) ---")
    for st in trace.stages:
        status = "✓" if st.all_successful else "✗"
        lines.append(f"\nStage {st.stage_index + 1} [{st.stage_type}] {status}")
        lines.append(f"  Agents: {', '.join(st.agents)}")
        lines.append(f"  Duration: {st.duration_seconds:.2f}s")

        if st.filter_applied:
            lines.append(f"  Filter: {st.filter_applied}")
            lines.append(f"    Before: {st.polymers_before_filter}")
            lines.append(f"    After: {st.polymers_after_filter}")

        for at in st.agent_traces:
            agent_status = "✓" if at.success else "✗"
            lines.append(f"  → {at.agent_name} {agent_status}")
            lines.append(f"      Iterations: {at.iterations}, Tool calls: {at.total_tool_calls}")
            lines.append(f"      Duration: {at.duration_seconds:.2f}s")
            if at.output_keys:
                lines.append(f"      Outputs: {at.output_keys}")
            if at.error:
                lines.append(f"      Error: {at.error}")

            if verbose and at.tool_calls:
                lines.append(f"      Tool calls:")
                for tc in at.tool_calls:
                    tc_status = "✓" if tc.success else "✗"
                    lines.append(f"        - {tc.tool_name} {tc_status} ({tc.duration_ms:.0f}ms)")
                    if tc.error:
                        lines.append(f"          Error: {tc.error}")

    # Summary stats
    lines.append(f"\n--- Summary ---")
    lines.append(f"Total Agents: {trace.total_agents_run}")
    lines.append(f"Total Tool Calls: {trace.total_tool_calls}")
    lines.append(f"Failed Stages: {trace.failed_stages}")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def analyze_trace_performance(trace: WorkflowTrace) -> Dict[str, Any]:
    """
    Analyze trace for performance insights.

    Returns dict with:
    - slowest_stage: Stage with longest duration
    - slowest_agent: Agent with longest duration
    - most_tool_calls: Agent with most tool calls
    - bottlenecks: List of potential performance issues
    """
    bottlenecks = []
    agent_durations = []
    stage_durations = []

    for st in trace.stages:
        stage_durations.append((st.stage_index, st.duration_seconds, st.agents))

        for at in st.agent_traces:
            agent_durations.append((at.agent_name, at.duration_seconds, at.iterations))

            # Check for potential issues
            if at.iterations >= 6:
                bottlenecks.append({
                    "type": "high_iterations",
                    "agent": at.agent_name,
                    "iterations": at.iterations,
                    "message": f"Agent {at.agent_name} used {at.iterations} iterations (may indicate stuck loop)",
                })

            if at.failed_tool_calls > 0:
                bottlenecks.append({
                    "type": "failed_tools",
                    "agent": at.agent_name,
                    "count": at.failed_tool_calls,
                    "message": f"Agent {at.agent_name} had {at.failed_tool_calls} failed tool calls",
                })

            if at.duration_seconds > 30:
                bottlenecks.append({
                    "type": "slow_agent",
                    "agent": at.agent_name,
                    "duration": at.duration_seconds,
                    "message": f"Agent {at.agent_name} took {at.duration_seconds:.1f}s (>30s threshold)",
                })

    # Find slowest
    slowest_stage = max(stage_durations, key=lambda x: x[1]) if stage_durations else None
    slowest_agent = max(agent_durations, key=lambda x: x[1]) if agent_durations else None

    # Find agent with most tool calls
    tool_call_counts = [
        (at.agent_name, at.total_tool_calls)
        for st in trace.stages
        for at in st.agent_traces
    ]
    most_tool_calls = max(tool_call_counts, key=lambda x: x[1]) if tool_call_counts else None

    return {
        "slowest_stage": {
            "index": slowest_stage[0],
            "duration": slowest_stage[1],
            "agents": slowest_stage[2],
        } if slowest_stage else None,
        "slowest_agent": {
            "name": slowest_agent[0],
            "duration": slowest_agent[1],
            "iterations": slowest_agent[2],
        } if slowest_agent else None,
        "most_tool_calls": {
            "agent": most_tool_calls[0],
            "count": most_tool_calls[1],
        } if most_tool_calls else None,
        "bottlenecks": bottlenecks,
        "efficiency_score": _calculate_efficiency_score(trace),
    }


def _calculate_efficiency_score(trace: WorkflowTrace) -> float:
    """
    Calculate efficiency score (0-1) based on:
    - Success rate
    - Iteration efficiency
    - Tool call success rate
    """
    if not trace.stages:
        return 0.0

    # Success component (40%)
    success_rate = 1.0 if trace.success else 0.0

    # Stage success rate (30%)
    stage_success_rate = sum(
        1 for st in trace.stages if st.all_successful
    ) / len(trace.stages)

    # Iteration efficiency (30%) - fewer iterations is better
    total_iterations = sum(
        at.iterations for st in trace.stages for at in st.agent_traces
    )
    total_agents = trace.total_agents_run
    avg_iterations = total_iterations / max(total_agents, 1)
    iteration_efficiency = max(0, 1 - (avg_iterations - 1) / 7)  # 1 iter = 1.0, 8 iter = 0.0

    return (
        success_rate * 0.4 +
        stage_success_rate * 0.3 +
        iteration_efficiency * 0.3
    )


# ============================================================
# PUBLICATION VISUALIZATIONS
# ============================================================

def trace_to_timeline_svg(
    trace: WorkflowTrace,
    output_path: str = None,
    title: str = None,
    figsize: tuple = (12, 6),
    show_tool_calls: bool = False,
) -> str:
    """
    Generate publication-ready timeline SVG from workflow trace.

    Creates a Gantt-style chart showing parallel/sequential execution.

    Args:
        trace: WorkflowTrace to visualize
        output_path: Path to save SVG (if None, returns path in plots/)
        title: Figure title (defaults to workflow name)
        figsize: Figure dimensions in inches
        show_tool_calls: Include tool call bars within agents

    Returns:
        Path to saved SVG file
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib required for visualization: pip install matplotlib")

    # Color scheme (colorblind-friendly)
    COLORS = {
        "profitability": "#4CAF50",  # Green
        "literature": "#2196F3",     # Blue
        "separation": "#FF9800",     # Orange
        "tea_lca": "#9C27B0",        # Purple
        "aggregator": "#607D8B",     # Gray
        "default": "#795548",        # Brown
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Collect all agents and their timing
    agent_bars = []  # (agent_name, start_time, duration, stage_idx, is_parallel)

    cumulative_time = 0
    for stage in trace.stages:
        stage_start = cumulative_time

        if stage.stage_type == "parallel":
            # Parallel agents start at same time
            max_duration = 0
            for at in stage.agent_traces:
                agent_bars.append((
                    at.agent_name,
                    stage_start,
                    at.duration_seconds,
                    stage.stage_index,
                    True,
                ))
                max_duration = max(max_duration, at.duration_seconds)
            cumulative_time += max_duration
        else:
            # Sequential
            for at in stage.agent_traces:
                agent_bars.append((
                    at.agent_name,
                    cumulative_time,
                    at.duration_seconds,
                    stage.stage_index,
                    False,
                ))
                cumulative_time += at.duration_seconds

    # Get unique agents in order
    seen = set()
    agent_order = []
    for name, _, _, _, _ in agent_bars:
        if name not in seen:
            agent_order.append(name)
            seen.add(name)

    agent_y = {name: i for i, name in enumerate(agent_order)}

    # Draw bars
    bar_height = 0.6
    for agent_name, start, duration, stage_idx, is_parallel in agent_bars:
        y = agent_y[agent_name]
        color = COLORS.get(agent_name, COLORS["default"])

        # Add hatching for parallel stages
        hatch = "//" if is_parallel else None

        rect = FancyBboxPatch(
            (start, y - bar_height/2),
            duration,
            bar_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor="black",
            linewidth=1,
            alpha=0.85,
            hatch=hatch,
        )
        ax.add_patch(rect)

        # Duration label
        if duration > 0.5:
            ax.text(
                start + duration/2, y,
                f"{duration:.1f}s",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white"
            )

    # Stage separators
    stage_boundaries = []
    cumulative = 0
    for stage in trace.stages:
        stage_dur = stage.duration_seconds
        if stage.stage_type == "parallel" and stage.agent_traces:
            stage_dur = max(at.duration_seconds for at in stage.agent_traces)
        cumulative += stage_dur
        stage_boundaries.append(cumulative)

    for boundary in stage_boundaries[:-1]:
        ax.axvline(x=boundary, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    # Formatting
    ax.set_yticks(range(len(agent_order)))
    ax.set_yticklabels(agent_order, fontsize=11)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_xlim(-0.2, trace.total_duration_seconds * 1.05)
    ax.set_ylim(-0.5, len(agent_order) - 0.5)

    # Title
    fig_title = title or f"Workflow: {trace.workflow_name}"
    query_preview = trace.query[:60] + "..." if len(trace.query) > 60 else trace.query
    ax.set_title(f"{fig_title}\n\"{query_preview}\"", fontsize=12, fontweight="bold")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="gray", alpha=0.5, label="Sequential"),
        mpatches.Patch(facecolor="gray", alpha=0.5, hatch="//", label="Parallel"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    # Grid
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    if output_path is None:
        import os
        os.makedirs("plots", exist_ok=True)
        timestamp = trace.timestamp.replace(":", "-").replace(".", "-")[:19]
        output_path = f"plots/workflow_trace_{trace.workflow_name}_{timestamp}.svg"

    plt.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved timeline SVG to {output_path}")
    return output_path


def traces_to_comparison_svg(
    traces: List[WorkflowTrace],
    output_path: str = None,
    title: str = "Workflow Performance Comparison",
    figsize: tuple = (10, 6),
) -> str:
    """
    Generate comparison chart across multiple workflow traces.

    Args:
        traces: List of WorkflowTrace objects to compare
        output_path: Path to save SVG
        title: Figure title
        figsize: Figure dimensions

    Returns:
        Path to saved SVG file
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib required for visualization: pip install matplotlib")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Data extraction
    names = [t.workflow_name for t in traces]
    durations = [t.total_duration_seconds for t in traces]
    tool_calls = [t.total_tool_calls for t in traces]
    efficiency = [_calculate_efficiency_score(t) for t in traces]

    x = np.arange(len(traces))
    bar_width = 0.6

    # Chart 1: Duration
    ax1 = axes[0]
    bars1 = ax1.bar(x, durations, bar_width, color="#2196F3", alpha=0.8)
    ax1.set_ylabel("Duration (s)")
    ax1.set_title("Execution Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax1.bar_label(bars1, fmt="%.1f", fontsize=8)

    # Chart 2: Tool calls
    ax2 = axes[1]
    bars2 = ax2.bar(x, tool_calls, bar_width, color="#FF9800", alpha=0.8)
    ax2.set_ylabel("Count")
    ax2.set_title("Tool Calls")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax2.bar_label(bars2, fmt="%d", fontsize=8)

    # Chart 3: Efficiency
    ax3 = axes[2]
    bars3 = ax3.bar(x, efficiency, bar_width, color="#4CAF50", alpha=0.8)
    ax3.set_ylabel("Score (0-1)")
    ax3.set_title("Efficiency Score")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.bar_label(bars3, fmt="%.2f", fontsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    # Save
    if output_path is None:
        import os
        os.makedirs("plots", exist_ok=True)
        output_path = "plots/workflow_comparison.svg"

    plt.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison SVG to {output_path}")
    return output_path


def trace_to_architecture_svg(
    workflow: Workflow,
    output_path: str = None,
    title: str = None,
    figsize: tuple = (10, 4),
) -> str:
    """
    Generate architecture diagram showing workflow structure.

    Args:
        workflow: Workflow definition to visualize
        output_path: Path to save SVG
        title: Figure title
        figsize: Figure dimensions

    Returns:
        Path to saved SVG file
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except ImportError:
        raise ImportError("matplotlib required for visualization: pip install matplotlib")

    COLORS = {
        "profitability": "#4CAF50",
        "literature": "#2196F3",
        "separation": "#FF9800",
        "tea_lca": "#9C27B0",
        "aggregator": "#607D8B",
        "default": "#795548",
    }

    fig, ax = plt.subplots(figsize=figsize)

    box_width = 1.5
    box_height = 0.6
    h_spacing = 2.5
    v_spacing = 1.0

    # Calculate positions
    stage_positions = []
    x = 0.5

    for stage in workflow.stages:
        if stage.is_parallel:
            # Stack vertically
            y_start = (len(stage.agents) - 1) * v_spacing / 2
            positions = []
            for i, agent in enumerate(stage.agents):
                y = y_start - i * v_spacing
                positions.append((agent, x, y))
            stage_positions.append(positions)
        else:
            stage_positions.append([(stage.agents[0], x, 0)])

        x += h_spacing

    # Draw boxes
    for stage_pos in stage_positions:
        for agent, bx, by in stage_pos:
            color = COLORS.get(agent, COLORS["default"])
            rect = FancyBboxPatch(
                (bx - box_width/2, by - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor=color,
                edgecolor="black",
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_patch(rect)
            ax.text(bx, by, agent, ha="center", va="center",
                   fontsize=10, fontweight="bold", color="white")

    # Draw arrows
    for i in range(len(stage_positions) - 1):
        current_stage = stage_positions[i]
        next_stage = stage_positions[i + 1]

        for _, cx, cy in current_stage:
            for _, nx, ny in next_stage:
                arrow = FancyArrowPatch(
                    (cx + box_width/2, cy),
                    (nx - box_width/2, ny),
                    arrowstyle="->",
                    mutation_scale=15,
                    color="gray",
                    linewidth=1.5,
                )
                ax.add_patch(arrow)

    # Add "Start" and "End" labels
    first_x = stage_positions[0][0][1]
    last_x = stage_positions[-1][0][1]

    ax.annotate("Start", (first_x - box_width/2 - 0.3, 0),
               fontsize=10, ha="right", va="center")
    ax.annotate("End", (last_x + box_width/2 + 0.3, 0),
               fontsize=10, ha="left", va="center")

    # Stage labels
    for i, stage_pos in enumerate(stage_positions):
        stage = workflow.stages[i]
        x = stage_pos[0][1]
        max_y = max(p[2] for p in stage_pos) + box_height/2 + 0.2

        label = "║" if stage.is_parallel else "→"
        ax.text(x, max_y + 0.15, f"Stage {i+1} {label}",
               ha="center", va="bottom", fontsize=8, color="gray")

    # Formatting
    ax.set_xlim(-0.5, x + 0.5)
    all_y = [p[2] for sp in stage_positions for p in sp]
    ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    fig_title = title or f"Workflow Architecture: {workflow.name}"
    ax.set_title(fig_title, fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()

    # Save
    if output_path is None:
        import os
        os.makedirs("plots", exist_ok=True)
        output_path = f"plots/workflow_architecture_{workflow.name}.svg"

    plt.savefig(output_path, format="svg", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved architecture SVG to {output_path}")
    return output_path


def export_trace_for_publication(
    traces: List[WorkflowTrace],
    output_dir: str = "plots/publication",
    workflows: List[Workflow] = None,
) -> Dict[str, str]:
    """
    Generate complete publication figure set from traces.

    Generates:
    - workflow_architecture.svg (system diagram for each workflow)
    - execution_timeline_{n}.svg (Gantt chart for each trace)
    - performance_comparison.svg (bar charts across traces)

    Args:
        traces: List of WorkflowTrace objects
        output_dir: Directory for output files
        workflows: Optional workflow definitions for architecture diagrams

    Returns:
        Dict mapping figure name to file path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    outputs = {}

    # Timeline for each trace
    for i, trace in enumerate(traces):
        path = trace_to_timeline_svg(
            trace,
            output_path=f"{output_dir}/execution_timeline_{i+1}.svg",
            title=f"Figure {i+1}: Execution Timeline",
        )
        outputs[f"timeline_{i+1}"] = path

    # Comparison if multiple traces
    if len(traces) > 1:
        path = traces_to_comparison_svg(
            traces,
            output_path=f"{output_dir}/performance_comparison.svg",
        )
        outputs["comparison"] = path

    # Architecture diagrams
    if workflows:
        for wf in workflows:
            path = trace_to_architecture_svg(
                wf,
                output_path=f"{output_dir}/architecture_{wf.name}.svg",
            )
            outputs[f"architecture_{wf.name}"] = path

    logger.info(f"Generated {len(outputs)} publication figures in {output_dir}")
    return outputs


# ============================================================
# OPENTELEMETRY INSTRUMENTATION
# ============================================================

# Optional OpenTelemetry integration - only active if opentelemetry is installed
_otel_tracer = None
_otel_enabled = False


def init_opentelemetry(
    service_name: str = "dissolve-workflow",
    endpoint: str = None,
    enable_console: bool = False,
) -> bool:
    """
    Initialize OpenTelemetry tracing for production monitoring.

    Requires: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

    Args:
        service_name: Service name for traces
        endpoint: OTLP endpoint (e.g., "http://localhost:4317")
        enable_console: Also print spans to console

    Returns:
        True if initialization successful
    """
    global _otel_tracer, _otel_enabled

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        # OTLP exporter (for Jaeger/Grafana)
        if endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info(f"OpenTelemetry: OTLP exporter configured for {endpoint}")
            except ImportError:
                logger.warning("OTLP exporter not available: pip install opentelemetry-exporter-otlp")

        # Console exporter (for debugging)
        if enable_console:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)
        _otel_tracer = trace.get_tracer("dissolve.workflow")
        _otel_enabled = True

        logger.info("OpenTelemetry tracing initialized")
        return True

    except ImportError as e:
        logger.warning(f"OpenTelemetry not available: {e}")
        return False


def get_otel_tracer():
    """Get the OpenTelemetry tracer (or None if not initialized)."""
    return _otel_tracer


def is_otel_enabled() -> bool:
    """Check if OpenTelemetry is enabled."""
    return _otel_enabled


class OTelWorkflowInstrumentation:
    """
    OpenTelemetry instrumentation mixin for WorkflowEngine.

    Usage:
        # Initialize OpenTelemetry
        init_opentelemetry(endpoint="http://localhost:4317")

        # Wrap engine execution
        instrumentation = OTelWorkflowInstrumentation()
        with instrumentation.workflow_span(workflow) as span:
            result = await engine.run_workflow(state, workflow)
    """

    def __init__(self):
        self.tracer = get_otel_tracer()

    def _set_trace_attributes(self, span, trace: WorkflowTrace):
        """Set span attributes from WorkflowTrace."""
        if span is None:
            return

        span.set_attribute("workflow.name", trace.workflow_name)
        span.set_attribute("workflow.success", trace.success)
        span.set_attribute("workflow.duration_seconds", trace.total_duration_seconds)
        span.set_attribute("workflow.stages_count", len(trace.stages))
        span.set_attribute("workflow.total_agents", trace.total_agents_run)
        span.set_attribute("workflow.total_tool_calls", trace.total_tool_calls)
        span.set_attribute("workflow.used_planner", trace.used_planner)

        if trace.predefined_confidence is not None:
            span.set_attribute("workflow.predefined_confidence", trace.predefined_confidence)
        if trace.planning_time_ms > 0:
            span.set_attribute("workflow.planning_time_ms", trace.planning_time_ms)
        if trace.error:
            span.set_attribute("workflow.error", trace.error)

    def workflow_span(self, workflow_name: str):
        """
        Context manager for workflow-level span.

        Usage:
            with instrumentation.workflow_span("tea_first") as span:
                # Execute workflow
                pass
        """
        if self.tracer is None:
            # Return dummy context manager
            from contextlib import nullcontext
            return nullcontext()

        return self.tracer.start_as_current_span(
            f"workflow.{workflow_name}",
            attributes={"workflow.name": workflow_name}
        )

    def stage_span(self, stage_index: int, stage_type: str, agents: List[str]):
        """Context manager for stage-level span."""
        if self.tracer is None:
            from contextlib import nullcontext
            return nullcontext()

        return self.tracer.start_as_current_span(
            f"stage.{stage_index}.{stage_type}",
            attributes={
                "stage.index": stage_index,
                "stage.type": stage_type,
                "stage.agents": ",".join(agents),
            }
        )

    def agent_span(self, agent_name: str):
        """Context manager for agent-level span."""
        if self.tracer is None:
            from contextlib import nullcontext
            return nullcontext()

        return self.tracer.start_as_current_span(
            f"agent.{agent_name}",
            attributes={"agent.name": agent_name}
        )

    def tool_span(self, tool_name: str, arguments: dict = None):
        """Context manager for tool-level span."""
        if self.tracer is None:
            from contextlib import nullcontext
            return nullcontext()

        attrs = {"tool.name": tool_name}
        if arguments:
            # Truncate large arguments
            attrs["tool.arguments"] = str(arguments)[:500]

        return self.tracer.start_as_current_span(
            f"tool.{tool_name}",
            attributes=attrs
        )

    def record_trace(self, trace: WorkflowTrace):
        """
        Record a completed WorkflowTrace as OpenTelemetry spans.

        Useful for post-hoc recording of traces (e.g., from saved trace data).
        """
        if self.tracer is None:
            return

        with self.workflow_span(trace.workflow_name) as workflow_span:
            self._set_trace_attributes(workflow_span, trace)

            for stage in trace.stages:
                with self.stage_span(stage.stage_index, stage.stage_type, stage.agents):
                    for agent_trace in stage.agent_traces:
                        with self.agent_span(agent_trace.agent_name) as agent_span:
                            if agent_span:
                                agent_span.set_attribute("agent.iterations", agent_trace.iterations)
                                agent_span.set_attribute("agent.duration_seconds", agent_trace.duration_seconds)
                                agent_span.set_attribute("agent.success", agent_trace.success)
                                agent_span.set_attribute("agent.tool_calls", agent_trace.total_tool_calls)

                            for tool_call in agent_trace.tool_calls:
                                with self.tool_span(tool_call.tool_name) as tool_span:
                                    if tool_span:
                                        tool_span.set_attribute("tool.duration_ms", tool_call.duration_ms)
                                        tool_span.set_attribute("tool.success", tool_call.success)
                                        if tool_call.error:
                                            tool_span.set_attribute("tool.error", tool_call.error)
