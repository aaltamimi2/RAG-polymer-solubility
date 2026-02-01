# Multi-Agent System Integration Guide

## Branch: `multiagent-1-dev`

This branch contains consolidated multi-agent enhancements including:
- P0-P2 enhancements (review/revision loops, parallel execution, knowledge graph)
- P3-P5 enhancements (tool communication, error recovery, observability)
- 3-way collaboration (Separation → Literature → TEA)
- Gemini message ordering fixes
- Iteration counter fixes for handoff timing

---

## Part 1: P0-P2 Enhancements

### P0: Review/Revision Loop (Critical)
**Purpose:** Quality gate before TEA analysis - validates separation results and triggers retries if needed.

**Files:**
- `multi_agent_system.py:150-320` - `separation_reviewer_node()`, `SEPARATION_QUALITY_THRESHOLDS`
- `agent_schemas.py:651-700` - `ReviewerFeedback` schema

**Integration Points:**
- Graph routing: Insert `separation_reviewer` node between `collab_separation_agent` → `collab_tea_agent`
- State fields: Add `separation_retry_count`, `reviewer_feedback`, `retry_params` to `MultiAgentState`

### P1: Checkpointing, Parallel Execution, Supervisor
**Files:**
- `multi_agent_system.py:80-150` - `CheckpointerConfig` class
- `multi_agent_system.py:200-280` - `parallel_orchestrator_node()`, `supervisor_decision_node()`
- `agent_schemas.py:702-724` - `SupervisorDecision` schema

### P2: Cross-Session Store & Knowledge Graph
**Files:**
- `multi_agent_system.py:320-400` - `SessionStore`, `PolymerKnowledgeGraph` classes

---

## Part 2: P3-P5 Enhancements (NEW)

### P3: Tool Communication & Validation

**Purpose:** Structured tool outputs and validated handoffs between agents.

**Files:**
- `agent_schemas.py:730-820` - Tool output schemas
- `agent_schemas.py:825-920` - Handoff validation
- `tools/registry.py` - Central tool registry

**Key Components:**

```python
# Tool Output Schemas - Validate tool results
from agent_schemas import (
    SeparationToolOutput,  # Solvents, selectivities, sequences
    TEAToolOutput,         # Cost, CAPEX, OPEX, payback
    LiteratureToolOutput,  # Papers, citations, relevance
    ComparisonToolOutput,  # Rankings, scores, best item
)

# Example: Validate separation tool output
output = SeparationToolOutput(
    tool_name="find_separation",
    solvents=["xylene", "toluene"],
    selectivities=[45.0, 38.0],
    best_sequence=["PE", "PP", "PS"],
    confidence=0.9
)

# Handoff Validation - Type-safe agent communication
from agent_schemas import validate_handoff, HANDOFF_CONTRACTS

result = validate_handoff(
    from_agent="separation",
    to_agent="tea_lca",
    task_params={"solvents": ["xylene"], "throughput_kg_hr": 100.0}
)
if not result.is_valid:
    print(f"Validation errors: {result.errors}")

# Tool Registry - Central tool management
from tools.registry import get_registry, ToolCategory, ToolCapability

registry = get_registry()
tools = registry.list_tools(category=ToolCategory.SEPARATION)
contract = registry.get_contract("find_optimal_separation_sequence")
```

**Pre-defined Handoff Contracts:**
- `separation_to_tea` - Requires: solvents, throughput_kg_hr
- `separation_to_literature` - Requires: search_topic
- `tea_to_aggregator` - Optional: cost_per_kg, best_solvent
- `literature_to_tea` - Requires: solvents

---

### P4: Error Recovery & Conditional Routing

**Purpose:** Graceful degradation and dynamic routing based on result quality.

**Files:**
- `agent_schemas.py:925-1020` - Error recovery schemas
- `agent_schemas.py:1025-1150` - Conditional routing

**Key Components:**

```python
# Partial Results - Pass incomplete data to downstream
from agent_schemas import PartialResult, ErrorContext

partial = PartialResult(
    agent="separation",
    completion_percentage=60.0,
    partial_data={"solvents": ["xylene"]},
    failed_step="optimize_sequence",
    can_continue=True,
    fallback_values={"best_sequence": ["PE", "PP"]}
)
context = partial.to_handoff_context()  # For downstream agent

# Error Context - Detailed failure information
error = ErrorContext(
    error_type="tool_failure",
    error_message="Database timeout",
    agent="separation",
    is_recoverable=True,
    recovery_action="retry"
)

# Conditional Routing - Dynamic agent selection
from agent_schemas import (
    RoutingCondition,
    RoutingRule,
    ConditionalRouter,
    QUALITY_BASED_ROUTER,
)

# Use pre-defined quality-based router
state = {"quality_score": 0.4, "retry_count": 0}
target, updates = QUALITY_BASED_ROUTER.route(state)
# target = "collab_separation_agent" (retry due to low quality)

# Custom routing rules
router = ConditionalRouter(
    name="custom_router",
    rules=[
        RoutingRule(
            name="high_quality",
            conditions=[RoutingCondition(field="quality_score", operator="gte", value=0.7)],
            target_agent="tea_agent",
            priority=10
        ),
    ],
    default_target="aggregator"
)

# Context Pruning - Manage context size
from agent_schemas import ContextBudget, prune_context

budget = ContextBudget(max_tokens=4000, max_messages=20)
pruned_state = prune_context(large_state, budget)
```

**Recovery Strategies:**
- `retry` - Re-run agent with modified parameters
- `skip` - Continue without this agent's results
- `fallback` - Use default/cached values
- `escalate` - Route to human or supervisor

---

### P5: Tool Chaining, Dependencies & Observability

**Purpose:** Declarative tool workflows, agent dependencies, and decision tracking.

**Files:**
- `agent_schemas.py:1155-1280` - Tool chaining
- `agent_schemas.py:1285-1400` - Dependency graph
- `agent_schemas.py:1405-1550` - Observability

**Key Components:**

```python
# Tool Chaining - Define tool execution sequences
from agent_schemas import ToolCall, ToolChain, SEPARATION_ANALYSIS_CHAIN

chain = ToolChain(
    name="analysis_workflow",
    tools=[
        ToolCall(tool_name="find_separation", output_key="sep_result"),
        ToolCall(tool_name="rank_solvents", depends_on=["sep_result"], output_key="ranking"),
        ToolCall(tool_name="calculate_selectivity", depends_on=["ranking"], output_key="final"),
    ],
    parallel_execution=False
)
execution_order = chain.get_execution_order()
# [["sep_result"], ["ranking"], ["final"]]

# Agent Dependency Graph - Declarative agent ordering
from agent_schemas import AgentDependency, AgentGraph, DEFAULT_AGENT_GRAPH

graph = AgentGraph(
    name="separation_workflow",
    agents=[
        AgentDependency(agent="router", depends_on=[]),
        AgentDependency(agent="separation", depends_on=["router"]),
        AgentDependency(agent="literature", depends_on=["router"]),  # Parallel with separation
        AgentDependency(agent="tea", depends_on=["separation"]),
        AgentDependency(agent="aggregator", depends_on=["tea", "literature"]),
    ]
)
parallel_groups = graph.get_parallel_groups()
# [["router"], ["separation", "literature"], ["tea"], ["aggregator"]]

errors = graph.validate()  # Check for cycles, missing deps

# Observability - Track agent decisions
from agent_schemas import (
    DecisionType,
    DecisionLog,
    AgentObserver,
    get_observer,
    log_agent_decision,
)

# Log a routing decision
decision_id = log_agent_decision(
    agent="separation_reviewer",
    decision_type=DecisionType.ROUTING,
    options=["tea_agent", "retry_separation"],
    chosen="tea_agent",
    reasoning="Quality score 0.85 exceeds threshold",
    confidence=0.9,
    state=current_state
)

# Get decision report
observer = get_observer()
report = observer.generate_report()
print(report)
```

**Pre-defined Tool Chains:**
- `SEPARATION_ANALYSIS_CHAIN` - find_separation → rank_solvents → calculate_selectivity
- `TEA_COMPARISON_CHAIN` - analyze_tea → compare_solvents

**Default Agent Graph:** `DEFAULT_AGENT_GRAPH`
- router → separation → separation_reviewer → tea_lca → aggregator
- router → literature (optional, parallel with separation)

---

## Part 3: 3-Way Collaboration

**Purpose:** Enable separation planning with literature verification and economic analysis.

**Flow:** Separation → Literature (STRAP-CORE) → TEA → Aggregator

**Router Detection:**
```python
# Triggers on queries with separation + literature + cost keywords
if has_separation_keyword and has_literature_keyword and has_cost_keyword:
    collaboration_specialists=["separation", "literature", "tea_lca"]
```

---

## Part 4: Gemini Message Ordering Fixes

**File:** `agent_sql_final_1212_patched.py` (lines ~11554-11640)

**Solution:** Two-pass sanitization with bridge HumanMessages.

---

## Part 5: Iteration Counter Fixes

**File:** `multi_agent_system.py` (MultiAgentState, agent nodes)

**Solution:** Added `sep_iteration_count`, `tea_iteration_count` with max limits.

---

## Test Coverage

| Phase | Component | Test File | Tests |
|-------|-----------|-----------|-------|
| P0 | Reviewer | `tests/test_reviewer.py` | 14 |
| P0 | Multi-loop | `tests/test_reviewer_loops.py` | 25 |
| P0 | Integration | `tests/test_reviewer_integration.py` | 10 |
| P1 | Checkpointer/Parallel/Supervisor | `tests/test_p1_enhancements.py` | 24 |
| P2 | SessionStore/KnowledgeGraph | `tests/test_p2_enhancements.py` | 33 |
| P3 | Tool Schemas/Validation/Registry | `tests/test_p3_p5_enhancements.py` | 26 |
| P4 | Error Recovery/Routing/Pruning | `tests/test_p3_p5_enhancements.py` | 21 |
| P5 | Chaining/Dependencies/Observability | `tests/test_p3_p5_enhancements.py` | 20 |
| **Total** | | | **173** |

Run all tests:
```bash
pytest tests/test_reviewer*.py tests/test_p1*.py tests/test_p2*.py tests/test_p3*.py -v
```

---

## Quick Start Integration

```python
# Import P0-P2 components
from multi_agent_system import (
    separation_reviewer_node,
    SEPARATION_QUALITY_THRESHOLDS,
    CheckpointerConfig,
    SessionStore,
    PolymerKnowledgeGraph,
)
from agent_schemas import ReviewerFeedback, SupervisorDecision

# Import P3-P5 components
from agent_schemas import (
    # P3: Tool Communication
    SeparationToolOutput, TEAToolOutput,
    validate_handoff, HANDOFF_CONTRACTS,
    # P4: Error Recovery & Routing
    PartialResult, ErrorContext,
    ConditionalRouter, QUALITY_BASED_ROUTER,
    ContextBudget, prune_context,
    # P5: Tool Chaining & Observability
    ToolChain, SEPARATION_ANALYSIS_CHAIN,
    AgentGraph, DEFAULT_AGENT_GRAPH,
    log_agent_decision, DecisionType,
)
from tools.registry import get_registry, ToolCategory

# Use in agent workflow
registry = get_registry()
tools = registry.list_tools(capability=ToolCapability.COMPUTE_SELECTIVITY)

# Validate handoff before sending
validation = validate_handoff("separation", "tea_lca", task_params)
if validation.is_valid:
    # Proceed with handoff
    pass

# Log decision for observability
log_agent_decision(
    agent="router",
    decision_type=DecisionType.ROUTING,
    options=["separation", "literature"],
    chosen="separation"
)
```

---

## Integration Commands

```bash
# From another worktree
git fetch origin
git merge origin/multiagent-1-dev

# Or cherry-pick specific commits
git log --oneline multiagent-1-dev  # Find commit hashes
git cherry-pick <hash>
```
