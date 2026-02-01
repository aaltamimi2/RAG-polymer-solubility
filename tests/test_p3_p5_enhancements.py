"""
Tests for P3-P5 Enhancements: Tool Communication, Error Recovery, Observability

P3: Tool Output Schemas, Handoff Validation, Tool Registry
P4: Error Recovery, Conditional Routing, Context Pruning
P5: Tool Chaining, Dependency Graph, Observability
"""

import pytest
import asyncio
from typing import Dict, Any
from datetime import datetime

from agent_schemas import (
    # P3: Tool Output Schemas
    ToolOutputBase,
    SeparationToolOutput,
    TEAToolOutput,
    LiteratureToolOutput,
    ComparisonToolOutput,
    # P3: Handoff Validation
    HandoffValidationResult,
    HandoffContract,
    HANDOFF_CONTRACTS,
    validate_handoff,
    # P4: Error Recovery
    PartialResult,
    ErrorContext,
    RecoveryStrategy,
    DEFAULT_RECOVERY_STRATEGIES,
    # P4: Conditional Routing
    RoutingCondition,
    RoutingRule,
    ConditionalRouter,
    QUALITY_BASED_ROUTER,
    # P4: Context Pruning
    ContextBudget,
    ContextSummary,
    prune_context,
    # P5: Tool Chaining
    ToolCall,
    ToolChain,
    SEPARATION_ANALYSIS_CHAIN,
    TEA_COMPARISON_CHAIN,
    # P5: Dependency Graph
    AgentDependency,
    AgentGraph,
    DEFAULT_AGENT_GRAPH,
    # P5: Observability
    DecisionType,
    DecisionLog,
    ObservabilityConfig,
    AgentObserver,
    get_observer,
    log_agent_decision,
)

from tools.registry import (
    ToolCategory,
    ToolCapability,
    ToolParameter,
    ToolContract,
    ToolRegistry,
    get_registry,
    register_tool,
)


# ============================================================
# P3: TOOL OUTPUT SCHEMA TESTS
# ============================================================

class TestToolOutputSchemas:
    """Tests for P3 tool output validation schemas."""

    def test_tool_output_base(self):
        """Test base tool output schema."""
        output = ToolOutputBase(
            tool_name="test_tool",
            success=True,
            confidence=0.95
        )
        assert output.tool_name == "test_tool"
        assert output.success is True
        assert output.confidence == 0.95

    def test_tool_output_base_failure(self):
        """Test tool output with failure."""
        output = ToolOutputBase(
            tool_name="test_tool",
            success=False,
            error_message="Database connection failed"
        )
        assert output.success is False
        assert "Database" in output.error_message

    def test_separation_tool_output(self):
        """Test separation tool output schema."""
        output = SeparationToolOutput(
            tool_name="find_optimal_separation",
            solvents=["xylene", "toluene"],
            selectivities=[45.0, 38.0],
            best_sequence=["PE", "PP", "PS"],
            algorithm_used="greedy",
            polymers_analyzed=["PE", "PP", "PS"],
            temperature=80.0,
            coverage_complete=True
        )
        assert len(output.solvents) == 2
        assert output.algorithm_used == "greedy"
        assert output.coverage_complete is True

    def test_separation_tool_output_quality(self):
        """Test separation output with quality indicators."""
        output = SeparationToolOutput(
            tool_name="separation",
            solvents=["xylene"],
            selectivities=[25.0, 15.0, 10.0],
            min_selectivity=10.0,
            max_selectivity=25.0,
            confidence=0.8
        )
        assert output.min_selectivity == 10.0
        assert output.max_selectivity == 25.0

    def test_tea_tool_output(self):
        """Test TEA tool output schema."""
        output = TEAToolOutput(
            tool_name="analyze_tea",
            solvent="xylene",
            cost_per_kg=2.50,
            total_capex=150000,
            annual_opex=25000,
            payback_years=3.5,
            throughput_kg_hr=100.0,
            recovery_rate=0.95
        )
        assert output.cost_per_kg == 2.50
        assert output.payback_years == 3.5

    def test_tea_tool_output_with_lca(self):
        """Test TEA output with LCA data."""
        output = TEAToolOutput(
            tool_name="tea_lca",
            solvent="ethanol",
            cost_per_kg=1.80,
            co2_kg_per_kg=0.5,
            energy_mj_per_kg=15.0
        )
        assert output.co2_kg_per_kg == 0.5

    def test_literature_tool_output(self):
        """Test literature tool output schema."""
        output = LiteratureToolOutput(
            tool_name="search_literature",
            papers_found=5,
            relevant_excerpts=["PE dissolves in xylene at 80C"],
            citations=[{"title": "Paper 1", "doi": "10.1000/abc"}],
            knowledgebase_used="strap_core",
            polymers_mentioned=["PE", "PP"],
            solvents_mentioned=["xylene"],
            relevance_score=0.85
        )
        assert output.papers_found == 5
        assert output.relevance_score == 0.85

    def test_comparison_tool_output(self):
        """Test comparison tool output schema."""
        output = ComparisonToolOutput(
            tool_name="compare_solvents",
            items_compared=["xylene", "toluene", "cyclohexane"],
            ranking=["xylene", "toluene", "cyclohexane"],
            scores={"xylene": 0.95, "toluene": 0.85, "cyclohexane": 0.75},
            best_item="xylene",
            criteria_used=["selectivity", "cost"]
        )
        assert output.best_item == "xylene"
        assert output.ranking[0] == "xylene"


# ============================================================
# P3: HANDOFF VALIDATION TESTS
# ============================================================

class TestHandoffValidation:
    """Tests for P3 handoff validation."""

    def test_handoff_validation_result_success(self):
        """Test successful validation result."""
        payload = {"solvents": ["xylene"], "throughput_kg_hr": 100.0}
        result = HandoffValidationResult.success(payload)
        assert result.is_valid is True
        assert result.validated_payload == payload

    def test_handoff_validation_result_failure(self):
        """Test failed validation result."""
        result = HandoffValidationResult.failure(["Missing required field: solvents"])
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_handoff_contract_validate_success(self):
        """Test contract validation success."""
        contract = HandoffContract(
            from_agent="separation",
            to_agent="tea_lca",
            required_fields=["solvents"],
            field_types={"solvents": "list"}
        )
        result = contract.validate({"solvents": ["xylene"]})
        assert result.is_valid is True

    def test_handoff_contract_validate_missing_field(self):
        """Test contract validation with missing field."""
        contract = HandoffContract(
            from_agent="separation",
            to_agent="tea_lca",
            required_fields=["solvents", "throughput_kg_hr"]
        )
        result = contract.validate({"solvents": ["xylene"]})
        assert result.is_valid is False
        assert "throughput_kg_hr" in result.errors[0]

    def test_handoff_contract_validate_wrong_type(self):
        """Test contract validation with wrong type."""
        contract = HandoffContract(
            from_agent="separation",
            to_agent="tea_lca",
            required_fields=["solvents"],
            field_types={"solvents": "list"}
        )
        result = contract.validate({"solvents": "xylene"})  # String instead of list
        assert result.is_valid is False
        assert "list" in result.errors[0]

    def test_handoff_contract_min_solvents(self):
        """Test contract with minimum solvents requirement."""
        contract = HandoffContract(
            from_agent="separation",
            to_agent="tea_lca",
            required_fields=["solvents"],
            min_solvents=2
        )
        result = contract.validate({"solvents": ["xylene"]})
        assert result.is_valid is False
        assert "at least 2 solvents" in result.errors[0]

    def test_predefined_contracts_exist(self):
        """Test that predefined contracts are available."""
        assert "separation_to_tea" in HANDOFF_CONTRACTS
        assert "separation_to_literature" in HANDOFF_CONTRACTS
        assert "tea_to_aggregator" in HANDOFF_CONTRACTS

    def test_validate_handoff_function(self):
        """Test validate_handoff convenience function."""
        result = validate_handoff(
            from_agent="separation",
            to_agent="tea_lca",
            task_params={"solvents": ["xylene", "toluene"], "throughput_kg_hr": 100.0}
        )
        assert result.is_valid is True

    def test_validate_handoff_unknown_contract(self):
        """Test validation with unknown contract warns but allows."""
        result = validate_handoff(
            from_agent="unknown",
            to_agent="agent",
            task_params={"data": "value"}
        )
        assert result.is_valid is True
        assert len(result.warnings) > 0


# ============================================================
# P3: TOOL REGISTRY TESTS
# ============================================================

class TestToolRegistry:
    """Tests for P3 tool registry."""

    def setup_method(self):
        """Clear registry before each test."""
        get_registry().clear()

    def test_registry_singleton(self):
        """Registry should be a singleton."""
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_register_tool(self):
        """Test basic tool registration."""
        registry = get_registry()
        registry.register(
            name="test_tool",
            implementation=lambda x: x,
            description="A test tool",
            category=ToolCategory.ANALYSIS
        )
        assert registry.get("test_tool") is not None

    def test_register_with_decorator(self):
        """Test registration using decorator."""
        @register_tool(
            name="decorated_tool",
            description="Tool registered with decorator",
            category=ToolCategory.SEPARATION,
            capabilities=[ToolCapability.COMPUTE_SELECTIVITY]
        )
        def my_tool(x: int) -> int:
            return x * 2

        tool = get_registry().get("decorated_tool")
        assert tool is not None
        assert tool.contract.category == ToolCategory.SEPARATION

    def test_list_tools_by_category(self):
        """Test listing tools by category."""
        registry = get_registry()
        registry.register(
            name="sep_tool",
            implementation=lambda: None,
            description="Separation tool",
            category=ToolCategory.SEPARATION
        )
        registry.register(
            name="tea_tool",
            implementation=lambda: None,
            description="TEA tool",
            category=ToolCategory.TEA
        )

        sep_tools = registry.list_tools(category=ToolCategory.SEPARATION)
        assert "sep_tool" in sep_tools
        assert "tea_tool" not in sep_tools

    def test_list_tools_by_capability(self):
        """Test listing tools by capability."""
        registry = get_registry()
        registry.register(
            name="query_tool",
            implementation=lambda: None,
            description="Query tool",
            category=ToolCategory.DATABASE,
            capabilities=[ToolCapability.QUERY_DATABASE]
        )

        db_tools = registry.list_tools(capability=ToolCapability.QUERY_DATABASE)
        assert "query_tool" in db_tools

    def test_get_contract(self):
        """Test getting tool contract."""
        registry = get_registry()
        registry.register(
            name="contract_tool",
            implementation=lambda: None,
            description="Tool with contract",
            category=ToolCategory.ANALYSIS,
            input_parameters=[
                ToolParameter(name="value", type="float", required=True)
            ]
        )

        contract = registry.get_contract("contract_tool")
        assert contract is not None
        assert len(contract.input_parameters) == 1

    def test_contract_validation(self):
        """Test contract input validation."""
        contract = ToolContract(
            name="validated_tool",
            description="Tool with validation",
            category=ToolCategory.ANALYSIS,
            input_parameters=[
                ToolParameter(name="value", type="float", required=True,
                             constraints={"min": 0, "max": 100})
            ]
        )

        # Valid input
        is_valid, errors = contract.validate_input({"value": 50.0})
        assert is_valid is True

        # Missing required
        is_valid, errors = contract.validate_input({})
        assert is_valid is False

        # Out of range
        is_valid, errors = contract.validate_input({"value": 150.0})
        assert is_valid is False

    def test_find_tools_for_task(self):
        """Test finding tools by task description."""
        registry = get_registry()
        registry.register(
            name="cost_analyzer",
            implementation=lambda: None,
            description="Analyze costs",
            category=ToolCategory.TEA,
            capabilities=[ToolCapability.ANALYZE_COSTS]
        )

        tools = registry.find_for_task("calculate the cost of separation")
        tool_names = [t.name for t in tools]
        assert "cost_analyzer" in tool_names

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a registered tool."""
        registry = get_registry()
        registry.register(
            name="adder",
            implementation=lambda a, b: a + b,
            description="Add two numbers",
            category=ToolCategory.ANALYSIS,
            input_parameters=[
                ToolParameter(name="a", type="float", required=True),
                ToolParameter(name="b", type="float", required=True)
            ]
        )

        result = await registry.execute("adder", {"a": 5, "b": 3})
        assert result["success"] is True
        assert result["result"] == 8


# ============================================================
# P4: ERROR RECOVERY TESTS
# ============================================================

class TestErrorRecovery:
    """Tests for P4 error recovery mechanisms."""

    def test_partial_result_creation(self):
        """Test creating partial result."""
        partial = PartialResult(
            agent="separation",
            completion_percentage=60.0,
            completed_steps=["query_database", "compute_selectivity"],
            partial_data={"solvents": ["xylene"]},
            failed_step="optimize_sequence",
            error_message="Timeout after 30s",
            can_continue=True,
            recovery_suggestions=["Try with fewer polymers"]
        )
        assert partial.completion_percentage == 60.0
        assert partial.can_continue is True

    def test_partial_result_to_handoff_context(self):
        """Test converting partial result to handoff context."""
        partial = PartialResult(
            agent="separation",
            completion_percentage=50.0,
            partial_data={"solvents": ["xylene"]},
            fallback_values={"best_sequence": ["PE", "PP"]}
        )
        context = partial.to_handoff_context()

        assert context["upstream_partial"] is True
        assert context["upstream_agent"] == "separation"
        assert "xylene" in context["available_data"]["solvents"]

    def test_error_context_creation(self):
        """Test creating error context."""
        error = ErrorContext(
            error_type="tool_failure",
            error_message="Database query failed",
            agent="separation",
            tool_name="query_database",
            is_recoverable=True,
            recovery_action="retry"
        )
        assert error.is_recoverable is True
        assert error.recovery_action == "retry"

    def test_recovery_strategy(self):
        """Test recovery strategy definition."""
        strategy = RecoveryStrategy(
            error_type="timeout",
            action="fallback",
            fallback_values={"solvents": ["xylene"]},
            min_completion_for_continue=30.0
        )
        assert strategy.action == "fallback"
        assert "xylene" in strategy.fallback_values["solvents"]

    def test_default_recovery_strategies(self):
        """Test default recovery strategies exist."""
        assert len(DEFAULT_RECOVERY_STRATEGIES) >= 3

        # Find retry strategy
        retry_strategies = [s for s in DEFAULT_RECOVERY_STRATEGIES if s.action == "retry"]
        assert len(retry_strategies) > 0


# ============================================================
# P4: CONDITIONAL ROUTING TESTS
# ============================================================

class TestConditionalRouting:
    """Tests for P4 conditional routing."""

    def test_routing_condition_eq(self):
        """Test equality condition."""
        condition = RoutingCondition(field="status", operator="eq", value="complete")
        assert condition.evaluate({"status": "complete"}) is True
        assert condition.evaluate({"status": "pending"}) is False

    def test_routing_condition_numeric(self):
        """Test numeric conditions."""
        condition_gt = RoutingCondition(field="quality_score", operator="gt", value=0.5)
        assert condition_gt.evaluate({"quality_score": 0.8}) is True
        assert condition_gt.evaluate({"quality_score": 0.3}) is False

        condition_lt = RoutingCondition(field="retry_count", operator="lt", value=2)
        assert condition_lt.evaluate({"retry_count": 1}) is True
        assert condition_lt.evaluate({"retry_count": 3}) is False

    def test_routing_condition_in(self):
        """Test 'in' operator."""
        condition = RoutingCondition(field="agent", operator="in", value=["separation", "tea"])
        assert condition.evaluate({"agent": "separation"}) is True
        assert condition.evaluate({"agent": "literature"}) is False

    def test_routing_condition_contains(self):
        """Test 'contains' operator."""
        condition = RoutingCondition(field="solvents", operator="contains", value="xylene")
        assert condition.evaluate({"solvents": ["xylene", "toluene"]}) is True
        assert condition.evaluate({"solvents": ["toluene"]}) is False

    def test_routing_rule_single_condition(self):
        """Test routing rule with single condition."""
        rule = RoutingRule(
            name="high_quality",
            conditions=[RoutingCondition(field="quality_score", operator="gte", value=0.7)],
            target_agent="tea_agent"
        )
        assert rule.evaluate({"quality_score": 0.8}) is True
        assert rule.evaluate({"quality_score": 0.5}) is False

    def test_routing_rule_multiple_conditions_and(self):
        """Test routing rule with AND conditions."""
        rule = RoutingRule(
            name="retry_eligible",
            conditions=[
                RoutingCondition(field="quality_score", operator="lt", value=0.5),
                RoutingCondition(field="retry_count", operator="lt", value=2)
            ],
            all_conditions=True,
            target_agent="separation_agent"
        )
        assert rule.evaluate({"quality_score": 0.3, "retry_count": 1}) is True
        assert rule.evaluate({"quality_score": 0.3, "retry_count": 3}) is False

    def test_routing_rule_multiple_conditions_or(self):
        """Test routing rule with OR conditions."""
        rule = RoutingRule(
            name="needs_attention",
            conditions=[
                RoutingCondition(field="has_errors", operator="eq", value=True),
                RoutingCondition(field="quality_score", operator="lt", value=0.3)
            ],
            all_conditions=False,
            target_agent="review_agent"
        )
        assert rule.evaluate({"has_errors": True, "quality_score": 0.8}) is True
        assert rule.evaluate({"has_errors": False, "quality_score": 0.2}) is True
        assert rule.evaluate({"has_errors": False, "quality_score": 0.8}) is False

    def test_conditional_router(self):
        """Test conditional router."""
        router = ConditionalRouter(
            name="test_router",
            rules=[
                RoutingRule(
                    name="high_quality",
                    conditions=[RoutingCondition(field="quality_score", operator="gte", value=0.7)],
                    target_agent="tea_agent",
                    priority=10
                ),
                RoutingRule(
                    name="retry",
                    conditions=[RoutingCondition(field="quality_score", operator="lt", value=0.5)],
                    target_agent="separation_agent",
                    priority=5
                )
            ],
            default_target="aggregator"
        )

        target, _ = router.route({"quality_score": 0.8})
        assert target == "tea_agent"

        target, _ = router.route({"quality_score": 0.3})
        assert target == "separation_agent"

        target, _ = router.route({"quality_score": 0.6})
        assert target == "aggregator"

    def test_quality_based_router(self):
        """Test pre-defined quality-based router."""
        target, _ = QUALITY_BASED_ROUTER.route({"quality_score": 0.8, "retry_count": 0})
        assert target == "collab_tea_agent"


# ============================================================
# P4: CONTEXT PRUNING TESTS
# ============================================================

class TestContextPruning:
    """Tests for P4 context pruning."""

    def test_context_budget_defaults(self):
        """Test default context budget."""
        budget = ContextBudget()
        assert budget.max_tokens == 4000
        assert budget.max_messages == 20

    def test_context_summary_from_results(self):
        """Test creating context summary from results."""
        results = {
            "solvents": ["xylene", "toluene", "cyclohexane", "hexane", "ethanol"],
            "selectivities": [45.0, 38.0, 25.0, 20.0, 15.0],
            "cost_per_kg": 2.50,
            "best_sequence": ["PE", "PP", "PS"],
            "raw_response": "Very long raw text..." * 100
        }

        budget = ContextBudget(max_tokens=500, max_results_per_agent=3)
        summary = ContextSummary.from_results(results, budget)

        # Should have pruned
        assert summary.compression_ratio < 1.0
        # High priority fields kept
        assert "solvents" in summary.key_values
        assert "cost_per_kg" in summary.key_values
        # Low priority fields dropped
        assert "raw_response" in summary.dropped_fields

    def test_context_summary_truncates_lists(self):
        """Test that lists are truncated."""
        results = {
            "solvents": ["s1", "s2", "s3", "s4", "s5"],
        }
        budget = ContextBudget(max_results_per_agent=2)
        summary = ContextSummary.from_results(results, budget)

        if "solvents" in summary.key_values:
            assert len(summary.key_values["solvents"]) <= 2

    def test_prune_context_function(self):
        """Test prune_context function."""
        state = {
            "collaboration_mode": "separation_tea",
            "shared_context": {"polymers": ["PE", "PP"]},
            "trace_id": "test-123",
            "separation_results": {
                "solvents": ["xylene"],
                "raw_response": "long text..." * 50
            },
            "messages": [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        }

        budget = ContextBudget(max_messages=10, max_tokens=1000)
        pruned = prune_context(state, budget)

        # Essential fields kept
        assert "collaboration_mode" in pruned
        assert "trace_id" in pruned

        # Messages truncated
        assert len(pruned["messages"]) <= 10


# ============================================================
# P5: TOOL CHAINING TESTS
# ============================================================

class TestToolChaining:
    """Tests for P5 tool chaining."""

    def test_tool_call_definition(self):
        """Test tool call definition."""
        call = ToolCall(
            tool_name="find_separation",
            parameters={"polymers": "PE,PP"},
            output_key="separation_result",
            extract_fields=["solvents", "best_sequence"]
        )
        assert call.tool_name == "find_separation"
        assert len(call.extract_fields) == 2

    def test_tool_chain_definition(self):
        """Test tool chain definition."""
        chain = ToolChain(
            name="analysis_chain",
            description="Complete analysis workflow",
            tools=[
                ToolCall(tool_name="tool1", output_key="result1"),
                ToolCall(tool_name="tool2", depends_on=["result1"], output_key="result2"),
            ],
            stop_on_error=True
        )
        assert len(chain.tools) == 2

    def test_tool_chain_execution_order(self):
        """Test tool chain execution order calculation."""
        chain = ToolChain(
            name="ordered_chain",
            tools=[
                ToolCall(tool_name="tool1", output_key="r1"),
                ToolCall(tool_name="tool2", depends_on=["r1"], output_key="r2"),
                ToolCall(tool_name="tool3", depends_on=["r2"], output_key="r3"),
            ]
        )

        order = chain.get_execution_order()
        # Should be 3 levels
        assert len(order) == 3
        assert order[0] == ["r1"]
        assert order[1] == ["r2"]
        assert order[2] == ["r3"]

    def test_tool_chain_parallel_execution_order(self):
        """Test execution order with parallel opportunities."""
        chain = ToolChain(
            name="parallel_chain",
            tools=[
                ToolCall(tool_name="setup", output_key="setup"),
                ToolCall(tool_name="analysis1", depends_on=["setup"], output_key="a1"),
                ToolCall(tool_name="analysis2", depends_on=["setup"], output_key="a2"),
                ToolCall(tool_name="combine", depends_on=["a1", "a2"], output_key="final"),
            ]
        )

        order = chain.get_execution_order()
        # Should be 3 levels: setup, [a1,a2], final
        assert len(order) == 3
        assert order[0] == ["setup"]
        assert set(order[1]) == {"a1", "a2"}
        assert order[2] == ["final"]

    def test_separation_analysis_chain(self):
        """Test pre-defined separation analysis chain."""
        order = SEPARATION_ANALYSIS_CHAIN.get_execution_order()
        assert len(order) >= 2

    def test_tea_comparison_chain(self):
        """Test pre-defined TEA comparison chain."""
        assert len(TEA_COMPARISON_CHAIN.tools) == 2


# ============================================================
# P5: DEPENDENCY GRAPH TESTS
# ============================================================

class TestDependencyGraph:
    """Tests for P5 agent dependency graph."""

    def test_agent_dependency_definition(self):
        """Test agent dependency definition."""
        dep = AgentDependency(
            agent="tea_lca",
            depends_on=["separation"],
            dependency_type="required",
            required_data=["solvents", "throughput_kg_hr"]
        )
        assert dep.agent == "tea_lca"
        assert "separation" in dep.depends_on

    def test_agent_graph_execution_order(self):
        """Test agent graph execution order."""
        graph = AgentGraph(
            name="test_graph",
            agents=[
                AgentDependency(agent="router", depends_on=[]),
                AgentDependency(agent="separation", depends_on=["router"]),
                AgentDependency(agent="tea", depends_on=["separation"]),
            ]
        )

        order = graph.get_execution_order()
        assert len(order) == 3
        assert order[0] == ["router"]
        assert order[1] == ["separation"]
        assert order[2] == ["tea"]

    def test_agent_graph_parallel_groups(self):
        """Test agent graph parallel groups."""
        graph = AgentGraph(
            name="parallel_graph",
            agents=[
                AgentDependency(agent="router", depends_on=[]),
                AgentDependency(agent="separation", depends_on=["router"]),
                AgentDependency(agent="literature", depends_on=["router"]),
                AgentDependency(agent="aggregator", depends_on=["separation", "literature"]),
            ]
        )

        groups = graph.get_parallel_groups()
        # Router first, then separation+literature in parallel, then aggregator
        assert len(groups) == 3
        assert set(groups[1]) == {"separation", "literature"}

    def test_agent_graph_validation(self):
        """Test agent graph validation."""
        # Valid graph
        valid_graph = AgentGraph(
            name="valid",
            agents=[
                AgentDependency(agent="a", depends_on=[]),
                AgentDependency(agent="b", depends_on=["a"]),
            ]
        )
        errors = valid_graph.validate()
        assert len(errors) == 0

        # Graph with missing dependency
        invalid_graph = AgentGraph(
            name="invalid",
            agents=[
                AgentDependency(agent="a", depends_on=["unknown"]),
            ]
        )
        errors = invalid_graph.validate()
        assert len(errors) > 0

    def test_default_agent_graph(self):
        """Test default agent graph is valid."""
        errors = DEFAULT_AGENT_GRAPH.validate()
        assert len(errors) == 0

        order = DEFAULT_AGENT_GRAPH.get_execution_order()
        assert len(order) > 0


# ============================================================
# P5: OBSERVABILITY TESTS
# ============================================================

class TestObservability:
    """Tests for P5 observability features."""

    def test_decision_log_creation(self):
        """Test creating decision log."""
        log = DecisionLog(
            decision_id="d001",
            agent="separation",
            decision_type=DecisionType.TOOL_SELECTION,
            options_considered=["tool1", "tool2", "tool3"],
            chosen_option="tool1",
            reasoning="Best selectivity score",
            confidence=0.9
        )
        assert log.decision_type == DecisionType.TOOL_SELECTION
        assert log.chosen_option == "tool1"

    def test_observability_config(self):
        """Test observability configuration."""
        config = ObservabilityConfig(
            enabled=True,
            log_decisions=True,
            log_tool_calls=True,
            max_logs_in_state=50
        )
        assert config.enabled is True

    def test_agent_observer_log_decision(self):
        """Test agent observer logging decisions."""
        observer = AgentObserver()

        decision_id = observer.log_decision(
            agent="separation",
            decision_type=DecisionType.ROUTING,
            options=["tea_agent", "literature_agent"],
            chosen="tea_agent",
            reasoning="TEA analysis needed",
            confidence=0.85
        )

        assert len(observer.logs) == 1
        assert observer.logs[0].decision_id == decision_id

    def test_agent_observer_update_outcome(self):
        """Test updating decision outcome."""
        observer = AgentObserver()

        decision_id = observer.log_decision(
            agent="separation",
            decision_type=DecisionType.TOOL_SELECTION,
            options=["tool1", "tool2"],
            chosen="tool1"
        )

        observer.update_outcome(decision_id, "Success", True)

        assert observer.logs[0].outcome == "Success"
        assert observer.logs[0].outcome_success is True

    def test_agent_observer_max_logs(self):
        """Test that observer respects max logs limit."""
        config = ObservabilityConfig(max_logs_in_state=5)
        observer = AgentObserver(config)

        for i in range(10):
            observer.log_decision(
                agent="test",
                decision_type=DecisionType.PARAMETER_CHOICE,
                options=["a", "b"],
                chosen="a"
            )

        assert len(observer.logs) == 5

    def test_agent_observer_generate_report(self):
        """Test generating decision report."""
        observer = AgentObserver()

        observer.log_decision(
            agent="separation",
            decision_type=DecisionType.TOOL_SELECTION,
            options=["tool1", "tool2"],
            chosen="tool1"
        )
        observer.log_decision(
            agent="tea",
            decision_type=DecisionType.ROUTING,
            options=["aggregator", "retry"],
            chosen="aggregator"
        )

        report = observer.generate_report()
        assert "Decision Report" in report
        assert "separation" in report
        assert "tea" in report

    def test_global_observer(self):
        """Test global observer access."""
        observer1 = get_observer()
        observer2 = get_observer()
        assert observer1 is observer2

    def test_log_agent_decision_convenience(self):
        """Test convenience function for logging."""
        # Clear existing logs
        get_observer().logs.clear()

        decision_id = log_agent_decision(
            agent="test_agent",
            decision_type=DecisionType.TERMINATION,
            options=["continue", "stop"],
            chosen="stop",
            reasoning="Task complete"
        )

        assert decision_id != ""
        assert len(get_observer().logs) >= 1


# ============================================================
# INTEGRATION TESTS
# ============================================================

class TestP3P5Integration:
    """Integration tests combining P3-P5 features."""

    def test_tool_output_to_handoff_validation(self):
        """Test tool output flowing through handoff validation."""
        # Tool produces output
        tool_output = SeparationToolOutput(
            tool_name="find_separation",
            solvents=["xylene", "toluene"],
            selectivities=[45.0, 38.0],
            best_sequence=["PE", "PP"],
            success=True
        )

        # Create handoff task params from tool output
        task_params = {
            "solvents": tool_output.solvents,
            "throughput_kg_hr": 100.0,
            "best_sequence": tool_output.best_sequence
        }

        # Validate handoff
        result = validate_handoff("separation", "tea_lca", task_params)
        assert result.is_valid is True

    def test_conditional_routing_with_tool_output(self):
        """Test routing decisions based on tool output quality."""
        # Simulate tool output quality metrics
        state = {
            "quality_score": 0.4,
            "solvents_count": 1,
            "retry_count": 0
        }

        target, updates = QUALITY_BASED_ROUTER.route(state)

        # Low quality should trigger retry
        assert target == "collab_separation_agent"

    def test_error_recovery_with_partial_results(self):
        """Test error recovery passing partial results."""
        # Simulate partial tool failure
        partial = PartialResult(
            agent="separation",
            completion_percentage=40.0,
            partial_data={"solvents": ["xylene"]},
            failed_step="optimize_sequence",
            error_message="Timeout",
            can_continue=True,
            fallback_values={"best_sequence": ["PE", "PP"]}
        )

        # Convert to handoff context
        context = partial.to_handoff_context()

        # Downstream can use partial data
        assert context["upstream_partial"] is True
        assert "xylene" in context["available_data"]["solvents"]

    def test_observability_tracks_routing_decisions(self):
        """Test that observability logs routing decisions."""
        observer = AgentObserver()
        observer.logs.clear()

        # Simulate routing decision
        state = {"quality_score": 0.8}
        target, _ = QUALITY_BASED_ROUTER.route(state)

        # Log the decision
        observer.log_decision(
            agent="router",
            decision_type=DecisionType.ROUTING,
            options=["separation_agent", "tea_agent", "aggregator"],
            chosen=target,
            reasoning="Quality score above threshold",
            state=state
        )

        assert len(observer.logs) == 1
        assert observer.logs[0].chosen_option == target


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
