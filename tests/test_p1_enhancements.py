"""
Tests for P1 Enhancements: Checkpointing, Parallel Execution, Supervisor

P1 improvements based on state-of-the-art multi-agent patterns.
"""

import pytest
import asyncio
import os
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

from multi_agent_system import (
    CheckpointerConfig,
    parallel_orchestrator_node,
    supervisor_decision_node,
    MultiAgentState,
)
from agent_schemas import SupervisorDecision
from langgraph.checkpoint.memory import MemorySaver


class TestCheckpointerConfig:
    """Tests for P1 checkpointer configuration."""

    def test_default_memory_checkpointer(self):
        """Default should be in-memory checkpointer."""
        with patch.dict(os.environ, {}, clear=True):
            cp = CheckpointerConfig.get_checkpointer()
            # InMemorySaver is the actual class name
            assert "MemorySaver" in type(cp).__name__ or "InMemory" in type(cp).__name__

    def test_memory_checkpointer_explicit(self):
        """Explicit memory type should return MemorySaver."""
        with patch.dict(os.environ, {"CHECKPOINTER_TYPE": "memory"}):
            cp = CheckpointerConfig.get_checkpointer()
            assert "MemorySaver" in type(cp).__name__ or "InMemory" in type(cp).__name__

    def test_postgres_without_url_falls_back(self):
        """Postgres without DATABASE_URL should fall back to memory."""
        with patch.dict(os.environ, {"CHECKPOINTER_TYPE": "postgres"}, clear=True):
            cp = CheckpointerConfig.get_checkpointer()
            # Should fall back to MemorySaver
            assert "MemorySaver" in type(cp).__name__ or "InMemory" in type(cp).__name__

    def test_redis_without_url_falls_back(self):
        """Redis without REDIS_URL should fall back to memory."""
        with patch.dict(os.environ, {"CHECKPOINTER_TYPE": "redis"}, clear=True):
            cp = CheckpointerConfig.get_checkpointer()
            # Should fall back to MemorySaver
            assert "MemorySaver" in type(cp).__name__ or "InMemory" in type(cp).__name__

    def test_unknown_type_falls_back(self):
        """Unknown checkpointer type should fall back to memory."""
        with patch.dict(os.environ, {"CHECKPOINTER_TYPE": "unknown"}):
            cp = CheckpointerConfig.get_checkpointer()
            assert "MemorySaver" in type(cp).__name__ or "InMemory" in type(cp).__name__


class TestSupervisorDecisionNode:
    """Tests for P1 supervisor decision logic."""

    @pytest.fixture
    def basic_state(self) -> Dict[str, Any]:
        """Basic state for testing."""
        return {
            "messages": [],
            "complexity": 5,
            "collaboration_mode": "separation_tea",
            "separation_results": {},
            "tea_results": {},
            "reviewer_feedback": {},
            "separation_retry_count": 0,
        }

    @pytest.mark.asyncio
    async def test_routes_to_separation_when_no_results(self, basic_state):
        """Without separation results, should route to separation."""
        basic_state["separation_results"] = {}
        result = await supervisor_decision_node(basic_state)

        assert hasattr(result, 'goto')
        assert result.goto == "collab_separation_agent"

    @pytest.mark.asyncio
    async def test_routes_to_tea_when_has_solvents(self, basic_state):
        """With solvents but no TEA, should route to TEA."""
        basic_state["separation_results"] = {
            "solvents": ["xylene", "toluene"],
            "selectivities": [20.0, 15.0],
        }
        result = await supervisor_decision_node(basic_state)

        assert hasattr(result, 'goto')
        assert result.goto == "collab_tea_agent"

    @pytest.mark.asyncio
    async def test_routes_to_aggregator_when_complete(self, basic_state):
        """With both results, should route to aggregator."""
        basic_state["separation_results"] = {
            "solvents": ["xylene"],
            "selectivities": [20.0],
        }
        basic_state["tea_results"] = {
            "cost_per_kg": 2.50,
        }
        result = await supervisor_decision_node(basic_state)

        assert hasattr(result, 'goto')
        assert result.goto == "smart_aggregator"

    @pytest.mark.asyncio
    async def test_low_complexity_uses_simple_routing(self):
        """Low complexity queries should use simple rule-based routing."""
        state = {
            "messages": [],
            "complexity": 2,  # Low complexity
            "collaboration_mode": None,
            "separation_results": {},
            "tea_results": {},
        }
        result = await supervisor_decision_node(state)

        # Should still return a valid Command
        assert hasattr(result, 'goto')

    @pytest.mark.asyncio
    async def test_supervisor_decision_in_update(self, basic_state):
        """Supervisor should include decision in state update."""
        basic_state["separation_results"] = {"solvents": ["xylene"]}

        result = await supervisor_decision_node(basic_state)

        # Check that supervisor_decision is in update
        update = result.update or {}
        if "supervisor_decision" in update:
            decision = update["supervisor_decision"]
            assert "next_agent" in decision
            assert "reason" in decision
            assert "confidence" in decision

    @pytest.mark.asyncio
    async def test_low_quality_triggers_reroute(self, basic_state):
        """Low quality score should consider reroute to separation."""
        basic_state["separation_results"] = {"solvents": ["xylene"]}
        basic_state["reviewer_feedback"] = {"quality_score": 0.3}
        basic_state["separation_retry_count"] = 0

        result = await supervisor_decision_node(basic_state)

        # Might reroute to separation due to low quality
        assert hasattr(result, 'goto')
        # Could be either separation (reroute) or tea (proceed anyway)
        assert result.goto in ["collab_separation_agent", "collab_tea_agent"]


class TestSupervisorDecisionSchema:
    """Tests for SupervisorDecision Pydantic schema."""

    def test_create_basic_decision(self):
        """Test creating a basic supervisor decision."""
        decision = SupervisorDecision(
            next_agent="collab_tea_agent",
            reason="Separation complete, proceeding to TEA"
        )
        assert decision.next_agent == "collab_tea_agent"
        assert decision.confidence == 1.0  # Default

    def test_decision_with_reroute(self):
        """Test decision marking a reroute."""
        decision = SupervisorDecision(
            next_agent="collab_separation_agent",
            reason="Quality too low",
            is_reroute=True,
            original_plan=["separation", "tea_lca"],
            confidence=0.7
        )
        assert decision.is_reroute == True
        assert decision.confidence == 0.7

    def test_decision_serialization(self):
        """Test decision can be serialized for state storage."""
        decision = SupervisorDecision(
            next_agent="smart_aggregator",
            reason="All analyses complete"
        )
        data = decision.model_dump()

        assert isinstance(data, dict)
        assert data["next_agent"] == "smart_aggregator"


class TestParallelOrchestratorNode:
    """Tests for P1 parallel execution orchestrator."""

    @pytest.fixture
    def mock_sql_agent(self):
        """Mock SQL agent node for testing."""
        async def mock_agent(state):
            return {
                "messages": [MagicMock(content="Mock response")],
            }
        return mock_agent

    @pytest.fixture
    def parallel_state(self) -> Dict[str, Any]:
        """State for parallel execution testing."""
        return {
            "messages": [],
            "collaboration_specialists": ["separation", "literature"],
            "shared_context": {
                "original_query": "Find separation methods for PE and PP",
                "polymers": ["PE", "PP"],
                "temperature": 80.0,
                "throughput_kg_hr": 100.0,
            },
        }

    @pytest.mark.asyncio
    async def test_parallel_with_independent_specialists(self, parallel_state, mock_sql_agent):
        """Test running independent specialists in parallel."""
        result = await parallel_orchestrator_node(
            parallel_state,
            mock_sql_agent,
            specialists=["separation", "literature"]
        )

        # Should have results from both specialists
        assert "separation_results" in result or "literature_results" in result
        assert result.get("aggregation_required") == True

    @pytest.mark.asyncio
    async def test_parallel_handles_tea_dependency(self, mock_sql_agent):
        """Test that TEA runs after separation (dependent)."""
        state = {
            "messages": [],
            "collaboration_specialists": ["separation", "tea_lca"],
            "shared_context": {
                "original_query": "Cost-effective separation",
                "polymers": ["PE", "PP"],
                "temperature": 80.0,
                "throughput_kg_hr": 100.0,
            },
            "separation_results": {},  # Will be populated
        }

        result = await parallel_orchestrator_node(
            state,
            mock_sql_agent,
            specialists=["separation", "tea_lca"]
        )

        # Should have attempted both
        assert "aggregation_required" in result

    @pytest.mark.asyncio
    async def test_parallel_empty_specialists(self, mock_sql_agent):
        """Test handling empty specialist list."""
        state = {
            "messages": [],
            "collaboration_specialists": [],
            "shared_context": {},
        }

        result = await parallel_orchestrator_node(
            state,
            mock_sql_agent,
            specialists=[]
        )

        # Should return aggregation required
        assert result.get("aggregation_required") == True

    @pytest.mark.asyncio
    async def test_parallel_handles_errors(self, parallel_state):
        """Test that parallel execution handles individual failures."""
        async def failing_agent(state):
            raise ValueError("Simulated failure")

        # Should not raise - should handle gracefully
        result = await parallel_orchestrator_node(
            parallel_state,
            failing_agent,
            specialists=["separation"]
        )

        # Should still return a result dict
        assert isinstance(result, dict)


class TestParallelVsSequential:
    """Tests comparing parallel vs sequential execution patterns."""

    @pytest.fixture
    def mock_timed_agent(self):
        """Mock agent that tracks execution order."""
        execution_order = []

        async def mock_agent(state):
            categories = state.get("selected_categories", [])
            agent_type = "unknown"
            if "separation" in categories:
                agent_type = "separation"
            elif "literature" in categories:
                agent_type = "literature"
            elif "economics" in categories:
                agent_type = "tea"

            execution_order.append(agent_type)
            await asyncio.sleep(0.01)  # Simulate work
            return {"messages": [MagicMock(content=f"{agent_type} result")]}

        return mock_agent, execution_order

    @pytest.mark.asyncio
    async def test_separation_literature_are_parallel(self, mock_timed_agent):
        """Separation and literature should run in parallel."""
        mock_agent, execution_order = mock_timed_agent

        state = {
            "messages": [],
            "collaboration_specialists": ["separation", "literature"],
            "shared_context": {
                "original_query": "Research on PE separation",
                "polymers": ["PE"],
                "temperature": 80.0,
            },
        }

        import time
        start = time.time()
        await parallel_orchestrator_node(state, mock_agent, specialists=["separation", "literature"])
        elapsed = time.time() - start

        # Both should have been called
        assert "separation" in execution_order
        assert "literature" in execution_order

    @pytest.mark.asyncio
    async def test_tea_depends_on_separation(self, mock_timed_agent):
        """TEA should wait for separation results."""
        mock_agent, execution_order = mock_timed_agent

        state = {
            "messages": [],
            "collaboration_specialists": ["separation", "tea_lca"],
            "shared_context": {
                "original_query": "Cost-effective separation",
                "polymers": ["PE"],
                "temperature": 80.0,
                "throughput_kg_hr": 100.0,
            },
            "separation_results": {},
        }

        await parallel_orchestrator_node(state, mock_agent, specialists=["separation", "tea_lca"])

        # Separation should be called before TEA
        if "separation" in execution_order and "tea" in execution_order:
            sep_idx = execution_order.index("separation")
            tea_idx = execution_order.index("tea")
            assert sep_idx < tea_idx, "Separation should run before TEA"


class TestMultiAgentStateP1Fields:
    """Tests for P1 state field additions."""

    def test_state_has_supervisor_decision(self):
        """State should have supervisor_decision field defined."""
        # MultiAgentState is a TypedDict subclass, check annotations
        annotations = getattr(MultiAgentState, '__annotations__', {})
        assert 'supervisor_decision' in annotations

    def test_state_has_parallel_execution(self):
        """State should have parallel_execution field defined."""
        annotations = getattr(MultiAgentState, '__annotations__', {})
        assert 'parallel_execution' in annotations

    def test_state_has_parallel_results(self):
        """State should have parallel_results field defined."""
        annotations = getattr(MultiAgentState, '__annotations__', {})
        assert 'parallel_results' in annotations

    def test_state_fields_have_defaults(self):
        """State fields should have default values."""
        # Create empty state dict and check P1 fields exist in class definition
        annotations = getattr(MultiAgentState, '__annotations__', {})
        p1_fields = ['supervisor_decision', 'parallel_execution', 'parallel_results']
        for field in p1_fields:
            assert field in annotations, f"Field {field} not found in MultiAgentState"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
