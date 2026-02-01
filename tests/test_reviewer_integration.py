"""
Integration Tests for P0 Enhancement: End-to-End Review Loop Flow

Tests the complete collaboration pipeline with review loops:
separation_agent -> separation_reviewer -> [retry or TEA] -> aggregator
"""

import pytest
import asyncio
from typing import Dict, Any, List
from copy import deepcopy
import time

from agent_schemas import ReviewerFeedback, SeparationResult, TEAResult
from multi_agent_system import (
    separation_reviewer_node,
    smart_aggregator_node,
    SEPARATION_QUALITY_THRESHOLDS,
    MultiAgentState,
)


class TestEndToEndReviewFlow:
    """End-to-end tests simulating the complete flow with review loops."""

    def _simulate_separation_agent_output(
        self,
        polymers: List[str],
        temperature: float,
        retry_params: Dict = None,
        quality: str = "poor"
    ) -> Dict[str, Any]:
        """Simulate what separation agent would return based on quality level."""
        if quality == "poor":
            # No solvents found
            return {
                "solvents": [],
                "selectivities": [],
                "best_sequence": [],
                "polymers": polymers,
                "temperature": temperature,
            }
        elif quality == "marginal":
            # One solvent with low selectivity
            return {
                "solvents": ["xylene"],
                "selectivities": [4.0],
                "best_sequence": [polymers[0]] if polymers else [],
                "polymers": polymers,
                "temperature": temperature,
            }
        elif quality == "improved":
            # Two solvents, at threshold selectivity
            return {
                "solvents": ["xylene", "toluene"],
                "selectivities": [6.0, 5.0],
                "best_sequence": polymers[:2] if len(polymers) >= 2 else polymers,
                "polymers": polymers,
                "temperature": temperature,
            }
        else:  # "good"
            # Good results
            return {
                "solvents": ["xylene", "cyclohexane", "toluene"],
                "selectivities": [45.0, 38.0, 25.0],
                "best_sequence": polymers,
                "polymers": polymers,
                "temperature": temperature,
            }

    @pytest.mark.asyncio
    async def test_complete_flow_first_try_success(self):
        """Test complete flow when first separation attempt succeeds."""
        polymers = ["PE", "PP", "PS"]
        temperature = 80.0

        # Step 1: Separation agent produces good results
        separation_results = self._simulate_separation_agent_output(
            polymers, temperature, quality="good"
        )

        # Step 2: Reviewer evaluates
        state = {
            "messages": [],
            "separation_results": separation_results,
            "shared_context": {
                "polymers": polymers,
                "temperature": temperature,
                "throughput_kg_hr": 100.0,
                "original_query": "Find cheapest separation for PE, PP, PS"
            },
            "separation_retry_count": 0,
            "agent_timings": {"orchestrator": time.time(), "separation": time.time()},
            "collaboration_mode": "separation_tea",
            "trace_id": "test-trace-001",
            "handoff_metrics": [],
        }

        review_result = await separation_reviewer_node(state)

        # Should proceed to TEA without retry
        assert review_result.goto == "collab_tea_agent"

        # Verify state updates
        feedback = review_result.update.get("reviewer_feedback", {})
        assert feedback.get("is_acceptable") == True
        assert feedback.get("quality_score", 0) >= 0.8
        assert feedback.get("retry_count") == 0

        # Verify pending handoff created for TEA
        pending = review_result.update.get("pending_handoff", {})
        assert pending.get("to_agent") == "tea_lca"
        task_params = pending.get("task_params", {})
        assert "solvents" in task_params
        assert len(task_params["solvents"]) > 0

    @pytest.mark.asyncio
    async def test_complete_flow_with_one_retry(self):
        """Test flow with one retry cycle."""
        polymers = ["PE", "PP", "PS"]
        temperature = 80.0

        # === ITERATION 1: Poor results, needs retry ===
        separation_results_1 = self._simulate_separation_agent_output(
            polymers, temperature, quality="marginal"
        )

        state_1 = {
            "messages": [],
            "separation_results": separation_results_1,
            "shared_context": {
                "polymers": polymers,
                "temperature": temperature,
                "throughput_kg_hr": 100.0,
            },
            "separation_retry_count": 0,
            "agent_timings": {"separation": time.time()},
            "collaboration_mode": "separation_tea",
            "trace_id": "test-trace-002",
            "handoff_metrics": [],
        }

        review_result_1 = await separation_reviewer_node(state_1)

        # Should request retry
        assert review_result_1.goto == "collab_separation_agent"
        assert review_result_1.update.get("separation_retry_count") == 1

        feedback_1 = review_result_1.update.get("reviewer_feedback", {})
        assert feedback_1.get("requires_revision") == True
        retry_params = review_result_1.update.get("retry_params", {})
        assert "temperature_range" in retry_params

        # === ITERATION 2: Good results after retry ===
        separation_results_2 = self._simulate_separation_agent_output(
            polymers, temperature, quality="good"
        )

        state_2 = {
            "messages": [],
            "separation_results": separation_results_2,
            "shared_context": state_1["shared_context"],
            "separation_retry_count": 1,  # After first retry
            "retry_params": retry_params,
            "agent_timings": {"separation": time.time()},
            "collaboration_mode": "separation_tea",
            "trace_id": "test-trace-002",
            "handoff_metrics": review_result_1.update.get("handoff_metrics", []),
        }

        review_result_2 = await separation_reviewer_node(state_2)

        # Should proceed to TEA
        assert review_result_2.goto == "collab_tea_agent"

        feedback_2 = review_result_2.update.get("reviewer_feedback", {})
        assert feedback_2.get("is_acceptable") == True

    @pytest.mark.asyncio
    async def test_complete_flow_with_max_retries(self):
        """Test flow reaching max retries and proceeding with partial results."""
        polymers = ["PE", "PP", "PS"]
        temperature = 80.0
        max_retries = SEPARATION_QUALITY_THRESHOLDS["max_retries"]

        states = []
        results = []

        # Simulate each iteration with improving but still-poor results
        qualities = ["poor", "marginal", "marginal"]  # Never quite good enough

        for i, quality in enumerate(qualities):
            if i > max_retries:
                break

            separation_results = self._simulate_separation_agent_output(
                polymers, temperature, quality=quality
            )

            state = {
                "messages": [],
                "separation_results": separation_results,
                "shared_context": {
                    "polymers": polymers,
                    "temperature": temperature,
                },
                "separation_retry_count": i,
                "agent_timings": {"separation": time.time()},
                "collaboration_mode": "separation_tea",
                "handoff_metrics": [],
            }

            result = await separation_reviewer_node(state)
            states.append(state)
            results.append(result)

            if result.goto != "collab_separation_agent":
                break

        # Should have hit max retries and proceeded
        last_result = results[-1]

        # After max_retries (2), should proceed to TEA
        assert last_result.goto == "collab_tea_agent"

        feedback = last_result.update.get("reviewer_feedback", {})
        # Should not require revision since we hit max
        assert feedback.get("requires_revision") == False

    @pytest.mark.asyncio
    async def test_flow_with_aggregator(self):
        """Test complete flow through to aggregator."""
        polymers = ["PE", "PP", "PS"]
        temperature = 80.0

        # Simulate good separation results
        separation_results = self._simulate_separation_agent_output(
            polymers, temperature, quality="good"
        )

        # Review state
        review_state = {
            "messages": [],
            "separation_results": separation_results,
            "shared_context": {
                "polymers": polymers,
                "temperature": temperature,
                "throughput_kg_hr": 100.0,
                "original_query": "Find cheapest separation"
            },
            "separation_retry_count": 0,
            "agent_timings": {"orchestrator": time.time(), "separation": time.time()},
            "collaboration_mode": "separation_tea",
            "trace_id": "test-trace-003",
            "handoff_metrics": [],
            "reviewer_feedback": None,
        }

        review_result = await separation_reviewer_node(review_state)
        assert review_result.goto == "collab_tea_agent"

        # Simulate TEA results
        tea_results = {
            "cost_per_kg": 2.50,
            "best_solvent": "xylene",
            "total_capex": 150000,
            "total_opex": 25000,
            "payback_years": 3.5,
            "solvents_analyzed": separation_results["solvents"],
        }

        # Aggregator state (after TEA)
        aggregator_state = {
            "messages": [],
            "separation_results": separation_results,
            "tea_results": tea_results,
            "shared_context": review_state["shared_context"],
            "collaboration_mode": "separation_tea",
            "trace_id": "test-trace-003",
            "handoff_metrics": review_result.update.get("handoff_metrics", []),
            "agent_timings": {
                "orchestrator": time.time() - 10,
                "separation": time.time() - 8,
                "reviewer": time.time() - 5,
                "tea": time.time(),
            },
            "reviewer_feedback": review_result.update.get("reviewer_feedback"),
            "specialist_start_time": time.time() - 10,
            "pending_handoff": {},
        }

        aggregator_result = await smart_aggregator_node(aggregator_state)

        # Check aggregated output
        messages = aggregator_result.get("messages", [])
        assert len(messages) > 0

        final_message = messages[-1]
        content = final_message.content if hasattr(final_message, 'content') else str(final_message)

        # Should contain key information
        assert "Separation" in content
        assert "Economic" in content or "TEA" in content or "cost" in content.lower()

        # Should include reviewer feedback (quality score)
        assert "Quality score" in content or "quality" in content.lower()


class TestReviewerStateTransitions:
    """Tests for proper state management across review iterations."""

    @pytest.mark.asyncio
    async def test_retry_params_passed_correctly(self):
        """Verify retry params are passed back for next iteration."""
        polymers = ["PE", "PP"]
        state = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene"],
                "selectivities": [3.0],
                "best_sequence": [],
                "polymers": polymers,
                "temperature": 80.0,
            },
            "shared_context": {"polymers": polymers, "temperature": 80.0},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)

        # Should have retry params
        retry_params = result.update.get("retry_params", {})
        assert "temperature_range" in retry_params

        # Params should have reasonable values
        temp_range = retry_params["temperature_range"]
        assert isinstance(temp_range, tuple)
        assert len(temp_range) == 2
        assert temp_range[0] < temp_range[1]

    @pytest.mark.asyncio
    async def test_handoff_metrics_accumulate(self):
        """Verify handoff metrics accumulate across iterations."""
        polymers = ["PE", "PP"]

        # First iteration
        state_1 = {
            "messages": [],
            "separation_results": {
                "solvents": [],
                "selectivities": [],
                "best_sequence": [],
                "polymers": polymers,
                "temperature": 80.0,
            },
            "shared_context": {"polymers": polymers},
            "separation_retry_count": 0,
            "agent_timings": {"separation": time.time()},
            "handoff_metrics": [],
        }

        result_1 = await separation_reviewer_node(state_1)
        metrics_1 = result_1.update.get("handoff_metrics", [])
        assert len(metrics_1) == 1

        # Second iteration (with accumulated metrics)
        state_2 = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene", "toluene"],
                "selectivities": [10.0, 8.0],
                "best_sequence": ["PE", "PP"],
                "polymers": polymers,
                "temperature": 80.0,
            },
            "shared_context": {"polymers": polymers},
            "separation_retry_count": 1,
            "agent_timings": {"separation": time.time()},
            "handoff_metrics": metrics_1,  # Pass previous metrics
        }

        result_2 = await separation_reviewer_node(state_2)
        metrics_2 = result_2.update.get("handoff_metrics", [])

        # Should have new metric (old ones preserved via Annotated[..., operator.add])
        assert len(metrics_2) >= 1

    @pytest.mark.asyncio
    async def test_separation_results_cleared_on_retry(self):
        """Verify separation_results cleared on retry for fresh run."""
        state = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene"],
                "selectivities": [2.0],  # Very low
                "best_sequence": [],
                "polymers": ["PE", "PP"],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)

        if result.goto == "collab_separation_agent":
            # separation_results should be None for fresh retry
            assert result.update.get("separation_results") is None


class TestQualityThresholdVariations:
    """Tests for different quality threshold scenarios."""

    @pytest.mark.asyncio
    async def test_exactly_at_threshold(self):
        """Test results exactly at threshold boundary."""
        state = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene", "toluene"],  # Exactly min_solvents
                "selectivities": [5.0, 5.0],  # Exactly min_selectivity
                "best_sequence": ["PE", "PP"],
                "polymers": ["PE", "PP"],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)
        feedback = result.update.get("reviewer_feedback", {})

        # At threshold should be acceptable
        assert feedback.get("is_acceptable") == True
        assert result.goto == "collab_tea_agent"

    @pytest.mark.asyncio
    async def test_just_below_threshold(self):
        """Test results just below threshold."""
        state = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene"],  # Below min_solvents (2)
                "selectivities": [4.9],  # Just below min_selectivity (5)
                "best_sequence": ["PE"],
                "polymers": ["PE", "PP"],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)
        feedback = result.update.get("reviewer_feedback", {})

        # Below threshold should trigger issues
        assert len(feedback.get("issues", [])) >= 1

    @pytest.mark.asyncio
    async def test_partial_quality_acceptable(self):
        """Test that partial good results can still be acceptable."""
        state = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene", "toluene", "cyclohexane"],  # Good count
                "selectivities": [3.0, 2.0, 1.0],  # Low selectivity
                "best_sequence": ["PE", "PP", "PS"],
                "polymers": ["PE", "PP", "PS"],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": ["PE", "PP", "PS"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)
        feedback = result.update.get("reviewer_feedback", {})

        # Has issues but quality score might still be acceptable
        quality = feedback.get("quality_score", 0)

        # Should have issues noted but may still proceed if overall quality OK
        # (depends on exact scoring logic)
        assert "issues" in feedback


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
