"""
Tests for P0 Enhancement: Review/Revision Loop

Tests the separation_reviewer_node and ReviewerFeedback schema.
"""

import pytest
import asyncio
from typing import Dict, Any

# Import test targets
from agent_schemas import ReviewerFeedback, SeparationResult
from multi_agent_system import (
    separation_reviewer_node,
    SEPARATION_QUALITY_THRESHOLDS,
    MultiAgentState,
)


class TestReviewerFeedbackSchema:
    """Tests for ReviewerFeedback Pydantic schema."""

    def test_create_feedback_defaults(self):
        """Test default values for ReviewerFeedback."""
        feedback = ReviewerFeedback()
        assert feedback.is_acceptable == True
        assert feedback.quality_score == 1.0
        assert feedback.issues == []
        assert feedback.requires_revision == False
        assert feedback.retry_count == 0

    def test_create_feedback_with_issues(self):
        """Test creating feedback with issues."""
        feedback = ReviewerFeedback(
            is_acceptable=False,
            quality_score=0.4,
            issues=["Too few solvents", "Low selectivity"],
            requires_revision=True,
            retry_count=1,
        )
        assert feedback.is_acceptable == False
        assert feedback.quality_score == 0.4
        assert len(feedback.issues) == 2
        assert feedback.requires_revision == True

    def test_quality_score_bounds(self):
        """Test quality score stays within bounds."""
        feedback = ReviewerFeedback(quality_score=0.0)
        assert feedback.quality_score == 0.0

        feedback = ReviewerFeedback(quality_score=1.0)
        assert feedback.quality_score == 1.0

    def test_retry_params(self):
        """Test retry parameters storage."""
        feedback = ReviewerFeedback(
            retry_params={
                "temperature_range": (60, 120),
                "retry_reason": "Low selectivity",
            }
        )
        assert feedback.retry_params["temperature_range"] == (60, 120)
        assert "retry_reason" in feedback.retry_params


class TestQualityThresholds:
    """Tests for quality threshold configuration."""

    def test_threshold_values(self):
        """Test quality threshold values are reasonable."""
        assert SEPARATION_QUALITY_THRESHOLDS["min_solvents"] >= 1
        assert SEPARATION_QUALITY_THRESHOLDS["min_selectivity"] >= 0
        assert SEPARATION_QUALITY_THRESHOLDS["max_retries"] >= 1
        assert SEPARATION_QUALITY_THRESHOLDS["temperature_expansion"] > 0

    def test_thresholds_not_too_strict(self):
        """Ensure thresholds allow reasonable results to pass."""
        # Should accept: 2+ solvents, 5%+ selectivity
        assert SEPARATION_QUALITY_THRESHOLDS["min_solvents"] <= 3
        assert SEPARATION_QUALITY_THRESHOLDS["min_selectivity"] <= 10


class TestSeparationReviewerLogic:
    """Tests for separation_reviewer_node decision logic."""

    @pytest.fixture
    def good_separation_results(self) -> Dict[str, Any]:
        """High-quality separation results."""
        return {
            "solvents": ["xylene", "cyclohexane", "toluene"],
            "selectivities": [45.0, 38.0, 25.0],
            "best_sequence": ["PE", "PP", "PS"],
            "polymers": ["PE", "PP", "PS"],
            "temperature": 80.0,
        }

    @pytest.fixture
    def poor_separation_results(self) -> Dict[str, Any]:
        """Low-quality separation results (should trigger revision)."""
        return {
            "solvents": ["xylene"],  # Only 1 solvent
            "selectivities": [3.0],  # Low selectivity
            "best_sequence": [],
            "polymers": ["PE", "PP"],
            "temperature": 80.0,
        }

    @pytest.fixture
    def marginal_separation_results(self) -> Dict[str, Any]:
        """Marginal results (borderline acceptable)."""
        return {
            "solvents": ["xylene", "toluene"],  # Exactly 2 solvents
            "selectivities": [5.0, 4.5],  # At threshold
            "best_sequence": ["PE", "PP"],
            "polymers": ["PE", "PP"],
            "temperature": 80.0,
        }

    def _create_state(self, separation_results: Dict, retry_count: int = 0) -> dict:
        """Create a test state dict."""
        return {
            "messages": [],
            "separation_results": separation_results,
            "shared_context": {
                "polymers": separation_results.get("polymers", []),
                "temperature": separation_results.get("temperature", 80.0),
                "throughput_kg_hr": 100.0,
            },
            "separation_retry_count": retry_count,
            "agent_timings": {"orchestrator": 0, "separation": 1},
        }

    @pytest.mark.asyncio
    async def test_good_results_accepted(self, good_separation_results):
        """Good results should be accepted and route to TEA."""
        state = self._create_state(good_separation_results)
        result = await separation_reviewer_node(state)

        # Check it's a Command object routing to collab_tea_agent
        assert hasattr(result, 'goto')
        assert result.goto == "collab_tea_agent"

        # Check feedback indicates acceptance
        feedback = result.update.get("reviewer_feedback", {})
        assert feedback.get("is_acceptable") == True
        assert feedback.get("quality_score", 0) >= 0.6

    @pytest.mark.asyncio
    async def test_poor_results_trigger_revision(self, poor_separation_results):
        """Poor results should trigger revision (route back to separation)."""
        state = self._create_state(poor_separation_results, retry_count=0)
        result = await separation_reviewer_node(state)

        # Should route back to separation for retry
        assert hasattr(result, 'goto')
        assert result.goto == "collab_separation_agent"

        # Check feedback indicates revision needed
        feedback = result.update.get("reviewer_feedback", {})
        assert feedback.get("requires_revision") == True
        assert feedback.get("is_acceptable") == False
        assert len(feedback.get("issues", [])) > 0

        # Retry count should increment
        assert result.update.get("separation_retry_count") == 1

    @pytest.mark.asyncio
    async def test_max_retries_proceeds_to_tea(self, poor_separation_results):
        """After max retries, should proceed to TEA even with poor results."""
        max_retries = SEPARATION_QUALITY_THRESHOLDS["max_retries"]
        state = self._create_state(poor_separation_results, retry_count=max_retries)
        result = await separation_reviewer_node(state)

        # Should proceed to TEA despite poor results
        assert hasattr(result, 'goto')
        assert result.goto == "collab_tea_agent"

    @pytest.mark.asyncio
    async def test_retry_params_set(self, poor_separation_results):
        """Retry should set modified parameters (expanded temperature)."""
        state = self._create_state(poor_separation_results, retry_count=0)
        result = await separation_reviewer_node(state)

        # Check retry params are set
        retry_params = result.update.get("retry_params", {})
        assert "temperature_range" in retry_params

        # Temperature should be expanded
        temp_range = retry_params["temperature_range"]
        original_temp = poor_separation_results["temperature"]
        expansion = SEPARATION_QUALITY_THRESHOLDS["temperature_expansion"]
        assert temp_range[0] <= original_temp - expansion + 1
        assert temp_range[1] >= original_temp + expansion - 1

    @pytest.mark.asyncio
    async def test_handoff_metrics_created(self, good_separation_results):
        """Reviewer should create handoff metrics for tracking."""
        state = self._create_state(good_separation_results)
        result = await separation_reviewer_node(state)

        # Check handoff metrics are present
        metrics = result.update.get("handoff_metrics", [])
        assert len(metrics) > 0

        metric = metrics[0]
        assert metric.get("from_agent") == "separation_reviewer"
        assert "quality_score" in metric

    @pytest.mark.asyncio
    async def test_marginal_results_quality_score(self, marginal_separation_results):
        """Marginal results should have intermediate quality score."""
        state = self._create_state(marginal_separation_results)
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        quality = feedback.get("quality_score", 0)

        # Marginal results should have quality between 0.4 and 0.9
        assert 0.4 <= quality <= 0.9


class TestReviewerIntegration:
    """Integration tests for reviewer in the collaboration pipeline."""

    @pytest.mark.asyncio
    async def test_empty_separation_results(self):
        """Handle missing separation results gracefully."""
        state = {
            "messages": [],
            "separation_results": {},
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)

        # Should route to retry or aggregator
        assert hasattr(result, 'goto')
        # With no solvents, should trigger revision or proceed with warning
        assert result.goto in ["collab_separation_agent", "collab_tea_agent", "smart_aggregator"]

    @pytest.mark.asyncio
    async def test_tea_task_request_created(self):
        """Reviewer should create TEATaskRequest when accepting."""
        good_results = {
            "solvents": ["xylene", "cyclohexane"],
            "selectivities": [45.0, 38.0],
            "best_sequence": ["PE", "PP"],
            "polymers": ["PE", "PP"],
            "temperature": 80.0,
        }
        state = {
            "messages": [],
            "separation_results": good_results,
            "shared_context": {"polymers": ["PE", "PP"], "throughput_kg_hr": 200.0},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)

        if result.goto == "collab_tea_agent":
            pending_handoff = result.update.get("pending_handoff", {})
            task_params = pending_handoff.get("task_params", {})

            # Should have TEA task parameters
            assert "solvents" in task_params
            assert len(task_params["solvents"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
