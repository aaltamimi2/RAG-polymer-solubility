"""
Comprehensive Tests for P0 Enhancement: Multi-Iteration Review/Revision Loops

Tests scenarios requiring 2-3 review cycles, temperature expansion,
and progressive quality improvement.
"""

import pytest
import asyncio
from typing import Dict, Any, List
from copy import deepcopy

# Import test targets
from agent_schemas import ReviewerFeedback, SeparationResult
from multi_agent_system import (
    separation_reviewer_node,
    SEPARATION_QUALITY_THRESHOLDS,
    MultiAgentState,
)


class TestMultipleRetryLoops:
    """Tests for scenarios requiring multiple review cycles."""

    def _create_state(
        self,
        separation_results: Dict,
        retry_count: int = 0,
        retry_params: Dict = None
    ) -> dict:
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
            "retry_params": retry_params or {},
            "agent_timings": {"orchestrator": 0, "separation": 1},
            "reviewer_feedback": None,
        }

    @pytest.mark.asyncio
    async def test_first_retry_cycle(self):
        """Test first retry when results are poor."""
        # Poor results - only 1 solvent, low selectivity
        poor_results = {
            "solvents": ["xylene"],
            "selectivities": [3.0],
            "best_sequence": [],
            "polymers": ["PE", "PP", "PS"],
            "temperature": 80.0,
        }

        state = self._create_state(poor_results, retry_count=0)
        result = await separation_reviewer_node(state)

        # Should trigger retry
        assert result.goto == "collab_separation_agent"
        assert result.update.get("separation_retry_count") == 1

        feedback = result.update.get("reviewer_feedback", {})
        assert feedback.get("requires_revision") == True
        assert feedback.get("retry_count") == 0  # Was 0 before this retry
        assert len(feedback.get("issues", [])) >= 1

        # Check temperature expansion
        retry_params = result.update.get("retry_params", {})
        assert "temperature_range" in retry_params
        temp_range = retry_params["temperature_range"]
        assert temp_range[0] < 80.0  # Lower bound expanded
        assert temp_range[1] > 80.0  # Upper bound expanded

    @pytest.mark.asyncio
    async def test_second_retry_cycle(self):
        """Test second retry when results are still poor."""
        # Slightly improved but still insufficient
        poor_results = {
            "solvents": ["xylene"],  # Still only 1 solvent
            "selectivities": [4.5],  # Slightly better but still below threshold
            "best_sequence": ["PE"],
            "polymers": ["PE", "PP", "PS"],
            "temperature": 80.0,
        }

        # State after first retry
        state = self._create_state(
            poor_results,
            retry_count=1,
            retry_params={"temperature_range": (60, 100)}
        )
        result = await separation_reviewer_node(state)

        # Should trigger second retry (retry_count was 1, now 2)
        assert result.goto == "collab_separation_agent"
        assert result.update.get("separation_retry_count") == 2

        feedback = result.update.get("reviewer_feedback", {})
        assert feedback.get("requires_revision") == True

        # Temperature should expand further
        retry_params = result.update.get("retry_params", {})
        temp_range = retry_params["temperature_range"]
        # Should be even wider than first retry
        assert temp_range[0] <= 60  # At least as wide as before
        assert temp_range[1] >= 100

    @pytest.mark.asyncio
    async def test_third_retry_exceeds_max(self):
        """Test that third retry exceeds max and proceeds to TEA."""
        # Still poor results after 2 retries
        poor_results = {
            "solvents": ["xylene"],
            "selectivities": [4.0],
            "best_sequence": [],
            "polymers": ["PE", "PP", "PS"],
            "temperature": 80.0,
        }

        # State after max retries (2)
        max_retries = SEPARATION_QUALITY_THRESHOLDS["max_retries"]
        state = self._create_state(
            poor_results,
            retry_count=max_retries,  # Already at max
            retry_params={"temperature_range": (40, 120)}
        )
        result = await separation_reviewer_node(state)

        # Should NOT retry - proceed to TEA despite poor results
        assert result.goto == "collab_tea_agent"

        feedback = result.update.get("reviewer_feedback", {})
        # Should not require revision since we've hit max retries
        assert feedback.get("requires_revision") == False

    @pytest.mark.asyncio
    async def test_full_retry_sequence_simulation(self):
        """Simulate a full sequence of retries with improving results."""
        # Simulate 3 iterations with progressively improving results
        iterations = [
            # Iteration 1: Very poor (triggers retry)
            {
                "solvents": [],
                "selectivities": [],
                "best_sequence": [],
                "polymers": ["PE", "PP", "PS"],
                "temperature": 80.0,
            },
            # Iteration 2: Slightly better (still triggers retry)
            {
                "solvents": ["xylene"],
                "selectivities": [3.0],
                "best_sequence": ["PE"],
                "polymers": ["PE", "PP", "PS"],
                "temperature": 80.0,
            },
            # Iteration 3: Good enough (should pass after max retries)
            {
                "solvents": ["xylene", "toluene"],
                "selectivities": [5.5, 4.0],
                "best_sequence": ["PE", "PP"],
                "polymers": ["PE", "PP", "PS"],
                "temperature": 80.0,
            },
        ]

        retry_count = 0
        retry_params = {}
        destinations = []

        for i, results in enumerate(iterations):
            state = self._create_state(results, retry_count, retry_params)
            result = await separation_reviewer_node(state)

            destinations.append(result.goto)

            if result.goto == "collab_separation_agent":
                # Retry triggered
                retry_count = result.update.get("separation_retry_count", retry_count + 1)
                retry_params = result.update.get("retry_params", {})
            else:
                # Proceeded to TEA or aggregator
                break

        # Should have retried twice then proceeded
        assert destinations[0] == "collab_separation_agent"  # First retry
        assert destinations[1] == "collab_separation_agent"  # Second retry
        assert destinations[2] == "collab_tea_agent"  # Finally proceeds

    @pytest.mark.asyncio
    async def test_early_success_no_retry(self):
        """Good results on first try should not trigger any retries."""
        good_results = {
            "solvents": ["xylene", "cyclohexane", "toluene"],
            "selectivities": [45.0, 38.0, 25.0],
            "best_sequence": ["PE", "PP", "PS"],
            "polymers": ["PE", "PP", "PS"],
            "temperature": 80.0,
        }

        state = self._create_state(good_results, retry_count=0)
        result = await separation_reviewer_node(state)

        # Should proceed directly to TEA
        assert result.goto == "collab_tea_agent"
        assert result.update.get("separation_retry_count", 0) == 0

        feedback = result.update.get("reviewer_feedback", {})
        assert feedback.get("is_acceptable") == True
        assert feedback.get("quality_score", 0) >= 0.8


class TestTemperatureExpansion:
    """Tests for temperature range expansion on retries."""

    def _create_state(self, temp: float, retry_count: int = 0) -> dict:
        return {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene"],  # Only 1 - triggers retry
                "selectivities": [3.0],
                "best_sequence": [],
                "polymers": ["PE", "PP"],
                "temperature": temp,
            },
            "shared_context": {
                "polymers": ["PE", "PP"],
                "temperature": temp,
            },
            "separation_retry_count": retry_count,
            "retry_params": {},
            "agent_timings": {},
        }

    @pytest.mark.asyncio
    async def test_expansion_from_low_temperature(self):
        """Test expansion when starting at low temperature."""
        state = self._create_state(temp=50.0, retry_count=0)
        result = await separation_reviewer_node(state)

        retry_params = result.update.get("retry_params", {})
        temp_range = retry_params.get("temperature_range", (0, 0))

        # Lower bound should not go below 40°C
        assert temp_range[0] >= 40
        # Upper bound should expand upward
        expansion = SEPARATION_QUALITY_THRESHOLDS["temperature_expansion"]
        assert temp_range[1] >= 50 + expansion - 1

    @pytest.mark.asyncio
    async def test_expansion_from_high_temperature(self):
        """Test expansion when starting at high temperature."""
        state = self._create_state(temp=160.0, retry_count=0)
        result = await separation_reviewer_node(state)

        retry_params = result.update.get("retry_params", {})
        temp_range = retry_params.get("temperature_range", (0, 0))

        # Upper bound should not exceed 180°C
        assert temp_range[1] <= 180
        # Lower bound should expand downward
        expansion = SEPARATION_QUALITY_THRESHOLDS["temperature_expansion"]
        assert temp_range[0] <= 160 - expansion + 1

    @pytest.mark.asyncio
    async def test_expansion_at_bounds(self):
        """Test expansion at temperature boundaries."""
        # Test at lower bound
        state = self._create_state(temp=45.0, retry_count=0)
        result = await separation_reviewer_node(state)

        retry_params = result.update.get("retry_params", {})
        temp_range = retry_params.get("temperature_range", (0, 0))
        assert temp_range[0] >= 40  # Should not go below minimum

        # Test at upper bound
        state = self._create_state(temp=175.0, retry_count=0)
        result = await separation_reviewer_node(state)

        retry_params = result.update.get("retry_params", {})
        temp_range = retry_params.get("temperature_range", (0, 0))
        assert temp_range[1] <= 180  # Should not exceed maximum


class TestQualityScoreProgression:
    """Tests for quality score calculation across iterations."""

    def _create_state(self, solvents: List[str], selectivities: List[float]) -> dict:
        return {
            "messages": [],
            "separation_results": {
                "solvents": solvents,
                "selectivities": selectivities,
                "best_sequence": ["PE", "PP"] if solvents else [],
                "polymers": ["PE", "PP", "PS"],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": ["PE", "PP", "PS"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

    @pytest.mark.asyncio
    async def test_quality_zero_solvents(self):
        """Zero solvents should have very low quality."""
        state = self._create_state(solvents=[], selectivities=[])
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        quality = feedback.get("quality_score", 1.0)
        assert quality < 0.5  # Very low quality

    @pytest.mark.asyncio
    async def test_quality_one_solvent(self):
        """One solvent should have low quality."""
        state = self._create_state(solvents=["xylene"], selectivities=[10.0])
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        quality = feedback.get("quality_score", 1.0)
        assert quality < 0.7  # Low quality (below threshold for needing 2)

    @pytest.mark.asyncio
    async def test_quality_two_solvents_low_selectivity(self):
        """Two solvents with low selectivity should have moderate quality."""
        state = self._create_state(solvents=["xylene", "toluene"], selectivities=[3.0, 2.0])
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        quality = feedback.get("quality_score", 1.0)
        # Has enough solvents but low selectivity
        assert 0.4 <= quality <= 0.8

    @pytest.mark.asyncio
    async def test_quality_three_solvents_good_selectivity(self):
        """Three solvents with good selectivity should have high quality."""
        state = self._create_state(
            solvents=["xylene", "toluene", "cyclohexane"],
            selectivities=[45.0, 30.0, 25.0]
        )
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        quality = feedback.get("quality_score", 1.0)
        assert quality >= 0.8  # High quality

    @pytest.mark.asyncio
    async def test_quality_progression_simulation(self):
        """Simulate quality improvement over iterations."""
        scenarios = [
            ([], [], "no_solvents"),
            (["xylene"], [2.0], "one_low"),
            (["xylene", "toluene"], [4.0, 3.0], "two_low"),
            (["xylene", "toluene"], [6.0, 5.0], "two_threshold"),
            (["xylene", "toluene", "cyclohexane"], [15.0, 10.0, 8.0], "three_good"),
        ]

        qualities = []
        for solvents, selectivities, label in scenarios:
            state = self._create_state(solvents, selectivities)
            result = await separation_reviewer_node(state)
            feedback = result.update.get("reviewer_feedback", {})
            qualities.append((label, feedback.get("quality_score", 0)))

        # Quality should generally increase
        print("\nQuality progression:")
        for label, q in qualities:
            print(f"  {label}: {q:.2f}")

        # Verify monotonic improvement (with some tolerance)
        for i in range(1, len(qualities)):
            # Each subsequent scenario should be >= previous (with small tolerance)
            assert qualities[i][1] >= qualities[i-1][1] - 0.1, \
                f"{qualities[i][0]} quality should be >= {qualities[i-1][0]} quality"


class TestIssueDetection:
    """Tests for specific issue detection in review."""

    def _create_state(self, **kwargs) -> dict:
        defaults = {
            "solvents": [],
            "selectivities": [],
            "best_sequence": [],
            "polymers": ["PE", "PP", "PS"],
            "temperature": 80.0,
        }
        defaults.update(kwargs)

        return {
            "messages": [],
            "separation_results": defaults,
            "shared_context": {"polymers": defaults["polymers"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

    @pytest.mark.asyncio
    async def test_issue_too_few_solvents(self):
        """Detect issue when too few solvents found."""
        state = self._create_state(solvents=["xylene"], selectivities=[10.0])
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        issues = feedback.get("issues", [])

        # Should have issue about solvents
        assert any("solvent" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_issue_low_selectivity(self):
        """Detect issue when selectivity is too low."""
        state = self._create_state(
            solvents=["xylene", "toluene"],
            selectivities=[2.0, 1.5]  # Below 5% threshold
        )
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        issues = feedback.get("issues", [])

        # Should have issue about selectivity
        assert any("selectivity" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_issue_incomplete_sequence(self):
        """Detect issue when sequence doesn't cover all polymers."""
        state = self._create_state(
            solvents=["xylene", "toluene"],
            selectivities=[10.0, 8.0],
            best_sequence=["PE"],  # Only 1 step for 3 polymers
            polymers=["PE", "PP", "PS"]
        )
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        issues = feedback.get("issues", [])

        # Should have issue about sequence
        assert any("sequence" in issue.lower() or "step" in issue.lower() for issue in issues)

    @pytest.mark.asyncio
    async def test_suggestions_provided(self):
        """Verify suggestions are provided for issues."""
        state = self._create_state(solvents=["xylene"], selectivities=[2.0])
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        suggestions = feedback.get("suggestions", [])

        # Should have at least one suggestion
        assert len(suggestions) > 0

    @pytest.mark.asyncio
    async def test_no_issues_for_good_results(self):
        """Good results should have no issues."""
        state = self._create_state(
            solvents=["xylene", "toluene", "cyclohexane"],
            selectivities=[30.0, 25.0, 20.0],
            best_sequence=["PE", "PP", "PS"]
        )
        result = await separation_reviewer_node(state)

        feedback = result.update.get("reviewer_feedback", {})
        issues = feedback.get("issues", [])

        # Should have no issues
        assert len(issues) == 0


class TestHandoffMetricsInLoops:
    """Tests for handoff metrics tracking across review loops."""

    def _create_state(self, quality: str = "poor") -> dict:
        if quality == "poor":
            results = {"solvents": [], "selectivities": [], "best_sequence": []}
        elif quality == "marginal":
            results = {"solvents": ["xylene"], "selectivities": [4.0], "best_sequence": ["PE"]}
        else:
            results = {"solvents": ["xylene", "toluene"], "selectivities": [15.0, 10.0], "best_sequence": ["PE", "PP"]}

        results["polymers"] = ["PE", "PP"]
        results["temperature"] = 80.0

        return {
            "messages": [],
            "separation_results": results,
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 0,
            "agent_timings": {"separation": 1.0},
        }

    @pytest.mark.asyncio
    async def test_metrics_on_retry(self):
        """Handoff metrics should be created on retry."""
        state = self._create_state(quality="poor")
        result = await separation_reviewer_node(state)

        metrics = result.update.get("handoff_metrics", [])
        assert len(metrics) > 0

        metric = metrics[0]
        assert metric.get("from_agent") == "separation_reviewer"
        assert metric.get("to_agent") == "collab_separation_agent"  # Retry destination
        assert metric.get("success") == False  # Failed quality check

    @pytest.mark.asyncio
    async def test_metrics_on_accept(self):
        """Handoff metrics should be created on accept."""
        state = self._create_state(quality="good")
        result = await separation_reviewer_node(state)

        metrics = result.update.get("handoff_metrics", [])
        assert len(metrics) > 0

        metric = metrics[0]
        assert metric.get("from_agent") == "separation_reviewer"
        assert metric.get("to_agent") == "collab_tea_agent"  # Accept destination
        assert metric.get("success") == True

    @pytest.mark.asyncio
    async def test_quality_score_in_metrics(self):
        """Quality score should be included in metrics."""
        state = self._create_state(quality="marginal")
        result = await separation_reviewer_node(state)

        metrics = result.update.get("handoff_metrics", [])
        assert len(metrics) > 0

        metric = metrics[0]
        assert "quality_score" in metric
        assert 0 <= metric["quality_score"] <= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_missing_separation_results(self):
        """Handle missing separation_results gracefully."""
        state = {
            "messages": [],
            "separation_results": None,
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)

        # Should handle gracefully and likely trigger retry or proceed
        assert hasattr(result, 'goto')
        assert result.goto in ["collab_separation_agent", "collab_tea_agent", "smart_aggregator"]

    @pytest.mark.asyncio
    async def test_empty_polymers_list(self):
        """Handle empty polymers list."""
        state = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene"],
                "selectivities": [10.0],
                "best_sequence": [],
                "polymers": [],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": []},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)
        assert hasattr(result, 'goto')

    @pytest.mark.asyncio
    async def test_very_high_retry_count(self):
        """Handle retry count higher than max (safety check)."""
        state = {
            "messages": [],
            "separation_results": {
                "solvents": [],
                "selectivities": [],
                "best_sequence": [],
                "polymers": ["PE", "PP"],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 100,  # Way over max
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)

        # Should proceed (not retry) since we're over max
        assert result.goto in ["collab_tea_agent", "smart_aggregator"]

    @pytest.mark.asyncio
    async def test_negative_selectivity(self):
        """Handle negative selectivity values."""
        state = {
            "messages": [],
            "separation_results": {
                "solvents": ["xylene", "toluene"],
                "selectivities": [-5.0, 10.0],  # Invalid negative
                "best_sequence": ["PE"],
                "polymers": ["PE", "PP"],
                "temperature": 80.0,
            },
            "shared_context": {"polymers": ["PE", "PP"]},
            "separation_retry_count": 0,
            "agent_timings": {},
        }

        result = await separation_reviewer_node(state)

        # Should handle gracefully
        assert hasattr(result, 'goto')
        feedback = result.update.get("reviewer_feedback", {})
        # Min selectivity should reflect the negative value
        assert feedback.get("min_selectivity") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
