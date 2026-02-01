"""
Tests for the tools module.

Tests the separation, optimization, visualization, and analysis tools.
"""

import pytest
import asyncio
import duckdb
import os
from pathlib import Path

# Import tools
from tools.separation import (
    SeparationStep,
    SeparationSequence,
    SeparationResult,
    SeparationStatus,
    GreedySeparator,
    DPSeparator,
    BranchAndBoundSeparator,
    find_best_separation,
)
from tools.optimization import (
    TemperatureOptimizer,
    ThroughputAnalyzer,
    OptimizationResult,
    OptimizationObjective,
)
from tools.analysis import (
    SelectivityCalculator,
    SolventRanker,
    PolymerCompatibilityMatrix,
    CompatibilityLevel,
    SelectivityMetrics,
)
from tools.visualization import (
    SeparationTreeVisualizer,
    SelectivityHeatmap,
    ProcessFlowDiagram,
    PlotConfig,
)


# Fixtures
@pytest.fixture
def db_connection():
    """Create a test database connection with sample data."""
    conn = duckdb.connect(':memory:')

    # Create sample solubility data
    conn.execute("""
        CREATE TABLE common_solvents_database (
            polymer VARCHAR,
            solvent VARCHAR,
            solubility____ DOUBLE,
            temperature___c_ DOUBLE
        )
    """)

    # Insert test data
    test_data = [
        # PS - dissolves well in toluene, xylene
        ('PS', 'toluene', 85.0, 100.0),
        ('PS', 'toluene', 90.0, 120.0),
        ('PS', 'xylene', 80.0, 100.0),
        ('PS', 'acetone', 20.0, 100.0),
        ('PS', 'water', 0.0, 100.0),

        # LDPE - dissolves in xylene, some in toluene
        ('LDPE', 'xylene', 70.0, 120.0),
        ('LDPE', 'toluene', 40.0, 100.0),
        ('LDPE', 'acetone', 5.0, 100.0),
        ('LDPE', 'water', 0.0, 100.0),

        # HDPE - similar to LDPE but less soluble
        ('HDPE', 'xylene', 50.0, 120.0),
        ('HDPE', 'toluene', 30.0, 100.0),
        ('HDPE', 'acetone', 2.0, 100.0),

        # PP - dissolves in xylene at high temp
        ('PP', 'xylene', 60.0, 140.0),
        ('PP', 'toluene', 25.0, 120.0),
        ('PP', 'acetone', 3.0, 100.0),

        # PVC - dissolves in THF
        ('PVC', 'thf', 75.0, 80.0),
        ('PVC', 'acetone', 30.0, 80.0),
        ('PVC', 'toluene', 15.0, 100.0),

        # PET - dissolves in phenol
        ('PET', 'phenol', 65.0, 120.0),
        ('PET', 'toluene', 5.0, 120.0),
        ('PET', 'xylene', 8.0, 120.0),
    ]

    conn.executemany(
        "INSERT INTO common_solvents_database VALUES (?, ?, ?, ?)",
        test_data
    )

    return conn


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for plots."""
    return str(tmp_path / "plots")


# =============================================================================
# Separation Module Tests
# =============================================================================

class TestSeparationStep:
    """Tests for SeparationStep dataclass."""

    def test_create_step(self):
        step = SeparationStep(
            step_number=1,
            target_polymer="PS",
            remaining_polymers=["LDPE", "HDPE"],
            solvent="toluene",
            selectivity=45.0,
            target_solubility=85.0,
            max_other_solubility=40.0,
            temperature=100.0,
        )
        assert step.target_polymer == "PS"
        assert step.selectivity == 45.0
        assert step.is_viable is True

    def test_selectivity_ratio(self):
        step = SeparationStep(
            step_number=1,
            target_polymer="PS",
            remaining_polymers=["LDPE"],
            solvent="toluene",
            selectivity=45.0,
            target_solubility=85.0,
            max_other_solubility=40.0,
            temperature=100.0,
        )
        assert step.selectivity_ratio == 85.0 / 40.0


class TestSeparationSequence:
    """Tests for SeparationSequence dataclass."""

    def test_sequence_metrics(self):
        steps = [
            SeparationStep(1, "PS", ["LDPE", "HDPE"], "toluene", 45.0, 85.0, 40.0, 100.0),
            SeparationStep(2, "LDPE", ["HDPE"], "xylene", 20.0, 70.0, 50.0, 120.0),
            SeparationStep(3, "HDPE", [], "N/A", 100.0, 100.0, 0.0, 120.0),
        ]
        seq = SeparationSequence(polymers=["PS", "LDPE", "HDPE"], steps=steps)

        assert seq.min_selectivity == 20.0
        assert len(seq.unique_solvents) == 2
        assert seq.status == SeparationStatus.SUCCESS

    def test_failed_status(self):
        steps = [
            SeparationStep(1, "PS", ["LDPE"], "water", -5.0, 0.0, 5.0, 100.0, is_viable=False),
        ]
        seq = SeparationSequence(polymers=["PS", "LDPE"], steps=steps)

        assert seq.status == SeparationStatus.FAILED

    def test_to_dict(self):
        steps = [
            SeparationStep(1, "PS", ["LDPE"], "toluene", 45.0, 85.0, 40.0, 100.0),
            SeparationStep(2, "LDPE", [], "N/A", 100.0, 100.0, 0.0, 120.0),
        ]
        seq = SeparationSequence(polymers=["PS", "LDPE"], steps=steps)

        d = seq.to_dict()
        assert d["sequence"] == ["PS", "LDPE"]
        assert "metrics" in d


class TestGreedySeparator:
    """Tests for GreedySeparator algorithm."""

    @pytest.mark.asyncio
    async def test_greedy_separation(self, db_connection):
        separator = GreedySeparator(db_connection)
        result = await separator.find_optimal_sequence(
            polymers=["PS", "LDPE", "HDPE"],
            temperature=100.0,
        )

        assert isinstance(result, SeparationResult)
        assert result.algorithm == "greedy"
        assert len(result.best_sequence.steps) == 3
        assert result.nodes_explored > 0

    @pytest.mark.asyncio
    async def test_greedy_two_polymers(self, db_connection):
        separator = GreedySeparator(db_connection)
        result = await separator.find_optimal_sequence(
            polymers=["PS", "LDPE"],
            temperature=100.0,
        )

        assert len(result.best_sequence.steps) == 2


class TestDPSeparator:
    """Tests for DPSeparator algorithm."""

    @pytest.mark.asyncio
    async def test_dp_separation(self, db_connection):
        separator = DPSeparator(db_connection)
        result = await separator.find_optimal_sequence(
            polymers=["PS", "LDPE", "HDPE"],
            temperature=100.0,
        )

        assert isinstance(result, SeparationResult)
        assert result.algorithm == "dynamic_programming"

    @pytest.mark.asyncio
    async def test_dp_too_many_polymers(self, db_connection):
        separator = DPSeparator(db_connection)

        with pytest.raises(ValueError, match="Too many polymers"):
            await separator.find_optimal_sequence(
                polymers=[f"P{i}" for i in range(15)],
                temperature=100.0,
            )


class TestFindBestSeparation:
    """Tests for the convenience function."""

    @pytest.mark.asyncio
    async def test_auto_algorithm_selection(self, db_connection):
        # Small set should use DP
        result = await find_best_separation(
            polymers=["PS", "LDPE", "HDPE"],
            db_connection=db_connection,
            temperature=100.0,
            algorithm="auto",
        )

        assert result.algorithm in ("dynamic_programming", "greedy", "branch_and_bound")


# =============================================================================
# Optimization Module Tests
# =============================================================================

class TestTemperatureOptimizer:
    """Tests for TemperatureOptimizer."""

    @pytest.mark.asyncio
    async def test_find_optimal_temperature(self, db_connection):
        optimizer = TemperatureOptimizer(db_connection)
        result = await optimizer.find_optimal_temperature(
            target_polymer="PS",
            other_polymers=["LDPE", "HDPE"],
            solvent="toluene",
        )

        assert isinstance(result, OptimizationResult)
        assert result.optimal_temperature > 0
        assert 0 <= result.feasibility_score <= 1


class TestThroughputAnalyzer:
    """Tests for ThroughputAnalyzer."""

    def test_estimate_dissolution_rate(self, db_connection):
        analyzer = ThroughputAnalyzer(db_connection)

        rate = analyzer.estimate_dissolution_rate(
            polymer="PS",
            solvent="toluene",
            temperature=100.0,
            selectivity=50.0,
        )

        assert rate > 0
        assert rate > analyzer.base_rate  # Higher temp should increase rate

    def test_analyze_sequence_throughput(self, db_connection):
        analyzer = ThroughputAnalyzer(db_connection)

        steps = [
            {"polymer": "PS", "solvent": "toluene", "temperature": 100.0, "selectivity": 50.0},
            {"polymer": "LDPE", "solvent": "xylene", "temperature": 120.0, "selectivity": 30.0},
        ]

        result = analyzer.analyze_sequence_throughput(steps)

        assert "overall_rate" in result
        assert "bottleneck_step" in result
        assert "recommendations" in result


# =============================================================================
# Analysis Module Tests
# =============================================================================

class TestSelectivityCalculator:
    """Tests for SelectivityCalculator."""

    def test_calculate_selectivity(self, db_connection):
        calc = SelectivityCalculator(db_connection)

        metrics = calc.calculate(
            target="PS",
            others=["LDPE", "HDPE"],
            solvent="toluene",
            temperature=100.0,
        )

        assert isinstance(metrics, SelectivityMetrics)
        assert metrics.target_polymer == "PS"
        assert metrics.selectivity > 0  # PS should have higher solubility in toluene

    def test_calculate_all_solvents(self, db_connection):
        calc = SelectivityCalculator(db_connection)

        results = calc.calculate_all_solvents(
            target="PS",
            others=["LDPE"],
            temperature=100.0,
        )

        assert len(results) > 0
        # Should be sorted by selectivity
        for i in range(len(results) - 1):
            assert results[i].selectivity >= results[i + 1].selectivity


class TestSolventRanker:
    """Tests for SolventRanker."""

    def test_rank_solvents(self, db_connection):
        calc = SelectivityCalculator(db_connection)
        ranker = SolventRanker(calc)

        scores = ranker.rank_solvents(
            target="PS",
            others=["LDPE", "HDPE"],
            temperature=100.0,
            top_k=5,
        )

        assert len(scores) <= 5
        for score in scores:
            assert 0 <= score.overall_score <= 1


class TestPolymerCompatibilityMatrix:
    """Tests for PolymerCompatibilityMatrix."""

    def test_build_matrix(self, db_connection):
        matrix_builder = PolymerCompatibilityMatrix(db_connection)

        matrix = matrix_builder.build_matrix(
            polymers=["PS", "LDPE", "HDPE"],
            temperature=100.0,
        )

        assert "PS" in matrix
        assert len(matrix["PS"]) > 0

    def test_find_challenging_pairs(self, db_connection):
        matrix_builder = PolymerCompatibilityMatrix(db_connection)

        pairs = matrix_builder.find_challenging_pairs(
            polymers=["PS", "LDPE", "HDPE"],
            temperature=100.0,
            threshold=50.0,
        )

        # Should find some challenging pairs
        assert isinstance(pairs, list)

    def test_get_compatibility_level(self, db_connection):
        matrix_builder = PolymerCompatibilityMatrix(db_connection)

        level = matrix_builder.get_compatibility_level(
            polymer="PS",
            solvent="toluene",
            temperature=100.0,
        )

        assert level in CompatibilityLevel
        assert level == CompatibilityLevel.EXCELLENT  # PS highly soluble in toluene


# =============================================================================
# Visualization Module Tests
# =============================================================================

class TestSeparationTreeVisualizer:
    """Tests for SeparationTreeVisualizer."""

    def test_create_tree(self, output_dir):
        config = PlotConfig(output_dir=output_dir)
        viz = SeparationTreeVisualizer(config)

        steps = [
            SeparationStep(1, "PS", ["LDPE", "HDPE"], "toluene", 45.0, 85.0, 40.0, 100.0),
            SeparationStep(2, "LDPE", ["HDPE"], "xylene", 20.0, 70.0, 50.0, 120.0),
            SeparationStep(3, "HDPE", [], "N/A", 100.0, 100.0, 0.0, 120.0),
        ]
        seq = SeparationSequence(polymers=["PS", "LDPE", "HDPE"], steps=steps)

        filepath = viz.create_tree([seq])

        assert Path(filepath).exists()
        assert filepath.endswith(".png")


class TestSelectivityHeatmap:
    """Tests for SelectivityHeatmap."""

    def test_create_heatmap(self, output_dir):
        config = PlotConfig(output_dir=output_dir)
        viz = SelectivityHeatmap(config)

        data = {
            "PS": {"toluene": 85.0, "xylene": 80.0, "acetone": 20.0},
            "LDPE": {"toluene": 40.0, "xylene": 70.0, "acetone": 5.0},
            "HDPE": {"toluene": 30.0, "xylene": 50.0, "acetone": 2.0},
        }

        filepath = viz.create_polymer_solvent_heatmap(data)

        assert Path(filepath).exists()


class TestProcessFlowDiagram:
    """Tests for ProcessFlowDiagram."""

    def test_create_flow_diagram(self, output_dir):
        config = PlotConfig(output_dir=output_dir)
        viz = ProcessFlowDiagram(config)

        steps = [
            SeparationStep(1, "PS", ["LDPE", "HDPE"], "toluene", 45.0, 85.0, 40.0, 100.0),
            SeparationStep(2, "LDPE", ["HDPE"], "xylene", 20.0, 70.0, 50.0, 120.0),
            SeparationStep(3, "HDPE", [], "N/A", 100.0, 100.0, 0.0, 120.0),
        ]
        seq = SeparationSequence(polymers=["PS", "LDPE", "HDPE"], steps=steps)

        filepath = viz.create_flow_diagram(seq)

        assert Path(filepath).exists()


# =============================================================================
# Integration Tests
# =============================================================================

class TestToolsIntegration:
    """Integration tests combining multiple tools."""

    @pytest.mark.asyncio
    async def test_full_separation_workflow(self, db_connection, output_dir):
        """Test complete workflow: analyze -> optimize -> visualize."""
        polymers = ["PS", "LDPE", "HDPE"]

        # 1. Find optimal separation sequence
        result = await find_best_separation(
            polymers=polymers,
            db_connection=db_connection,
            temperature=100.0,
            algorithm="greedy",
        )

        assert result.best_sequence is not None

        # 2. Analyze throughput
        analyzer = ThroughputAnalyzer(db_connection)
        steps_for_analysis = [
            {
                "polymer": s.target_polymer,
                "solvent": s.solvent,
                "temperature": s.temperature,
                "selectivity": s.selectivity,
            }
            for s in result.best_sequence.steps
        ]
        throughput = analyzer.analyze_sequence_throughput(steps_for_analysis)

        assert throughput["overall_rate"] > 0

        # 3. Create visualization
        config = PlotConfig(output_dir=output_dir)
        viz = SeparationTreeVisualizer(config)
        plot_path = viz.create_tree([result.best_sequence])

        assert Path(plot_path).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
