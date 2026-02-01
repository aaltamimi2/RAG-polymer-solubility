"""
Tests for LangChain tool wrappers.

Tests that the @tool decorated functions work correctly with
the modular tools and can be used by agents.
"""

import pytest
import duckdb
from pathlib import Path

# Import tools
from tools.langchain_tools import (
    set_db_connection,
    get_db_connection,
    find_optimal_separation_sequence,
    compare_separation_algorithms,
    optimize_separation_temperature,
    analyze_sequence_throughput,
    calculate_selectivity_detailed,
    rank_solvents_for_separation,
    build_compatibility_matrix,
    find_challenging_polymer_pairs,
    create_separation_tree_plot,
    create_selectivity_heatmap,
    create_process_flow_diagram,
    ADVANCED_SEPARATION_TOOLS,
)


@pytest.fixture(scope="module")
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
        ('LDPE', 'toluene', 50.0, 120.0),
        ('LDPE', 'acetone', 5.0, 100.0),
        ('LDPE', 'water', 0.0, 100.0),

        # HDPE - similar to LDPE but less soluble
        ('HDPE', 'xylene', 50.0, 120.0),
        ('HDPE', 'toluene', 30.0, 100.0),
        ('HDPE', 'toluene', 35.0, 120.0),
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

    # Set the connection for all tools
    set_db_connection(conn)

    yield conn

    # Cleanup
    set_db_connection(None)


@pytest.fixture
def output_dir(tmp_path):
    """Create temporary output directory for plots."""
    return str(tmp_path / "plots")


# =============================================================================
# Tool Collection Tests
# =============================================================================

class TestToolCollection:
    """Tests for the tool collection."""

    def test_all_tools_present(self):
        assert len(ADVANCED_SEPARATION_TOOLS) == 11

    def test_tools_are_invokable(self):
        for tool in ADVANCED_SEPARATION_TOOLS:
            # LangChain tools use .invoke() method
            assert hasattr(tool, 'invoke')

    def test_tools_have_descriptions(self):
        for tool in ADVANCED_SEPARATION_TOOLS:
            # LangChain tools have a description attribute
            assert hasattr(tool, 'description') or hasattr(tool, 'func')


# =============================================================================
# Separation Algorithm Tool Tests
# =============================================================================

class TestFindOptimalSeparationSequence:
    """Tests for find_optimal_separation_sequence tool."""

    def test_basic_separation(self, db_connection):
        result = find_optimal_separation_sequence.invoke({
            "polymers": "PS,LDPE,HDPE",
            "temperature": 100.0,
            "algorithm": "greedy",
        })

        assert "Optimal Separation Sequence" in result
        assert "PS" in result or "LDPE" in result or "HDPE" in result
        assert "Selectivity" in result

    def test_auto_algorithm(self, db_connection):
        result = find_optimal_separation_sequence.invoke({
            "polymers": "PS,LDPE",
            "temperature": 100.0,
            "algorithm": "auto",
        })

        assert "Optimal Separation Sequence" in result

    def test_error_on_single_polymer(self, db_connection):
        result = find_optimal_separation_sequence.invoke({
            "polymers": "PS",
            "temperature": 100.0,
        })

        assert "Error" in result
        assert "at least 2 polymers" in result

    def test_error_on_too_many_polymers(self, db_connection):
        polymers = ",".join([f"P{i}" for i in range(15)])
        result = find_optimal_separation_sequence.invoke({
            "polymers": polymers,
            "temperature": 100.0,
        })

        assert "Error" in result
        assert "Too many" in result


class TestCompareSeparationAlgorithms:
    """Tests for compare_separation_algorithms tool."""

    def test_compare_algorithms(self, db_connection):
        result = compare_separation_algorithms.invoke({
            "polymers": "PS,LDPE,HDPE",
            "temperature": 100.0,
        })

        assert "Algorithm Comparison" in result
        assert "Greedy" in result
        assert "Dynamic Programming" in result
        assert "Conclusion" in result


# =============================================================================
# Temperature Optimization Tool Tests
# =============================================================================

class TestOptimizeSeparationTemperature:
    """Tests for optimize_separation_temperature tool."""

    def test_basic_optimization(self, db_connection):
        result = optimize_separation_temperature.invoke({
            "target_polymer": "PS",
            "other_polymers": "LDPE,HDPE",
            "solvent": "toluene",
        })

        assert "Temperature Optimization" in result
        assert "Optimal Temperature" in result


class TestAnalyzeSequenceThroughput:
    """Tests for analyze_sequence_throughput tool."""

    def test_throughput_analysis(self, db_connection):
        result = analyze_sequence_throughput.invoke({
            "polymers": "PS,LDPE,HDPE",
            "temperature": 100.0,
        })

        assert "Throughput Analysis" in result
        assert "Bottleneck" in result
        assert "kg/hr" in result


# =============================================================================
# Analysis Tool Tests
# =============================================================================

class TestCalculateSelectivityDetailed:
    """Tests for calculate_selectivity_detailed tool."""

    def test_basic_selectivity(self, db_connection):
        result = calculate_selectivity_detailed.invoke({
            "target_polymer": "PS",
            "other_polymers": "LDPE,HDPE",
            "solvent": "toluene",
            "temperature": 100.0,
        })

        assert "Selectivity Analysis" in result
        assert "Target Polymer" in result
        assert "Solvent" in result


class TestRankSolventsForSeparation:
    """Tests for rank_solvents_for_separation tool."""

    def test_solvent_ranking(self, db_connection):
        result = rank_solvents_for_separation.invoke({
            "target_polymer": "PS",
            "other_polymers": "LDPE",
            "temperature": 100.0,
            "top_k": 5,
        })

        assert "Solvent Ranking" in result
        assert "Overall" in result or "Selectivity" in result


class TestBuildCompatibilityMatrix:
    """Tests for build_compatibility_matrix tool."""

    def test_basic_matrix(self, db_connection):
        result = build_compatibility_matrix.invoke({
            "polymers": "PS,LDPE,HDPE",
            "temperature": 100.0,
        })

        assert "Compatibility Matrix" in result
        # Should have table structure
        assert "|" in result


class TestFindChallengingPolymerPairs:
    """Tests for find_challenging_polymer_pairs tool."""

    def test_find_pairs(self, db_connection):
        result = find_challenging_polymer_pairs.invoke({
            "polymers": "PS,LDPE,HDPE,PP",
            "temperature": 100.0,
            "selectivity_threshold": 30.0,
        })

        assert "Challenging Polymer Pairs" in result


# =============================================================================
# Visualization Tool Tests
# =============================================================================

class TestVisualizationTools:
    """Tests for visualization tools."""

    def test_separation_tree_plot(self, db_connection, tmp_path, monkeypatch):
        # Monkeypatch the output directory
        import tools.visualization as viz_module
        monkeypatch.setattr(viz_module.PlotConfig, "__init__",
                          lambda self, **kwargs: setattr(self, 'output_dir', str(tmp_path)) or
                                                 setattr(self, 'format', 'png') or
                                                 setattr(self, 'dpi', 100) or
                                                 setattr(self, 'figsize', (10, 6)) or
                                                 setattr(self, 'style', 'default') or
                                                 setattr(self, 'color_palette', 'viridis'))

        result = create_separation_tree_plot.invoke({
            "polymers": "PS,LDPE,HDPE",
            "temperature": 100.0,
        })

        assert "Separation Tree" in result
        assert "saved to" in result.lower() or "Plot" in result

    def test_selectivity_heatmap(self, db_connection, tmp_path, monkeypatch):
        import tools.visualization as viz_module
        monkeypatch.setattr(viz_module.PlotConfig, "__init__",
                          lambda self, **kwargs: setattr(self, 'output_dir', str(tmp_path)) or
                                                 setattr(self, 'format', 'png') or
                                                 setattr(self, 'dpi', 100) or
                                                 setattr(self, 'figsize', (10, 6)) or
                                                 setattr(self, 'style', 'default') or
                                                 setattr(self, 'color_palette', 'viridis'))

        result = create_selectivity_heatmap.invoke({
            "polymers": "PS,LDPE,HDPE",
            "temperature": 100.0,
        })

        assert "Heatmap" in result

    def test_process_flow_diagram(self, db_connection, tmp_path, monkeypatch):
        import tools.visualization as viz_module
        monkeypatch.setattr(viz_module.PlotConfig, "__init__",
                          lambda self, **kwargs: setattr(self, 'output_dir', str(tmp_path)) or
                                                 setattr(self, 'format', 'png') or
                                                 setattr(self, 'dpi', 100) or
                                                 setattr(self, 'figsize', (10, 6)) or
                                                 setattr(self, 'style', 'default') or
                                                 setattr(self, 'color_palette', 'viridis'))

        result = create_process_flow_diagram.invoke({
            "polymers": "PS,LDPE,HDPE",
            "temperature": 100.0,
        })

        assert "Process Flow" in result


# =============================================================================
# Integration Tests
# =============================================================================

class TestAgentIntegration:
    """Tests for agent integration."""

    def test_tool_invoke_interface(self, db_connection):
        """Test that tools work with LangChain's invoke interface."""
        # All tools should be invokable with dict input
        result = find_optimal_separation_sequence.invoke({
            "polymers": "PS,LDPE",
        })
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_description_for_agent(self, db_connection):
        """Test that tools have proper descriptions for agent use."""
        tool = find_optimal_separation_sequence

        # LangChain tools expose description for agent reasoning
        desc = str(tool.description) if hasattr(tool, 'description') else ""
        assert "separation" in desc.lower() or "polymer" in desc.lower()

    def test_workflow_sequence(self, db_connection):
        """Test a realistic workflow sequence."""
        # 1. Find challenging pairs first
        pairs_result = find_challenging_polymer_pairs.invoke({
            "polymers": "PS,LDPE,HDPE,PP",
            "selectivity_threshold": 50.0,
        })
        assert "Challenging" in pairs_result

        # 2. Get optimal sequence
        seq_result = find_optimal_separation_sequence.invoke({
            "polymers": "PS,LDPE,HDPE",
            "algorithm": "auto",
        })
        assert "Sequence" in seq_result

        # 3. Analyze throughput
        throughput_result = analyze_sequence_throughput.invoke({
            "polymers": "PS,LDPE,HDPE",
        })
        assert "Throughput" in throughput_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
