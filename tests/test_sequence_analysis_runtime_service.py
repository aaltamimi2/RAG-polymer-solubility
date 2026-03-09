import pytest


@pytest.mark.asyncio
async def test_find_optimal_separation_prefers_safest_valid_candidate():
    from strap.services.sequence_analysis_runtime_service import find_optimal_separation

    def fake_get_solubility(polymer, solvent, temperature):
        table = {
            ("A", "S1", 25.0): 70.0,
            ("B", "S1", 25.0): 10.0,
            ("A", "S2", 25.0): 80.0,
            ("B", "S2", 25.0): 20.0,
        }
        return table.get((polymer, solvent, temperature), 0.0)

    async def fake_lookup(solvents, table):
        return {
            "S1": {"g_score": 6.0},
            "S2": {"g_score": 8.5},
        }

    class FakeConnection:
        def execute(self, query, params):
            class Result:
                @staticmethod
                def fetchdf():
                    import pandas as pd

                    return pd.DataFrame([])

            return Result()

    result = await find_optimal_separation(
        target="A",
        remaining=["B"],
        available_temps=[25.0],
        rank_by="safety",
        used_solvents=None,
        get_solubility=fake_get_solubility,
        get_available_solvents=lambda: ["S1", "S2"],
        solvent_table="demo_table",
        lookup_solvent_properties=fake_lookup,
        connection=FakeConnection(),
        min_selectivity_threshold=5.0,
    )

    assert result["solvent"] == "S2"
    assert result["g_score"] == 8.5


@pytest.mark.asyncio
async def test_build_greedy_integrated_results_builds_single_result():
    from strap.services.sequence_analysis_runtime_service import build_greedy_integrated_results

    async def fake_find_optimal_separation(*, target, remaining, used_solvents):
        solvent_map = {
            "A": {"solvent": "S1", "selectivity": 20.0, "temperature": 25.0},
            "B": {"solvent": "S2", "selectivity": 10.0, "temperature": 30.0},
            "C": {"solvent": "S3", "selectivity": 5.0, "temperature": 35.0},
        }
        return solvent_map[target]

    results = await build_greedy_integrated_results(
        ["A", "B", "C"],
        find_optimal_separation_fn=fake_find_optimal_separation,
    )

    assert len(results) == 1
    assert results[0]["sequence"][0] == "A"
    assert results[0]["min_selectivity"] == 10.0


@pytest.mark.asyncio
async def test_run_exhaustive_integrated_analysis_sorts_results():
    from strap.services.sequence_analysis_runtime_service import run_exhaustive_integrated_analysis

    async def fake_analyze(sequence):
        return {
            "sequence": sequence,
            "min_selectivity": 5.0 if sequence[0] == "A" else 12.0,
        }

    results = await run_exhaustive_integrated_analysis(
        [("A", "B"), ("B", "A")],
        analyze_sequence_fn=fake_analyze,
        concurrency=2,
    )

    assert [result["sequence"] for result in results] == [("B", "A"), ("A", "B")]
