import pytest


def test_rank_by_safety_prefers_high_gscore_viable_candidates():
    from strap.services.sequence_runtime_service import rank_by_safety

    ranked = rank_by_safety(
        [
            {"solvent": "A", "selectivity": 8.0},
            {"solvent": "B", "selectivity": 7.0},
            {"solvent": "C", "selectivity": 3.0},
        ],
        min_selectivity=5.0,
        gscore_map={"A": 6.0, "B": 8.0, "C": 9.5},
    )

    assert [entry["solvent"] for entry in ranked] == ["B", "A"]


def test_build_greedy_scheme_variant_uses_alternate_first_pick():
    from strap.services.sequence_runtime_service import build_greedy_scheme_variant, rank_by_selectivity

    selectivity_map = {
        ("A", ("B", "C")): [
            {"solvent": "S1", "selectivity": 20.0},
            {"solvent": "S2", "selectivity": 15.0},
        ],
        ("B", ("A", "C")): [
            {"solvent": "S3", "selectivity": 18.0},
        ],
        ("C", ("A", "B")): [
            {"solvent": "S4", "selectivity": 5.0},
        ],
        ("A", ("C",)): [{"solvent": "S1", "selectivity": 12.0}],
        ("C", ("A",)): [{"solvent": "S4", "selectivity": 2.0}],
    }

    def fake_get_all_selectivity(target, others, temperature):
        return list(selectivity_map.get((target, tuple(others)), []))

    scheme = build_greedy_scheme_variant(
        polymer_list=["A", "B", "C"],
        temperature=80.0,
        get_all_selectivity=fake_get_all_selectivity,
        rank_fn=rank_by_selectivity,
        name="demo",
        tag="D1",
        first_step_pick=1,
        bp_map={"S3": 90.0},
        gscore_map={},
        logp_map={},
    )

    assert scheme["seq"][0] == "B"
    assert scheme["steps"][0]["solvent"] == "S3"


@pytest.mark.asyncio
async def test_find_top_solvents_for_target_applies_filters_and_properties():
    from strap.services.sequence_runtime_service import find_top_solvents_for_target

    def fake_get_solubility(polymer, solvent, temperature):
        table = {
            ("A", "S1", 60.0): 80.0,
            ("B", "S1", 60.0): 10.0,
            ("A", "S2", 60.0): 50.0,
            ("B", "S2", 60.0): 45.0,
            ("A", "S3", 60.0): 70.0,
            ("B", "S3", 60.0): 5.0,
        }
        if (polymer, solvent, temperature) in table:
            return table[(polymer, solvent, temperature)]
        if polymer == "A" and solvent in {"S1", "S2", "S3"} and temperature in {25.0, 30.0, 35.0}:
            return 60.0 + (temperature - 25.0)
        if polymer == "B" and solvent in {"S1", "S2", "S3"} and temperature in {25.0, 30.0, 35.0}:
            return 5.0
        return 0.0

    async def fake_lookup(solvents, table):
        return {
            "S3": {"bp": 80.0, "logp": 1.2},
        }

    results = await find_top_solvents_for_target(
        target="A",
        remaining=["B"],
        temperature=60.0,
        top_k=2,
        used_solvents={"S1"},
        excluded_solvents={"S2"},
        min_selectivity=5.0,
        solvent_column="solvent",
        polymer_column="polymer",
        get_solubility=fake_get_solubility,
        get_available_solvents_for_polymer=lambda polymer: ["S1", "S2", "S3"],
        solvent_table="demo_table",
        lookup_solvent_properties=fake_lookup,
    )

    assert [entry["solvent"] for entry in results] == ["S3"]
    assert results[0]["bp"] == 80.0
    assert results[0]["logp"] == 1.2


@pytest.mark.asyncio
async def test_find_top_solvents_for_target_labels_above_range_temperature():
    from strap.services.sequence_runtime_service import find_top_solvents_for_target

    def fake_get_solubility(polymer, solvent, temperature):
        if solvent != "S1":
            return None
        if polymer == "A":
            return 100.0 if temperature == 170.0 else 10.0
        if polymer == "B":
            return 1.0
        return None

    async def fake_lookup(solvents, table):
        return {"S1": {"bp": 220.0}}

    results = await find_top_solvents_for_target(
        target="A",
        remaining=["B"],
        temperature=170.0,
        top_k=1,
        used_solvents=None,
        excluded_solvents=None,
        min_selectivity=5.0,
        solvent_column="solvent",
        polymer_column="polymer",
        get_solubility=fake_get_solubility,
        get_available_solvents_for_polymer=lambda polymer: ["S1"],
        solvent_table="demo_table",
        lookup_solvent_properties=fake_lookup,
    )

    assert results[0]["temperature_extrapolation"] == "above_fit"
    assert results[0]["optimal_temp"] == 170.0
    assert results[0]["optimal_temp_extrapolation"] == "above_fit"


@pytest.mark.asyncio
async def test_load_scheme_property_maps_matches_abbreviations():
    from strap.services.sequence_runtime_service import load_scheme_property_maps

    class FakeConnection:
        def execute(self, query):
            class Result:
                @staticmethod
                def fetchdf():
                    import pandas as pd

                    return pd.DataFrame(
                        [
                            {"solvent_common_name": "tetrahydrofuran", "g_score": 7.5},
                        ]
                    )

            return Result()

    async def fake_lookup(solvents, table):
        return {
            "THF": {"bp": 66.0, "logp": 0.5},
        }

    bp_map, logp_map, gscore_map = await load_scheme_property_maps(
        all_solvents=["THF"],
        solvent_table="demo_table",
        lookup_solvent_properties=fake_lookup,
        connection=FakeConnection(),
        abbreviation_map={"thf": "tetrahydrofuran"},
    )

    assert bp_map["THF"] == 66.0
    assert logp_map["THF"] == 0.5
    assert gscore_map["THF"] == 7.5
