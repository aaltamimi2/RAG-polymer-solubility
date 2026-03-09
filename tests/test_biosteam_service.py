"""Tests for BioSTEAM service-layer helpers."""

import json


def test_expand_solvents_expands_known_keyword():
    from strap.services.biosteam_service import expand_solvents

    solvents = expand_solvents("all_pe", "PE")

    assert "Toluene" in solvents
    assert "p-Xylene" in solvents
    assert len(solvents) > 5


def test_expand_solvents_resolves_aliases():
    from strap.services.biosteam_service import expand_solvents

    solvents = expand_solvents("THF, toluene", "PE")

    assert "Tetrahydrofuran" in solvents
    assert "Toluene" in solvents


def test_prioritize_batch_solvents_moves_stable_pe_candidates_first():
    from strap.services.biosteam_service import prioritize_batch_solvents

    ordered = prioritize_batch_solvents(
        ["sec-Butyl Acetate", "p-Xylene", "Heptane", "Toluene"],
        "PE",
    )

    assert ordered[:3] == ["Heptane", "Toluene", "p-Xylene"]


def test_expand_solvents_uses_csv_driven_pet_catalog():
    from strap.services.biosteam_service import expand_solvents

    solvents = expand_solvents("all_pet", "PET")

    assert "N,N-Dimethylformamide" in solvents
    assert "Dimethyl sulfoxide" in solvents


def test_build_single_config_applies_optional_fields():
    from strap.services.biosteam_service import build_single_config

    config = build_single_config(
        solvent="Toluene",
        target_plastic="PE",
        energy_case="C2",
        dissolution_temp_c=115,
        precipitation_temp_c=30,
        solvent_price=1.25,
    )

    assert config["solvent"] == "Toluene"
    assert config["energy_case"] == "C2"
    assert config["dissolution_temperature_c"] == 115
    assert config["precipitation_temperature_c"] == 30
    assert config["solvent_price"] == 1.25


def test_parse_json_array_rejects_non_array():
    from strap.services.biosteam_service import parse_json_array

    try:
        parse_json_array('{"solvent": "Toluene"}', field_name="scenarios_json")
    except ValueError as exc:
        assert "scenarios_json" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-array JSON input")


def test_json_tool_error_uses_standard_envelope():
    from strap.services.biosteam_service import json_tool_error

    raw = json_tool_error("bad input", tool_name="demo")
    parsed = json.loads(raw)

    assert parsed["display"] == "bad input"
    assert parsed["data"]["success"] is False
    assert parsed["data"]["error"] == "bad input"
    assert parsed["data"]["tool_name"] == "demo"


def test_extract_successful_results_handles_multi_polymer_shape():
    from strap.services.biosteam_service import extract_successful_results

    payload = {
        "per_polymer": [
            {"result": {"success": True, "solvent": "Toluene"}},
            {"result": {"success": False, "solvent": "Xylene"}},
        ]
    }

    results = extract_successful_results(payload)

    assert results == [{"success": True, "solvent": "Toluene"}]
