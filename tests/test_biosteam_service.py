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


def test_build_single_config_omits_null_optional_temperatures():
    from strap.services.biosteam_service import build_single_config

    config = build_single_config(
        solvent="Cyclohexane",
        target_plastic="PE",
        target_plastic_percent=60.0,
        processing_capacity=1000.0,
        dissolution_temp_c=None,
        precipitation_temp_c=None,
    )

    assert config["solvent"] == "Cyclohexane"
    assert "dissolution_temperature_c" not in config
    assert "precipitation_temperature_c" not in config


def test_resolve_to_biosteam_maps_gvl_to_canonical_name():
    from strap.solvent_registry import resolve_to_biosteam

    assert resolve_to_biosteam("gvl") == "gamma-Valerolactone"
    assert resolve_to_biosteam("gamma-valerolactone") == "gamma-Valerolactone"


def test_build_single_config_canonicalizes_gvl_alias_for_biosteam():
    from strap.services.biosteam_service import build_single_config

    config = build_single_config(
        solvent="gvl",
        target_plastic="EVOH",
        target_plastic_percent=40.0,
        processing_capacity=1000.0,
        dissolution_temp_c=100.0,
    )

    assert config["solvent"] == "gamma-Valerolactone"
    assert config["solvent_input_name"] == "gvl"
    assert config["dissolution_temperature_c"] == 100.0


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


def test_worker_target_plastic_map_keeps_native_evoh_and_pc():
    from strap.vendor.biosteam_worker import _TARGET_PLASTIC_MAP

    assert _TARGET_PLASTIC_MAP["EVOH"] == "EVOH"
    assert _TARGET_PLASTIC_MAP["PC"] == "PC"
