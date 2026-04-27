import json

import pytest


def test_safety_card_uses_local_bp_and_curated_peroxide_without_pubchem():
    from strap.tools.safety_card import get_solvent_safety_card

    parsed = json.loads(
        get_solvent_safety_card("THF", operating_temp_c=60, include_pubchem=False)
    )

    profile = parsed["data"]["safety_profile"]
    assert parsed["data"]["success"] is True
    assert profile["identity"]["name"] == "Tetrahydrofuran (THF)"
    assert profile["physical_properties"]["boiling_point_c"] == pytest.approx(65.0)
    assert profile["peroxide_risk"]["peroxide_former_class"] == "Class B"
    assert profile["process_temperature_assessment"]["boiling_margin_c"] == pytest.approx(5.0)
    assert "near_normal_boiling_point" in profile["process_temperature_assessment"]["flags"]
    assert parsed["display"].startswith("╭")
    assert "DISSOLVE SAFETY CARD" in parsed["display"]
    assert "RISK: HIGH" in parsed["display"]
    assert "Peroxide / Storage" in parsed["display"]


def test_safety_card_normalizes_noisy_query_before_local_property_lookup():
    from strap.services.solvent_safety_service import build_solvent_safety_profile

    profile = build_solvent_safety_profile("can you dodecane", include_pubchem=False)

    assert profile["identity"]["name"] == "Dodecane"
    assert profile["identity"]["cas_number"] == "112-40-3"
    assert profile["physical_properties"]["boiling_point_c"] == pytest.approx(216.3)
    assert profile["sources"]["local_properties"] == "data/Solvent_Data.csv"


def test_safety_profile_flags_heated_toluene_with_pubchem_physical_data(monkeypatch):
    from strap.services import solvent_safety_service as service

    monkeypatch.setattr(
        service,
        "fetch_pubchem_physical_properties",
        lambda _cid: {
            "flash_point_c": 4.4,
            "flash_point_raw": "4.4 C",
            "autoignition_c": 480.0,
            "autoignition_raw": "480 C",
            "pubchem_boiling_point_c": 110.6,
            "pubchem_boiling_point_raw": "110.6 C",
            "vapor_pressure_kpa": 3.79,
            "vapor_pressure_temp_c": 25.0,
            "vapor_pressure_raw": "28.4 mmHg",
            "raw_headings": {},
        },
    )
    monkeypatch.setattr(
        service,
        "_fetch_pubchem_hazards",
        lambda _cid: {
            "ghs": {
                "signal_word": "Danger",
                "pictograms": ["Flammable"],
                "hazard_statements": ["Highly flammable liquid and vapor"],
            },
            "toxicity": {"ld50_values": ["oral rat LD50 5580 mg/kg"]},
        },
    )

    profile = service.build_solvent_safety_profile(
        "toluene",
        operating_temp_c=110.0,
        include_pubchem=True,
    )

    assessment = profile["process_temperature_assessment"]
    assert profile["physical_properties"]["flash_point_c"] == pytest.approx(4.4)
    assert "near_normal_boiling_point" in assessment["flags"]
    assert "above_flash_point" in assessment["flags"]
    assert assessment["risk_level"] == "high"
    assert profile["ghs"]["signal_word"] == "Danger"
    assert profile["toxicity"]["ld50_values"][0].startswith("oral rat")


def test_direct_safety_parser_extracts_single_and_multi_solvents():
    from strap.direct_fast_path import _extract_operating_temperature, _extract_safety_solvents

    assert _extract_safety_solvents("safety card for THF at 60 C") == ["Tetrahydrofuran (THF)"]
    assert _extract_safety_solvents("can you show me the dodecane safety card") == ["Dodecane"]
    assert _extract_operating_temperature("safety card for THF at 60 C") == pytest.approx(60.0)
    assert _extract_safety_solvents("autoemission temperature of toluene") == ["Toluene"]
    assert _extract_safety_solvents(
        "compare safety profile for heptane, cyclohexane, and toluene at 110 C"
    ) == ["Heptane", "Cyclohexane", "Toluene"]


def test_safety_comparison_uses_terminal_card_layout():
    from strap.tools.safety_card import compare_solvent_safety_cards

    parsed = json.loads(
        compare_solvent_safety_cards(
            "THF, tetrahydropyran",
            operating_temp_c=65,
            include_pubchem=False,
        )
    )

    assert parsed["data"]["success"] is True
    assert parsed["display"].startswith("╭")
    assert "DISSOLVE SAFETY COMPARISON" in parsed["display"]
    assert "Ranked Profiles" in parsed["display"]
    assert "Tetrahydrofuran (THF)" in parsed["display"]
