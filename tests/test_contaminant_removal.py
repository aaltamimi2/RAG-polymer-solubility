from __future__ import annotations

import json


def test_contaminant_data_service_parses_families_and_known_entries():
    from strap.services.contaminant_data_service import (
        expand_requested_contaminants,
        get_contaminant_family,
        get_logd_entry,
        get_miscibility_entry,
        list_supported_contaminant_families,
    )

    families = list_supported_contaminant_families()
    assert families == ["PFAS", "Phthalates"]

    supported, unsupported, expanded_families = expand_requested_contaminants(["phthalates"])
    assert unsupported == []
    assert "Phthalates" in expanded_families
    assert "di-n-butyl phthalate (DBP)" in supported

    miscibility = get_miscibility_entry("Acetone", "di-n-butyl phthalate (DBP)", regime="rt")
    assert miscibility is not None
    assert miscibility["miscible"] is True

    logd = get_logd_entry("dichloromethane", "di-n-butyl phthalate (DBP)")
    assert logd is not None
    assert round(logd["logd"], 2) == 1.57
    assert get_contaminant_family("di-n-butyl phthalate (DBP)") == "Phthalates"


def test_screen_leaching_candidates_prefers_positive_logd_and_non_dissolving_proxy(monkeypatch):
    from strap.services import contaminant_screening_service as service

    monkeypatch.setattr(
        service,
        "expand_requested_contaminants",
        lambda contaminants: (["di-n-butyl phthalate (DBP)"], [], ["Phthalates"]),
    )
    monkeypatch.setattr(service, "_resolve_polymer_or_none", lambda polymer: polymer.upper())
    monkeypatch.setattr(service, "get_supported_solvents_for_contaminants", lambda contaminants: ["acetone", "methanol"])
    monkeypatch.setattr(service, "_choose_leaching_temperature", lambda solvent, max_temperature_c: (55.0, "t_higher"))
    monkeypatch.setattr(service, "get_boiling_point", lambda solvent: {"acetone": 56.0, "methanol": 64.6}[solvent])

    def fake_screen(solvent, contaminants, *, regime):
        if solvent == "acetone":
            return ([{"contaminant": contaminants[0], "miscible": True, "logd": 0.69}], 0.69, True, True)
        return ([{"contaminant": contaminants[0], "miscible": True, "logd": -0.61}], -0.61, True, False)

    monkeypatch.setattr(service, "_screen_contaminants_for_solvent", fake_screen)

    def fake_polymer_status(polymer, solvent, temperature_c):
        status = "non_dissolving_proxy_swelling_candidate" if solvent == "acetone" else "non_dissolving_low_swelling_confidence"
        solubility = 3.0 if solvent == "acetone" else 0.2
        return service._PolymerStatus(
            polymer=polymer,
            solvent=solvent,
            temperature_c=temperature_c,
            supported=True,
            solubility_wt_pct=solubility,
            status=status,
        )

    monkeypatch.setattr(service, "_classify_polymer_behavior", fake_polymer_status)

    result = service.screen_leaching_candidates(
        target_polymer="PET",
        contaminants=["di-n-butyl phthalate (DBP)"],
    )

    assert result["recommended_solvents"] == ["acetone"]
    assert result["candidate_solvents"][0]["solvent"] == "acetone"
    assert result["candidate_solvents"][0]["passes"] is True


def test_screen_contaminant_leaching_returns_standard_envelope(monkeypatch):
    from strap.tools import contaminant_removal

    monkeypatch.setattr(
        contaminant_removal,
        "_screen_leaching",
        lambda **kwargs: {
            "mode": "leaching",
            "target_polymer": "PET",
            "other_polymers": [],
            "contaminants": ["di-n-butyl phthalate (DBP)"],
            "supported_contaminants": ["di-n-butyl phthalate (DBP)"],
            "unsupported_contaminants": [],
            "candidate_solvents": [
                {
                    "solvent": "acetone",
                    "passes": True,
                    "operating_temperature_c": 55.0,
                    "contaminant_logd_min": 0.69,
                    "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                }
            ],
            "recommended_solvents": ["acetone"],
            "decision_basis": ["positive contaminant logD"],
            "caveats": ["proxy swelling"],
        },
    )

    parsed = json.loads(
        contaminant_removal.screen_contaminant_leaching(
            target_polymer="PET",
            contaminants="di-n-butyl phthalate (DBP)",
        )
    )

    assert parsed["data"]["tool_name"] == "screen_contaminant_leaching"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["recommended_solvents"] == ["acetone"]


def test_list_supported_contaminants_returns_standard_envelope():
    from strap.tools.contaminant_removal import list_supported_contaminants

    parsed = json.loads(list_supported_contaminants("PFAS"))

    assert parsed["data"]["tool_name"] == "list_supported_contaminants"
    assert parsed["data"]["success"] is True
    assert "PFAS" in parsed["data"]["supported_families"]
    assert parsed["data"]["n_contaminants"] > 0


def test_plan_contaminant_wash_steps_prefers_single_step_when_tradeoff_gain_is_small(monkeypatch):
    from strap.services import contaminant_screening_service as service

    pfas = "Perfluorobutanoic acid"
    phthalate = "di-n-butyl phthalate (DBP)"
    mode_result = {
        "mode": "leaching",
        "supported_contaminants": [pfas, phthalate],
        "candidate_solvents": [
            {
                "solvent": "broad-one-step",
                "mode": "leaching",
                "operating_temperature_c": 25.0,
                "boiling_point_c": 80.0,
                "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                "other_polymer_status": {},
                "contaminants": [
                    {"contaminant": pfas, "miscible": True, "logd": 0.7},
                    {"contaminant": phthalate, "miscible": True, "logd": 0.8},
                ],
            },
            {
                "solvent": "pfas-only",
                "mode": "leaching",
                "operating_temperature_c": 25.0,
                "boiling_point_c": 90.0,
                "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                "other_polymer_status": {},
                "contaminants": [
                    {"contaminant": pfas, "miscible": True, "logd": 0.9},
                    {"contaminant": phthalate, "miscible": False, "logd": -0.2},
                ],
            },
            {
                "solvent": "phthalate-only",
                "mode": "leaching",
                "operating_temperature_c": 25.0,
                "boiling_point_c": 92.0,
                "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                "other_polymer_status": {},
                "contaminants": [
                    {"contaminant": pfas, "miscible": False, "logd": -0.2},
                    {"contaminant": phthalate, "miscible": True, "logd": 0.9},
                ],
            },
        ],
    }

    profiles = {
        "broad-one-step": {"price_usd_kg": 1.5, "g_score": 6.0, "gsk_class": "Esters"},
        "pfas-only": {"price_usd_kg": 0.5, "g_score": 8.0, "gsk_class": "Alcohols"},
        "phthalate-only": {"price_usd_kg": 0.5, "g_score": 8.0, "gsk_class": "Alcohols"},
    }
    monkeypatch.setattr(
        service,
        "_lookup_solvent_tradeoff_profile",
        lambda solvent, *, conn: profiles[solvent],
    )

    planned = service.plan_contaminant_wash_steps(mode_result=mode_result)

    assert planned["recommended_wash_plan"]["n_steps"] == 1
    assert planned["recommended_wash_plan"]["covered_contaminants"] == [pfas, phthalate]
    assert any(
        plan["n_steps"] == 2 and "best_multi_step" in plan["selection_labels"]
        for plan in planned["wash_step_plans"]
    )


def test_plan_contaminant_wash_steps_prefers_multi_step_when_single_step_is_harsh(monkeypatch):
    from strap.services import contaminant_screening_service as service

    pfas = "Perfluorobutanoic acid"
    phthalate = "di-n-butyl phthalate (DBP)"
    mode_result = {
        "mode": "leaching",
        "supported_contaminants": [pfas, phthalate],
        "candidate_solvents": [
            {
                "solvent": "harsh-one-step",
                "mode": "leaching",
                "operating_temperature_c": 25.0,
                "boiling_point_c": 85.0,
                "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                "other_polymer_status": {},
                "contaminants": [
                    {"contaminant": pfas, "miscible": True, "logd": 0.6},
                    {"contaminant": phthalate, "miscible": True, "logd": 0.7},
                ],
            },
            {
                "solvent": "cheap-safe-pfas",
                "mode": "leaching",
                "operating_temperature_c": 25.0,
                "boiling_point_c": 95.0,
                "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                "other_polymer_status": {},
                "contaminants": [
                    {"contaminant": pfas, "miscible": True, "logd": 0.9},
                    {"contaminant": phthalate, "miscible": False, "logd": -0.3},
                ],
            },
            {
                "solvent": "cheap-safe-phthalate",
                "mode": "leaching",
                "operating_temperature_c": 25.0,
                "boiling_point_c": 94.0,
                "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                "other_polymer_status": {},
                "contaminants": [
                    {"contaminant": pfas, "miscible": False, "logd": -0.3},
                    {"contaminant": phthalate, "miscible": True, "logd": 0.9},
                ],
            },
        ],
    }

    profiles = {
        "harsh-one-step": {"price_usd_kg": 4.8, "g_score": 1.0, "gsk_class": "Chlorinated"},
        "cheap-safe-pfas": {"price_usd_kg": 0.4, "g_score": 9.0, "gsk_class": "Alcohols"},
        "cheap-safe-phthalate": {"price_usd_kg": 0.4, "g_score": 9.0, "gsk_class": "Alcohols"},
    }
    monkeypatch.setattr(
        service,
        "_lookup_solvent_tradeoff_profile",
        lambda solvent, *, conn: profiles[solvent],
    )

    planned = service.plan_contaminant_wash_steps(mode_result=mode_result)

    assert planned["recommended_wash_plan"]["n_steps"] == 2
    assert planned["recommended_wash_plan"]["full_coverage"] is True
    assert {
        step["solvent"]
        for step in planned["recommended_wash_plan"]["steps"]
    } == {"cheap-safe-pfas", "cheap-safe-phthalate"}


def test_screen_leaching_candidates_emits_multi_step_plan_when_no_single_solvent_covers_all(monkeypatch):
    from strap.services import contaminant_screening_service as service

    pfas = "Perfluorobutanoic acid"
    phthalate = "di-n-butyl phthalate (DBP)"

    monkeypatch.setattr(
        service,
        "expand_requested_contaminants",
        lambda contaminants: ([pfas, phthalate], [], ["PFAS", "Phthalates"]),
    )
    monkeypatch.setattr(service, "_resolve_polymer_or_none", lambda polymer: polymer.upper())
    monkeypatch.setattr(service, "get_supported_solvents_for_contaminants", lambda contaminants: ["wash-a", "wash-b"])
    monkeypatch.setattr(service, "_choose_leaching_temperature", lambda solvent, max_temperature_c: (25.0, "rt"))
    monkeypatch.setattr(service, "get_boiling_point", lambda solvent: {"wash-a": 70.0, "wash-b": 72.0}[solvent])

    def fake_screen(solvent, contaminants, *, regime):
        if solvent == "wash-a":
            rows = [
                {"contaminant": pfas, "miscible": True, "logd": 0.8},
                {"contaminant": phthalate, "miscible": False, "logd": -0.1},
            ]
            return rows, -0.1, False, False
        rows = [
            {"contaminant": pfas, "miscible": False, "logd": -0.2},
            {"contaminant": phthalate, "miscible": True, "logd": 0.9},
        ]
        return rows, -0.2, False, False

    monkeypatch.setattr(service, "_screen_contaminants_for_solvent", fake_screen)
    monkeypatch.setattr(
        service,
        "_classify_polymer_behavior",
        lambda polymer, solvent, temperature_c: service._PolymerStatus(
            polymer=polymer,
            solvent=solvent,
            temperature_c=temperature_c,
            supported=True,
            solubility_wt_pct=3.0,
            status="non_dissolving_proxy_swelling_candidate",
        ),
    )
    profiles = {
        "wash-a": {"price_usd_kg": 0.8, "g_score": 8.0, "gsk_class": "Alcohols"},
        "wash-b": {"price_usd_kg": 0.9, "g_score": 7.5, "gsk_class": "Esters"},
    }
    monkeypatch.setattr(
        service,
        "_lookup_solvent_tradeoff_profile",
        lambda solvent, *, conn: profiles[solvent],
    )

    result = service.screen_leaching_candidates(
        target_polymer="LDPE",
        other_polymers="EVOH",
        contaminants="PFAS, Phthalates",
    )

    assert result["recommended_solvents"] == []
    assert result["recommended_wash_plan"]["n_steps"] == 2
    assert result["recommended_wash_plan"]["full_coverage"] is True
    assert {
        step["solvent"]
        for step in result["recommended_wash_plan"]["steps"]
    } == {"wash-a", "wash-b"}


def test_wash_tradeoff_profile_uses_ml_gscore_fallback(monkeypatch):
    from strap.services import contaminant_screening_service as service

    monkeypatch.setattr(
        service,
        "lookup_local_solvent_market_data",
        lambda solvent: {"price_usd_kg": 1.2, "price_source": "test"},
    )
    monkeypatch.setattr(
        service,
        "get_cross_database_properties",
        lambda solvent, conn: {"g_score": None, "gsk_class": None},
    )
    monkeypatch.setattr(
        service,
        "lookup_local_gscore_data",
        lambda solvent: {
            "solvent_name": "ExampleSolvent",
            "classification": None,
            "g_score": 7.4,
            "g_score_uncertainty": 0.35,
            "source": "GreenSolventDB_10k",
            "ml_predicted": True,
        },
    )

    profile = service._lookup_solvent_tradeoff_profile("ExampleSolvent", conn=None)

    assert profile["g_score"] == 7.4
    assert profile["g_score_source"] == "GreenSolventDB_10k"
    assert profile["g_score_uncertainty"] == 0.35


def test_plan_contaminant_wash_steps_marks_ml_gscore_source(monkeypatch):
    from strap.services import contaminant_screening_service as service

    pfas = "Perfluorobutanoic acid"
    mode_result = {
        "mode": "leaching",
        "supported_contaminants": [pfas],
        "candidate_solvents": [
            {
                "solvent": "ml-only-solvent",
                "mode": "leaching",
                "operating_temperature_c": 25.0,
                "boiling_point_c": 80.0,
                "target_polymer_status": "non_dissolving_proxy_swelling_candidate",
                "other_polymer_status": {},
                "contaminants": [
                    {"contaminant": pfas, "miscible": True, "logd": 0.7},
                ],
            }
        ],
    }

    monkeypatch.setattr(
        service,
        "_lookup_solvent_tradeoff_profile",
        lambda solvent, *, conn: {
            "price_usd_kg": 1.0,
            "price_source": "test",
            "g_score": 7.1,
            "gsk_class": None,
            "g_score_source": "GreenSolventDB_10k",
            "g_score_uncertainty": 0.22,
        },
    )

    planned = service.plan_contaminant_wash_steps(mode_result=mode_result)

    step = planned["recommended_wash_plan"]["steps"][0]
    assert step["g_score"] == 7.1
    assert step["g_score_source"] == "GreenSolventDB_10k"
    assert step["g_score_uncertainty"] == 0.22
