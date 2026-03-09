from __future__ import annotations

import json


def test_contaminant_data_service_parses_families_and_known_entries():
    from strap.services.contaminant_data_service import (
        expand_requested_contaminants,
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
