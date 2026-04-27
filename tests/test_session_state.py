"""Tests for durable DISSOLVE CLI session context."""

from __future__ import annotations

import json


def test_session_context_extracts_feedstock_and_followup(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    thread_id = "case-feedstock"
    query = (
        "For a mixed plastic feedstock of 8000 tonnes/year composed of "
        "34% LDPE, 33% EVOH, and 33% PET under scenario A, estimate CAPEX/OPEX/GWP."
    )

    context = session_state.load_session_context(thread_id)
    context = session_state.update_session_context_from_text(context, query)
    session_state.save_session_context(thread_id, context)

    saved = json.loads((tmp_path / thread_id / "context.json").read_text(encoding="utf-8"))
    assert saved["feedstock"]["capacity_mt_yr"] == 8000
    assert saved["feedstock"]["composition_wt_pct"] == {
        "EVOH": 33.0,
        "LDPE": 34.0,
        "PET": 33.0,
    }
    assert saved["process"]["scenario"] == "A"
    assert "target_plastic_percent" not in saved["process"]

    followup = "Now run it under C2 and focus on GWP."
    assert session_state.should_inject_session_context(followup, saved) is True
    injected = session_state.inject_session_context(followup, saved)
    assert "Session context" in injected
    assert "capacity=8,000 MT/yr" in injected
    assert "LDPE=34%" in injected
    assert "User request:" in injected


def test_session_context_avoids_injection_for_fully_specified_new_query(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.update_session_context_from_text(
        session_state.load_session_context("thread-a"),
        "For a feedstock of 8000 tonnes/year composed of 60% PE and 40% EVOH under scenario A.",
    )
    new_query = (
        "For a feedstock of 50000 tonnes/year composed of 100% PET under scenario B, "
        "estimate MSP and GWP."
    )

    assert session_state.should_inject_session_context(new_query, context) is False


def test_session_context_extracts_assistant_solvent_candidates_for_followup(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.load_session_context("solvent-followup")
    context = session_state.update_session_context_from_text(
        context,
        (
            "Common solvents for dissolving LDPE include alkanes such as "
            "cyclohexane, hexane, n-heptane, and dodecane."
        ),
        role="assistant",
    )

    assert context["process"]["solvent_candidates"] == [
        "Cyclohexane",
        "Hexane",
        "Heptane",
        "Dodecane",
    ]
    followup = "what is the solubility of LDPE in each of these solvents up to 80C"
    assert session_state.should_inject_session_context(followup, context) is True
    injected = session_state.inject_session_context(followup, context)
    assert "solvent_candidates=" in injected
    assert "Cyclohexane" in injected
    assert "User request:" in injected


def test_session_context_tracks_latest_displayed_recommendation_table(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.load_session_context("displayed-recommendations")
    context = session_state.update_session_context_from_text(
        context,
        (
            "Separation Analysis (Up to 100C)\n\n"
            "  Polymer   Recommended Solvent           Potential Selectivity\n"
            "  LDPE      Dodecane                      Good selectivity\n"
            "  EVOH      Dimethylsulfoxide (DMSO)      Strong candidate\n"
            "  PET       N,N-Dimethylformamide (DMF)   Effective but caution\n\n"
            "Engineering Considerations: DMF can dissolve multiple components."
        ),
        role="assistant",
    )

    latest = context["analysis"]["last_solvent_candidate_table"]
    assert latest["source"] == "assistant_displayed_recommendations"
    assert latest["polymers"] == ["LDPE", "EVOH", "PET"]
    assert latest["solvents"] == ["Dodecane", "Dimethyl sulfoxide", "N,N-Dimethylformamide"]
    injected = session_state.inject_session_context("plot each of these solvents", context)
    assert "Last solvent candidates: polymers=LDPE, EVOH, PET" in injected
    assert "solvents=Dodecane, Dimethyl sulfoxide, N,N-Dimethylformamide" in injected


def test_session_context_filters_triethylamine_from_accumulated_candidates(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.update_session_context_from_text(
        session_state.load_session_context("triethylamine-filter"),
        "Common LDPE solvents include dodecane and triethylamine.",
        role="assistant",
    )

    assert context["process"]["solvent_candidates"] == ["Dodecane"]


def test_session_context_does_not_treat_deg_c_as_diethylene_glycol(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.update_session_context_from_text(
        session_state.load_session_context("deg-c"),
        (
            "For a multilayer feedstock containing LDPE, EVOH, and PET, identify "
            "solvents for dissolving any component below 100 deg C."
        ),
    )

    assert "solvent_candidates" not in context.get("process", {})


def test_query_context_does_not_treat_deg_c_as_diethylene_glycol():
    from strap.query_context import extract_query_context

    assert extract_query_context("Use solvents below 100 deg C.").solvents == ()
    assert extract_query_context("Use diethylene glycol as a solvent.").solvents == ("diethylene_glycol",)


def test_session_context_repairs_wrapped_output_dir_and_temperature_units(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    query = (
        "Use cyclohexane for LDPE at 212 fahrenehit and save to "
        f"{tmp_path}/case-1/01-ldpe-evoh-p\n"
        "  et\n"
        "    -solubility/json."
    )
    context = session_state.update_session_context_from_text(
        session_state.load_session_context("wrapped-path-temp"),
        query,
    )

    assert abs(context["process"]["dissolution_temp_c"] - 100.0) < 1e-6
    assert context["process"]["output_dir"] == str(
        tmp_path / "case-1" / "01-ldpe-evoh-pet-solubility" / "json"
    )


def test_session_context_normalizes_degree_word_and_precipitation_temperatures(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    dissolution = session_state.update_session_context_from_text(
        session_state.load_session_context("degrees-f"),
        "Use cyclohexane for LDPE at 212 degrees Fahrenheit.",
    )
    assert abs(dissolution["process"]["dissolution_temp_c"] - 100.0) < 1e-6

    precipitation = session_state.update_session_context_from_text(
        session_state.load_session_context("precip-f"),
        "Set precipitation temperature 50 F.",
    )
    assert abs(precipitation["process"]["precipitation_temp_c"] - 10.0) < 1e-6
    assert "dissolution_temp_c" not in precipitation["process"]

    cooling = session_state.update_session_context_from_text(
        session_state.load_session_context("cooling-k"),
        "Set cooling temperature 323.15 K.",
    )
    assert abs(cooling["process"]["precipitation_temp_c"] - 50.0) < 1e-6
    assert "dissolution_temp_c" not in cooling["process"]


def test_session_context_still_extracts_diethylene_glycol_full_name(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.update_session_context_from_text(
        session_state.load_session_context("diethylene-glycol"),
        "Show PET solubility in diethylene glycol.",
    )

    assert context["process"]["solvent_candidates"] == ["diethylene_glycol"]


def test_session_context_tracks_last_solubility_lookup_without_false_composition(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.load_session_context("plot-followup")
    context = session_state.update_session_context_from_text(
        context,
        "what is the solubility of EVOH in DMF from room temp to 80C",
    )
    context = session_state.update_session_context_from_text(
        context,
        "The predicted solubility of EVOH in DMF increases from approximately 0.16% at 25°C to 6.63% at 80°C.",
        role="assistant",
    )

    assert context["analysis"]["last_solubility_lookup"] == {
        "polymer": "EVOH",
        "solvents": ["N,N-Dimethylformamide"],
        "temperature_min_c": 25.0,
        "temperature_max_c": 80.0,
    }
    assert context["feedstock"].get("composition_wt_pct") in (None, {})
    injected = session_state.inject_session_context("plot it", context)
    assert "Last solubility lookup" in injected


def test_session_context_tracks_requested_output_dir(tmp_path):
    from strap.session_state import build_session_context_block, update_session_context_from_text

    context = {
        "schema_version": "1.1",
        "thread_id": "test",
        "feedstock": {},
        "process": {},
        "analysis": {},
        "artifacts": [],
        "route_decisions": [],
        "run_ledgers": [],
        "last_user_query": "",
    }

    updated = update_session_context_from_text(
        context,
        f"Plot LDPE/EVOH/PET solubility and save structured output to {tmp_path}.",
        role="user",
    )

    assert updated["process"]["output_dir"] == str(tmp_path)
    assert f"output_dir={tmp_path}" in build_session_context_block(updated)


def test_session_context_does_not_overwrite_process_temperature_from_plot_range():
    from strap.session_state import update_session_context_from_text

    context = {
        "schema_version": "1.1",
        "thread_id": "test",
        "feedstock": {"polymers": ["LDPE", "EVOH", "PET"]},
        "process": {"dissolution_temp_c": 90.0},
        "analysis": {},
        "artifacts": [],
        "route_decisions": [],
        "run_ledgers": [],
        "last_user_query": "",
    }

    updated = update_session_context_from_text(
        context,
        "plot the solubility of these polymers in these solvents from room temperature to 100C",
        role="user",
    )

    assert updated["process"]["dissolution_temp_c"] == 90.0


def test_session_context_extracts_actual_process_temperature_from_stage_text():
    from strap.session_state import update_session_context_from_text

    context = {
        "schema_version": "1.1",
        "thread_id": "test",
        "feedstock": {},
        "process": {},
        "analysis": {},
        "artifacts": [],
        "route_decisions": [],
        "run_ledgers": [],
        "last_user_query": "",
    }

    updated = update_session_context_from_text(
        context,
        "Stage 1: Use 1,4-dimethylbenzene at 90C to dissolve LDPE.",
        role="assistant",
    )

    assert updated["process"]["dissolution_temp_c"] == 90.0


def test_session_context_injects_for_plot_range_correction(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.load_session_context("plot-correction")
    context = session_state.update_session_context_from_text(
        context,
        "For an LDPE/EVOH/PET feedstock, what is the solubility of EVOH in DMF from room temp to 153C?",
    )
    context = session_state.update_session_context_from_text(
        context,
        "The predicted solubility of EVOH in DMF was plotted from 25C to 153C.",
        role="assistant",
    )

    followup = "no just plot from 25C to 90C"
    assert session_state.should_inject_session_context(followup, context) is True
    injected = session_state.inject_session_context(followup, context)
    assert "Last solubility lookup" in injected
    assert "polymer=EVOH" in injected
    assert "solvents=N,N-Dimethylformamide" in injected
    assert "User request:\nno just plot from 25C to 90C" in injected


def test_session_context_persists_direct_artifacts_for_followups(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.load_session_context("artifact-followup")
    context = session_state.update_session_context_from_direct_metadata(
        context,
        {
            "strap_route_decision": {
                "route_id": "route_1",
                "mode": "direct_tool",
                "intent": "solvent_candidate_lookup",
                "allowed_tools": ["list_available_solvents"],
                "model_call_budget": 0,
                "tool_call_budget": 1,
            },
            "strap_run_ledger": {
                "route_id": "route_1",
                "status": "ok",
                "model_calls": 0,
                "tool_calls": 1,
                "tools": ["list_available_solvents"],
            },
            "strap_artifacts": [
                {
                    "artifact_id": "artifact_1",
                    "type": "solvent_candidate_table",
                    "producer": "list_available_solvents",
                    "entities": {
                        "polymer": "EVOH",
                        "solvents": ["dimethylformamide", "dimethylsulfoxide", "isopropylamine"],
                    },
                    "data": {},
                    "row_order": ["dimethylformamide", "dimethylsulfoxide", "isopropylamine"],
                }
            ],
        },
    )

    assert context["route_decisions"][0]["route_id"] == "route_1"
    assert context["analysis"]["last_solvent_candidate_table"]["polymer"] == "EVOH"
    assert session_state.should_inject_session_context("plot the top 2 of those up to 100C", context) is True
    injected = session_state.inject_session_context("plot the top 2 of those up to 100C", context)
    assert "Last solvent candidates" in injected
    assert "polymer=EVOH" in injected
    assert "dimethylformamide" in injected


def test_session_context_persists_plot_artifact_for_range_correction(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.load_session_context("plot-artifact")
    context = session_state.update_session_context_from_direct_metadata(
        context,
        {
            "strap_artifacts": [
                {
                    "artifact_id": "artifact_plot",
                    "type": "plot_artifact",
                    "producer": "plot_solubility_vs_temperature",
                    "entities": {
                        "plot_type": "solubility_vs_temperature",
                        "polymer": "EVOH",
                        "solvents": ["dimethylformamide"],
                    },
                    "data": {
                        "path": "/tmp/evoh.png",
                        "temperature_min_c": 25.0,
                        "temperature_max_c": 153.0,
                    },
                    "row_order": ["dimethylformamide"],
                }
            ]
        },
    )

    assert session_state.should_inject_session_context("no just plot from 25C to 90C", context) is True
    injected = session_state.inject_session_context("no just plot from 25C to 90C", context)
    assert "Last plot artifact" in injected
    assert "temperature_range=25-153 C" in injected
    assert "path=/tmp/evoh.png" in injected


def test_session_context_persists_optimization_result_artifact(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    context = session_state.load_session_context("optimization-artifact")
    payload = {
        "analysis_type": "point_optimum",
        "scenario": "A",
        "feed_composition": {"PE": 0.6, "EVOH": 0.4},
        "profit": 12_136_242.52,
        "total_cost": 6_953_357.48,
        "emissions": 9_211.0833,
        "circularity_score": 0.6408,
        "optimal_washes": ["PE-o-Xylene @ 143.5C", "EVOH-Dimethyl sulfoxide @ 145C"],
    }

    context = session_state.update_session_context_from_direct_metadata(
        context,
        {
            "strap_artifacts": [
                {
                    "artifact_id": "artifact_opt",
                    "type": "optimization_point_result",
                    "producer": "optimization-engineer",
                    "entities": {"analysis_type": "point_optimum"},
                    "data": {"payload": payload},
                    "row_order": [],
                }
            ]
        },
    )

    saved_payload = context["analysis"]["last_optimization_result"]["payload"]
    assert saved_payload["analysis_type"] == "point_optimum"
    assert saved_payload["total_cost"] == 6_953_357.48
    injected = session_state.inject_session_context("plot this result", context)
    assert "Last optimization result:" in injected
    assert '"analysis_type":"point_optimum"' in injected
    assert "PE-o-Xylene @ 143.5C" in injected


def test_session_context_persists_multi_polymer_plot_output_dir(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    output_dir = tmp_path / "cli-plots"
    context = session_state.load_session_context("multi-plot-artifact")
    context = session_state.update_session_context_from_direct_metadata(
        context,
        {
            "strap_artifacts": [
                {
                    "artifact_id": "artifact_plot",
                    "type": "plot_artifact",
                    "producer": "plot_solubility_vs_temperature",
                    "entities": {
                        "plot_type": "solubility_vs_temperature",
                        "polymer": "LDPE",
                        "polymers": ["LDPE", "EVOH", "PET"],
                        "solvents": ["dodecane", "1,2-dimethylbenzene", "toluene"],
                    },
                    "data": {
                        "path": str(output_dir / "plot.png"),
                        "output_dir": str(output_dir),
                        "temperature_min_c": 25.0,
                        "temperature_max_c": 150.0,
                    },
                    "row_order": ["dodecane", "1,2-dimethylbenzene", "toluene"],
                }
            ]
        },
    )

    injected = session_state.inject_session_context("no replot from 25C to 100C", context)

    assert "polymers=LDPE, EVOH, PET" in injected
    assert "output_dir=" + str(output_dir) in injected


def test_session_transcript_jsonl(tmp_path, monkeypatch):
    from strap import session_state

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    session_state.append_transcript_event("thread-a", "user", "hello", injected_context=False)
    session_state.append_transcript_event("thread-a", "assistant", "world")

    lines = (tmp_path / "thread-a" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["role"] == "user"
    assert json.loads(lines[0])["metadata"]["injected_context"] is False
    assert json.loads(lines[1])["content"] == "world"
