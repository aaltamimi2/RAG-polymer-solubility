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
    assert "polymer=EVOH" in injected
    assert "temperature_range=25-80 C" in injected


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
