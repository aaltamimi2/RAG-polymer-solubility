from __future__ import annotations

from strap.handoff_adapters import build_typed_handoff
from strap.handoff_models import HandoffRecord, HandoffScope


def _scope() -> HandoffScope:
    return HandoffScope(invocation_id="inv", run_id="run", thread_id="thread")


def test_build_typed_handoff_separation_to_optimization_includes_solvent_filters():
    source = HandoffRecord(
        handoff_id="handoff-sep-1",
        scope=_scope(),
        producer="separation-engineer",
        consumer="orchestrator",
        contract="separation.route.v1",
        status="ok",
        payload={
            "agent": "separation-engineer",
            "schema_version": "1.0",
            "polymers": ["LDPE", "EVOH"],
            "steps": [
                {"step": 1, "polymer": "LDPE", "solvent": "Toluene", "temperature_c": 110.0},
                {"step": 2, "polymer": "EVOH", "solvent": "Ethylene Glycol", "temperature_c": 90.0},
            ],
            "solvent_mapping": {"LDPE": "Toluene", "EVOH": "Ethylene Glycol"},
            "top_solvents": ["Toluene", "Xylene", "Ethylene Glycol"],
            "top_k_sequences": [
                {"rank": 1, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Toluene", "EVOH": "Ethylene Glycol"}},
                {"rank": 2, "sequence": ["LDPE", "EVOH"], "solvent_mapping": {"LDPE": "Xylene", "EVOH": "Pyridazine"}},
            ],
        },
        created_at="2026-04-17T18:00:00Z",
    )

    typed = build_typed_handoff(
        source,
        "optimization-engineer",
        scope_user_query="Find a separation route and then optimize waste management for the shortlisted solvents.",
    )

    assert typed is not None
    contract, payload, task_prompt = typed

    assert contract == "optimization_route_context.v1"
    assert payload["polymer_solvent_filters"]["PE"] == ["Toluene", "Xylene"]
    assert payload["polymer_solvent_filters"]["EVOH"] == ["Ethylene Glycol", "Pyridazine"]
    assert "candidate_solvents" in payload
    assert "polymer_solvent_filters_json" in task_prompt
    assert "candidate_solvents" in task_prompt
