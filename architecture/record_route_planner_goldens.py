"""Record live route-planner payloads as golden fixtures for offline tests.

Run this whenever the planner prompt in ``strap/route_planner.py`` (or the
planner model) changes, then re-run ``tests/test_route_planner_goldens.py``:

    python architecture/record_route_planner_goldens.py

Requires GOOGLE_API_KEY (loaded from .env). This is the ONLY routing-eval
entry point that spends API calls; everything downstream replays the recorded
payloads model-free.

Sources recorded:
- every query in docs/subagent_query_bank-v1.xlsx (with expected_route labels)
- the case-study turn scripts under case-studies/case-1/
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT / "src"))

from langchain.chat_models import init_chat_model  # noqa: E402

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from strap.route_planner import (  # noqa: E402
    LLMRoutePlannerBackend,
    build_session_digest,
    validate_route_payload,
)

PLANNER_MODEL = "google_genai:gemini-3-flash-preview"
GOLDENS_PATH = _ROOT / "tests" / "fixtures" / "route_planner_goldens.json"
QUERY_BANK_PATH = _ROOT / "docs" / "subagent_query_bank-v1.xlsx"

CASE_STUDY_QUERIES = {
    "cs1_t1_solvent_discovery": "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, identify solvents that are promising for dissolving any one of the components below 100 deg C. Focus on separation-engineering usefulness, not just listing every solvent. Save any structured output to /home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/01-ldpe-evoh-pet-solubility/json.",
    "cs1_t2_solubility_plot": "Using the same LDPE/EVOH/PET feedstock, plot the predicted solubility or solubility-related response of all three polymers in dodecane and o-xylene from 25 to 100 deg C. Save the figure and structured data under /home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/01-ldpe-evoh-pet-solubility.",
    "cs1_t3_state_map": "Now generate a separation state map for all 3! possible LDPE/EVOH/PET separation sequences under a maximum processing temperature of 100 deg C. Save the state map figure and structured sequence-ranking output under the same case-study folder.",
    "cs1_t4_followup": "From that state map, which separation sequence maximizes predicted separation efficiency across each step under the 100 deg C constraint? Use the structured sequence results rather than re-describing the feedstock.",
    "cs2_t1_data_coverage": "For a co-mingled feedstock containing HDPE, LDPE, PP, PVC, PA6, PA66, PC, EVOH, PET, and PS, report which polymers have usable DISSOLVE solubility/separation data and identify promising solvent families below 100 deg C. Save structured output under /home/aaltamimi2/langchain-STRAP-v10-core/case-studies/case-1/02-complex-feedstock/json.",
    "cs2_t2_sequence_planning": "Using only the polymers and solvent candidates with usable data, propose feasible separation sequence options under a maximum temperature of 100 deg C. If the full search space is too large, explain the pruning strategy and save structured outputs and figures under the same case-study folder.",
    "cs2_t3_objective_comparison": "Compare the top feasible sequence options under separation efficiency, solvent greenness, and cost or operating-burden proxies. Save all generated figures and structured outputs under the same case-study folder.",
}


def _sep_completed_history(first_query: str, followup: str, *, failed: bool = False) -> list:
    result = (
        ToolMessage(content="Tool error: solver crashed", tool_call_id="t1", status="error")
        if failed
        else ToolMessage(
            content=(
                '<STRUCTURED_RESULT>{"agent":"separation-engineer","schema_version":"1.0",'
                '"best_sequence":["LDPE","EVOH"],"steps":[{"step":1,"polymer":"LDPE","solvent":"Toluene",'
                '"temperature_c":95.0}],"solvent_mapping":{"LDPE":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["LDPE","EVOH"]}],"polymers":["LDPE","EVOH","PET"]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="t1",
        )
    )
    return [
        HumanMessage(content=first_query),
        AIMessage(content="", tool_calls=[{
            "id": "t1", "name": "task",
            "args": {"subagent_type": "separation-engineer", "description": "run separation"},
        }]),
        result,
        HumanMessage(content=followup),
    ]


def _biosteam_completed_history(followup: str) -> list:
    return [
        HumanMessage(content="Run BioSTEAM TEA for PE with toluene and with DMSO under case C1."),
        AIMessage(content="", tool_calls=[{
            "id": "t1", "name": "task",
            "args": {"subagent_type": "biosteam-analyst", "description": "run TEA for both solvents"},
        }]),
        ToolMessage(
            content=(
                '<STRUCTURED_RESULT>{"agent":"biosteam-analyst","schema_version":"1.0",'
                '"target_plastic":"PE","energy_case":"C1",'
                '"results":[{"solvent":"Toluene","msp":1.42},{"solvent":"DMSO","msp":1.61}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="t1",
        ),
        HumanMessage(content=followup),
    ]


# Multi-turn follow-up scenarios: (id, message history, expectations).
# The digest is built deterministically from the history; the planner must
# route follow-ups against that session state.
SESSION_FOLLOWUP_SCENARIOS: list[dict] = [
    {
        "id": "followup_answer_from_results",
        "messages": _sep_completed_history(
            "Generate a separation state map for all LDPE/EVOH/PET sequences under 100 C.",
            "From that state map, which separation sequence maximizes predicted separation "
            "efficiency? Use the structured sequence results rather than re-describing the feedstock.",
        ),
        "expected_mode": "orchestrator",
    },
    {
        "id": "followup_new_stage_tea",
        "messages": _sep_completed_history(
            "Design a separation sequence for LDPE/EVOH/PET below 120 C.",
            "Now estimate MSP and GWP for that route.",
        ),
        "expected_mode": "specialists",
        "expected_specialists": ["biosteam-analyst"],
        "forbidden_specialists": ["separation-engineer"],
    },
    {
        "id": "followup_new_stage_safety",
        "messages": _sep_completed_history(
            "Design a separation sequence for LDPE/EVOH/PET below 120 C.",
            "Add safety scoring for the solvents used in that route.",
        ),
        "expected_mode": "specialists",
        "expected_specialists": ["safety-analyst"],
        "forbidden_specialists": ["separation-engineer"],
    },
    {
        "id": "followup_redo_with_new_params",
        "messages": _sep_completed_history(
            "Design a separation sequence for LDPE/EVOH/PET below 120 C.",
            "Redo the separation with a 140 C ceiling instead.",
        ),
        "expected_mode": "specialists",
        "expected_specialists": ["separation-engineer"],
    },
    {
        "id": "followup_compare_existing",
        "messages": _biosteam_completed_history(
            "Which of the two solvents had the lower MSP, and by how much?"
        ),
        "expected_mode": "orchestrator",
    },
    {
        "id": "followup_retry_after_failure",
        "messages": _sep_completed_history(
            "Design a separation sequence for LDPE/EVOH/PET below 120 C.",
            "The separation step failed — please try it again.",
            failed=True,
        ),
        "expected_mode": "specialists",
        "expected_specialists": ["separation-engineer"],
    },
    {
        "id": "followup_plot_prior_tea",
        "messages": _biosteam_completed_history(
            "Plot the cost breakdown comparison from that TEA."
        ),
        "expected_mode": "specialists",
        "expected_specialists_any": ["visualization-specialist", "biosteam-analyst"],
    },
]


def collect_entries() -> list[dict]:
    entries: list[dict] = []
    xl = pd.ExcelFile(QUERY_BANK_PATH)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if "query" not in df.columns:
            continue
        for _, row in df.iterrows():
            query = str(row.get("query") or "").strip()
            if not query:
                continue
            entries.append({
                "id": f"bank_{sheet.split()[0]}_{row.name}",
                "source": f"query_bank:{sheet}",
                "query": query,
                "expected_label": str(row.get("expected_route_or_subagents") or ""),
            })
    for key, query in CASE_STUDY_QUERIES.items():
        entries.append({"id": key, "source": "case_study", "query": query, "expected_label": ""})
    for scenario in SESSION_FOLLOWUP_SCENARIOS:
        messages = scenario["messages"]
        query = messages[-1].content
        entries.append({
            "id": scenario["id"],
            "source": "session_followup",
            "query": query,
            "expected_label": "",
            "session_digest": build_session_digest(messages),
            "expected_mode": scenario.get("expected_mode"),
            "expected_specialists": scenario.get("expected_specialists"),
            "expected_specialists_any": scenario.get("expected_specialists_any"),
            "forbidden_specialists": scenario.get("forbidden_specialists"),
        })
    return entries


def main() -> None:
    backend = LLMRoutePlannerBackend(init_chat_model(PLANNER_MODEL))
    entries = collect_entries()

    def record(entry: dict) -> dict:
        payload = backend(entry["query"], session_digest=entry.get("session_digest"))
        plan = validate_route_payload(entry["query"], payload) if payload else None
        return {**entry, "payload": payload, "validated": plan.explain() if plan else None}

    with ThreadPoolExecutor(max_workers=4) as pool:
        recorded = list(pool.map(record, entries))

    errors = [entry["id"] for entry in recorded if entry["payload"] is None]
    if errors:
        raise SystemExit(f"backend errors for {errors}; goldens NOT written")

    GOLDENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDENS_PATH.write_text(json.dumps(
        {
            "recorded_with": PLANNER_MODEL,
            "entries": recorded,
        },
        indent=2,
    ))
    print(f"recorded={len(recorded)} -> {GOLDENS_PATH}")
    for entry in recorded:
        if entry["source"] in {"case_study", "session_followup"}:
            validated = entry["validated"] or {}
            steps = [step["subagent"] for step in validated.get("steps", [])]
            print(f'{entry["id"]}: mode={validated.get("mode")} steps={steps}')


if __name__ == "__main__":
    main()
