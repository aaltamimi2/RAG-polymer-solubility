"""Model-free end-to-end routing + handoff trace for one query.

Shows every routing artifact for a single query, in order:

  Stage A (planning):   raw planner payload -> validated RoutePlan -> advisory hint
  Stage B (execution):  synthesized first dispatch -> producer structured result
                        -> derived handoff (contract, payload, consumer task_prompt)
                        -> auto-dispatched downstream task() -> router guards
  Stage C (budgets):    the token/tool budgets that bound each subagent

No model calls are made anywhere (model access is hard-blocked); the planner
payload is replayed from tests/fixtures/route_planner_goldens.json when the
query is recorded there, otherwise the deterministic keyword fallback is used.

Usage:
    python architecture/route_trace_demo.py                     # default chain demo
    python architecture/route_trace_demo.py --golden-id cs1_t2_solubility_plot
    python architecture/route_trace_demo.py --query "..."       # fallback-only trace
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from strap.handoff_store import initialize_handoff_scope, store_agent_result
from strap.handoffs import build_handoff_for_consumer
from strap.route_planner import RoutePlan, RoutePlanner, activate_route_plan, validate_route_payload
from strap.routing import RoutingMiddleware
from strap.routing_classifier import _build_hint_from_matches
from strap.routing_guards import _build_initial_route_task_response
from strap.routing_handoff_state import _get_pending_required_handoff, _get_ready_downstream_handoff
from strap.routing_message_state import _get_ordered_plan
from strap.subagent_config import load_subagent_specs
from strap.testing_utils import block_model_access

GOLDENS = _ROOT / "tests" / "fixtures" / "route_planner_goldens.json"
DEFAULT_GOLDEN = "bank_10_1"  # separation -> optimization -> visualization chain

SEP_RESULT_PAYLOAD = {
    "agent": "separation-engineer",
    "schema_version": "1.0",
    "polymers": ["LDPE", "PP", "EVOH"],
    "best_sequence": ["LDPE", "PP", "EVOH"],
    "steps": [
        {"step": 1, "polymer": "LDPE", "solvent": "Toluene", "temperature_c": 95.0, "selectivity_pct": 84.2},
        {"step": 2, "polymer": "PP", "solvent": "Xylene", "temperature_c": 98.0, "selectivity_pct": 71.5},
    ],
    "solvent_mapping": {"LDPE": "Toluene", "PP": "Xylene"},
    "top_k_sequences": [
        {"rank": 1, "sequence": ["LDPE", "PP", "EVOH"], "min_selectivity": 71.5,
         "solvent_mapping": {"LDPE": "Toluene", "PP": "Xylene"}},
    ],
}


def _hr(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def _task_call(tool_call_id: str, subagent: str, description: str = "") -> dict:
    return {
        "id": tool_call_id,
        "name": "task",
        "args": {"subagent_type": subagent, "description": description or f"Run {subagent}."},
    }


def _load_golden(golden_id: str) -> dict | None:
    if not GOLDENS.exists():
        return None
    for entry in json.loads(GOLDENS.read_text())["entries"]:
        if entry["id"] == golden_id:
            return entry
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-id", default=DEFAULT_GOLDEN)
    parser.add_argument("--query", default=None, help="trace an arbitrary query via keyword fallback")
    args = parser.parse_args()

    with block_model_access():
        golden = None if args.query else _load_golden(args.golden_id)
        if golden is not None:
            query = golden["query"]
        elif args.query:
            query = args.query
        else:
            raise SystemExit(f"golden id {args.golden_id!r} not found in {GOLDENS}")

        _hr("STAGE A1 — THE QUERY")
        print(query)

        if golden is not None:
            _hr("STAGE A2 — RAW PLANNER PAYLOAD (recorded live Gemini output, replayed offline)")
            print(json.dumps(golden["payload"], indent=2))
            plan = validate_route_payload(query, golden["payload"])
        else:
            _hr("STAGE A2 — NO GOLDEN RECORDED: deterministic keyword fallback path")
            plan = None

        planner = RoutePlanner(backend=(lambda q: golden["payload"]) if golden else None)
        plan = planner.plan(query)
        activate_route_plan(plan)

        _hr("STAGE A3 — VALIDATED ROUTE PLAN (structural checks, cycle/exclusion repair)")
        print(json.dumps(plan.explain(), indent=2))

        middleware = RoutingMiddleware(planner=planner)
        rules = plan.to_rules()

        _hr("STAGE A4 — ADVISORY HINT APPENDED TO THE ORCHESTRATOR SYSTEM PROMPT")
        hint = _build_hint_from_matches(rules, query_text=query) if rules else "(none — direct/orchestrator mode)"
        print(hint or "(no hint)")

        if not plan.is_specialists:
            _hr("PLAN IS NOT A SPECIALIST WORKFLOW — trace ends (direct/orchestrator mode)")
            return

        _hr("STAGE B1 — ORDERED EXECUTION PLAN (step ids, dependency edges)")
        messages: list = [HumanMessage(content=query)]
        for step in _get_ordered_plan(messages, allowed_rules=rules):
            print(f"  {step['subagent']:<28} depends_on={tuple(step.get('depends_on') or ())}")

        _hr("STAGE B2 — MODEL STOPS WITHOUT DISPATCHING? Router synthesizes the first task()")
        synthesized = _build_initial_route_task_response(messages, rules)
        first_call = synthesized.result[0].tool_calls[0]
        print(json.dumps({"name": first_call["name"], "args": first_call["args"]}, indent=2)[:600])

        producer = rules[0]["subagent"]
        consumer = next(
            (r["subagent"] for r in rules[1:] if producer in tuple(r.get("depends_on") or ())),
            rules[1]["subagent"] if len(rules) > 1 else None,
        )

        _hr(f"STAGE B3 — {producer} RETURNS; result stored as validated handoff record")
        scratch = Path(tempfile.mkdtemp(prefix="route_trace_"))
        initialize_handoff_scope(user_query=query, artifact_root=scratch)
        result_record = store_agent_result(
            producer=producer,
            payload=SEP_RESULT_PAYLOAD,
            source_tool_call_id="tc_step1",
        )
        print(f"  handoff_id={result_record.handoff_id} contract={result_record.contract} "
              f"status={result_record.status}")

        messages.extend([
            AIMessage(content="", tool_calls=[_task_call("tc_step1", producer)]),
            ToolMessage(
                content=f"<STRUCTURED_RESULT>{json.dumps(SEP_RESULT_PAYLOAD)}</STRUCTURED_RESULT>",
                tool_call_id="tc_step1",
            ),
        ])

        _hr("STAGE B4 — ROUTER DETECTS THE NEXT REQUIRED HANDOFF EDGE")
        pending = _get_pending_required_handoff(messages, rules)
        print(f"  pending producer->consumer edge: {pending}")

        if consumer:
            _hr(f"STAGE B5 — DERIVED HANDOFF BUILT FOR {consumer} (REAL adapter output)")
            derived = build_handoff_for_consumer(
                consumer=consumer,
                source_handoff_id=result_record.handoff_id,
                producer=producer,
            )
            print(f"  contract:     {derived.contract}")
            print(f"  status:       {derived.status}")
            payload_json = json.dumps(derived.payload)
            print(f"  payload:      {len(payload_json)} chars (~{len(payload_json) // 4} tokens), "
                  f"keys={sorted(derived.payload.keys())}")
            print(f"  task_prompt ({len(derived.task_prompt or '')} chars) — VERBATIM consumer prompt:")
            print("  " + "-" * 70)
            for line in (derived.task_prompt or "(empty)").splitlines():
                print(f"  | {line}")
            print("  " + "-" * 70)
            print("  NOTE: the payload travels in graph STATE (strap_handoff_payload), not")
            print("  in the consumer's prompt; guardrails inject it into tool args directly.")

            messages.extend([
                AIMessage(content="", tool_calls=[{
                    "id": "bh1", "name": "build_handoff",
                    "args": {"consumer": consumer, "producer": producer},
                }]),
                ToolMessage(
                    content=json.dumps({"ok": True, "handoff": {
                        "handoff_id": derived.handoff_id, "producer": producer,
                        "consumer": consumer, "status": "ok",
                        "task_prompt": derived.task_prompt,
                    }}),
                    tool_call_id="bh1",
                ),
            ])

            _hr("STAGE B6 — READY HANDOFF AUTO-DISPATCH (what task() call the router forces)")
            ready = _get_ready_downstream_handoff(messages, rules)
            if ready:
                print(json.dumps({
                    "task.subagent_type": ready.get("consumer"),
                    "task.handoff_id": ready.get("handoff_id"),
                    "task.description": (ready.get("task_prompt") or "")[:220] + "...",
                }, indent=2))
            else:
                print("  (no ready handoff — consumer may need more upstream edges)")

        _hr("STAGE B7 — GUARDS: off-plan and premature calls are refused")
        from unittest.mock import MagicMock

        class _ToolReq:
            def __init__(self, tool_call):
                self.tool_call = tool_call
                self.state = {"messages": messages}
                self.runtime = MagicMock()
                self.tool = None

        off_plan = middleware.wrap_tool_call(
            _ToolReq(_task_call("tc_bad", "statistics-ml")), handler=lambda r: "ALLOWED"
        )
        verdict = off_plan.content if isinstance(off_plan, ToolMessage) else off_plan
        print(f"  task(statistics-ml) mid-workflow -> {str(verdict)[:160]}")

        _hr("STAGE C — PER-SUBAGENT BUDGETS BOUNDING TOKEN USAGE (from YAML guardrails)")
        for spec in load_subagent_specs():
            name = spec.get("name")
            if name not in {r["subagent"] for r in rules}:
                continue
            guard = spec.get("guardrails") or {}
            print(f"  {name:<28} max_tool_calls={guard.get('max_tool_calls')} "
                  f"token_budget={guard.get('token_budget')} "
                  f"truncate_tool_results_after={guard.get('truncate_tool_results_after')}")

        print("\nTrace complete — zero model calls were made (block_model_access active).")


if __name__ == "__main__":
    main()
