"""Phase 6: orchestration state-machine matrix (model-free).

Walks each workflow shape end-to-end through RoutingMiddleware with
planner-sourced plans (stub payloads), asserting every transition:

  single      dispatch -> result -> completion short-circuit
  chain       synthesized dispatch -> result -> pending handoff -> ready
              handoff -> forced downstream dispatch -> completion directive
  join        downstream blocked until ALL upstream handoffs exist
  retry       failed dispatch -> retry allowed -> success supersedes failure
  guards      off-plan task blocked; write_todos blocked pre-dispatch;
              direct plans block task() entirely
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from strap.route_planner import RoutePlanner, clear_active_route_plans
from strap.routing import RoutingMiddleware
from strap.routing_handoff_state import _get_pending_required_handoff, _get_ready_downstream_handoff
from strap.routing_message_state import _get_ordered_plan
from strap.routing_progress import _get_effective_completed_task_ids, _get_effective_failed_task_ids
from strap.routing_guards import _build_initial_route_task_response
from strap.testing_utils import block_model_access

OUT = _ROOT / "architecture" / "test_results" / "subagent_validation_offline_20260701"

CHECKS: list[tuple[str, bool, str]] = []


def check(shape: str, ok: bool, detail: str) -> None:
    CHECKS.append((shape, ok, detail))


def payload_planner(steps: list[dict], mode: str = "specialists") -> RoutePlanner:
    payload = {"mode": mode, "steps": steps, "excluded_subagents": [],
               "confidence": "high", "rationale": "phase6"}
    return RoutePlanner(backend=lambda q: payload)


def _task(tool_call_id: str, subagent: str, description: str = "run") -> dict:
    return {"id": tool_call_id, "name": "task",
            "args": {"subagent_type": subagent, "description": description}}


def _result(agent: str) -> str:
    return f'<STRUCTURED_RESULT>{{"agent":"{agent}","schema_version":"1.0","no_data":true}}</STRUCTURED_RESULT>'


def _handoff_msgs(call_id: str, producer: str, consumer: str) -> list:
    return [
        AIMessage(content="", tool_calls=[{
            "id": call_id, "name": "build_handoff",
            "args": {"consumer": consumer, "producer": producer},
        }]),
        ToolMessage(content=json.dumps({"ok": True, "handoff": {
            "handoff_id": f"h_{call_id}", "producer": producer, "consumer": consumer,
            "status": "ok", "task_prompt": f"Continue with {consumer}.",
        }}), tool_call_id=call_id),
    ]


def tool_req(middleware, messages, tool_call):
    request = MagicMock()
    request.tool_call = tool_call
    request.state = {"messages": messages}
    return middleware.wrap_tool_call(request, handler=lambda r: "ALLOWED")


def model_emits(middleware, messages, tool_call):
    """Drive after_model as if the orchestrator just emitted this tool call."""
    attempt = messages + [AIMessage(content="", tool_calls=[tool_call])]
    update = middleware.after_model({"messages": attempt}, MagicMock())
    if update is None:
        return "ALLOWED"
    blocked = [m for m in update["messages"] if m.tool_call_id == tool_call["id"]]
    return blocked[0] if blocked else "ALLOWED"


def run() -> None:
    with block_model_access():
        # ---------------- SHAPE: single specialist ----------------
        clear_active_route_plans()
        planner = payload_planner([{"subagent": "separation-engineer", "objective": "route", "depends_on": []}])
        middleware = RoutingMiddleware(planner=planner)
        query = "Design a separation for LDPE/PP below 100 C (phase6 single)"
        messages = [HumanMessage(content=query)]
        rules = middleware._get_allowed_rules(messages)
        check("single", [r["subagent"] for r in rules] == ["separation-engineer"], "plan projected")

        # Single-specialist routes are ADVISORY by design: the orchestrator may
        # answer directly with core tools, so no dispatch is force-synthesized.
        start = _build_initial_route_task_response(messages, rules)
        check("single", start is None,
              "single-specialist route stays advisory (no forced dispatch)")

        messages += [AIMessage(content="", tool_calls=[_task("t1", "separation-engineer")]),
                     ToolMessage(content=_result("separation-engineer"), tool_call_id="t1")]
        completed = _get_effective_completed_task_ids(messages)
        check("single", "t1" in completed, "result counted as completed")
        repeat = model_emits(middleware, messages, _task("t2", "separation-engineer"))
        check("single", isinstance(repeat, ToolMessage) and repeat.status == "error",
              "repeat dispatch after completion blocked")

        # ---------------- SHAPE: sequential chain ----------------
        clear_active_route_plans()
        planner = payload_planner([
            {"subagent": "separation-engineer", "objective": "route", "depends_on": []},
            {"subagent": "biosteam-analyst", "objective": "cost it", "depends_on": ["separation-engineer"]},
        ])
        middleware = RoutingMiddleware(planner=planner)
        query = "Design a route for LDPE/PP then run TEA (phase6 chain)"
        messages = [HumanMessage(content=query)]
        rules = middleware._get_allowed_rules(messages)
        ordered = [s["subagent"] for s in _get_ordered_plan(messages, allowed_rules=rules)]
        check("chain", ordered == ["separation-engineer", "biosteam-analyst"], f"ordered plan {ordered}")

        start = _build_initial_route_task_response(messages, rules)
        started = start is not None and start.result[0].tool_calls[0]["args"]["subagent_type"] == "separation-engineer"
        check("chain", started, "multi-specialist route force-synthesizes the first dispatch")

        premature = model_emits(middleware, messages, _task("tb0", "biosteam-analyst"))
        check("chain", isinstance(premature, ToolMessage) and premature.status == "error",
              "downstream dispatch before producer blocked")

        messages += [AIMessage(content="", tool_calls=[_task("ts1", "separation-engineer")]),
                     ToolMessage(content=_result("separation-engineer"), tool_call_id="ts1")]
        pending = _get_pending_required_handoff(messages, rules)
        check("chain", pending == ("separation-engineer", "biosteam-analyst"), f"pending edge {pending}")

        messages += _handoff_msgs("bh1", "separation-engineer", "biosteam-analyst")
        ready = _get_ready_downstream_handoff(messages, rules)
        check("chain", bool(ready) and ready.get("consumer") == "biosteam-analyst",
              "ready handoff detected for downstream dispatch")

        off_plan = tool_req(middleware, messages, _task("tx", "statistics-ml"))
        check("chain", isinstance(off_plan, ToolMessage) and off_plan.status == "error",
              "off-plan specialist blocked mid-workflow")

        messages += [AIMessage(content="", tool_calls=[_task("tb1", "biosteam-analyst")]),
                     ToolMessage(content=_result("biosteam-analyst"), tool_call_id="tb1")]
        completed = _get_effective_completed_task_ids(messages)
        check("chain", {"ts1", "tb1"} <= completed, "both steps completed")

        # ---------------- SHAPE: join (viz needs sep + opt) ----------------
        clear_active_route_plans()
        planner = payload_planner([
            {"subagent": "separation-engineer", "objective": "", "depends_on": []},
            {"subagent": "optimization-engineer", "objective": "", "depends_on": ["separation-engineer"]},
            {"subagent": "visualization-specialist", "objective": "",
             "depends_on": ["separation-engineer", "optimization-engineer"]},
        ])
        middleware = RoutingMiddleware(planner=planner)
        query = "Route, optimize, and visualize LDPE/PP (phase6 join)"
        messages = [HumanMessage(content=query)]
        rules = middleware._get_allowed_rules(messages)

        messages += [AIMessage(content="", tool_calls=[_task("j1", "separation-engineer")]),
                     ToolMessage(content=_result("separation-engineer"), tool_call_id="j1")]
        messages += _handoff_msgs("jbh1", "separation-engineer", "optimization-engineer")
        messages += [AIMessage(content="", tool_calls=[_task("j2", "optimization-engineer")]),
                     ToolMessage(content=_result("optimization-engineer"), tool_call_id="j2")]
        # only ONE of the two required upstream handoffs for viz exists:
        messages += _handoff_msgs("jbh2", "separation-engineer", "visualization-specialist")
        early_viz = model_emits(middleware, messages, _task("j3", "visualization-specialist"))
        check("join", isinstance(early_viz, ToolMessage) and early_viz.status == "error",
              "join consumer blocked until all upstream handoffs exist")

        messages += _handoff_msgs("jbh3", "optimization-engineer", "visualization-specialist")
        # A dispatch ignoring the handoff prompt is CORRECTED (handoff-first
        # discipline); dispatching with the handoff-provided prompt is allowed.
        bare_viz = model_emits(middleware, messages, _task("j4", "visualization-specialist"))
        check("join", isinstance(bare_viz, ToolMessage) and "handoff-provided task prompt" in bare_viz.content,
              "bare join dispatch corrected toward handoff-provided prompt")
        # Follow the guard's corrections to a fully validated dispatch:
        # 1) the composed multi-handoff description, 2) the validated handoff_id.
        required_description = ""
        if isinstance(bare_viz, ToolMessage):
            match = re.search(r'description="(.*)"', bare_viz.content, re.DOTALL)
            if match:
                required_description = match.group(1)
        second = model_emits(middleware, messages,
                             _task("j5", "visualization-specialist", required_description))
        required_handoff = ""
        if isinstance(second, ToolMessage):
            match = re.search(r'handoff_id="([^"]+)"', second.content)
            if match:
                required_handoff = match.group(1)
        final_call = {
            "id": "j6", "name": "task",
            "args": {"subagent_type": "visualization-specialist",
                     "description": required_description,
                     "handoff_id": required_handoff},
        }
        final = model_emits(middleware, messages, final_call)
        check("join", final == "ALLOWED",
              "join dispatch with guard-required prompt + handoff_id allowed")

        # ---------------- SHAPE: failure / retry ----------------
        clear_active_route_plans()
        planner = payload_planner([{"subagent": "separation-engineer", "objective": "", "depends_on": []}])
        middleware = RoutingMiddleware(planner=planner)
        query = "Separate LDPE from PP (phase6 retry)"
        messages = [HumanMessage(content=query),
                    AIMessage(content="", tool_calls=[_task("f1", "separation-engineer")]),
                    ToolMessage(content="Tool error: solver crashed", tool_call_id="f1", status="error")]
        failed = _get_effective_failed_task_ids(messages)
        check("retry", "f1" in failed, "failed dispatch tracked")
        retry = tool_req(middleware, messages, _task("f2", "separation-engineer"))
        check("retry", retry == "ALLOWED", "retry of failed dispatch allowed")
        messages += [AIMessage(content="", tool_calls=[_task("f2", "separation-engineer")]),
                     ToolMessage(content=_result("separation-engineer"), tool_call_id="f2")]
        completed = _get_effective_completed_task_ids(messages)
        failed = _get_effective_failed_task_ids(messages)
        check("retry", "f2" in completed and "f1" not in completed, "success supersedes failure")

        # ---------------- SHAPE: guards on direct plans ----------------
        clear_active_route_plans()
        planner = payload_planner([], mode="direct")
        middleware = RoutingMiddleware(planner=planner)
        query = "What solvents dissolve LDPE? (phase6 direct)"
        messages = [HumanMessage(content=query)]
        blocked = tool_req(middleware, messages, _task("d1", "separation-engineer"))
        check("direct", isinstance(blocked, ToolMessage) and "direct core-tool lookup" in blocked.content,
              "task() blocked under direct plan")

        # write_todos before first dispatch on a multi-specialist route
        clear_active_route_plans()
        planner = payload_planner([
            {"subagent": "separation-engineer", "objective": "", "depends_on": []},
            {"subagent": "biosteam-analyst", "objective": "", "depends_on": ["separation-engineer"]},
        ])
        middleware = RoutingMiddleware(planner=planner)
        messages = [HumanMessage(content="Route then TEA for PE/EVOH (phase6 todos)")]
        todos = tool_req(middleware, messages, {"id": "w1", "name": "write_todos", "args": {"todos": []}})
        check("guards", isinstance(todos, ToolMessage) and todos.status == "error",
              "write_todos blocked before first dispatch")

    passed = sum(1 for _, ok, _ in CHECKS if ok)
    doc = {
        "summary": {"checks": len(CHECKS), "passed": passed, "failed": len(CHECKS) - passed},
        "checks": [{"shape": s, "ok": ok, "detail": d} for s, ok, d in CHECKS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase6_state_machine.json").write_text(json.dumps(doc, indent=2))
    print(json.dumps(doc["summary"], indent=2))
    for shape, ok, detail in CHECKS:
        print(f"[{'ok' if ok else 'FAIL'}] {shape:<8} {detail}")


if __name__ == "__main__":
    run()
