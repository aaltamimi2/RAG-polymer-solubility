"""Live multistage stress test of the DISSOLVE harness.

Runs the REAL agent (Gemini Flash orchestrator + Flash route planner) on a
hard multistage query that must be decomposed into staged specialist work:

    stage 1  separation-engineer: operating parameters (solvent shortlist +
             temperatures) for the feed
    stage 2  optimization-engineer: cost-vs-emissions Pareto over exactly those
             candidates (live SCIP, workbook TEA)
    stage 3  identify the Pareto-dominant (knee) point and report it

Captures and documents the full trace: the route plan, every task() dispatch,
handoff builds/attachments, router-guard interventions, structured results,
timings, and the LangSmith run URL.

USAGE
    python architecture/stress_test_multistage.py [--query-file F] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")
os.environ.setdefault("DISSOLVE_TYPED_PLANNER", "off")  # exercise the subagent path

DEFAULT_QUERY = (
    "For a mixed plastic feedstock of 8000 tonnes/year composed of 60% PE and 40% EVOH: "
    "first have the separation engineer propose operating parameters — the top 3 solvent "
    "candidates per polymer with recommended dissolution temperatures below 140 C. "
    "Then pass exactly those shortlisted candidates to the optimization engineer to run "
    "the cost-versus-emissions Pareto analysis with at least 1 STRAP wash step. "
    "Finally identify the Pareto-dominant knee point and report its selected solvents, "
    "total cost, and emissions, alongside the cheapest point for comparison."
)

MODEL = os.getenv("STRAP_MODEL", "google_genai:gemini-3.5-flash")


def _msg_record(msg) -> dict:
    kwargs = getattr(msg, "additional_kwargs", {}) or {}
    record = {
        "type": getattr(msg, "type", msg.__class__.__name__),
        "content_chars": len(str(getattr(msg, "content", "") or "")),
        "content": str(getattr(msg, "content", "") or "")[:4000],
    }
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        record["tool_calls"] = [
            {"name": tc.get("name"), "id": tc.get("id"),
             "args_keys": sorted((tc.get("args") or {}).keys()),
             "subagent_type": (tc.get("args") or {}).get("subagent_type"),
             "handoff_id": (tc.get("args") or {}).get("handoff_id"),
             "description_head": str((tc.get("args") or {}).get("description") or "")[:200]}
            for tc in tool_calls
        ]
    if getattr(msg, "tool_call_id", None):
        record["tool_call_id"] = msg.tool_call_id
    if getattr(msg, "status", None):
        record["status"] = msg.status
    interesting = {k: v for k, v in kwargs.items() if k.startswith("strap_")}
    if interesting:
        record["strap_kwargs"] = {
            k: (v if isinstance(v, (str, int, float, bool)) else type(v).__name__)
            for k, v in interesting.items()
        }
    return record


def _analyze(messages, query: str) -> dict:
    dispatches, handoffs, guards, structured, errors = [], [], [], [], []
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            if tc.get("name") == "task":
                args = tc.get("args") or {}
                dispatches.append({
                    "subagent": args.get("subagent_type"),
                    "with_handoff": bool(args.get("handoff_id")),
                    "handoff_id": args.get("handoff_id"),
                })
            elif tc.get("name") == "build_handoff":
                args = tc.get("args") or {}
                handoffs.append({"producer": args.get("producer"), "consumer": args.get("consumer")})
        content = str(getattr(msg, "content", "") or "")
        if getattr(msg, "type", "") == "tool":
            if content.startswith("Router guard"):
                guards.append(content[:220])
            if getattr(msg, "status", None) == "error" and not content.startswith("Router guard"):
                errors.append(content[:220])
            match = re.search(r"<STRUCTURED_RESULT>\s*(\{.*?\})\s*</STRUCTURED_RESULT>", content, re.DOTALL)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    structured.append({"agent": payload.get("agent"),
                                       "keys": sorted(payload.keys())[:14]})
                except json.JSONDecodeError:
                    structured.append({"agent": "?", "keys": ["<unparseable>"]})

    final_answer = ""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and str(getattr(msg, "content", "") or "").strip():
            final_answer = str(msg.content)
            break

    from strap.route_planner import get_active_route_plan

    plan = get_active_route_plan(query)
    return {
        "route_plan": plan.explain() if plan else None,
        "task_dispatches": dispatches,
        "handoff_builds": handoffs,
        "router_guard_interventions": guards,
        "tool_errors": errors,
        "structured_results": structured,
        "final_answer_chars": len(final_answer),
        "final_answer": final_answer,
        "final_mentions": {
            "knee_or_dominant": bool(re.search(r"knee|dominant", final_answer, re.I)),
            "cost": bool(re.search(r"cost", final_answer, re.I)),
            "emissions": bool(re.search(r"emission", final_answer, re.I)),
            "solvent": bool(re.search(r"solvent", final_answer, re.I)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) if args.out else _ROOT / "architecture" / "test_results" / f"stress_multistage_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from langchain_core.messages import HumanMessage

    from strap.agent import create_dissolve_agent
    from strap.langsmith_tracing import get_langsmith_client, langsmith_trace, resolve_run_links

    print(f"model={MODEL}  typed_planner={os.getenv('DISSOLVE_TYPED_PLANNER')}")
    agent = create_dissolve_agent(MODEL, enable_persistence=False)

    def _run_agent(query: str) -> tuple[list, str | None]:
        """Stream the run so a crash still leaves the partial message history."""
        last_state: dict = {}
        error = None
        try:
            for state in agent.stream(
                {"messages": [HumanMessage(content=query)]},
                {"recursion_limit": 150},
                stream_mode="values",
            ):
                last_state = state
        except Exception as exc:  # noqa: BLE001 — document the crash, keep the trace
            error = f"{type(exc).__name__}: {exc}"
        return list(last_state.get("messages", []) or []), error

    trace_links = {}
    started = time.time()
    if langsmith_trace is not None and get_langsmith_client() is not None:
        with langsmith_trace(
            "DISSOLVE multistage stress test",
            run_type="chain",
            project_name=os.getenv("LANGSMITH_PROJECT", "strap-agent"),
            inputs={"message": args.query, "model_name": MODEL},
            metadata={"entrypoint": "stress_test_multistage", "model_name": MODEL},
            tags=["dissolve", "stress-test", "multistage"],
        ) as traced_run:
            messages, run_error = _run_agent(args.query)
            traced_run.end(outputs={"message_count": len(messages), "error": run_error})
        trace_links = resolve_run_links(traced_run) or {}
    else:
        print("WARNING: LangSmith tracing NOT active")
        messages, run_error = _run_agent(args.query)
    elapsed = time.time() - started
    (out_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(_msg_record(m)) for m in messages))
    analysis = _analyze(messages, args.query)
    analysis.update({
        "query": args.query,
        "model": MODEL,
        "elapsed_seconds": round(elapsed, 1),
        "n_messages": len(messages),
        "run_error": run_error,
        "langsmith": trace_links,
    })
    (out_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))

    print(json.dumps({k: v for k, v in analysis.items() if k != "final_answer"}, indent=2)[:4000])
    print(f"\nfinal answer head:\n{analysis['final_answer'][:1200]}")
    print(f"\ntrace dir: {out_dir}")


if __name__ == "__main__":
    main()
