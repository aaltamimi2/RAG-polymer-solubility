"""Live validation: multi-turn context buildup, persistent memory, durable
thread resume, and guarded mid-turn replanning.

PART 1 — continuity + memory (one thread, real Gemini):
  turn 1  screen solvents for LDPE/EVOH (specialist stage)
  turn 2  follow-up that is meaningless without turn 1 ("now run the
          cost-vs-emissions Pareto on those candidates...") — session digest
          must route ONLY the new stage
  turn 3  "remember that I always want MSP reported..." — agent should call
          save_memory (Claude-Code-style markdown fact file + index)
  turn 4  NEW agent instance (fresh graph, same sqlite checkpoint db) resumes
          the same thread — proves conversation state survives a "restart"

PART 2 — mid-turn replanning (fresh thread):
  the standing multistage stress query with DISSOLVE_SUBAGENT_RECURSION_LIMIT
  forced tiny so the first specialist exhausts its step budget → the router
  must revise the plan mid-turn (plan source becomes "planner_revision") and
  the run must end with a graceful answer, not a crash.

USAGE
    python architecture/stress_test_memory_replan.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")
os.environ.setdefault("DISSOLVE_TYPED_PLANNER", "off")

MODEL = os.getenv("STRAP_MODEL", "google_genai:gemini-3.5-flash")

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = _ROOT / "architecture" / "test_results" / f"memory_replan_{stamp}"
OUT.mkdir(parents=True, exist_ok=True)

from langchain_core.messages import HumanMessage

from strap.agent import create_dissolve_agent
from strap.memory_store import list_memories, memory_root
from strap.route_planner import get_active_route_plan

report: dict = {"model": MODEL, "memory_root": str(memory_root())}


def run_turn(agent, thread_id: str, text: str, label: str) -> dict:
    started = time.time()
    result = agent.invoke(
        {"messages": [HumanMessage(content=text)]},
        {"configurable": {"thread_id": thread_id}, "recursion_limit": 150},
    )
    messages = result.get("messages", [])
    final = ""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "ai" and str(getattr(msg, "content", "") or "").strip():
            final = str(msg.content)
            break
    plan = get_active_route_plan(text)
    record = {
        "label": label,
        "query": text,
        "elapsed_s": round(time.time() - started, 1),
        "n_messages": len(messages),
        "plan_mode": plan.mode if plan else None,
        "plan_steps": plan.subagent_names() if plan else None,
        "plan_source": plan.source if plan else None,
        "final_head": final[:600],
    }
    print(json.dumps(record, indent=1)[:1200], flush=True)
    return record


def part1() -> None:
    thread_id = f"memtest-{uuid.uuid4().hex[:8]}"
    report["part1_thread"] = thread_id
    agent = create_dissolve_agent(MODEL, enable_persistence=True)
    print(f"[part1] thread={thread_id} checkpointer={type(agent.checkpointer).__name__}", flush=True)
    report["checkpointer"] = type(agent.checkpointer).__name__

    report["turn1"] = run_turn(
        agent, thread_id,
        "Screen solvent candidates to separate LDPE and EVOH below 130 C — top 3 per polymer with temperatures.",
        "turn1_screen",
    )
    report["turn2"] = run_turn(
        agent, thread_id,
        "Now run the cost-versus-emissions Pareto on those candidates with at least 1 STRAP wash and report the knee point.",
        "turn2_followup",
    )
    report["turn3"] = run_turn(
        agent, thread_id,
        "Please remember going forward: I always want MSP reported alongside any economics you give me.",
        "turn3_remember",
    )
    memories = [(m.name, m.memory_type, m.description) for m in list_memories()]
    report["memories_after_turn3"] = memories
    print("[part1] memories:", memories, flush=True)

    # "restart": a brand-new agent instance; only the sqlite checkpoint db is shared
    agent2 = create_dissolve_agent(MODEL, enable_persistence=True)
    report["turn4_resume"] = run_turn(
        agent2, thread_id,
        "Quick recap: what did we establish in this conversation so far?",
        "turn4_resume_after_restart",
    )


def part2() -> None:
    os.environ["DISSOLVE_SUBAGENT_RECURSION_LIMIT"] = "25"
    try:
        thread_id = f"replantest-{uuid.uuid4().hex[:8]}"
        report["part2_thread"] = thread_id
        agent = create_dissolve_agent(MODEL, enable_persistence=True)
        query = (
            "For a mixed plastic feedstock of 8000 tonnes/year composed of 60% PE and 40% EVOH: "
            "first have the separation engineer propose the top 3 solvent candidates per polymer "
            "below 140 C, then run the cost-versus-emissions Pareto on exactly those candidates "
            "and report the knee point."
        )
        record = run_turn(agent, thread_id, query, "part2_forced_step_budget")
        report["part2"] = record
        report["part2_replanned"] = record.get("plan_source") == "planner_revision"
    finally:
        os.environ.pop("DISSOLVE_SUBAGENT_RECURSION_LIMIT", None)


def main() -> None:
    part1()
    part2()
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nreport: {OUT / 'report.json'}", flush=True)
    checks = {
        "turn2_planned_new_stage_only": bool(report.get("turn2", {}).get("plan_steps") is not None
                                             and "separation-engineer" not in (report["turn2"]["plan_steps"] or [])),
        "memory_saved": bool(report.get("memories_after_turn3")),
        "resume_after_restart_answered": bool(report.get("turn4_resume", {}).get("final_head")),
        "replan_applied": bool(report.get("part2_replanned")),
    }
    report["checks"] = checks
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
