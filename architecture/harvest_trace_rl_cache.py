"""Harvest last night's live multistage traces into the offline RL cache.

This is the ONE network step for the trace-RL work: it reads our *own* logged
LangSmith traces (no model inference) and writes a self-contained JSON cache
that case study 05 replays fully offline. Run once; commit the cache.

USAGE
    python architecture/harvest_trace_rl_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from strap.eval.trace_ingest import TraceRunSpec, harvest_trace_cache

# The multistage stress runs, in the order the harness fixes landed. Each run is
# the SAME query; config_note records which fixes were live, making runs 2->6 a
# natural experiment for offline policy evaluation.
SPECS = [
    TraceRunSpec(
        run_id="019f3aa0-be3d-7dd3-b6ef-bd08ff7e7fa2",
        label="run2_budgets_dead",
        config_note="pre-fix: subagent budgets silently dead; 5 dispatches, 2 wasted retries; infeasible",
        order=2,
    ),
    TraceRunSpec(
        run_id="019f3aac-ce0a-79c2-9bf1-8976ab970427",
        label="run3_clean_staged",
        config_note="seed_guard_state + handoff-visibility + budget-trip synthesis; clean 2-dispatch flow; infeasible",
        order=3,
    ),
    TraceRunSpec(
        run_id="019f3aae-be4b-7b22-ac8e-925d4f4bee86",
        label="run4_workbook_shortlist",
        config_note="workbook-constrained shortlist; still infeasible -> exposed candidate-admission defect",
        order=4,
    ),
    TraceRunSpec(
        run_id="019f3ab9-04cd-7731-9e84-e7ff27de53af",
        label="run5_baseline_fallback",
        config_note="baseline-fallback fix; first feasible Pareto front produced",
        order=5,
    ),
    TraceRunSpec(
        run_id="019f3abd-52a2-77c1-82e8-b20c48d53fb2",
        label="run6_knee_reported",
        config_note="enriched fallback (washes+knee/cheapest); complete deliverable",
        order=6,
    ),
]

OUT = _ROOT / "case-studies" / "05-offline-rl-from-traces" / "data" / "trace_rl_cache.json"


def main() -> None:
    cache = harvest_trace_cache(SPECS, OUT)
    print(f"harvested {len(cache['runs'])} runs -> {OUT}")
    for run in cache["runs"]:
        agents = sorted(run["structured_results"].keys())
        ledgers = {a: l.get("prompt_tokens", 0) + l.get("output_tokens", 0)
                   for a, l in run["ledger_by_agent"].items()}
        print(f"  {run['label']}: agents={agents} tokens={ledgers}")


if __name__ == "__main__":
    main()
