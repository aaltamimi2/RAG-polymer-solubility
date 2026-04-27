#!/usr/bin/env python
"""Case Study #2 — MSP vs GWP Pareto Front for PE Dissolution.

Trace campaign script: screens all PE solvents (30 Tier-1 validated +
extended Tier-2 curated solvents from data/solvent-econ-lca-summary.csv)
across 3 energy cases via the DISSOLVE agent, captures LangSmith traces,
and identifies the MSP vs GWP Pareto front.

Solvent data uses the 3-tier fallback system in biosteam_runner.py:
  Tier 1: 16 ecoinvent-validated (Branch-TEA) + 16 class-average extended
  Tier 2: 100 curated solvents from agent swarm web research (price + LCA)
  Tier 3: Chemical-class average fallback (13 classes)

Queries run through the biosteam-analyst subagent with LangSmith trace
capture, following the established trace campaign pattern from
architecture/test_harness.py.

Usage:
    # List all queries
    python run_cs2_pareto_traces.py --list

    # Run a single query (0-based index)
    python run_cs2_pareto_traces.py -q 0

    # Run all queries
    python run_cs2_pareto_traces.py

    # Skip trace visualizations
    python run_cs2_pareto_traces.py --no-viz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
# trace20-cs2-pareto/ → query_traces/ → architecture/ → project root
_THIS_DIR = Path(__file__).resolve().parent
_ARCH_DIR = _THIS_DIR.parent.parent          # architecture/
_ROOT_DIR = _ARCH_DIR.parent                 # project root
sys.path.insert(0, str(_ROOT_DIR / "src"))
sys.path.insert(0, str(_ARCH_DIR))
# Also check the v8 repo src/ for strap module
_V8_SRC = Path("/home/aaltamimi2/langchain-STRAP-v8/src")
if _V8_SRC.is_dir():
    sys.path.insert(0, str(_V8_SRC))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(str(_ROOT_DIR / ".env"))

from langsmith import Client as LangSmithClient  # noqa: E402

from test_harness import (  # noqa: E402
    TestQuery,
    run_query,
    print_summary,
    generate_trace_visuals,
)

# ── Query definitions ─────────────────────────────────────────────────

Q1_BATCH = TestQuery(
    name="cs2-batch-all-pe-solvents",
    query=(
        "Run a BioSTEAM batch simulation for all PE solvents across all three "
        "energy cases (C1, C2, C3). Include both the original validated solvents "
        "and the newly added Tier-2 curated solvents (Dimethyl Carbonate, NMP, "
        "Cyclohexanone, Anisole, etc.). Rank by MSP and report GWP for each. "
        "Note which solvents use Tier-1 validated LCA vs Tier-2 curated vs "
        "Tier-3 class-average data."
    ),
    pattern="single",
    expected_subagents=["biosteam-analyst"],
    recursion_limit=250,
    description="Batch: all PE solvents × 3 energy cases (3-tier LCA)",
)

Q2_SCENARIO = TestQuery(
    name="cs2-top5-scenario-compare",
    query=(
        "Compare BioSTEAM scenarios for Toluene, Heptane, Hexane, Xylene, "
        "Cyclohexane, Dimethyl Carbonate, and NMP for PE recovery across "
        "C1, C2, C3 energy cases. Include both original and newly curated "
        "solvents. Which solvent gives the lowest MSP? Which gives the "
        "lowest GWP?"
    ),
    pattern="single",
    expected_subagents=["biosteam-analyst"],
    recursion_limit=150,
    description="Scenario comparison: top solvents (validated + curated)",
)

Q3_TORNADO_MSP = TestQuery(
    name="cs2-tornado-toluene-msp",
    query=(
        "Run a tornado sensitivity analysis on Toluene for PE recovery under "
        "C1 to identify which parameters drive MSP the most."
    ),
    pattern="single",
    expected_subagents=["biosteam-analyst"],
    recursion_limit=150,
    description="Tornado: Toluene MSP drivers under C1",
)

Q4_TORNADO_GWP = TestQuery(
    name="cs2-tornado-heptane-gwp",
    query=(
        "Run a tornado sensitivity analysis on Heptane for PE recovery under "
        "C1 to identify which parameters drive GWP the most."
    ),
    pattern="single",
    expected_subagents=["biosteam-analyst"],
    recursion_limit=150,
    description="Tornado: Heptane GWP drivers under C1",
)

Q5_SWEEP = TestQuery(
    name="cs2-sweep-dissolution-temp",
    query=(
        "How does MSP change with dissolution temperature for Toluene in PE "
        "recovery under C1? Sweep from 80 to 130 degrees C."
    ),
    pattern="single",
    expected_subagents=["biosteam-analyst"],
    recursion_limit=150,
    description="Parameter sweep: dissolution temp 80–130°C",
)

Q6_PARALLEL = TestQuery(
    name="cs2-batch-safety-parallel",
    query=(
        "Run BioSTEAM batch for the top 5 PE solvents under C1 (include "
        "Dimethyl Carbonate and NMP if they rank in the top), and "
        "simultaneously get GSK safety G-scores for each. Which solvents are "
        "both economically viable and safe?"
    ),
    pattern="parallel",
    expected_subagents=["biosteam-analyst", "safety-analyst"],
    recursion_limit=200,
    description="Parallel: BioSTEAM batch + GSK safety (incl. new solvents)",
)

QUERIES: list[TestQuery] = [Q1_BATCH, Q2_SCENARIO, Q3_TORNADO_MSP,
                            Q4_TORNADO_GWP, Q5_SWEEP, Q6_PARALLEL]

# ── Main ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Case Study #2 — MSP vs GWP Pareto Front trace campaign"
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all queries without running",
    )
    parser.add_argument(
        "--query", "-q", default=None,
        help="Run specific query by index (0-based) or name",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip trace visualization generation",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: this trace directory)",
    )
    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        print("\nCase Study #2 — Pareto Front Trace Queries:")
        print(f"{'#':<4s} {'Name':<34s} {'Pattern':<10s} {'Description'}")
        print(f"{'-'*4} {'-'*34} {'-'*10} {'-'*45}")
        for i, tq in enumerate(QUERIES):
            print(f"{i:<4d} {tq.name:<34s} {tq.pattern:<10s} {tq.description}")
        print(f"\nTotal: {len(QUERIES)} queries")
        return

    # ── Output directory ──
    output_dir = Path(args.output_dir) if args.output_dir else _THIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Select queries ──
    if args.query is not None:
        try:
            idx = int(args.query)
            queries = [QUERIES[idx]]
        except (ValueError, IndexError):
            queries = [q for q in QUERIES if q.name == args.query]
            if not queries:
                print(f"Unknown query: {args.query}")
                print("Use --list to see available queries.")
                sys.exit(1)
    else:
        queries = list(QUERIES)

    # ── Create agent ──
    print("Loading DISSOLVE agent...")
    from strap.agent import create_dissolve_agent
    agent = create_dissolve_agent()
    print("Agent ready.\n")

    ls_client = LangSmithClient()

    # ── Run queries ──
    results = []
    for tq in queries:
        result = run_query(agent, tq, ls_client)

        if not args.no_viz:
            result = generate_trace_visuals(result, ls_client, output_dir)

        results.append(result)

    # ── Save results JSON ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"results_{timestamp}.json"

    payload = {
        "case_study": "CS2 — MSP vs GWP Pareto Front for PE Dissolution",
        "timestamp": timestamp,
        "n_queries": len(results),
        "results": [asdict(r) for r in results],
    }

    # Also save the full agent answers for figure script parsing
    answers = {}
    for r in results:
        answers[r.name] = r.answer_preview

    payload["answers"] = answers

    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Print summary ──
    print_summary(results)

    # ── Campaign summary ──
    total_time = sum(r.wall_time_s for r in results)
    n_ok = sum(1 for r in results if r.routing_match)
    n_err = sum(1 for r in results if r.error)
    print(f"Campaign complete: {n_ok}/{len(results)} routing matches, "
          f"{n_err} errors, {total_time:.0f}s total wall time")
    print(f"Results JSON: {results_path}")
    print(f"Use: python cs2_pareto_figures.py {results_path}")


if __name__ == "__main__":
    main()
