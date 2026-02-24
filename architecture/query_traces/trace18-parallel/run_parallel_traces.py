#!/usr/bin/env python
"""Run parallel execution trace campaign -- 10 queries.

Each query triggers a configured parallel pair from subagents.yaml,
exercising separation||safety, biosteam||safety, and scholar||patent
dispatch patterns.

Usage:
    python run_parallel_traces.py           # run all 10
    python run_parallel_traces.py --list    # list queries
    python run_parallel_traces.py -q 0      # run query #0
    python run_parallel_traces.py --no-viz  # skip trace PNGs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "architecture"))

from dotenv import load_dotenv
load_dotenv(str(_ROOT / ".env"))

from langsmith import Client as LangSmithClient
from strap.agent import create_dissolve_agent
from test_harness import TestQuery, run_query, print_summary, generate_trace_visuals

# ── 10 Parallel Queries ──────────────────────────────────────────────

QUERIES: list[TestQuery] = [
    TestQuery(
        name="par-sep-safety-1",
        query=(
            "What solvents selectively dissolve LDPE over PP at 110\u00b0C? "
            "Include GSK safety G-scores and PubChem hazard data for each "
            "recommended solvent."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: separation-engineer || safety-analyst",
    ),
    TestQuery(
        name="par-sep-safety-2",
        query=(
            "Plan a separation scheme for PS/PVC mixed waste using selective "
            "dissolution. Assess the safety profiles and LD50 toxicity of "
            "each recommended solvent."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: separation-engineer || safety-analyst",
    ),
    TestQuery(
        name="par-sep-safety-3",
        query=(
            "Find solvents for separating EVOH from PET in a barrier film. "
            "Check which solvents have the best environmental and safety "
            "ratings."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: separation-engineer || safety-analyst",
    ),
    TestQuery(
        name="par-sep-safety-4",
        query=(
            "Separate HDPE from LDPE using selective dissolution at "
            "atmospheric pressure. Rank the top solvents by both "
            "selectivity and GSK G-score."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: separation-engineer || safety-analyst",
    ),
    TestQuery(
        name="par-bio-safety-1",
        query=(
            "Run a BioSTEAM simulation for Toluene/PE under C1 and assess "
            "Toluene's safety profile including GSK G-score and PubChem "
            "hazards."
        ),
        pattern="parallel",
        expected_subagents=["biosteam-analyst", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: biosteam-analyst || safety-analyst",
    ),
    TestQuery(
        name="par-bio-safety-2",
        query=(
            "Compare the techno-economics for Xylene vs Heptane for PE "
            "recovery under C1, and compare their safety G-scores and "
            "PubChem hazard classifications."
        ),
        pattern="parallel",
        expected_subagents=["biosteam-analyst", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: biosteam-analyst || safety-analyst",
    ),
    TestQuery(
        name="par-bio-safety-3",
        query=(
            "Run a BioSTEAM batch comparison of Toluene, Cyclohexane, and "
            "Tetrahydrofuran for PS recovery. Also get safety data for all "
            "three solvents."
        ),
        pattern="parallel",
        expected_subagents=["biosteam-analyst", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: biosteam-analyst || safety-analyst",
    ),
    TestQuery(
        name="par-scholar-patent-1",
        query=(
            "Search Google Scholar for academic papers on polyethylene "
            "dissolution recycling processes. Also search for related "
            "patents on solvent-based PE recycling."
        ),
        pattern="parallel",
        expected_subagents=["scholar-researcher", "patent-researcher"],
        recursion_limit=250,
        description="Parallel: scholar-researcher || patent-researcher",
    ),
    TestQuery(
        name="par-scholar-patent-2",
        query=(
            "Find recent academic publications on selective dissolution for "
            "mixed plastic waste separation. Also search patents for "
            "multilayer film delamination using solvents."
        ),
        pattern="parallel",
        expected_subagents=["scholar-researcher", "patent-researcher"],
        recursion_limit=250,
        description="Parallel: scholar-researcher || patent-researcher",
    ),
    TestQuery(
        name="par-scholar-patent-3",
        query=(
            "Search for academic literature on EVOH barrier layer recovery "
            "methods. Also search for patents on ethylene vinyl alcohol "
            "recycling processes."
        ),
        pattern="parallel",
        expected_subagents=["scholar-researcher", "patent-researcher"],
        recursion_limit=250,
        description="Parallel: scholar-researcher || patent-researcher",
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="DISSOLVE parallel trace campaign (10 queries)"
    )
    parser.add_argument("--list", "-l", action="store_true",
                        help="List queries without running")
    parser.add_argument("--query", "-q", default=None,
                        help="Run specific query by index or name")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip trace visualization generation")
    args = parser.parse_args()

    if args.list:
        print("\nParallel Trace Queries:")
        print(f"{'#':<4s} {'Name':<28s} {'Description'}")
        print(f"{'-'*4} {'-'*28} {'-'*50}")
        for i, tq in enumerate(QUERIES):
            print(f"{i:<4d} {tq.name:<28s} {tq.description}")
        return

    # Select queries
    if args.query is not None:
        try:
            idx = int(args.query)
            queries = [QUERIES[idx]]
        except ValueError:
            queries = [q for q in QUERIES if q.name == args.query]
            if not queries:
                print(f"Unknown query: {args.query}")
                return
    else:
        queries = QUERIES

    ls = LangSmithClient()
    output_dir = _DIR
    results = []

    for tq in queries:
        agent = create_dissolve_agent()
        result = run_query(agent, tq, ls)
        if not args.no_viz:
            result = generate_trace_visuals(result, ls, output_dir)
        results.append(result)

    # Save results JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = _DIR / f"results_{ts}.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "campaign": "parallel",
            "n_queries": len(results),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
