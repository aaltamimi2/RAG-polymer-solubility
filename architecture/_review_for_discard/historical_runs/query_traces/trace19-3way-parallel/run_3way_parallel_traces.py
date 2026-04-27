#!/usr/bin/env python
"""Run 3-way parallel execution trace campaign -- 10 queries.

Each query triggers a configured parallel_3way group from subagents.yaml,
exercising separation||safety||biosteam, separation||safety||visualization,
biosteam||safety||visualization, and scholar||patent||rag dispatch patterns.

Usage:
    python run_3way_parallel_traces.py           # run all 10
    python run_3way_parallel_traces.py --list    # list queries
    python run_3way_parallel_traces.py -q 0      # run query #0
    python run_3way_parallel_traces.py --no-viz  # skip trace PNGs
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

# ── 10 Three-Way Parallel Queries ────────────────────────────────────

QUERIES: list[TestQuery] = [
    # Group A: separation + safety + biosteam (3 queries)
    TestQuery(
        name="3way-sep-safe-bio-1",
        query=(
            "Find the optimal separation sequence for LDPE/PP/HDPE mixed waste, "
            "assess the GSK safety G-scores and PubChem hazards for each "
            "recommended solvent, and run BioSTEAM TEA simulations under energy "
            "case C1 to estimate MSP and GWP for the top solvent."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst", "biosteam-analyst"],
        recursion_limit=250,
        description="3-way parallel: separation || safety || biosteam",
    ),
    TestQuery(
        name="3way-sep-safe-bio-2",
        query=(
            "Plan a selective dissolution scheme to separate PS from PVC at "
            "atmospheric pressure. Get LD50 toxicity and GSK G-score data for "
            "each candidate solvent. Also run a BioSTEAM batch comparison of "
            "the viable solvents for PS recovery under C1."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst", "biosteam-analyst"],
        recursion_limit=250,
        description="3-way parallel: separation || safety || biosteam",
    ),
    TestQuery(
        name="3way-sep-safe-bio-3",
        query=(
            "Identify solvents that selectively dissolve EVOH from PE in a "
            "multilayer film waste stream. Evaluate each solvent's safety "
            "profile (GSK G-score, PubChem GHS hazards). Run a BioSTEAM "
            "process simulation for the safest viable solvent under energy "
            "case C2."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst", "biosteam-analyst"],
        recursion_limit=250,
        description="3-way parallel: separation || safety || biosteam",
    ),
    # Group B: separation + safety + visualization (3 queries)
    TestQuery(
        name="3way-sep-safe-viz-1",
        query=(
            "Find the optimal separation sequence for PC/PS/PET mixed waste "
            "using selective dissolution. Assess GSK safety G-scores for each "
            "recommended solvent. Create a selectivity heatmap showing "
            "polymer-solvent selectivity across the waste stream."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst", "visualization-specialist"],
        recursion_limit=250,
        description="3-way parallel: separation || safety || visualization",
    ),
    TestQuery(
        name="3way-sep-safe-viz-2",
        query=(
            "Plan a 3-polymer separation cascade for LDPE/PET/EVOH/PC. Check "
            "PubChem hazard data and LD50 toxicity for all candidate solvents. "
            "Then create a process flow diagram showing the dissolution stages "
            "with safety annotations."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst", "visualization-specialist"],
        recursion_limit=250,
        description="3-way parallel: separation || safety || visualization",
    ),
    TestQuery(
        name="3way-sep-safe-viz-3",
        query=(
            "Find solvents for separating PVC from PS below their boiling "
            "points at atmospheric pressure. Get GSK G-scores for each "
            "solvent. Plot solubility vs temperature curves for both polymers "
            "in the top 3 solvents from 25 to 130 degrees C."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst", "visualization-specialist"],
        recursion_limit=250,
        description="3-way parallel: separation || safety || visualization",
    ),
    # Group C: biosteam + safety + visualization (2 queries)
    TestQuery(
        name="3way-bio-safe-viz-1",
        query=(
            "Run a BioSTEAM batch comparison of Toluene, Xylene, and Heptane "
            "for PE recovery under energy case C1. Get GSK safety G-scores "
            "and PubChem hazard data for all three solvents. Visualize the "
            "BioSTEAM results as a cost breakdown chart."
        ),
        pattern="parallel",
        expected_subagents=["biosteam-analyst", "safety-analyst", "visualization-specialist"],
        recursion_limit=250,
        description="3-way parallel: biosteam || safety || visualization",
    ),
    TestQuery(
        name="3way-bio-safe-viz-2",
        query=(
            "Run BioSTEAM simulations for Tetrahydrofuran and Cyclohexane "
            "dissolving PS under C1. Assess both solvents' safety profiles "
            "including LD50 toxicity. Plot the MSP and GWP comparison as a "
            "scenario bar chart."
        ),
        pattern="parallel",
        expected_subagents=["biosteam-analyst", "safety-analyst", "visualization-specialist"],
        recursion_limit=250,
        description="3-way parallel: biosteam || safety || visualization",
    ),
    # Group D: scholar + patent + rag (2 queries)
    TestQuery(
        name="3way-scholar-patent-rag-1",
        query=(
            "Search Google Scholar for recent papers on polyolefin dissolution "
            "recycling with green solvents. Search patents for solvent-based "
            "polyethylene recovery processes. Then query the RAG index to "
            "summarize what dissolution temperatures and solvents the indexed "
            "literature recommends for PE."
        ),
        pattern="parallel",
        expected_subagents=["scholar-researcher", "patent-researcher", "rag-analyst"],
        recursion_limit=250,
        description="3-way parallel: scholar || patent || rag",
    ),
    TestQuery(
        name="3way-scholar-patent-rag-2",
        query=(
            "Find academic publications on EVOH barrier layer dissolution and "
            "recovery methods. Search for patents on ethylene vinyl alcohol "
            "recycling. Then ask the RAG knowledge base what conditions are "
            "needed for selective EVOH dissolution from multilayer packaging."
        ),
        pattern="parallel",
        expected_subagents=["scholar-researcher", "patent-researcher", "rag-analyst"],
        recursion_limit=250,
        description="3-way parallel: scholar || patent || rag",
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="DISSOLVE 3-way parallel trace campaign (10 queries)"
    )
    parser.add_argument("--list", "-l", action="store_true",
                        help="List queries without running")
    parser.add_argument("--query", "-q", default=None,
                        help="Run specific query by index or name")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip trace visualization generation")
    args = parser.parse_args()

    if args.list:
        print("\n3-Way Parallel Trace Queries:")
        print(f"{'#':<4s} {'Name':<30s} {'Description'}")
        print(f"{'-'*4} {'-'*30} {'-'*50}")
        for i, tq in enumerate(QUERIES):
            print(f"{i:<4d} {tq.name:<30s} {tq.description}")
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
            "campaign": "3-way-parallel",
            "n_queries": len(results),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
