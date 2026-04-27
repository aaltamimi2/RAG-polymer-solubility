#!/usr/bin/env python
"""Run sequential execution trace campaign -- 10 queries.

Each query triggers a configured sequential pair from subagents.yaml,
exercising separation->TEA, separation->viz, stats->viz, scholar->RAG,
and stats->RAG chains.

Usage:
    python run_sequential_traces.py           # run all 10
    python run_sequential_traces.py --list    # list queries
    python run_sequential_traces.py -q 0      # run query #0
    python run_sequential_traces.py --no-viz  # skip trace PNGs
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

# ── 10 Sequential Queries ────────────────────────────────────────────

QUERIES: list[TestQuery] = [
    TestQuery(
        name="seq-sep-tea-1",
        query=(
            "Plan a separation sequence for LDPE/PP/HDPE mixed waste using "
            "selective dissolution, then run a BioSTEAM TEA for the "
            "top-ranked solvent under C1."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer", "biosteam-analyst"],
        recursion_limit=250,
        description="Sequential: separation-engineer -> biosteam-analyst",
    ),
    TestQuery(
        name="seq-sep-tea-2",
        query=(
            "Find the best solvent for separating EVOH from PE in a "
            "multilayer film waste. Then estimate the MSP and GWP via "
            "BioSTEAM simulation."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer", "biosteam-analyst"],
        recursion_limit=250,
        description="Sequential: separation-engineer -> biosteam-analyst",
    ),
    TestQuery(
        name="seq-sep-viz-1",
        query=(
            "Find the optimal separation sequence for PS/PET/PC at up to "
            "120\u00b0C, then create a selectivity heatmap showing the results."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer", "visualization-specialist"],
        recursion_limit=250,
        description="Sequential: separation-engineer -> visualization-specialist",
    ),
    TestQuery(
        name="seq-sep-viz-2",
        query=(
            "Plan a separation cascade for LDPE/HDPE/PP, then create a "
            "process flow diagram showing the dissolution stages."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer", "visualization-specialist"],
        recursion_limit=250,
        description="Sequential: separation-engineer -> visualization-specialist",
    ),
    TestQuery(
        name="seq-stats-viz-1",
        query=(
            "Look up the glass transition temperature for polycarbonate, "
            "then plot solubility vs temperature curves for PC in its "
            "three best solvents."
        ),
        pattern="sequential",
        expected_subagents=["statistics-ml", "visualization-specialist"],
        recursion_limit=250,
        description="Sequential: statistics-ml -> visualization-specialist",
    ),
    TestQuery(
        name="seq-stats-viz-2",
        query=(
            "Run a correlation analysis between solvent boiling point and "
            "LDPE solubility at 110\u00b0C, then plot the results."
        ),
        pattern="sequential",
        expected_subagents=["statistics-ml", "visualization-specialist"],
        recursion_limit=250,
        description="Sequential: statistics-ml -> visualization-specialist",
    ),
    TestQuery(
        name="seq-scholar-rag-1",
        query=(
            "Search Google Scholar for recent publications on polyolefin "
            "dissolution in terpene-based green solvents. Save relevant "
            "papers to RAG, then summarize the key findings from the "
            "indexed literature."
        ),
        pattern="sequential",
        expected_subagents=["scholar-researcher", "rag-analyst"],
        recursion_limit=250,
        description="Sequential: scholar-researcher -> rag-analyst",
    ),
    TestQuery(
        name="seq-scholar-rag-2",
        query=(
            "Search arXiv for papers on solvent-based multilayer packaging "
            "recycling. Ingest the best papers, then ask the literature "
            "what temperatures are used for EVOH dissolution."
        ),
        pattern="sequential",
        expected_subagents=["scholar-researcher", "rag-analyst"],
        recursion_limit=250,
        description="Sequential: scholar-researcher -> rag-analyst",
    ),
    TestQuery(
        name="seq-stats-rag",
        query=(
            "Predict the solubility of PMMA in Acetone at 80\u00b0C using "
            "the ML model, then search the RAG index for literature data "
            "on PMMA dissolution to validate the prediction."
        ),
        pattern="sequential",
        expected_subagents=["statistics-ml", "rag-analyst"],
        recursion_limit=250,
        description="Sequential: statistics-ml -> rag-analyst",
    ),
    TestQuery(
        name="seq-sep-tea-3",
        query=(
            "Find the optimal 4-polymer separation sequence for "
            "LDPE/HDPE/PS/PP, then run multi-polymer BioSTEAM recovery "
            "with the recommended solvents under C1."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer", "biosteam-analyst"],
        recursion_limit=250,
        description="Sequential: separation-engineer -> biosteam-analyst",
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="DISSOLVE sequential trace campaign (10 queries)"
    )
    parser.add_argument("--list", "-l", action="store_true",
                        help="List queries without running")
    parser.add_argument("--query", "-q", default=None,
                        help="Run specific query by index or name")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip trace visualization generation")
    args = parser.parse_args()

    if args.list:
        print("\nSequential Trace Queries:")
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
            "campaign": "sequential",
            "n_queries": len(results),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
