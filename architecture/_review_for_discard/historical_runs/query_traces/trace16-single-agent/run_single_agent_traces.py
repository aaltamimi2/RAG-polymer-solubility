#!/usr/bin/env python
"""Run single-agent execution trace campaign -- 10 queries.

Each query targets exactly one subagent, covering all subagents plus
the new BioSTEAM sensitivity tools (Monte Carlo, tornado, sweep).

Usage:
    python run_single_agent_traces.py           # run all 10
    python run_single_agent_traces.py --list    # list queries
    python run_single_agent_traces.py -q 0      # run query #0
    python run_single_agent_traces.py --no-viz  # skip trace PNGs
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

# ── 10 Single-Agent Queries ──────────────────────────────────────────

QUERIES: list[TestQuery] = [
    TestQuery(
        name="single-separation",
        query=(
            "Find the optimal separation sequence for a PP/PS/PVC mixed "
            "waste stream using selective dissolution at atmospheric pressure."
        ),
        pattern="single",
        expected_subagents=["separation-engineer"],
        recursion_limit=150,
        description="Single: separation-engineer",
    ),
    TestQuery(
        name="single-safety",
        query=(
            "Assess the GSK safety G-scores, PubChem GHS hazards, and LD50 "
            "toxicity data for Toluene, Xylene, and Tetrahydrofuran."
        ),
        pattern="single",
        expected_subagents=["safety-analyst"],
        recursion_limit=150,
        description="Single: safety-analyst",
    ),
    TestQuery(
        name="single-biosteam-sim",
        query=(
            "Run a rigorous BioSTEAM STRAP process simulation for Toluene "
            "dissolving PE under energy case C1 at 20,000 MT/yr capacity."
        ),
        pattern="single",
        expected_subagents=["biosteam-analyst"],
        recursion_limit=150,
        description="Single: biosteam-analyst (simulation)",
    ),
    TestQuery(
        name="single-biosteam-batch",
        query=(
            "Compare all PE solvents in BioSTEAM under energy case C1. "
            "Rank them by MSP."
        ),
        pattern="single",
        expected_subagents=["biosteam-analyst"],
        recursion_limit=150,
        description="Single: biosteam-analyst (batch comparison)",
    ),
    TestQuery(
        name="single-uncertainty",
        query=(
            "Run a Monte Carlo uncertainty analysis for Toluene/PE recovery "
            "under C1. How confident are we in the MSP estimate?"
        ),
        pattern="single",
        expected_subagents=["biosteam-analyst"],
        recursion_limit=150,
        description="Single: biosteam-analyst (Monte Carlo)",
    ),
    TestQuery(
        name="single-tornado",
        query=(
            "Which parameter drives MSP the most for Xylene PE recovery "
            "under C1? Run a tornado sensitivity analysis."
        ),
        pattern="single",
        expected_subagents=["biosteam-analyst"],
        recursion_limit=150,
        description="Single: biosteam-analyst (tornado)",
    ),
    TestQuery(
        name="single-sweep",
        query=(
            "How does MSP change with solvent price for Heptane PE recovery? "
            "Sweep solvent price from $0.50 to $2.00/kg."
        ),
        pattern="single",
        expected_subagents=["biosteam-analyst"],
        recursion_limit=150,
        description="Single: biosteam-analyst (parameter sweep)",
    ),
    TestQuery(
        name="single-viz",
        query=(
            "Plot solubility vs temperature curves for LDPE in Toluene, "
            "Xylene, and Heptane from 25 to 150\u00b0C."
        ),
        pattern="single",
        expected_subagents=["visualization-specialist"],
        recursion_limit=150,
        description="Single: visualization-specialist",
    ),
    TestQuery(
        name="single-stats-ml",
        query=(
            "Look up the glass transition temperature for polystyrene. Then "
            "predict the solubility of PS in Toluene at 110\u00b0C using the ML model."
        ),
        pattern="single",
        expected_subagents=["statistics-ml"],
        recursion_limit=150,
        description="Single: statistics-ml",
    ),
    TestQuery(
        name="single-rag",
        query=(
            "Search the RAG index for information about EVOH dissolution "
            "conditions and solvent selection from the indexed literature."
        ),
        pattern="single",
        expected_subagents=["rag-analyst"],
        recursion_limit=150,
        description="Single: rag-analyst",
    ),
]


def main():
    parser = argparse.ArgumentParser(
        description="DISSOLVE single-agent trace campaign (10 queries)"
    )
    parser.add_argument("--list", "-l", action="store_true",
                        help="List queries without running")
    parser.add_argument("--query", "-q", default=None,
                        help="Run specific query by index or name")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip trace visualization generation")
    args = parser.parse_args()

    if args.list:
        print("\nSingle-Agent Trace Queries:")
        print(f"{'#':<4s} {'Name':<28s} {'Description'}")
        print(f"{'-'*4} {'-'*28} {'-'*40}")
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
            "campaign": "single-agent",
            "n_queries": len(results),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
