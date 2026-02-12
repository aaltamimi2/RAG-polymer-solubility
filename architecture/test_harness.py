"""Multi-agent workflow test harness for DISSOLVE.

Runs predefined test queries, captures LangSmith traces, extracts metrics,
and generates publication-quality trace visualizations.

Usage:
    # Run all multi-agent queries
    python architecture/test_harness.py

    # Run a specific query by index (0-based) or name
    python architecture/test_harness.py --query 0
    python architecture/test_harness.py --query "parallel-sep-safety"

    # List available queries without running
    python architecture/test_harness.py --list

    # Visualize only (from previous results JSON)
    python architecture/test_harness.py --visualize-only results.json

    # Dry run — validate routing classification only
    python architecture/test_harness.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Ensure src/ is importable
_ARCH_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _ARCH_DIR.parent
sys.path.insert(0, str(_ROOT_DIR / "src"))

from dotenv import load_dotenv
load_dotenv(str(_ROOT_DIR / ".env"))

from langsmith import Client as LangSmithClient


# ── Test query definitions ────────────────────────────────────────────

@dataclass
class TestQuery:
    """A single test query with metadata."""
    name: str
    query: str
    pattern: str              # "parallel", "sequential", "3-agent", "cross-domain"
    expected_subagents: list[str]
    recursion_limit: int
    description: str = ""


MULTI_AGENT_QUERIES: list[TestQuery] = [
    # ── 2-agent parallel ──
    TestQuery(
        name="parallel-sep-safety",
        query=(
            "What solvents selectively dissolve PS over PVC at 120°C? "
            "Include GSK safety scores and PubChem hazard data for each recommended solvent."
        ),
        pattern="parallel",
        expected_subagents=["separation-engineer", "safety-analyst"],
        recursion_limit=250,
        description="Parallel: separation-engineer + safety-analyst",
    ),
    # ── 2-agent sequential: sep → TEA ──
    TestQuery(
        name="seq-sep-tea",
        query=(
            "Find an optimal separation sequence for a LDPE/HDPE/PP mixed waste stream "
            "using selective dissolution at atmospheric pressure. "
            "Then run a techno-economic analysis on the solvent recovery for the best option."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer", "tea-lca-analyst"],
        recursion_limit=250,
        description="Sequential: separation → TEA cost analysis",
    ),
    # ── 2-agent sequential: sep → viz ──
    TestQuery(
        name="seq-sep-viz",
        query=(
            "Find the optimal separation sequence for PS, PMMA, and PET "
            "at up to 120°C, then create a selectivity heatmap showing the results."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer", "visualization-specialist"],
        recursion_limit=250,
        description="Sequential: separation → visualization",
    ),
    # ── 2-agent sequential: scholar → RAG ──
    TestQuery(
        name="seq-scholar-rag",
        query=(
            "Do a Google Scholar literature search for recent publications on polyolefin "
            "dissolution in terpene-based solvents. Save the most relevant papers "
            "to the RAG index and then ask the indexed literature to summarize key findings."
        ),
        pattern="sequential",
        expected_subagents=["scholar-researcher", "rag-analyst"],
        recursion_limit=250,
        description="Sequential: literature search → RAG ingestion + Q&A",
    ),
    # ── 2-agent sequential: stats → viz ──
    TestQuery(
        name="seq-stats-viz",
        query=(
            "Look up the glass transition temperature for polycarbonate, then "
            "plot solubility vs temperature curves for its three best solvents."
        ),
        pattern="sequential",
        expected_subagents=["statistics-ml", "visualization-specialist"],
        recursion_limit=250,
        description="Sequential: Tg lookup → solubility plot",
    ),
    # ── 3-agent: sep + safety + TEA ──
    TestQuery(
        name="3agent-sep-safety-tea",
        query=(
            "Find the optimal separation sequence for LDPE and HDPE using selective dissolution. "
            "Assess the safety G-scores and PubChem hazards of each recommended solvent. "
            "Then run a techno-economic analysis on the operating costs for the safest option."
        ),
        pattern="3-agent",
        expected_subagents=["separation-engineer", "safety-analyst", "tea-lca-analyst"],
        recursion_limit=250,
        description="3-agent chain: separation → safety → TEA",
    ),
    # ── 3-agent: sep + safety + viz ──
    TestQuery(
        name="3agent-sep-safety-viz",
        query=(
            "Separate PS from PVC using selective dissolution — show the selectivity data, "
            "safety profiles for the top 3 solvents, and create a comparison dashboard."
        ),
        pattern="3-agent",
        expected_subagents=["separation-engineer", "safety-analyst", "visualization-specialist"],
        recursion_limit=250,
        description="3-agent chain: separation → safety → dashboard",
    ),
    # ── Cross-domain: RAG + separation ──
    TestQuery(
        name="cross-rag-sep",
        query=(
            "Search the RAG index for information about EVOH dissolution conditions "
            "in our indexed literature. Then plan a separation scheme for an EVOH/LDPE "
            "mixed stream using selective dissolution."
        ),
        pattern="cross-domain",
        expected_subagents=["rag-analyst", "separation-engineer"],
        recursion_limit=250,
        description="Cross-domain: RAG literature → separation planning",
    ),
    # ── Ambiguous: multi-criteria comparison ──
    TestQuery(
        name="ambiguous-multi-criteria",
        query=(
            "Compare toluene and xylene for selective dissolution of HDPE at 120°C — "
            "I need selectivity data, GSK safety G-scores, and a techno-economic "
            "analysis of the operating costs for solvent recovery."
        ),
        pattern="3-agent",
        expected_subagents=["separation-engineer", "safety-analyst", "tea-lca-analyst"],
        recursion_limit=250,
        description="Ambiguous routing: selectivity + safety + cost comparison",
    ),
    # ── 3-scheme separation (regression test for v6 optimization) ──
    TestQuery(
        name="regression-3scheme-9poly",
        query=(
            "Find the optimal separation sequence for a mixed polymer waste stream "
            "containing PS, PVC, LDPE, HDPE, PP, EVOH, Nylon6, Nylon66, and PET. "
            "Use selective dissolution at atmospheric pressure. "
            "Propose THREE different sets of solvents and conditions."
        ),
        pattern="sequential",
        expected_subagents=["separation-engineer"],
        recursion_limit=150,
        description="Regression: 9-polymer 3-scheme (should use multi-scheme tool)",
    ),
]


# ── Metric extraction (mirrors scaling_benchmark.py) ──────────────────

def extract_metrics(messages: list) -> dict:
    """Extract token counts, tool calls, and subagent info from messages."""
    total_input = 0
    total_output = 0
    n_tool_calls = 0
    tool_names: list[str] = []
    subagents_invoked: list[str] = []

    for msg in messages:
        if msg.type == "ai":
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)

            for tc in getattr(msg, "tool_calls", None) or []:
                name = tc.get("name", "")
                tool_names.append(name)
                n_tool_calls += 1
                if name == "task":
                    sa = tc.get("args", {}).get("subagent_type", "")
                    if sa:
                        subagents_invoked.append(sa)

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "n_tool_calls": n_tool_calls,
        "n_messages": len(messages),
        "tool_names": tool_names,
        "subagents_invoked": subagents_invoked,
    }


def _extract_text(content) -> str:
    """Extract plain text from AI message content (handles Gemini list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


# ── LangSmith trace capture ──────────────────────────────────────────

def get_latest_trace_id(
    client: LangSmithClient,
    project: str = "strap-agent",
    after_time: datetime | None = None,
) -> str | None:
    """Get the trace ID of the most recent root run.

    If after_time is provided, only considers traces started after that time.
    Polls up to 15 seconds for the trace to appear (LangSmith has ingestion lag).
    """
    import time as _time

    for attempt in range(6):
        try:
            runs = list(client.list_runs(
                project_name=project,
                is_root=True,
                limit=5,
            ))
        except Exception as e:
            print(f"  LangSmith query failed (attempt {attempt + 1}): {e}")
            _time.sleep(2)
            continue

        if runs:
            if after_time:
                # Filter to runs started after our timestamp
                # Handle timezone-aware vs naive datetimes
                candidates = []
                for r in runs:
                    if not r.start_time:
                        continue
                    r_time = r.start_time
                    a_time = after_time
                    # Make both aware or both naive for comparison
                    if r_time.tzinfo is None and a_time.tzinfo is not None:
                        a_time = a_time.replace(tzinfo=None)
                    elif r_time.tzinfo is not None and a_time.tzinfo is None:
                        r_time = r_time.replace(tzinfo=None)
                    if r_time > a_time:
                        candidates.append(r)
                if candidates:
                    candidates.sort(key=lambda r: r.start_time, reverse=True)
                    return str(candidates[0].trace_id)
            else:
                return str(runs[0].trace_id)

        if attempt < 5:
            _time.sleep(3)

    return None


# ── Query runner ──────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Full result from running a single test query."""
    name: str
    query: str
    pattern: str
    expected_subagents: list[str]
    actual_subagents: list[str]
    wall_time_s: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    n_tool_calls: int
    n_messages: int
    tool_names: list[str]
    trace_id: str | None
    answer_preview: str
    routing_match: bool        # did actual subagents match expected?
    timestamp: str
    error: str | None = None
    waterfall_png: str | None = None
    swimlane_png: str | None = None


def run_query(agent, tq: TestQuery, ls_client: LangSmithClient) -> QueryResult:
    """Run a single test query and capture all metrics + trace ID."""
    from datetime import timezone

    print(f"\n{'='*70}")
    print(f"Running: {tq.name}")
    print(f"Pattern: {tq.pattern} | Expected: {', '.join(tq.expected_subagents)}")
    print(f"Query: {tq.query[:100]}...")
    print(f"{'='*70}")

    before_time = datetime.now(tz=timezone.utc)
    error = None
    answer = ""
    messages = []

    t0 = time.time()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": tq.query}]},
            {"recursion_limit": tq.recursion_limit},
        )
        messages = result.get("messages", [])

        # Extract answer
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                answer = _extract_text(msg.content)
                break
    except Exception as e:
        error = str(e)
        print(f"  ERROR: {e}")
    wall_time = time.time() - t0

    # Extract metrics
    metrics = extract_metrics(messages) if messages else {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        "n_tool_calls": 0, "n_messages": 0, "tool_names": [],
        "subagents_invoked": [],
    }

    # Capture trace ID from LangSmith
    print("  Fetching LangSmith trace ID...")
    trace_id = get_latest_trace_id(ls_client, after_time=before_time)
    if trace_id:
        print(f"  Trace ID: {trace_id}")
    else:
        print("  WARNING: Could not capture trace ID")

    # Check routing match
    actual_set = set(metrics["subagents_invoked"])
    expected_set = set(tq.expected_subagents)
    routing_match = expected_set.issubset(actual_set)

    # Truncate answer for preview
    answer_preview = answer[:500] + "..." if len(answer) > 500 else answer

    # Print summary
    print(f"\n  Duration:    {wall_time:.1f}s")
    print(f"  Tokens:      {metrics['total_tokens']:,} ({metrics['input_tokens']:,} in, {metrics['output_tokens']:,} out)")
    print(f"  Messages:    {metrics['n_messages']}")
    print(f"  Tool calls:  {metrics['n_tool_calls']}")
    print(f"  Subagents:   {metrics['subagents_invoked']}")
    print(f"  Routing OK:  {'YES' if routing_match else 'NO — expected ' + str(tq.expected_subagents)}")
    if error:
        print(f"  Error:       {error}")

    return QueryResult(
        name=tq.name,
        query=tq.query,
        pattern=tq.pattern,
        expected_subagents=tq.expected_subagents,
        actual_subagents=metrics["subagents_invoked"],
        wall_time_s=round(wall_time, 1),
        total_tokens=metrics["total_tokens"],
        input_tokens=metrics["input_tokens"],
        output_tokens=metrics["output_tokens"],
        n_tool_calls=metrics["n_tool_calls"],
        n_messages=metrics["n_messages"],
        tool_names=metrics["tool_names"],
        trace_id=trace_id,
        answer_preview=answer_preview,
        routing_match=routing_match,
        timestamp=datetime.now().isoformat(),
        error=error,
    )


# ── Trace visualization ──────────────────────────────────────────────

def generate_trace_visuals(
    result: QueryResult,
    ls_client: LangSmithClient,
    output_dir: Path,
) -> QueryResult:
    """Generate stacked-card trace PNG for a result."""
    if not result.trace_id:
        print(f"  Skipping visualization for {result.name} — no trace ID")
        return result

    # Import the stacked-card visualization module
    sys.path.insert(0, str(_ARCH_DIR))
    from visualize_trace_cards import fetch_trace_structure, draw_trace_cards

    trace_dir = output_dir / result.name
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Stacked-card diagram (clean style matching trace7)
    try:
        print(f"  Generating stacked-card trace for {result.name}...")
        data = fetch_trace_structure(ls_client, result.trace_id)
        card_path = str(trace_dir / f"{result.name}_trace.png")
        title = f"DISSOLVE — {result.name}  |  {result.pattern}"
        draw_trace_cards(data, card_path, title=title)
        result.waterfall_png = card_path
    except Exception as e:
        print(f"  WARNING: Trace visualization failed: {e}")

    return result


# ── Routing dry run ───────────────────────────────────────────────────

def dry_run():
    """Validate routing classification for all test queries."""
    from langchain_core.messages import HumanMessage
    from strap.routing import classify_query

    print("\nRouting Dry Run")
    print("=" * 70)

    for tq in MULTI_AGENT_QUERIES:
        hint = classify_query([HumanMessage(content=tq.query)])
        if hint is None:
            matched = "(no routing hint)"
        else:
            # Extract agent names from hint
            import re
            agents_in_hint = re.findall(r'"([\w-]+)"', hint)
            matched = ", ".join(agents_in_hint[:4])

        expected = ", ".join(tq.expected_subagents)

        # Check if expected agents appear in the hint
        ok = all(a in (hint or "") for a in tq.expected_subagents)
        status = "OK" if ok else "MISMATCH"

        print(f"\n  [{status:8s}] {tq.name}")
        print(f"            Expected:  {expected}")
        print(f"            Detected:  {matched}")
        if hint and not ok:
            print(f"            Hint:      {hint[:120]}...")


# ── Summary table ─────────────────────────────────────────────────────

def print_summary(results: list[QueryResult]):
    """Print a formatted summary table of all results."""
    print(f"\n\n{'='*100}")
    print("MULTI-AGENT TEST RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"{'Name':<28s} {'Pattern':<14s} {'Time':>7s} {'Tokens':>10s} {'Msgs':>5s} {'Tools':>6s} {'Subagents':<30s} {'Route':>5s}")
    print(f"{'-'*28} {'-'*14} {'-'*7} {'-'*10} {'-'*5} {'-'*6} {'-'*30} {'-'*5}")

    for r in results:
        agents_str = ",".join(r.actual_subagents) if r.actual_subagents else "(none)"
        if len(agents_str) > 29:
            agents_str = agents_str[:26] + "..."
        route = "OK" if r.routing_match else "MISS"
        err = " ERR" if r.error else ""

        print(
            f"{r.name:<28s} {r.pattern:<14s} {r.wall_time_s:>6.1f}s "
            f"{r.total_tokens:>10,} {r.n_messages:>5} {r.n_tool_calls:>6} "
            f"{agents_str:<30s} {route:>5s}{err}"
        )

    # Totals
    total_time = sum(r.wall_time_s for r in results)
    total_tokens = sum(r.total_tokens for r in results)
    n_ok = sum(1 for r in results if r.routing_match)
    n_err = sum(1 for r in results if r.error)
    print(f"{'-'*100}")
    print(f"{'TOTAL':<28s} {'':14s} {total_time:>6.1f}s {total_tokens:>10,} {'':5s} {'':6s} {'':30s} {n_ok}/{len(results)}")
    if n_err:
        print(f"  Errors: {n_err}")
    print()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DISSOLVE multi-agent workflow test harness"
    )
    parser.add_argument(
        "--query", "-q", default=None,
        help="Run specific query by index (0-based) or name"
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all available test queries"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate routing only — don't run the agent"
    )
    parser.add_argument(
        "--visualize-only", metavar="JSON",
        help="Generate visuals from a previous results JSON file"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip trace visualization generation"
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory for results (default: architecture/test_results/)"
    )
    parser.add_argument(
        "-p", "--project", default="strap-agent",
        help="LangSmith project name"
    )
    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        print("\nAvailable test queries:")
        print(f"{'#':<4s} {'Name':<28s} {'Pattern':<14s} {'Description'}")
        print(f"{'-'*4} {'-'*28} {'-'*14} {'-'*40}")
        for i, tq in enumerate(MULTI_AGENT_QUERIES):
            print(f"{i:<4d} {tq.name:<28s} {tq.pattern:<14s} {tq.description}")
        return

    # ── Dry run mode ──
    if args.dry_run:
        dry_run()
        return

    # ── Output directory ──
    output_dir = Path(args.output_dir) if args.output_dir else _ARCH_DIR / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    ls_client = LangSmithClient()

    # ── Visualize-only mode ──
    if args.visualize_only:
        with open(args.visualize_only) as f:
            data = json.load(f)
        results = [QueryResult(**r) for r in data["results"]]
        for r in results:
            generate_trace_visuals(r, ls_client, output_dir)
        print("\nVisualization complete.")
        return

    # ── Select queries to run ──
    if args.query is not None:
        try:
            idx = int(args.query)
            queries = [MULTI_AGENT_QUERIES[idx]]
        except ValueError:
            queries = [q for q in MULTI_AGENT_QUERIES if q.name == args.query]
            if not queries:
                print(f"Unknown query: {args.query}")
                print("Use --list to see available queries.")
                return
    else:
        queries = MULTI_AGENT_QUERIES

    # ── Create agent ──
    print("Loading DISSOLVE agent...")
    from strap.agent import create_dissolve_agent
    agent = create_dissolve_agent()
    print("Agent ready.\n")

    # ── Run queries ──
    results: list[QueryResult] = []
    for tq in queries:
        result = run_query(agent, tq, ls_client)

        if not args.no_viz:
            result = generate_trace_visuals(result, ls_client, output_dir)

        results.append(result)

    # ── Save results JSON ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"test_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "n_queries": len(results),
            "results": [asdict(r) for r in results],
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Print summary ──
    print_summary(results)


if __name__ == "__main__":
    main()
