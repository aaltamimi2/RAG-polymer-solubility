"""Benchmark DISSOLVE agent scaling behavior across polymer counts.

Runs a series of queries with increasing polymer counts and measures
wall-clock time, token usage, tool calls, and message counts.

Usage:
    python architecture/scaling_benchmark.py
    python architecture/scaling_benchmark.py --dry-run      # validate routing only
    python architecture/scaling_benchmark.py -o custom.png   # custom output path
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

# ── Constants ──────────────────────────────────────────────────────────

ALL_POLYMERS = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "Nylon6", "Nylon66", "PET"]
MIN_POLYMERS = 2
RECURSION_LIMIT = 50


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Configuration for one benchmark series."""
    name: str
    query_template: str          # format string with {polymer_list}
    polymers: list[str] = field(default_factory=lambda: ALL_POLYMERS.copy())
    min_count: int = MIN_POLYMERS
    recursion_limit: int = RECURSION_LIMIT
    color: str = "#2C3E50"
    marker: str = "o"


@dataclass
class BenchmarkResult:
    """Metrics from a single query run."""
    n_polymers: int
    polymers_used: list[str]
    query: str
    wall_time_s: float
    total_tokens: int
    input_tokens: int
    output_tokens: int
    n_tool_calls: int
    n_messages: int
    tool_names: list[str]
    routed_to_subagent: bool
    error: str | None = None


# ── Predefined configs ─────────────────────────────────────────────────

THERMODYNAMIC_CONFIG = BenchmarkConfig(
    name="thermodynamic-only",
    query_template=(
        "From the available solvents, which best dissolve each of {polymer_list} "
        "at 120C while rejecting the others? Rank by solubility ratio."
    ),
    color="#2C3E50",
    marker="o",
)

# Future configs — uncomment/add as needed:
#
# SAFETY_CONFIG = BenchmarkConfig(
#     name="with-safety-subagent",
#     query_template=(
#         "Find the optimal separation for {polymer_list}. "
#         "Then check the safety G-scores of the recommended solvents."
#     ),
#     recursion_limit=250,
#     color="#E74C3C",
#     marker="s",
# )


# ── Metric extraction ─────────────────────────────────────────────────

def extract_metrics(messages: list) -> dict:
    """Extract token counts, tool calls, and routing info from messages."""
    total_input = 0
    total_output = 0
    n_tool_calls = 0
    tool_names: list[str] = []
    routed_to_subagent = False

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
                    routed_to_subagent = True

    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_input + total_output,
        "n_tool_calls": n_tool_calls,
        "tool_names": tool_names,
        "routed_to_subagent": routed_to_subagent,
        "n_messages": len(messages),
    }


# ── Routing validation ────────────────────────────────────────────────

def validate_query_routing(query: str, expect_no_routing: bool = True) -> bool:
    """Pre-flight check: ensure query does/doesn't trigger subagent routing."""
    from langchain_core.messages import HumanMessage
    from strap.routing import classify_query

    hint = classify_query([HumanMessage(content=query)])

    if expect_no_routing and hint is not None:
        print(f"  WARNING: Query triggers routing! Hint: {hint[:120]}...")
        return False
    if not expect_no_routing and hint is None:
        print(f"  WARNING: Query does NOT trigger routing (expected it to).")
        return False
    return True


# ── Query generation ──────────────────────────────────────────────────

def generate_query(template: str, polymers: list[str]) -> str:
    """Format a query template with a polymer list."""
    return template.format(polymer_list=", ".join(polymers))


# ── Single query runner ───────────────────────────────────────────────

def run_single_query(
    agent,
    query: str,
    n_polymers: int,
    polymers_used: list[str],
    recursion_limit: int = RECURSION_LIMIT,
) -> BenchmarkResult:
    """Run a single query and collect metrics."""
    print(f"  [{n_polymers} polymers] Running...", end=" ", flush=True)

    t0 = time.time()
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"recursion_limit": recursion_limit},
        )
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"ERROR ({elapsed:.1f}s): {e}")
        return BenchmarkResult(
            n_polymers=n_polymers, polymers_used=polymers_used,
            query=query, wall_time_s=elapsed,
            total_tokens=0, input_tokens=0, output_tokens=0,
            n_tool_calls=0, n_messages=0, tool_names=[],
            routed_to_subagent=False, error=str(e),
        )

    metrics = extract_metrics(result["messages"])
    br = BenchmarkResult(
        n_polymers=n_polymers,
        polymers_used=polymers_used,
        query=query,
        wall_time_s=elapsed,
        total_tokens=metrics["total_tokens"],
        input_tokens=metrics["input_tokens"],
        output_tokens=metrics["output_tokens"],
        n_tool_calls=metrics["n_tool_calls"],
        n_messages=metrics["n_messages"],
        tool_names=metrics["tool_names"],
        routed_to_subagent=metrics["routed_to_subagent"],
    )

    status = "SUBAGENT" if br.routed_to_subagent else "OK"
    print(
        f"{status} | {elapsed:.1f}s | {br.total_tokens:,} tok | "
        f"{br.n_tool_calls} tools | {br.n_messages} msgs"
    )
    if br.routed_to_subagent:
        print(f"    WARNING: Routed to subagent! Tools: {br.tool_names}")

    return br


# ── Series runner ─────────────────────────────────────────────────────

def run_benchmark_series(
    config: BenchmarkConfig,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """Run a full benchmark series from min_count to len(polymers)."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {config.name}")
    print(f"Polymers: {', '.join(config.polymers)}")
    print(f"Range: {config.min_count} -> {len(config.polymers)}")
    print(f"{'=' * 60}\n")

    # Phase 1: generate + validate queries
    queries = []
    for n in range(config.min_count, len(config.polymers) + 1):
        subset = config.polymers[:n]
        query = generate_query(config.query_template, subset)
        queries.append((n, subset, query))

    print("Phase 1: Routing validation")
    all_valid = True
    for n, _, query in queries:
        ok = validate_query_routing(query, expect_no_routing=True)
        print(f"  {n} polymers: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_valid = False

    if not all_valid:
        print("\nROUTING VALIDATION FAILED. Fix the query template.")
        return []

    if dry_run:
        print("\nDry run — all queries pass routing validation.")
        for n, _, query in queries:
            print(f"\n  [{n}] {query}")
        return []

    # Phase 2: run queries
    print("\nPhase 2: Loading agent...")
    logging.disable(logging.CRITICAL)

    from strap.agent import create_dissolve_agent
    agent = create_dissolve_agent()
    print("Agent loaded.\n")

    results = []
    for n, subset, query in queries:
        r = run_single_query(agent, query, n, subset, config.recursion_limit)
        results.append(r)

    return results


# ── Plotting ──────────────────────────────────────────────────────────

def plot_results(
    series: dict[str, tuple[BenchmarkConfig, list[BenchmarkResult]]],
    output_path: str,
) -> None:
    """Generate a publication-quality scaling plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for name, (config, results) in series.items():
        valid = [r for r in results if r.error is None]
        if not valid:
            continue

        xs = [r.n_polymers for r in valid]
        ys = [r.wall_time_s for r in valid]
        tokens = [r.total_tokens for r in valid]

        ax.plot(
            xs, ys,
            color=config.color,
            marker=config.marker,
            markersize=8,
            linewidth=2,
            label=config.name,
            zorder=3,
        )

        for x, y, tok in zip(xs, ys, tokens):
            if tok >= 1_000_000:
                label = f"{tok / 1_000_000:.1f}M"
            elif tok >= 1_000:
                label = f"{tok / 1_000:.0f}K"
            else:
                label = str(tok)
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
                color=config.color,
                fontweight="bold",
            )

    ax.set_xlabel("Number of Polymers", fontsize=12)
    ax.set_ylabel("End-to-End Query Time (seconds)", fontsize=12)
    ax.set_title("DISSOLVE Agent Scaling Behavior", fontsize=14, fontweight="bold")
    ax.set_xticks(range(MIN_POLYMERS, len(ALL_POLYMERS) + 1))

    if len(series) > 1:
        ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


# ── Summary + JSON export ─────────────────────────────────────────────

def print_summary(results: list[BenchmarkResult], name: str) -> None:
    """Print a summary table."""
    print(f"\n{'=' * 70}")
    print(f"Summary: {name}")
    print(f"{'=' * 70}")
    print(f"{'N':>3} | {'Time (s)':>8} | {'Tokens':>10} | {'Tools':>5} | {'Msgs':>4} | {'Subagent':>8}")
    print(f"{'-' * 3}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 5}-+-{'-' * 4}-+-{'-' * 8}")
    for r in results:
        sa = "YES" if r.routed_to_subagent else "no"
        err = " ERR" if r.error else ""
        print(
            f"{r.n_polymers:>3} | {r.wall_time_s:>8.1f} | "
            f"{r.total_tokens:>10,} | {r.n_tool_calls:>5} | "
            f"{r.n_messages:>4} | {sa:>8}{err}"
        )


def save_results_json(results: list[BenchmarkResult], output_path: str) -> None:
    """Save raw results as JSON for later analysis."""
    data = []
    for r in results:
        data.append({
            "n_polymers": r.n_polymers,
            "polymers": r.polymers_used,
            "wall_time_s": r.wall_time_s,
            "total_tokens": r.total_tokens,
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "n_tool_calls": r.n_tool_calls,
            "n_messages": r.n_messages,
            "tool_names": r.tool_names,
            "routed_to_subagent": r.routed_to_subagent,
            "error": r.error,
            "query": r.query,
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DISSOLVE agent scaling behavior"
    )
    parser.add_argument(
        "-o", "--output",
        default=str(Path(__file__).parent / "scaling_benchmark.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--json", default=None,
        help="Output JSON path (default: same stem as PNG)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate routing without running queries",
    )
    args = parser.parse_args()

    json_path = args.json or str(Path(args.output).with_suffix(".json"))

    configs = {
        "thermodynamic-only": THERMODYNAMIC_CONFIG,
    }

    all_series: dict[str, tuple[BenchmarkConfig, list[BenchmarkResult]]] = {}
    for name, config in configs.items():
        results = run_benchmark_series(config, dry_run=args.dry_run)
        if results:
            print_summary(results, name)
            all_series[name] = (config, results)

    if not args.dry_run and all_series:
        all_results = []
        for _, (_, results) in all_series.items():
            all_results.extend(results)
        save_results_json(all_results, json_path)
        plot_results(all_series, args.output)


if __name__ == "__main__":
    main()
