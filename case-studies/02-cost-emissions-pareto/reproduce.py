"""Cost-vs-emissions / cost-vs-circularity Pareto landscapes for the STRAP
waste-management superstructure — v10 port and fix of the v9 case-matrix work.

THE PROBLEM THIS FIXES
    Many v9 Pareto runs returned a single "broken" point. That is not a solver
    bug — it is what the strict frontier collapses to when the min-cost design
    and the min-(other-objective) design coincide. Three things determine
    whether a *rich* frontier exists, and this case study makes each explicit:

    1. Scenario. Scenario A's technology/emission parameters make the cheapest
       design also the cleanest, so the cost-vs-emissions frontier is a single
       point. Scenario B prices the clean and dirty stage-3 technologies
       apart, producing a genuine trade-off.
    2. Objective. Stage-3 technology selection is all-or-nothing in the
       superstructure (one technology takes the whole residual stream), so
       cost-vs-emissions resolves to the two technology corners. Circularity
       varies with the recovered-mass mix, giving a multi-point frontier.
    3. What you plot. The strict frontier is small, but the *landscape* of all
       feasible designs is always rich (~18 points here). Plotting the
       landscape with the frontier highlighted is the honest, informative
       visual — the fix for "it only shows one point".

REPRODUCTION
    Data in data/*.json is live output from v10's run_waste_management_pareto
    (SCIP solver, workbook-backed TEA — no BioSTEAM/API calls). Re-solve with
    --live to regenerate it; by default this script replays the committed JSON.

USAGE
    python case-studies/02-cost-emissions-pareto/reproduce.py          # replay
    python case-studies/02-cost-emissions-pareto/reproduce.py --live   # re-solve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "case-studies" / "_shared"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import casestudy_style as style  # noqa: E402

DATA = _HERE / "data"
FIGURES = _HERE / "figures"

# name -> (title, y_metric, y_label, run kwargs for --live)
PANELS = {
    "A_emissions_degenerate": (
        "Scenario A · cost vs emissions",
        "emissions", "Emissions (t CO₂e/yr)",
        dict(feed_composition_json={"PE": 0.6, "EVOH": 0.4}, scenario="A",
             y_metric="emissions", n_points=20),
    ),
    "B_emissions_two_corner": (
        "Scenario B · cost vs emissions",
        "emissions", "Emissions (t CO₂e/yr)",
        dict(feed_composition_json={"PE": 0.6, "EVOH": 0.4}, scenario="B",
             y_metric="emissions", n_points=20),
    ),
    "B_circularity_rich": (
        "Scenario B · cost vs circularity (≥1 wash)",
        "circularity_score", "Circularity score",
        dict(feed_composition_json={"PE": 0.5, "EVOH": 0.5}, scenario="B",
             y_metric="circularity", n_points=24, min_active_washes=1, max_active_washes=2),
    ),
}


def regenerate_live() -> None:
    from strap.tools.waste_optimization import run_waste_management_pareto

    DATA.mkdir(exist_ok=True)
    for name, (_title, _ym, _yl, kwargs) in PANELS.items():
        raw = run_waste_management_pareto(feed=8000.0, x_metric="total_cost", **kwargs)
        payload = json.loads(raw)
        (DATA / f"{name}.json").write_text(
            json.dumps(payload.get("data", payload), indent=1)
        )
        print(f"  re-solved {name}")


def _nondominated(points: list[tuple[float, float]], *, maximize_y: bool) -> list[tuple[float, float]]:
    """Non-dominated set for (minimise cost, minimise/maximise y).

    Computed directly from the landscape so every frontier point shares the
    landscape's cost basis. This is the fix for v9's "broken" single-point
    frontiers: the engine's native frontier could include an anchor design
    with capital_cost reported as 0 (cost basis ~27x below the built designs),
    which dominated everything and collapsed the frontier. Recomputing from the
    landscape drops that phantom point and recovers the true — often richer —
    trade-off (e.g. Scenario B emissions: 2 native points -> 3 real points).
    """
    ordered = sorted(points, key=lambda cy: (cy[0], -cy[1] if maximize_y else cy[1]))
    frontier: list[tuple[float, float]] = []
    best = -float("inf") if maximize_y else float("inf")
    for cost, y in ordered:
        improves = (y > best + 1e-9) if maximize_y else (y < best - 1e-6)
        if improves:
            frontier.append((cost, y))
            best = y
    return frontier


def _points(entry: dict, y_key: str, *, maximize_y: bool):
    """Return (landscape, frontier) as lists of (cost, y) tuples.

    Frontier is recomputed from the landscape on a consistent cost basis
    rather than read from the engine's (cost-basis-inconsistent) points[].
    """
    landscape = [
        (p["total_cost"], p.get(y_key))
        for p in entry.get("landscape_points", [])
        if p.get("total_cost") is not None and p.get(y_key) is not None
    ]
    frontier = _nondominated(landscape, maximize_y=maximize_y)
    return landscape, frontier


def render() -> dict:
    style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    summary = {}

    for ax, (name, (title, y_key, y_label, _kw)) in zip(axes, PANELS.items()):
        entry = json.loads((DATA / f"{name}.json").read_text())
        maximize_y = y_key == "circularity_score"
        landscape, frontier = _points(entry, y_key, maximize_y=maximize_y)

        if landscape:
            lx, ly = zip(*[(c / 1e6, y) for c, y in landscape])
            ax.scatter(lx, ly, s=34, color="#b8c2cc", edgecolor="#8a97a3",
                       linewidth=0.4, zorder=2, label=f"Feasible designs ({len(landscape)})")
        if frontier:
            fx, fy = zip(*[(c / 1e6, y) for c, y in frontier])
            ax.plot(fx, fy, "-", color=style.FRONTIER, lw=1.6, zorder=3)
            ax.scatter(fx, fy, s=95, color=style.ACCENT_OPTIMAL, edgecolor="white",
                       linewidth=1.0, zorder=4, marker="D",
                       label=f"Pareto frontier ({len(frontier)})")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Total cost (M$/yr)")
        ax.set_ylabel(y_label)
        ax.legend(loc="best")

        summary[name] = {
            "y_metric": entry.get("y_metric"),
            "landscape_points": len(landscape),
            "frontier_points": len(frontier),
            "engine_native_frontier_points": entry.get("n_points_feasible"),
        }

    fig.suptitle("STRAP superstructure: Pareto landscapes and frontiers\n"
                 "single-point frontiers are degenerate corners, not solver failures — "
                 "the design landscape stays rich",
                 fontsize=14, fontweight="bold", y=1.02, color=style.INK)
    style.caption(fig, "Live SCIP optimizer over workbook-backed TEA (no BioSTEAM/API). "
                       "Diamonds: non-dominated frontier; grey: all feasible designs.")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "pareto_landscapes.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "pareto_landscapes.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="re-solve with the live SCIP optimizer before rendering")
    args = parser.parse_args()
    if args.live:
        regenerate_live()
    summary = render()
    print(json.dumps({
        "panels": summary,
        "figure": "case-studies/02-cost-emissions-pareto/figures/pareto_landscapes.png",
    }, indent=2))


if __name__ == "__main__":
    main()
