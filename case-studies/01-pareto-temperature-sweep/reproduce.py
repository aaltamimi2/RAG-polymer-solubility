"""Reproduce the selectivity-vs-cost Pareto frontier across the STRAP
dissolution separation sequence — v10 port of the v9 dp_lattice_sweep flagship.

WHAT THIS SHOWS
    For a fixed 8-polymer separation sequence (each step recovers one polymer
    with a chosen solvent), the dissolution temperature at each step trades
    selectivity against minimum selling price (MSP). Raising a step's
    temperature usually lifts solubility (and MSP via energy) while changing
    how cleanly that polymer separates from the ones still in the stream.
    The Pareto frontier answers: "to guarantee a minimum selectivity of X%
    at every step, what is the lowest achievable average MSP, and at which
    per-step temperatures?"

REPRODUCTION MODEL (no API calls)
    - Selectivity curves are recomputed LIVE from v10's validated solubility
      interpolation engine (strap.solubility.get_solubility) for each step's
      (target polymer, chosen solvent) against the polymers still in the
      stream. This is the v10 improvement over v9, which used a bespoke
      selectivity cache.
    - MSP curves are REPLAYED from 217 real BioSTEAM simulations that were run
      once in v9 (data/biosteam_sims_cache.json) — no BioSTEAM/API reruns.
    - The frontier is recomputed and cross-checked against the v9 reference
      (data/reference_frontier_v9.json).

USAGE
    python case-studies/01-pareto-temperature-sweep/reproduce.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "case-studies" / "_shared"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import casestudy_style as style  # noqa: E402
from strap.solubility import get_solubility  # noqa: E402

DATA = _HERE / "data"
FIGURES = _HERE / "figures"
TEMP_STEP_C = 5.0


# ---------------------------------------------------------------------------
# Load replay data
# ---------------------------------------------------------------------------

def load_sequence_and_msp() -> tuple[list[dict], dict[int, list[tuple[float, float]]]]:
    """Return the DP sequence and per-step {temp -> MSP} from the cached sims."""
    cache = json.loads((DATA / "biosteam_sims_cache.json").read_text())
    sequence = cache["step_data_summary"]  # [{step_idx, polymer, solvent, dp_temp}]
    configs, index, results = cache["configs"], cache["config_index"], cache["results"]

    msp_by_step: dict[int, list[tuple[float, float]]] = {}
    for i, (step_idx, _temp_idx) in enumerate(index):
        result = results[i]
        if not result.get("success"):
            continue
        temp = float(configs[i]["dissolution_temperature_c"])
        msp = result.get("tea", {}).get("msp_usd_per_kg")
        if msp is None:
            continue
        msp_by_step.setdefault(step_idx, []).append((temp, float(msp)))
    for step_idx in msp_by_step:
        msp_by_step[step_idx].sort()
    return sequence, msp_by_step


# ---------------------------------------------------------------------------
# Recompute selectivity curves from v10's interpolation engine
# ---------------------------------------------------------------------------

def selectivity_at(target: str, others: list[str], solvent: str, temp_c: float) -> float | None:
    """Selectivity of `solvent` for `target` vs the max of `others` at temp.

    Uses v10's validated solubility engine. Returns None if the target
    solubility is unavailable (e.g. no data for that pair).
    """
    target_sol = get_solubility(target, solvent, temp_c, method="auto")
    if target_sol is None:
        return None
    other_sols = [
        s for s in (get_solubility(other, solvent, temp_c, method="auto") for other in others)
        if s is not None
    ]
    max_other = max(other_sols) if other_sols else 0.0
    return float(target_sol) - float(max_other)


_POLYMER_ALIASES = {"Nylon6": "NYLON6", "Nylon66": "NYLON66"}


def build_step_curves(sequence: list[dict], msp_by_step: dict) -> list[dict]:
    """For each step, compute selectivity vs temperature and attach MSP."""
    ordered_polymers = [step["polymer"] for step in sequence]
    step_curves: list[dict] = []
    for step in sequence:
        step_idx = step["step_idx"]
        target = _POLYMER_ALIASES.get(step["polymer"], step["polymer"])
        # Polymers still in the stream at this step are the ones not yet recovered.
        remaining = [
            _POLYMER_ALIASES.get(p, p) for p in ordered_polymers[step_idx + 1:]
        ]
        temps = [t for t, _ in msp_by_step.get(step_idx, [])]
        if not temps:
            temps = list(np.arange(25.0, 160.0 + 1e-9, TEMP_STEP_C))
        msp_lookup = dict(msp_by_step.get(step_idx, []))
        selectivities, msps = [], []
        for temp in temps:
            sel = selectivity_at(target, remaining, step["solvent"], temp)
            selectivities.append(sel)
            msps.append(msp_lookup.get(temp))
        # DP-optimal selectivity at the recorded dp_temp for annotation.
        dp_sel = selectivity_at(target, remaining, step["solvent"], float(step["dp_temp"]))
        step_curves.append({
            "step_idx": step_idx,
            "polymer": step["polymer"],
            "solvent": step["solvent"],
            "dp_temp": float(step["dp_temp"]),
            "dp_selectivity": dp_sel,
            "temp_grid": temps,
            "selectivities": selectivities,
            "msps": msps,
        })
    return step_curves


# ---------------------------------------------------------------------------
# Pareto frontier: cheapest temperatures meeting a minimum-selectivity floor
# ---------------------------------------------------------------------------

def compute_frontier(step_curves: list[dict], threshold_step: float = 0.5) -> list[dict]:
    costed = [s for s in step_curves if any(m is not None for m in s["msps"])
              and any(v is not None for v in s["selectivities"])]
    if not costed:
        return []
    all_sels = [v for s in costed for v in s["selectivities"] if v is not None]
    hi, lo = max(all_sels), min(all_sels)

    frontier = []
    for threshold in np.arange(hi, lo - 1, -threshold_step):
        temps, msps, sels = [], [], []
        feasible = True
        for s in costed:
            valid = [
                (t, v, m)
                for t, v, m in zip(s["temp_grid"], s["selectivities"], s["msps"])
                if v is not None and m is not None and v >= threshold
            ]
            if not valid:
                feasible = False
                break
            t, v, m = min(valid, key=lambda x: x[2])  # cheapest meeting the floor
            temps.append(t); sels.append(v); msps.append(m)
        if feasible:
            frontier.append({
                "threshold": float(threshold),
                "min_sel": float(min(sels)),
                "avg_msp": float(sum(msps) / len(msps)),
                "sum_msp": float(sum(msps)),
                "step_temps": temps,
                "step_sels": sels,
                "step_msps": msps,
            })
    # Deduplicate by (min_sel, avg_msp), keeping the highest threshold.
    seen, deduped = set(), []
    for point in frontier:
        key = (round(point["min_sel"], 2), round(point["avg_msp"], 4))
        if key not in seen:
            seen.add(key)
            deduped.append(point)
    return deduped


def knee_point(frontier: list[dict]) -> dict | None:
    if len(frontier) < 3:
        return None
    xs = np.array([p["min_sel"] for p in frontier])
    ys = np.array([p["avg_msp"] for p in frontier])
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]
    dx, dy = x1 - x0, y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        return None
    dist = np.abs(dy * xs - dx * ys + x1 * y0 - y1 * x0) / length
    return frontier[int(np.argmax(dist))]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def render_main_figure(step_curves, frontier, out_stem: Path) -> None:
    fig = plt.figure(figsize=(17, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.34, wspace=0.24)

    ax_sel = fig.add_subplot(grid[0, 0])
    ax_msp = fig.add_subplot(grid[1, 0])
    ax_par = fig.add_subplot(grid[:, 1])

    for i, s in enumerate(step_curves):
        color = style.SERIES_COLORS[i % len(style.SERIES_COLORS)]
        label = f"S{s['step_idx'] + 1}: {s['polymer']} / {s['solvent']}"
        ts = [t for t, v in zip(s["temp_grid"], s["selectivities"]) if v is not None]
        vs = [v for v in s["selectivities"] if v is not None]
        if ts:
            ax_sel.plot(ts, vs, "o-", color=color, ms=3, lw=1.6, label=label)
            ax_sel.axvline(s["dp_temp"], color=color, alpha=0.25, ls="--", lw=0.8)
        mt = [t for t, m in zip(s["temp_grid"], s["msps"]) if m is not None]
        mm = [m for m in s["msps"] if m is not None]
        if mt:
            ax_msp.plot(mt, mm, "s-", color=color, ms=3, lw=1.6, label=label)
            ax_msp.axvline(s["dp_temp"], color=color, alpha=0.25, ls="--", lw=0.8)

    ax_sel.set(xlabel="Dissolution temperature (°C)", ylabel="Selectivity (%)",
               title="Per-step selectivity vs temperature")
    ax_sel.legend(loc="upper left", ncol=2)
    ax_msp.set(xlabel="Dissolution temperature (°C)", ylabel="MSP ($/kg)",
               title="Per-step minimum selling price vs temperature")
    ax_msp.legend(loc="upper left", ncol=2)

    if frontier:
        xs = [p["min_sel"] for p in frontier]
        ys = [p["avg_msp"] for p in frontier]
        ax_par.fill_between(xs, ys, max(ys), alpha=0.08, color=style.FILL)
        ax_par.plot(xs, ys, "o-", color=style.FRONTIER, ms=4, lw=2, label="Pareto frontier")

        dp_optimal = frontier[0]
        cheapest = min(frontier, key=lambda p: p["avg_msp"])
        ax_par.plot(dp_optimal["min_sel"], dp_optimal["avg_msp"], "*",
                    color=style.ACCENT_OPTIMAL, ms=20, zorder=5,
                    label=f"Max-selectivity ({dp_optimal['min_sel']:.1f}%, ${dp_optimal['avg_msp']:.2f}/kg)")
        ax_par.plot(cheapest["min_sel"], cheapest["avg_msp"], "D",
                    color=style.ACCENT_CHEAPEST, ms=11, zorder=5,
                    label=f"Cheapest ({cheapest['min_sel']:.1f}%, ${cheapest['avg_msp']:.2f}/kg)")
        knee = knee_point(frontier)
        if knee:
            ax_par.plot(knee["min_sel"], knee["avg_msp"], "^",
                        color=style.ACCENT_KNEE, ms=13, zorder=5,
                        label=f"Knee ({knee['min_sel']:.1f}%, ${knee['avg_msp']:.2f}/kg)")
        ax_par.invert_xaxis()  # better selectivity to the left

    ax_par.set(xlabel="Guaranteed minimum selectivity across all steps (%)",
               ylabel="Average MSP across steps ($/kg)",
               title="Pareto frontier: selectivity vs cost")
    ax_par.legend(loc="upper right")

    sequence = " → ".join(s["polymer"] for s in step_curves)
    fig.suptitle(f"Temperature-resolved separation economics\n{sequence}",
                 fontsize=15, fontweight="bold", y=0.99, color=style.INK)
    style.caption(fig, "Selectivity recomputed live from v10 interpolation engine; "
                       "MSP replayed from 217 cached BioSTEAM simulations (no API calls).")

    FIGURES.mkdir(exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def render_step_detail(step_curves, out_stem: Path) -> None:
    n = len(step_curves)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows), squeeze=False)

    for i, s in enumerate(step_curves):
        ax = axes[divmod(i, ncols)]
        color = style.SERIES_COLORS[i % len(style.SERIES_COLORS)]
        ts = [t for t, v in zip(s["temp_grid"], s["selectivities"]) if v is not None]
        vs = [v for v in s["selectivities"] if v is not None]
        ax.plot(ts, vs, "o-", color=color, ms=3, lw=1.6)
        ax.set_ylabel("Selectivity (%)", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.axvline(s["dp_temp"], color="#888888", ls="--", lw=0.9)

        ax2 = ax.twinx()
        mt = [t for t, m in zip(s["temp_grid"], s["msps"]) if m is not None]
        mm = [m for m in s["msps"] if m is not None]
        ax2.plot(mt, mm, "s--", color="#555555", ms=3, lw=1.3, alpha=0.85)
        ax2.set_ylabel("MSP ($/kg)", color="#555555")
        ax2.tick_params(axis="y", labelcolor="#555555")
        ax2.grid(False)

        ax.set_title(f"S{s['step_idx'] + 1}: {s['polymer']} / {s['solvent']}", fontsize=10.5)
        ax.set_xlabel("Temperature (°C)")

    for j in range(n, nrows * ncols):
        axes[divmod(j, ncols)].axis("off")

    fig.suptitle("Per-step selectivity (colour) and MSP (grey) vs dissolution temperature",
                 fontsize=13, fontweight="bold", y=1.0, color=style.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-check against the v9 reference frontier
# ---------------------------------------------------------------------------

def cross_check(frontier: list[dict]) -> dict:
    reference = json.loads((DATA / "reference_frontier_v9.json").read_text())
    ref_points = reference.get("pareto_points", [])
    ref_msp_range = (
        (min(p["avg_msp"] for p in ref_points), max(p["avg_msp"] for p in ref_points))
        if ref_points else (None, None)
    )
    new_msp_range = (
        (min(p["avg_msp"] for p in frontier), max(p["avg_msp"] for p in frontier))
        if frontier else (None, None)
    )
    return {
        "reference_points": len(ref_points),
        "reproduced_points": len(frontier),
        "reference_avg_msp_range": ref_msp_range,
        "reproduced_avg_msp_range": new_msp_range,
    }


def main() -> None:
    style.apply_style()
    sequence, msp_by_step = load_sequence_and_msp()
    step_curves = build_step_curves(sequence, msp_by_step)
    frontier = compute_frontier(step_curves)

    render_main_figure(step_curves, frontier, FIGURES / "pareto_selectivity_vs_cost")
    render_step_detail(step_curves, FIGURES / "per_step_detail")

    check = cross_check(frontier)
    result = {
        "sequence": [s["polymer"] for s in step_curves],
        "n_steps": len(step_curves),
        "n_msp_simulations_replayed": sum(len(v) for v in msp_by_step.values()),
        "frontier_points": len(frontier),
        "cross_check_vs_v9": check,
        "figures": [str(p.relative_to(_ROOT)) for p in sorted(FIGURES.glob("*.png"))],
    }
    (_HERE / "data" / "reproduced_frontier.json").write_text(
        json.dumps({"frontier": frontier, "sequence": sequence}, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
