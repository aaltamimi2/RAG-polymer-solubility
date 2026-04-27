"""Pareto Temperature Optimization for Separation Sequences.

Starts from the DP-optimal separation sequence (max min-selectivity)
and sweeps dissolution temperature at each step to find the tradeoff
between selectivity (quality) and MSP (cost).

The polymer order and solvent assignments are FIXED — only temperature
varies per step. Produces:
  1. Per-step (temperature → selectivity, MSP) curves
  2. Global Pareto frontier (min_selectivity vs average MSP)

Usage:
    python architecture/pareto_temperature.py              # full run (~35 min)
    python architecture/pareto_temperature.py --sel-only    # selectivity sweep only (no BioSTEAM)
    python architecture/pareto_temperature.py --load CACHE  # reload cached BioSTEAM results
"""

import sys
import json
import time
import argparse
from pathlib import Path

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from strap.solubility import (
    get_all_solvents_selectivity,
    get_solubility,
    get_boiling_point,
)

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "dp_lattice_sweep"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────
STEP_COLORS = [
    "#e74c3c", "#e67e22", "#f1c40f", "#2ecc71",
    "#1abc9c", "#3498db", "#9b59b6", "#e84393",
]


# =====================================================================
# Step 1: DP with temperature sweep (self-contained)
# =====================================================================

def build_dp_temp_sweep(polymers, temp_range, banned_solvents=None):
    """Run bitmask DP over a temperature range, maximizing min-selectivity.

    For each (target, remaining_set) pair, picks the (solvent, temperature)
    with the highest selectivity across all temperatures in *temp_range*.

    Returns the same dict structure as the base build_dp_data, but
    sel_cache values are (solvent, temperature, selectivity) tuples.
    """
    n = len(polymers)
    full_mask = (1 << n) - 1
    temps = list(temp_range)
    _banned = {s.lower() for s in (banned_solvents or [])}

    print(f"  Precomputing selectivities for {n} polymers, "
          f"{len(temps)} temperatures ({temps[0]}-{temps[-1]}°C)...")
    t0 = time.time()

    # Precompute selectivities for ALL valid (target_idx, mask) pairs
    sel_cache = {}  # (target_idx, mask) -> (solvent, temp, selectivity)
    for tidx in range(n):
        for mask in range(1, 1 << n):
            if not (mask & (1 << tidx)):
                continue
            others_mask = mask ^ (1 << tidx)
            if others_mask == 0:
                continue
            target = polymers[tidx]
            others = [polymers[i] for i in range(n) if others_mask & (1 << i)]
            best = ("N/A", temps[0], 0.0)
            for t in temps:
                results = get_all_solvents_selectivity(target, others, t)
                if not results:
                    continue
                for r in results:
                    if r["solvent"].lower() in _banned:
                        continue
                    bp = get_boiling_point(r["solvent"])
                    if bp is not None and t > bp - 5:
                        continue
                    if r["selectivity"] > best[2]:
                        best = (r["solvent"], t, r["selectivity"])
                    break  # results sorted desc; first valid is best
            sel_cache[(tidx, mask)] = best

    dt = time.time() - t0
    print(f"  Precomputed {len(sel_cache):,} entries in {dt:.1f}s")

    # DP: dp[mask] = (min_selectivity, last_removed_idx, came_from_mask)
    dp = {}
    for i in range(n):
        rem = full_mask ^ (1 << i)
        _, _, sel = sel_cache.get((i, full_mask), ("N/A", 0.0, 0.0))
        if rem not in dp or sel > dp[rem][0]:
            dp[rem] = (sel, i, full_mask)

    for mask in range(full_mask - 1, -1, -1):
        if mask not in dp:
            continue
        cur_min = dp[mask][0]
        if mask == 0:
            continue
        pc = bin(mask).count("1")
        if pc == 1:
            idx = next(i for i in range(n) if mask & (1 << i))
            if 0 not in dp or cur_min > dp[0][0]:
                dp[0] = (cur_min, idx, mask)
            continue
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            new_mask = mask ^ (1 << i)
            _, _, sel = sel_cache.get((i, mask), ("N/A", 0.0, 0.0))
            new_min = min(cur_min, sel)
            if new_mask not in dp or new_min > dp[new_mask][0]:
                dp[new_mask] = (new_min, i, mask)

    # Reconstruct optimal path
    path = []
    cur = 0
    visited = set()
    while cur in dp and cur not in visited:
        visited.add(cur)
        _, ridx, came = dp[cur]
        solv, tmp, sel = sel_cache.get((ridx, came), ("N/A", 0.0, 0.0))
        path.append({
            "from_mask": came, "to_mask": cur,
            "removed_idx": ridx, "removed": polymers[ridx],
            "solvent": solv, "temperature": tmp, "selectivity": sel,
        })
        if came == full_mask:
            break
        cur = came
    path.reverse()

    return {
        "polymers": polymers, "n": n, "full_mask": full_mask,
        "sel_cache": sel_cache, "dp": dp, "path": path,
        "opt_min": dp.get(0, (0.0,))[0],
        "n_precomputed": len(sel_cache),
    }


# =====================================================================
# Step 2-3: Per-step temperature grids + selectivity
# =====================================================================

def compute_step_selectivities(path, polymers):
    """For each dissolution step, compute selectivity at every feasible temperature.

    Returns:
        list of dicts, one per dissolution step:
        {
            "step_idx": int,
            "polymer": str,
            "solvent": str,
            "dp_temp": float,        # DP-optimal temperature
            "dp_selectivity": float,  # selectivity at DP-optimal temp
            "others": list[str],      # remaining polymers at this step
            "temp_grid": list[float],
            "selectivities": list[float],  # selectivity at each temp
        }
    """
    n = len(polymers)
    steps = []

    for i, step in enumerate(path):
        from_mask = step["from_mask"]
        # Skip isolation step (last polymer remaining)
        if bin(from_mask).count("1") <= 1:
            continue

        target = step["removed"]
        solvent = step["solvent"]
        # Remaining polymers at this step (EXCLUDING the target)
        others = [polymers[j] for j in range(n)
                  if (from_mask & (1 << j)) and j != step["removed_idx"]]

        # Temperature bounds
        bp = get_boiling_point(solvent)
        t_lo = 25.0
        t_hi = min(bp - 5.0, 160.0) if bp else 160.0
        t_hi = max(t_hi, t_lo + 5)  # ensure at least one point

        temp_grid = list(np.arange(t_lo, t_hi + 2.5, 5.0))
        # Ensure we include the DP-optimal temp
        dp_temp = step["temperature"]
        if dp_temp not in temp_grid and t_lo <= dp_temp <= t_hi:
            temp_grid.append(dp_temp)
            temp_grid.sort()

        selectivities = []
        for t in temp_grid:
            target_sol = get_solubility(target, solvent, t)
            if target_sol is None or target_sol <= 0:
                selectivities.append(0.0)
                continue
            max_other = 0.0
            for other in others:
                other_sol = get_solubility(other, solvent, t)
                if other_sol is not None:
                    max_other = max(max_other, other_sol)
            selectivities.append(target_sol - max_other)

        steps.append({
            "step_idx": i,
            "polymer": target,
            "solvent": solvent,
            "dp_temp": dp_temp,
            "dp_selectivity": step["selectivity"],
            "others": others,
            "temp_grid": temp_grid,
            "selectivities": selectivities,
        })

    return steps


# =====================================================================
# Step 4-5: BioSTEAM batch execution
# =====================================================================

def build_pareto_configs(step_data_list):
    """Build BioSTEAM config dicts for all (step, temperature) combinations.

    Solvent names from the solubility database (lowercase interp keys like
    ``propyleneglycol``) are resolved to BioSTEAM-compatible names (like
    ``Propylene Glycol``) via the solvent registry.

    Returns:
        configs: list of config dicts
        config_index: list of (step_idx, temp_idx) tuples for result mapping
    """
    from strap.solvent_registry import resolve_to_biosteam
    from strap.vendor.biosteam_runner import (
        _SOLVENT_DEFAULTS, _csv_lookup, _curated_lookup,
    )

    configs = []
    config_index = []

    for sd in step_data_list:
        interp_key = sd["solvent"]  # lowercase interp key from solubility DB
        polymer = sd["polymer"]

        # Resolve to BioSTEAM-compatible name
        bst_name = resolve_to_biosteam(interp_key) or interp_key
        # Get solvent price from defaults (keyed by BioSTEAM name) or fallbacks
        defaults = _SOLVENT_DEFAULTS.get(bst_name)
        if defaults:
            price = defaults[0]
        else:
            curated = _curated_lookup(bst_name)
            if curated and curated.get("price") is not None:
                price = curated["price"]
            else:
                price = 1.50

        for ti, temp in enumerate(sd["temp_grid"]):
            cfg = {
                "solvent": bst_name,
                "target_plastic": polymer,
                "energy_case": "C1",
                "dissolution_temperature_c": temp,
                "solvent_price": price,
                "precipitation_temperature_c": 25.0,
                "solvent_loss_pct": 0.01,
                "feedstock_distance_km": 0.0,
                "processing_capacity": 20_000,
            }
            configs.append(cfg)
            config_index.append((sd["step_idx"], ti))

    return configs, config_index


def run_all_sims(configs, max_parallel=3, timeout=120):
    """Run all BioSTEAM simulations."""
    from strap.vendor.biosteam_runner import run_batch_simulations
    return run_batch_simulations(configs, max_parallel=max_parallel,
                                timeout_per_sim=timeout)


# =====================================================================
# Step 6: Merge results into step curves
# =====================================================================

def build_step_curves(step_data_list, configs, config_index, results):
    """Attach MSP results to step data.

    Returns updated step_data_list where each step gets a "msps" list
    aligned with temp_grid.
    """
    # Pre-fill MSP arrays with None
    for sd in step_data_list:
        sd["msps"] = [None] * len(sd["temp_grid"])

    # Map results back to steps
    for (si, ti), result in zip(config_index, results):
        sd = next(s for s in step_data_list if s["step_idx"] == si)
        if result.get("success"):
            msp = result.get("tea", {}).get("msp_usd_per_kg")
            sd["msps"][ti] = msp

    return step_data_list


# =====================================================================
# Step 7: Pareto frontier
# =====================================================================

def compute_pareto_frontier(step_data_list, threshold_step=0.5):
    """Sweep min-selectivity threshold and find cheapest temperature config.

    For each threshold:
      - At each step, pick the lowest-MSP temperature with selectivity >= threshold
      - Sum per-step MSPs to get total MSP

    Steps with NO valid MSP results (e.g. BioSTEAM failures) are excluded
    from the cost computation but their selectivity still constrains the
    min-selectivity metric.

    Returns:
        list of dicts: [{"threshold", "avg_msp", "max_msp", "sum_msp",
                         "step_temps", "step_msps", "step_sels"}, ...]
    """
    # Filter to steps with at least one valid MSP
    costed_steps = [sd for sd in step_data_list
                    if any(m is not None for m in (sd.get("msps") or []))]
    # Steps with no MSP data — we still use their selectivity as a constraint
    uncosted_steps = [sd for sd in step_data_list
                      if not any(m is not None for m in (sd.get("msps") or []))]

    if uncosted_steps:
        names = [f"{sd['polymer']}/{sd['solvent']}" for sd in uncosted_steps]
        print(f"  WARNING: {len(uncosted_steps)} step(s) have no MSP data "
              f"(excluded from cost): {', '.join(names)}")
        # For uncosted steps, use the DP-optimal selectivity as a fixed constraint
        # (we can't optimize their temperature since we have no cost data)
        fixed_sels = [sd["dp_selectivity"] for sd in uncosted_steps]
    else:
        fixed_sels = []

    if not costed_steps:
        print("  ERROR: no steps have valid MSP data — cannot build Pareto frontier")
        return []

    # Find the global selectivity range from costed steps
    all_sels = []
    for sd in costed_steps:
        all_sels.extend(sd["selectivities"])
    max_sel = max(all_sels) if all_sels else 0
    min_sel = min(all_sels) if all_sels else 0

    pareto = []
    thresholds = np.arange(max_sel + 1, min_sel - 1, -threshold_step)

    for threshold in thresholds:
        step_temps = []
        step_msps = []
        step_sels = []
        feasible = True

        for sd in costed_steps:
            # Find valid (temp, sel, msp) points at this threshold
            valid = []
            for j, (t, sel, msp) in enumerate(
                zip(sd["temp_grid"], sd["selectivities"], sd["msps"])
            ):
                if sel >= threshold and msp is not None:
                    valid.append((t, sel, msp))

            if not valid:
                feasible = False
                break

            # Pick cheapest
            best = min(valid, key=lambda x: x[2])
            step_temps.append(best[0])
            step_sels.append(best[1])
            step_msps.append(best[2])

        if feasible:
            n_steps = len(costed_steps)
            # Include fixed selectivities from uncosted steps in min-sel
            all_step_sels = step_sels + fixed_sels
            pareto.append({
                "threshold": float(threshold),
                "min_sel": min(all_step_sels),
                "avg_msp": sum(step_msps) / n_steps,
                "max_msp": max(step_msps),
                "sum_msp": sum(step_msps),
                "step_temps": step_temps,
                "step_msps": step_msps,
                "step_sels": step_sels,
            })

    # Deduplicate by (min_sel, avg_msp) — keep first (highest threshold)
    seen = set()
    deduped = []
    for p in pareto:
        key = (round(p["min_sel"], 2), round(p["avg_msp"], 4))
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    return deduped


# =====================================================================
# Step 8: Visualization
# =====================================================================

def plot_pareto(step_data_list, pareto, dp_path, output_path):
    """Two-panel figure: per-step curves (left) + global Pareto (right)."""

    n_steps = len(step_data_list)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.35, wspace=0.30)

    # ── Panel A (top-left): Per-step selectivity vs temperature ──
    ax_sel = fig.add_subplot(gs[0, 0])
    for i, sd in enumerate(step_data_list):
        color = STEP_COLORS[i % len(STEP_COLORS)]
        label = f"S{i+1}: {sd['polymer']} / {sd['solvent']}"
        ax_sel.plot(sd["temp_grid"], sd["selectivities"],
                    "o-", color=color, markersize=3, linewidth=1.5, label=label)
        # Mark DP-optimal
        ax_sel.axvline(sd["dp_temp"], color=color, alpha=0.3, linestyle="--",
                       linewidth=0.8)
    ax_sel.set_xlabel("Dissolution Temperature (°C)", fontsize=10)
    ax_sel.set_ylabel("Selectivity (%)", fontsize=10)
    ax_sel.set_title("Per-Step Selectivity vs Temperature", fontsize=12,
                     fontweight="bold")
    ax_sel.legend(fontsize=7, loc="upper left", ncol=2)
    ax_sel.grid(True, alpha=0.3)

    # ── Panel B (bottom-left): Per-step MSP vs temperature ──
    ax_msp = fig.add_subplot(gs[1, 0])
    for i, sd in enumerate(step_data_list):
        if sd.get("msps") is None:
            continue
        color = STEP_COLORS[i % len(STEP_COLORS)]
        valid_t = [t for t, m in zip(sd["temp_grid"], sd["msps"]) if m is not None]
        valid_m = [m for m in sd["msps"] if m is not None]
        if not valid_t:
            continue
        label = f"S{i+1}: {sd['polymer']} / {sd['solvent']}"
        ax_msp.plot(valid_t, valid_m, "s-", color=color, markersize=3,
                    linewidth=1.5, label=label)
        ax_msp.axvline(sd["dp_temp"], color=color, alpha=0.3, linestyle="--",
                       linewidth=0.8)
    ax_msp.set_xlabel("Dissolution Temperature (°C)", fontsize=10)
    ax_msp.set_ylabel("MSP ($/kg)", fontsize=10)
    ax_msp.set_title("Per-Step MSP vs Temperature", fontsize=12, fontweight="bold")
    ax_msp.legend(fontsize=7, loc="upper left", ncol=2)
    ax_msp.grid(True, alpha=0.3)

    # ── Panel C (right, spanning both rows): Global Pareto frontier ──
    ax_p = fig.add_subplot(gs[:, 1])

    if pareto:
        min_sels = [p["min_sel"] for p in pareto]
        avg_msps = [p["avg_msp"] for p in pareto]

        ax_p.plot(min_sels, avg_msps, "o-", color="#2c3e50", markersize=4,
                  linewidth=2, label="Pareto frontier")

        # Mark the DP-optimal point (highest min-sel)
        dp_point = pareto[0]  # highest threshold = DP optimal
        ax_p.plot(dp_point["min_sel"], dp_point["avg_msp"], "*",
                  color="#e74c3c", markersize=18, zorder=5,
                  label=f"DP optimal ({dp_point['min_sel']:.1f}%, "
                        f"${dp_point['avg_msp']:.2f}/kg)")

        # Mark the cheapest feasible point
        cheapest = min(pareto, key=lambda p: p["avg_msp"])
        ax_p.plot(cheapest["min_sel"], cheapest["avg_msp"], "D",
                  color="#27ae60", markersize=12, zorder=5,
                  label=f"Cheapest ({cheapest['min_sel']:.1f}%, "
                        f"${cheapest['avg_msp']:.2f}/kg)")

        # Find knee point (max distance from line connecting endpoints)
        if len(pareto) >= 3:
            x0, y0 = min_sels[0], avg_msps[0]
            x1, y1 = min_sels[-1], avg_msps[-1]
            dx, dy = x1 - x0, y1 - y0
            line_len = np.sqrt(dx**2 + dy**2)
            if line_len > 0:
                dists = [
                    abs(dy * p["min_sel"] - dx * p["avg_msp"]
                        + x1 * y0 - y1 * x0) / line_len
                    for p in pareto
                ]
                knee_idx = np.argmax(dists)
                knee = pareto[knee_idx]
                ax_p.plot(knee["min_sel"], knee["avg_msp"], "^",
                          color="#f39c12", markersize=12, zorder=5,
                          label=f"Knee ({knee['min_sel']:.1f}%, "
                                f"${knee['avg_msp']:.2f}/kg)")

        # Shade the Pareto improvement region
        ax_p.fill_between(
            min_sels, avg_msps,
            [max(avg_msps)] * len(avg_msps),
            alpha=0.08, color="#3498db",
        )

    ax_p.set_xlabel("Min Selectivity Across All Steps (%)", fontsize=11)
    ax_p.set_ylabel("Average MSP Across Steps ($/kg)", fontsize=11)
    ax_p.set_title("Pareto Frontier: Selectivity vs Cost", fontsize=13,
                   fontweight="bold")
    ax_p.legend(fontsize=9, loc="upper right")
    ax_p.grid(True, alpha=0.3)
    ax_p.invert_xaxis()  # higher selectivity (better) on left

    # ── Sequence info annotation ──
    seq_str = " → ".join(s["polymer"] for s in step_data_list)
    fig.suptitle(
        f"Pareto Temperature Optimization\n{seq_str}",
        fontsize=14, fontweight="bold", y=0.98, color="#2c3e50",
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved: {output_path}")


def plot_step_detail(step_data_list, output_path):
    """Detailed per-step subplots with dual y-axes (selectivity + MSP)."""
    n_steps = len(step_data_list)
    ncols = min(4, n_steps)
    nrows = (n_steps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for i, sd in enumerate(step_data_list):
        row, col = divmod(i, ncols)
        ax1 = axes[row, col]
        color_sel = "#3498db"
        color_msp = "#e74c3c"

        # Selectivity
        ax1.plot(sd["temp_grid"], sd["selectivities"], "o-",
                 color=color_sel, markersize=4, linewidth=1.5)
        ax1.set_xlabel("Temperature (°C)", fontsize=9)
        ax1.set_ylabel("Selectivity (%)", color=color_sel, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=color_sel)
        ax1.axvline(sd["dp_temp"], color="gray", linestyle="--", alpha=0.5,
                    label=f"DP opt: {sd['dp_temp']}°C")

        # MSP (right axis)
        if sd.get("msps"):
            ax2 = ax1.twinx()
            valid_t = [t for t, m in zip(sd["temp_grid"], sd["msps"])
                       if m is not None]
            valid_m = [m for m in sd["msps"] if m is not None]
            if valid_t:
                ax2.plot(valid_t, valid_m, "s-", color=color_msp,
                         markersize=4, linewidth=1.5)
                ax2.set_ylabel("MSP ($/kg)", color=color_msp, fontsize=9)
                ax2.tick_params(axis="y", labelcolor=color_msp)

        ax1.set_title(
            f"Step {i+1}: {sd['polymer']}\n{sd['solvent']}",
            fontsize=10, fontweight="bold",
        )
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.2)

    # Hide unused subplots
    for i in range(n_steps, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Per-Step Temperature Response", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Pareto temperature optimization")
    parser.add_argument("--sel-only", action="store_true",
                        help="Only compute selectivity (skip BioSTEAM)")
    parser.add_argument("--load", type=str, default=None,
                        help="Load cached BioSTEAM results from JSON")
    parser.add_argument("--save-cache", type=str, default=None,
                        help="Save BioSTEAM results to JSON for reuse")
    parser.add_argument("--workers", type=int, default=3,
                        help="Max parallel BioSTEAM workers (default 3)")
    args = parser.parse_args()

    # ── Step 1: Extract optimal sequence ──
    polymers = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "Nylon6", "Nylon66", "PET"]
    temp_range = range(25, 165, 5)

    print("=" * 70)
    print("STEP 1: Building DP (temp sweep 25-160°C, benzene banned)")
    print("=" * 70)
    t0 = time.time()
    data = build_dp_temp_sweep(polymers, temp_range, banned_solvents={"benzene"})
    dt = time.time() - t0
    print(f"  DP build time: {dt:.1f}s")
    print(f"  Optimal min-selectivity: {data['opt_min']:.1f}%")

    path = data["path"]
    print(f"\n  Optimal sequence ({len(path)} steps):")
    for i, step in enumerate(path):
        is_iso = bin(step["from_mask"]).count("1") <= 1
        if is_iso:
            print(f"    Step {i+1}: {step['removed']:>8}  (isolation)")
        else:
            print(f"    Step {i+1}: {step['removed']:>8}  "
                  f"← {step['solvent']:<20} @ {step['temperature']:>5.0f}°C  "
                  f"sel={step['selectivity']:>6.1f}%")

    # ── Step 2-3: Per-step selectivity curves ──
    print(f"\n{'=' * 70}")
    print("STEP 2-3: Computing per-step selectivity curves")
    print("=" * 70)
    step_data = compute_step_selectivities(path, polymers)
    print(f"  {len(step_data)} dissolution steps (excluding isolation)")
    for sd in step_data:
        n_temps = len(sd["temp_grid"])
        sel_range = (min(sd["selectivities"]), max(sd["selectivities"]))
        print(f"    Step {sd['step_idx']+1}: {sd['polymer']}/{sd['solvent']}  "
              f"{n_temps} temps ({sd['temp_grid'][0]:.0f}-{sd['temp_grid'][-1]:.0f}°C)  "
              f"sel range: {sel_range[0]:.1f}–{sel_range[1]:.1f}%")

    if args.sel_only:
        print("\n  --sel-only: skipping BioSTEAM simulations")
        # Still plot selectivity-only figure
        out_sel = str(OUTPUT_DIR / "pareto_selectivity_only.png")
        # Create a simple selectivity-only plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, sd in enumerate(step_data):
            color = STEP_COLORS[i % len(STEP_COLORS)]
            label = f"S{i+1}: {sd['polymer']} / {sd['solvent']}"
            ax.plot(sd["temp_grid"], sd["selectivities"], "o-",
                    color=color, markersize=4, linewidth=1.5, label=label)
            ax.axvline(sd["dp_temp"], color=color, alpha=0.3, linestyle="--")
        ax.set_xlabel("Dissolution Temperature (°C)")
        ax.set_ylabel("Selectivity (%)")
        ax.set_title("Per-Step Selectivity vs Temperature (no BioSTEAM)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.savefig(out_sel, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Saved: {out_sel}")
        return

    # ── Step 4-5: BioSTEAM simulations ──
    configs, config_index = build_pareto_configs(step_data)
    print(f"\n{'=' * 70}")
    print(f"STEP 4-5: Running {len(configs)} BioSTEAM simulations")
    print("=" * 70)

    if args.load:
        print(f"  Loading cached results from {args.load}")
        with open(args.load) as f:
            cache = json.load(f)
        results = cache["results"]
        print(f"  Loaded {len(results)} results")
    else:
        t1 = time.time()
        results = run_all_sims(configs, max_parallel=args.workers)
        dt1 = time.time() - t1
        n_ok = sum(1 for r in results if r.get("success"))
        print(f"  Completed: {n_ok}/{len(results)} succeeded in {dt1:.1f}s")

        # Save cache
        cache_path = args.save_cache or str(
            OUTPUT_DIR / "pareto_biosteam_cache.json"
        )
        with open(cache_path, "w") as f:
            json.dump({
                "configs": configs,
                "config_index": config_index,
                "results": results,
                "step_data_summary": [
                    {"step_idx": sd["step_idx"], "polymer": sd["polymer"],
                     "solvent": sd["solvent"], "dp_temp": sd["dp_temp"]}
                    for sd in step_data
                ],
            }, f, indent=2, default=str)
        print(f"  Cached results to {cache_path}")

    # ── Step 6: Build step curves ──
    print(f"\n{'=' * 70}")
    print("STEP 6: Building per-step (temperature, selectivity, MSP) curves")
    print("=" * 70)
    step_data = build_step_curves(step_data, configs, config_index, results)
    for sd in step_data:
        valid_msps = [m for m in sd["msps"] if m is not None]
        if valid_msps:
            msp_range = (min(valid_msps), max(valid_msps))
            print(f"    Step {sd['step_idx']+1}: {sd['polymer']}/{sd['solvent']}  "
                  f"MSP range: ${msp_range[0]:.3f}–${msp_range[1]:.3f}/kg  "
                  f"({len(valid_msps)}/{len(sd['msps'])} sims OK)")
        else:
            print(f"    Step {sd['step_idx']+1}: {sd['polymer']}/{sd['solvent']}  "
                  f"NO valid MSP results!")

    # ── Step 7: Pareto frontier ──
    print(f"\n{'=' * 70}")
    print("STEP 7: Computing Pareto frontier")
    print("=" * 70)
    pareto = compute_pareto_frontier(step_data)
    print(f"  {len(pareto)} Pareto points computed")
    if pareto:
        best_sel = pareto[0]
        cheapest = min(pareto, key=lambda p: p["avg_msp"])
        print(f"  DP-optimal point:  min_sel={best_sel['min_sel']:.1f}%  "
              f"avg_MSP=${best_sel['avg_msp']:.3f}/kg")
        print(f"  Cheapest point:    min_sel={cheapest['min_sel']:.1f}%  "
              f"avg_MSP=${cheapest['avg_msp']:.3f}/kg")
        savings = best_sel["avg_msp"] - cheapest["avg_msp"]
        sel_loss = best_sel["min_sel"] - cheapest["min_sel"]
        print(f"  Potential saving:  ${savings:.3f}/kg "
              f"for {sel_loss:.1f}% selectivity loss")

    # ── Step 8: Visualization ──
    print(f"\n{'=' * 70}")
    print("STEP 8: Generating plots")
    print("=" * 70)

    out_main = str(OUTPUT_DIR / "pareto_temperature.png")
    plot_pareto(step_data, pareto, path, out_main)

    out_detail = str(OUTPUT_DIR / "pareto_step_detail.png")
    plot_step_detail(step_data, out_detail)

    # Save Pareto data as JSON
    pareto_json = str(OUTPUT_DIR / "pareto_frontier.json")
    with open(pareto_json, "w") as f:
        json.dump({
            "sequence": [
                {"step": i+1, "polymer": sd["polymer"], "solvent": sd["solvent"],
                 "dp_temp": sd["dp_temp"], "dp_selectivity": sd["dp_selectivity"]}
                for i, sd in enumerate(step_data)
            ],
            "pareto_points": pareto,
        }, f, indent=2)
    print(f"  Saved: {pareto_json}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
