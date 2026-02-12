"""Visualize the Dynamic Programming (Bitmask DP) separation algorithm.

Shows the bitmask state-space lattice, selectivity-colored transitions,
and the optimal path. Uses 4 polymers for visual clarity.
Same aesthetic as plot_greedy_algorithm.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv
load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

from strap.solubility import get_all_solvents_selectivity


# ── Color helpers (matching greedy style) ─────────────────────────

def sel_color(sel):
    if sel is None:
        return "#95a5a6"
    if sel > 30:
        return "#27ae60"
    if sel > 10:
        return "#f39c12"
    if sel > 0:
        return "#e67e22"
    return "#e74c3c"


# ── Build DP data with full state tracking ────────────────────────

def build_dp_data(polymers, temperature=120.0):
    """Run bitmask DP, recording all selectivities and optimal path."""
    n = len(polymers)
    full_mask = (1 << n) - 1

    # Precompute selectivities for ALL valid (target_idx, mask) pairs
    sel_cache = {}  # (target_idx, mask) -> (solvent, selectivity)
    for tidx in range(n):
        for mask in range(1, 1 << n):
            if not (mask & (1 << tidx)):
                continue
            others_mask = mask ^ (1 << tidx)
            if others_mask == 0:
                continue
            target = polymers[tidx]
            others = [polymers[i] for i in range(n) if others_mask & (1 << i)]
            results = get_all_solvents_selectivity(target, others, temperature)
            if results:
                sel_cache[(tidx, mask)] = (results[0]["solvent"], results[0]["selectivity"])
            else:
                sel_cache[(tidx, mask)] = ("N/A", 0.0)

    # DP: dp[mask] = (min_selectivity, last_removed_idx, came_from_mask)
    INF = float("inf")
    dp = {}

    # Initialize: remove one polymer from the full set
    for i in range(n):
        rem = full_mask ^ (1 << i)
        _, sel = sel_cache.get((i, full_mask), ("N/A", 0.0))
        if rem not in dp or sel > dp[rem][0]:
            dp[rem] = (sel, i, full_mask)

    # Fill table: process masks in decreasing order
    for mask in range(full_mask - 1, -1, -1):
        if mask not in dp:
            continue
        cur_min = dp[mask][0]
        if mask == 0:
            continue
        pc = bin(mask).count("1")
        if pc == 1:
            # Last polymer remaining: isolation step (no selectivity needed)
            idx = next(i for i in range(n) if mask & (1 << i))
            if 0 not in dp or cur_min > dp[0][0]:
                dp[0] = (cur_min, idx, mask)
            continue
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            new_mask = mask ^ (1 << i)
            _, sel = sel_cache.get((i, mask), ("N/A", 0.0))
            new_min = min(cur_min, sel)
            if new_mask not in dp or new_min > dp[new_mask][0]:
                dp[new_mask] = (new_min, i, mask)

    # Reconstruct optimal path (reverse then flip)
    path = []
    cur = 0
    visited = set()
    while cur in dp and cur not in visited:
        visited.add(cur)
        _, ridx, came = dp[cur]
        solv, sel = sel_cache.get((ridx, came), ("N/A", 0.0))
        path.append({
            "from_mask": came, "to_mask": cur,
            "removed_idx": ridx, "removed": polymers[ridx],
            "solvent": solv, "selectivity": sel,
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


# ── Lattice plot ──────────────────────────────────────────────────

def plot_dp_lattice(data, output_path):
    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    dp = data["dp"]
    sel_cache = data["sel_cache"]
    path = data["path"]

    def mask_label(mask):
        names = [polymers[i] for i in range(n) if mask & (1 << i)]
        return ", ".join(names) if names else "∅"

    # Layout: diamond lattice
    lvl_y = 3.2   # vertical spacing between levels
    node_sp = 2.8  # horizontal spacing between nodes at same level

    positions = {}
    for level in range(n + 1):
        nodes = sorted(m for m in range(1 << n) if bin(m).count("1") == level)
        cnt = len(nodes)
        for i, mask in enumerate(nodes):
            x = (i - (cnt - 1) / 2) * node_sp
            y = level * lvl_y
            positions[mask] = (x, y)

    # Optimal edges & nodes
    opt_edges = {(e["from_mask"], e["to_mask"]) for e in path}
    opt_nodes = set()
    for e in path:
        opt_nodes.add(e["from_mask"])
        opt_nodes.add(e["to_mask"])

    # Edge selectivities for non-optimal edges (for coloring)
    def edge_sel(parent_mask, child_mask):
        """Get the selectivity for removing a polymer from parent to child."""
        diff = parent_mask ^ child_mask
        ridx = next(i for i in range(n) if diff & (1 << i))
        key = (ridx, parent_mask)
        if key in sel_cache:
            return sel_cache[key][1]
        return 0.0

    # Figure sizing
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad_x, pad_y = 3.0, 2.5
    fig_w = max(14, max(xs) - min(xs) + 2 * pad_x)
    fig_h = max(ys) - min(ys) + 2 * pad_y + 4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y - 1, max(ys) + pad_y + 2)
    ax.axis("off")

    # ── Title ──
    top = max(ys) + pad_y + 0.8
    ax.text(0, top + 0.4,
            "DYNAMIC PROGRAMMING (BITMASK DP) SEPARATOR",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="#2c3e50")
    n_lookups = data["n_precomputed"]
    n_factorial = int(np.prod(range(1, n + 1)))
    total_ops = n**2 * (1 << n) * 32  # n²·2ⁿ·s (s=32 solvents)
    ax.text(0, top - 0.3,
            f"{n} polymers | {n_lookups} selectivity lookups (n·2ⁿ⁻¹) | "
            f"O(n²·2ⁿ) ≈ {total_ops:,} total operations | "
            f"Optimal (vs {n_factorial} orderings)",
            ha="center", va="center", fontsize=15, color="#2c3e50")

    # ── Draw non-optimal edges ──
    for level in range(1, n + 1):
        for mask in range(1 << n):
            if bin(mask).count("1") != level:
                continue
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                child = mask ^ (1 << i)
                if (mask, child) in opt_edges:
                    continue
                x1, y1 = positions[mask]
                x2, y2 = positions[child]
                sel = edge_sel(mask, child)
                c = sel_color(sel)
                ax.plot([x1, x2], [y1, y2], "-",
                        color=c, linewidth=0.7, alpha=0.25, zorder=1)

    # ── Draw optimal path edges (thick, labeled) ──
    for edge in path:
        fm, tm = edge["from_mask"], edge["to_mask"]
        x1, y1 = positions[fm]
        x2, y2 = positions[tm]
        sel = edge["selectivity"]
        c = sel_color(sel)

        # Thick colored line
        ax.annotate("", xy=(x2, y2 + 0.42), xytext=(x1, y1 - 0.42),
                     arrowprops=dict(arrowstyle="-|>", lw=3.0, color=c,
                                     mutation_scale=18),
                     zorder=3)

        # Edge label: polymer removed + solvent + selectivity
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = x2 - x1
        off_x = 0.8 if dx >= 0 else -0.8
        ha_align = "left" if off_x > 0 else "right"
        is_isolation = edge["selectivity"] == 0.0 and edge["solvent"] == "N/A"
        if is_isolation:
            label = f"−{edge['removed']}\n(isolated)"
        else:
            label = f"−{edge['removed']}\n{edge['solvent']} ({sel:.1f}%)"
        ax.text(mx + off_x, my, label, ha=ha_align, va="center",
                fontsize=13, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c,
                          alpha=0.92, lw=1.2),
                zorder=6)

    # ── Draw nodes ──
    r = 0.60
    for mask, (x, y) in positions.items():
        is_opt = mask in opt_nodes
        is_full = mask == full_mask
        is_empty = mask == 0

        if is_full:
            fc, ec, tc = "#3498db", "#2c3e50", "white"
        elif is_empty:
            fc, ec, tc = "#2ecc71", "#2c3e50", "white"
        elif is_opt:
            fc, ec, tc = "#ffeaa7", "#f39c12", "#2c3e50"
        else:
            fc, ec, tc = "#f5f6fa", "#dcdde1", "#95a5a6"

        lw = 2.5 if (is_opt or is_full or is_empty) else 0.8
        circle = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                             linewidth=lw, zorder=4)
        ax.add_patch(circle)

        lbl = mask_label(mask)
        fs = 9 if len(lbl) > 15 else (10 if len(lbl) > 10 else 12)
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=fs, fontweight="bold" if is_opt else "normal",
                color=tc, zorder=5)

    # ── Level labels (right side) ──
    rx = max(xs) + pad_x - 0.8
    for level in range(n + 1):
        ly = level * lvl_y
        if level == n:
            label = f"FULL SET\n({level} polymers)"
        elif level == 0:
            label = "ALL SEPARATED\n(0 remaining)"
        else:
            from math import comb
            label = f"{comb(n, level)} states\n({level} remaining)"
        ax.text(rx, ly, label, ha="center", va="center",
                fontsize=13, color="#2c3e50", style="italic")

    # ── Result box at bottom ──
    seq_str = " → ".join(e["removed"] for e in path)
    ry = min(ys) - pad_y
    rw = max(xs) - min(xs) + 4
    result = FancyBboxPatch((-rw / 2, ry - 0.35), rw, 0.7,
                             boxstyle="round,pad=0.12",
                             facecolor="#2ecc71", edgecolor="#2c3e50",
                             linewidth=2.5)
    ax.add_patch(result)
    ax.text(0, ry,
            f"OPTIMAL SEQUENCE: {seq_str}  |  "
            f"Min selectivity: {data['opt_min']:.1f}%",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white")

    # ── Legend ──
    items = [
        mpatches.Patch(fc="#3498db", ec="black", label="Start (full mixture)"),
        mpatches.Patch(fc="#ffeaa7", ec="#f39c12", label="Optimal path state"),
        mpatches.Patch(fc="#f5f6fa", ec="#dcdde1", label="Other state"),
        mpatches.Patch(fc="#2ecc71", ec="black", label="Goal (all separated)"),
        mpatches.Patch(fc="#27ae60", ec="black", label="Selectivity > 30%"),
        mpatches.Patch(fc="#f39c12", ec="black", label="Selectivity 10–30%"),
        mpatches.Patch(fc="#e67e22", ec="black", label="Selectivity 0–10%"),
    ]
    ax.legend(handles=items, loc="upper left", fontsize=12,
              frameon=True, fancybox=True, edgecolor="#bdc3c7",
              title="Legend", title_fontsize=14,
              bbox_to_anchor=(0.0, 1.0))

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    HERE = Path(__file__).parent
    poly4 = ["PS", "PVC", "LDPE", "HDPE"]

    print("Building DP data for 4 polymers...")
    data = build_dp_data(poly4, temperature=120.0)
    print(f"  Selectivity lookups: {data['n_precomputed']} (n·2^(n-1))")
    np_ = len(poly4)
    print(f"  Total operations: ~{np_**2 * (1 << np_) * 32:,} (n²·2ⁿ·solvents)")
    print(f"  DP states filled: {len(data['dp'])}")
    print(f"  Optimal min selectivity: {data['opt_min']:.1f}%")
    print(f"  Optimal path: {' → '.join(e['removed'] for e in data['path'])}")

    out = HERE / "dp_algorithm_visual.png"
    plot_dp_lattice(data, str(out))
