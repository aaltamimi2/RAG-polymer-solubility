"""LEGACY: Academic visualization only. Not used by agent runtime.

Visualize the Dynamic Programming (Bitmask DP) separation algorithm.

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


# ── GSK G-score loader ────────────────────────────────────────────

def _load_gsk_scores():
    """Load GSK G-scores into a {lowercase_name: score} dict."""
    import csv
    gsk_path = Path(__file__).resolve().parent.parent / "data" / "GSK_dataset.csv"
    scores = {}
    with open(gsk_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("solvent_common_name", "").strip().lower()
            try:
                scores[name] = float(row["G-score"])
            except (ValueError, KeyError):
                pass
    # Abbreviation mappings for solubility model solvent names
    _ABBREV = {
        "dmf": "dimethylformamide", "thf": "tetrahydrofuran",
        "dcm": "dichloromethane", "ch2cl2": "dichloromethane",
        "chcl3": "chloroform", "meoh": "methanol", "etoh": "ethanol",
        "1,2-dimethylbenzene": "o-xylene", "1,4-dimethylbenzene": "p-xylene",
        "n-heptane": "heptane", "n-hexane": "hexane",
        "glycol": "ethylene glycol", "h2o": "water",
        "propanone": "acetone", "butanone": "methyl ethyl ketone",
        "ethylacetate": "ethyl acetate", "dimethylsulfoxide": "dimethyl sulfoxide",
        "dimethylformamide": "n,n-dimethylformamide",
        "acetylacetone": "2,4-pentanedione",
    }
    for abbr, full in _ABBREV.items():
        if full in scores and abbr not in scores:
            scores[abbr] = scores[full]
    return scores


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


# ── Safety-optimized DP ──────────────────────────────────────────

def build_dp_data_safety(polymers, temperature=120.0, min_sel=5.0):
    """Run bitmask DP optimizing for GSK safety (maximize min G-score).

    At each step, picks the solvent with the highest G-score among those
    with selectivity >= *min_sel*. Falls back to best-selectivity solvent
    if no safe option meets the threshold.
    """
    gsk = _load_gsk_scores()
    n = len(polymers)
    full_mask = (1 << n) - 1

    # Precompute: for each (target_idx, mask), pick safest viable solvent
    sel_cache = {}   # (tidx, mask) -> (solvent, selectivity, gscore)
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
            if not results:
                sel_cache[(tidx, mask)] = ("N/A", 0.0, 0.0)
                continue
            # Annotate with G-scores
            for r in results:
                r["gscore"] = gsk.get(r["solvent"].lower(), 0.0)
            # Filter: selectivity >= threshold AND has a G-score
            viable = [r for r in results
                      if r["selectivity"] >= min_sel and r["gscore"] > 0]
            if not viable:
                viable = [r for r in results if r["gscore"] > 0]
            if not viable:
                viable = results[:1]  # fallback to best selectivity
            best = max(viable, key=lambda r: r["gscore"])
            sel_cache[(tidx, mask)] = (
                best["solvent"], best["selectivity"], best["gscore"])

    # DP: maximize min G-score along path
    INF = float("inf")
    dp = {}

    for i in range(n):
        rem = full_mask ^ (1 << i)
        _, sel, gs = sel_cache.get((i, full_mask), ("N/A", 0.0, 0.0))
        if rem not in dp or gs > dp[rem][0]:
            dp[rem] = (gs, i, full_mask)

    for mask in range(full_mask - 1, -1, -1):
        if mask not in dp:
            continue
        cur_min_gs = dp[mask][0]
        if mask == 0:
            continue
        pc = bin(mask).count("1")
        if pc == 1:
            idx = next(i for i in range(n) if mask & (1 << i))
            if 0 not in dp or cur_min_gs > dp[0][0]:
                dp[0] = (cur_min_gs, idx, mask)
            continue
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            new_mask = mask ^ (1 << i)
            _, sel, gs = sel_cache.get((i, mask), ("N/A", 0.0, 0.0))
            new_min = min(cur_min_gs, gs)
            if new_mask not in dp or new_min > dp[new_mask][0]:
                dp[new_mask] = (new_min, i, mask)

    # Reconstruct path
    path = []
    cur = 0
    visited = set()
    while cur in dp and cur not in visited:
        visited.add(cur)
        _, ridx, came = dp[cur]
        solv, sel, gs = sel_cache.get((ridx, came), ("N/A", 0.0, 0.0))
        path.append({
            "from_mask": came, "to_mask": cur,
            "removed_idx": ridx, "removed": polymers[ridx],
            "solvent": solv, "selectivity": sel, "gscore": gs,
        })
        if came == full_mask:
            break
        cur = came
    path.reverse()

    return {
        "polymers": polymers, "n": n, "full_mask": full_mask,
        "sel_cache": sel_cache, "dp": dp, "path": path,
        "opt_min_gs": dp.get(0, (0.0,))[0],
    }


# ── Lattice plot ──────────────────────────────────────────────────

def plot_dp_lattice(data, output_path, *, start_label_above=False,
                    show_level_labels=True, readable_all=False,
                    alt_path=None, alt_label="Safety-optimized"):
    """Plot DP lattice. If *alt_path* (list of edge dicts) is given,
    draw it as a second highlighted path in blue/teal."""
    from math import comb
    from matplotlib.collections import LineCollection

    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    dp = data["dp"]
    sel_cache = data["sel_cache"]
    path = data["path"]

    max_level_count = max(comb(n, k) for k in range(n + 1))
    large = max_level_count > 20

    def mask_label(mask):
        names = [polymers[i] for i in range(n) if mask & (1 << i)]
        return ", ".join(names) if names else "\u2205"

    # ── Adaptive layout constants ──
    if large:
        lvl_y = 1.6
        node_sp = max(0.28, 14.0 / max_level_count)
        r_norm, r_opt, r_end = 0.08, 0.38, 0.50
        edge_lw, edge_alpha = 0.35, 0.18
        opt_lw = 3.5
        lbl_fs, node_fs = 22, 20
        title_fs, sub_fs, level_fs = 44, 28, 24
        result_fs_top, result_fs_bot = 24, 28
        legend_fs = 22
        save_dpi = 200
    else:
        lvl_y = 3.2
        node_sp = 2.8
        r_norm = r_opt = r_end = 0.60
        edge_lw, edge_alpha = 0.7, 0.25
        opt_lw = 3.0
        lbl_fs, node_fs = 17, 16
        title_fs, sub_fs, level_fs = 26, 19, 17
        result_fs_top = result_fs_bot = 20
        legend_fs = 16
        save_dpi = 200

    # ── Positions ──
    positions = {}
    for level in range(n + 1):
        nodes = sorted(m for m in range(1 << n) if bin(m).count("1") == level)
        cnt = len(nodes)
        for i, mask in enumerate(nodes):
            x = (i - (cnt - 1) / 2) * node_sp
            y = level * lvl_y
            positions[mask] = (x, y)

    opt_edges = {(e["from_mask"], e["to_mask"]) for e in path}
    opt_nodes = set()
    for e in path:
        opt_nodes.add(e["from_mask"])
        opt_nodes.add(e["to_mask"])

    alt_edges = set()
    alt_nodes = set()
    if alt_path:
        alt_edges = {(e["from_mask"], e["to_mask"]) for e in alt_path}
        for e in alt_path:
            alt_nodes.add(e["from_mask"])
            alt_nodes.add(e["to_mask"])
    highlighted_edges = opt_edges | alt_edges

    def edge_sel(parent_mask, child_mask):
        diff = parent_mask ^ child_mask
        ridx = next(i for i in range(n) if diff & (1 << i))
        key = (ridx, parent_mask)
        return sel_cache[key][1] if key in sel_cache else 0.0

    # ── Figure sizing ──
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
            ha="center", va="center", fontsize=title_fs, fontweight="bold",
            color="#2c3e50")
    n_lookups = data["n_precomputed"]
    n_factorial = int(np.prod(range(1, n + 1)))
    total_ops = n ** 2 * (1 << n) * 32
    ax.text(0, top - 0.3,
            f"{n} polymers | {n_lookups:,} selectivity lookups (n\u00b72\u207f\u207b\u00b9) | "
            f"O(n\u00b2\u00b72\u207f) \u2248 {total_ops:,} total operations | "
            f"Optimal (vs {n_factorial:,} orderings)",
            ha="center", va="center", fontsize=sub_fs, color="#2c3e50")

    # ── Non-optimal edges ──
    if large:
        segments, seg_colors = [], []
        for level in range(1, n + 1):
            for mask in range(1 << n):
                if bin(mask).count("1") != level:
                    continue
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue
                    child = mask ^ (1 << i)
                    if (mask, child) in highlighted_edges:
                        continue
                    x1, y1 = positions[mask]
                    x2, y2 = positions[child]
                    segments.append([(x1, y1), (x2, y2)])
                    seg_colors.append(sel_color(edge_sel(mask, child)))
        lc = LineCollection(segments, colors=seg_colors,
                            linewidths=edge_lw, alpha=edge_alpha, zorder=1)
        ax.add_collection(lc)
    else:
        for level in range(1, n + 1):
            for mask in range(1 << n):
                if bin(mask).count("1") != level:
                    continue
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue
                    child = mask ^ (1 << i)
                    if (mask, child) in highlighted_edges:
                        continue
                    x1, y1 = positions[mask]
                    x2, y2 = positions[child]
                    c = sel_color(edge_sel(mask, child))
                    ax.plot([x1, x2], [y1, y2], "-",
                            color=c, linewidth=edge_lw, alpha=edge_alpha,
                            zorder=1)

    # ── Optimal path edges (thick, labeled) ──
    for edge in path:
        fm, tm = edge["from_mask"], edge["to_mask"]
        x1, y1 = positions[fm]
        x2, y2 = positions[tm]
        sel = edge["selectivity"]
        c = sel_color(sel)

        if large:
            r1 = r_end if (fm == full_mask or fm == 0) else r_opt
            r2 = r_end if (tm == full_mask or tm == 0) else r_opt
            gap1, gap2 = r1 + 0.04, r2 + 0.04
        else:
            gap1 = gap2 = 0.42
        ax.annotate("", xy=(x2, y2 + gap2), xytext=(x1, y1 - gap1),
                     arrowprops=dict(arrowstyle="-|>", lw=opt_lw, color=c,
                                     mutation_scale=18 if not large else 20),
                     zorder=3)

        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = x2 - x1
        off = 2.0 if large else 0.8
        # PP label: place in line with to-node (level 4), just slightly above
        if large and edge["removed"] == "PP":
            mx = x2
            my = y2 + 0.15
            off_x = -(r_opt + 0.20)
        else:
            off_x = off if dx >= 0 else -off
        ha_align = "left" if off_x > 0 else "right"
        is_iso = sel == 0.0 and edge["solvent"] == "N/A"
        if is_iso:
            label = f"\u2212{edge['removed']}\n(isolated)"
        else:
            label = f"\u2212{edge['removed']}\n{edge['solvent']} ({sel:.1f}%)"
        ax.text(mx + off_x, my, label, ha=ha_align, va="center",
                fontsize=lbl_fs, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c,
                          alpha=0.92, lw=1.2),
                zorder=6)

    # ── Alt path edges (teal, dashed-style labels) ──
    _ALT_COLOR = "#00b894"  # green for safety path
    if alt_path:
        for edge in alt_path:
            fm, tm = edge["from_mask"], edge["to_mask"]
            if (fm, tm) in opt_edges:
                continue  # shared edge, already drawn
            x1, y1 = positions[fm]
            x2, y2 = positions[tm]
            gap1 = gap2 = 0.42 if not large else (r_opt + 0.04)
            ax.annotate("", xy=(x2, y2 + gap2), xytext=(x1, y1 - gap1),
                         arrowprops=dict(arrowstyle="-|>", lw=opt_lw, color=_ALT_COLOR,
                                         mutation_scale=18, linestyle="--"),
                         zorder=3)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx = x2 - x1
            off = 2.0 if large else 0.8
            off_x = -off if dx >= 0 else off  # opposite side of selectivity labels
            ha_align = "left" if off_x > 0 else "right"
            sel = edge.get("selectivity", 0.0)
            gs = edge.get("gscore", 0.0)
            is_iso = sel == 0.0 and edge["solvent"] == "N/A"
            if is_iso:
                label = f"\u2212{edge['removed']}\n(isolated)"
            else:
                label = f"\u2212{edge['removed']}\n{edge['solvent']} (G:{gs:.1f})"
            ax.text(mx + off_x, my, label, ha=ha_align, va="center",
                    fontsize=lbl_fs, color=_ALT_COLOR, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=_ALT_COLOR,
                              alpha=0.92, lw=1.2),
                    zorder=6)

    # ── Nodes ──
    for mask, (x, y) in positions.items():
        is_opt = mask in opt_nodes
        is_alt = mask in alt_nodes and mask not in opt_nodes
        is_full = mask == full_mask
        is_empty = mask == 0

        if is_full:
            fc, ec = "#3498db", "#2c3e50"
            tc = "#2c3e50" if start_label_above else "white"
            r = r_end
        elif is_empty:
            fc, ec, tc = "#2ecc71", "#2c3e50", "white"
            r = r_end
        elif is_opt:
            fc, ec, tc = "#ffeaa7", "#f39c12", "#2c3e50"
            r = r_opt
        elif is_alt:
            fc, ec, tc = "#b8f0d8", _ALT_COLOR, "#2c3e50"
            r = r_opt
        else:
            fc, ec = "#f5f6fa", "#dcdde1"
            tc = "#2c3e50" if readable_all else "#95a5a6"
            r = r_norm

        lw = 3.0 if (is_opt or is_alt or is_full or is_empty) else (0.3 if large else 0.8)
        circle = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                             linewidth=lw, zorder=4)
        ax.add_patch(circle)

        # Labels: all nodes for small n, only endpoints for large n
        if large:
            if is_full:
                ax.text(x, y, "FULL", ha="center", va="center",
                        fontsize=node_fs, fontweight="bold", color=tc, zorder=5)
            elif is_empty:
                ax.text(x, y, "\u2205", ha="center", va="center",
                        fontsize=node_fs + 2, fontweight="bold", color=tc, zorder=5)
        else:
            lbl = mask_label(mask)
            if start_label_above and is_full:
                # Place label above the blue node
                ax.text(x, y + r + 0.25, lbl, ha="center", va="bottom",
                        fontsize=node_fs + 2, fontweight="bold",
                        color=tc, zorder=5)
            else:
                base_fs = (node_fs + 3) if readable_all else node_fs
                fs = (base_fs - 3) if len(lbl) > 15 else ((base_fs - 2) if len(lbl) > 10 else base_fs)
                ax.text(x, y, lbl, ha="center", va="center",
                        fontsize=fs, fontweight="bold" if (is_opt or readable_all) else "normal",
                        color=tc, zorder=5)

    # ── Level labels (right side) ──
    if show_level_labels:
        rx = max(xs) + pad_x - 0.8
        for level in range(n + 1):
            ly = level * lvl_y
            if level == n:
                label = f"FULL SET\n({level} polymers)"
            elif level == 0:
                label = "ALL SEPARATED\n(0 remaining)"
            else:
                label = f"{comb(n, level)} states\n({level} remaining)"
            ax.text(rx, ly, label, ha="center", va="center",
                    fontsize=level_fs, color="#2c3e50", style="italic")

    # ── Result box(es) ──
    seq_str = " \u2192 ".join(e["removed"] for e in path)
    ry = min(ys) - pad_y
    rw = max(xs) - min(xs) + 4

    if alt_path:
        # Two result boxes stacked: selectivity (orange) on top, safety (blue) below
        rh = 0.7
        # Selectivity box
        ry_sel = ry + 0.05
        box_sel = FancyBboxPatch((-rw / 2, ry_sel - rh / 2), rw, rh,
                                  boxstyle="round,pad=0.12",
                                  facecolor="#f39c12", edgecolor="#2c3e50",
                                  linewidth=2.5)
        ax.add_patch(box_sel)
        ax.text(0, ry_sel,
                f"SELECTIVITY-OPTIMAL: {seq_str}  |  "
                f"Min selectivity: {data['opt_min']:.1f}%",
                ha="center", va="center", fontsize=result_fs_top,
                fontweight="bold", color="white")
        # Safety box
        alt_seq = " \u2192 ".join(e["removed"] for e in alt_path)
        alt_min_gs = min((e.get("gscore", 0) for e in alt_path if e.get("gscore", 0) > 0), default=0)
        alt_min_sel = min((e.get("selectivity", 0) for e in alt_path), default=0)
        ry_safe = ry_sel - rh - 0.25
        box_safe = FancyBboxPatch((-rw / 2, ry_safe - rh / 2), rw, rh,
                                   boxstyle="round,pad=0.12",
                                   facecolor=_ALT_COLOR, edgecolor="#2c3e50",
                                   linewidth=2.5)
        ax.add_patch(box_safe)
        ax.text(0, ry_safe,
                f"SAFETY-OPTIMAL: {alt_seq}  |  "
                f"Min G-score: {alt_min_gs:.1f}  |  Min sel: {alt_min_sel:.1f}%",
                ha="center", va="center", fontsize=result_fs_top,
                fontweight="bold", color="white")
    elif large:
        rh = 1.0
        result = FancyBboxPatch((-rw / 2, ry - rh / 2), rw, rh,
                                 boxstyle="round,pad=0.12",
                                 facecolor="#2ecc71", edgecolor="#2c3e50",
                                 linewidth=2.5)
        ax.add_patch(result)
        ax.text(0, ry + 0.15,
                f"OPTIMAL SEQUENCE: {seq_str}",
                ha="center", va="center", fontsize=result_fs_top,
                fontweight="bold", color="white")
        ax.text(0, ry - 0.22,
                f"Min selectivity: {data['opt_min']:.1f}%",
                ha="center", va="center", fontsize=result_fs_bot,
                fontweight="bold", color="white")
    else:
        rh = 0.7
        result = FancyBboxPatch((-rw / 2, ry - rh / 2), rw, rh,
                                 boxstyle="round,pad=0.12",
                                 facecolor="#2ecc71", edgecolor="#2c3e50",
                                 linewidth=2.5)
        ax.add_patch(result)
        ax.text(0, ry,
                f"OPTIMAL SEQUENCE: {seq_str}  |  "
                f"Min selectivity: {data['opt_min']:.1f}%",
                ha="center", va="center", fontsize=result_fs_top,
                fontweight="bold", color="white")

    # ── Legend ──
    items = [
        mpatches.Patch(fc="#3498db", ec="black", label="Start (full mixture)"),
        mpatches.Patch(fc="#ffeaa7", ec="#f39c12", label="Selectivity-optimal path"),
    ]
    if alt_path:
        items.append(mpatches.Patch(fc="#bee5fc", ec=_ALT_COLOR, label=alt_label + " path"))
    items += [
        mpatches.Patch(fc="#f5f6fa", ec="#dcdde1", label="Other state"),
        mpatches.Patch(fc="#2ecc71", ec="black", label="Goal (all separated)"),
        mpatches.Patch(fc="#27ae60", ec="black", label="Selectivity > 30%"),
        mpatches.Patch(fc="#f39c12", ec="black", label="Selectivity 10\u201330%"),
        mpatches.Patch(fc="#e67e22", ec="black", label="Selectivity 0\u201310%"),
    ]
    ax.legend(handles=items, loc="upper left", fontsize=legend_fs,
              frameon=True, fancybox=True, edgecolor="#bdc3c7",
              title="Legend", title_fontsize=legend_fs + 2,
              bbox_to_anchor=(0.0, 1.0))

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Cascade plot (for large n) ────────────────────────────────────

def plot_dp_cascade(data, output_path):
    """Optimal-path cascade for large polymer sets (n > 5)."""
    from math import comb, factorial

    n = data["n"]
    polymers = data["polymers"]
    path = data["path"]
    sel_cache = data["sel_cache"]

    # Build node list from optimal path
    nodes = []
    fm = path[0]["from_mask"]
    nodes.append((fm, [polymers[i] for i in range(n) if fm & (1 << i)]))
    for e in path:
        m = e["to_mask"]
        nodes.append((m, [polymers[i] for i in range(n) if m & (1 << i)]))

    nn = len(nodes)

    # Layout constants
    gap = 2.4
    nw, nh = 10.5, 0.60

    # Y positions (top = high, bottom = low)
    y_pos = [(nn - 1 - i) * gap for i in range(nn)]

    # Figure bounds
    x_min, x_max = -nw / 2 - 3.5, nw / 2 + 5.5
    y_min = y_pos[-1] - 3.8
    y_max = y_pos[0] + 4.5

    scale = 0.80
    fw = (x_max - x_min) * scale
    fh = (y_max - y_min) * scale

    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ──
    ty = y_pos[0] + 3.5
    ax.text(0, ty, "DYNAMIC PROGRAMMING (BITMASK DP) SEPARATOR",
            ha="center", fontsize=18, fontweight="bold", color="#2c3e50")

    n_lookups = data["n_precomputed"]
    total_ops = n ** 2 * (1 << n) * 32
    ax.text(0, ty - 1.0,
            f"{n} polymers | {n_lookups:,} selectivity lookups (n\u00b72\u207f\u207b\u00b9) | "
            f"O(n\u00b2\u00b72\u207f) \u2248 {total_ops:,} ops | "
            f"Optimal (vs {factorial(n):,} orderings)",
            ha="center", fontsize=11, color="#7f8c8d")

    # ── Nodes ──
    for i, (mask, rem) in enumerate(nodes):
        y = y_pos[i]
        is_full = (i == 0)
        is_empty = (i == nn - 1)

        if is_full:
            fc, ec, tc = "#3498db", "#2c3e50", "white"
        elif is_empty:
            fc, ec, tc = "#2ecc71", "#2c3e50", "white"
        else:
            fc, ec, tc = "#ffeaa7", "#f39c12", "#2c3e50"

        box = FancyBboxPatch(
            (-nw / 2, y - nh / 2), nw, nh,
            boxstyle="round,pad=0.12",
            facecolor=fc, edgecolor=ec, linewidth=2.0, zorder=4)
        ax.add_patch(box)

        lbl = ", ".join(rem) if rem else "\u2205 (all separated)"
        fs = 10 if len(lbl) <= 55 else 8.5
        ax.text(0, y, lbl, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=5)

    # ── Edges + labels (right side) ──
    label_x = nw / 2 + 1.2
    for i, edge in enumerate(path):
        y1, y2 = y_pos[i], y_pos[i + 1]
        sel = edge["selectivity"]
        c = sel_color(sel)

        # Arrow
        ax.annotate("",
                    xy=(0, y2 + nh / 2 + 0.06),
                    xytext=(0, y1 - nh / 2 - 0.06),
                    arrowprops=dict(arrowstyle="-|>", lw=2.5, color=c,
                                    mutation_scale=15),
                    zorder=3)

        # Label
        my = (y1 + y2) / 2
        is_iso = sel == 0.0 and edge["solvent"] == "N/A"
        if is_iso:
            lbl = f"\u2212{edge['removed']}\n(isolated)"
        else:
            lbl = f"\u2212{edge['removed']}\n{edge['solvent']} ({sel:.1f}%)"

        ax.text(label_x, my, lbl, ha="left", va="center",
                fontsize=10, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c,
                          alpha=0.92, lw=1.0),
                zorder=6)

        # Dashed connector
        ax.plot([nw / 2, label_x - 0.15], [my, my],
                color=c, lw=0.6, alpha=0.4, linestyle="--", zorder=2)

    # ── Level annotations (left side) ──
    ann_x = -(nw / 2 + 1.5)
    for i in range(nn):
        k = len(nodes[i][1])
        y = y_pos[i]
        if k == n:
            lbl = f"FULL SET\n({k} polymers)"
        elif k == 0:
            lbl = "ALL SEPARATED\n(0 remaining)"
        else:
            lbl = f"{comb(n, k)} states\n({k} remaining)"
        ax.text(ann_x, y, lbl, ha="center", va="center",
                fontsize=9, color="#95a5a6", style="italic")

    # ── Result box ──
    seq_str = " \u2192 ".join(e["removed"] for e in path)
    ry = y_pos[-1] - 2.2
    rw = nw + 2
    rh = 1.0
    result = FancyBboxPatch(
        (-rw / 2, ry - rh / 2), rw, rh,
        boxstyle="round,pad=0.12",
        facecolor="#2ecc71", edgecolor="#2c3e50", linewidth=2.5)
    ax.add_patch(result)

    ax.text(0, ry + 0.15,
            f"OPTIMAL SEQUENCE: {seq_str}",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white", zorder=5)
    ax.text(0, ry - 0.22,
            f"Min selectivity: {data['opt_min']:.1f}%",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color="white", zorder=5)

    # ── Legend ──
    items = [
        mpatches.Patch(fc="#3498db", ec="black", label="Start (full mixture)"),
        mpatches.Patch(fc="#ffeaa7", ec="#f39c12", label="Optimal path state"),
        mpatches.Patch(fc="#2ecc71", ec="black", label="Goal (all separated)"),
        mpatches.Patch(fc="#27ae60", ec="black", label="Selectivity > 30%"),
        mpatches.Patch(fc="#f39c12", ec="black", label="Selectivity 10\u201330%"),
        mpatches.Patch(fc="#e67e22", ec="black", label="Selectivity 0\u201310%"),
    ]
    ax.legend(handles=items, loc="upper left", fontsize=9,
              frameon=True, fancybox=True, edgecolor="#bdc3c7",
              title="Legend", title_fontsize=11)

    fig.tight_layout(rect=[0, 0.01, 1, 0.99])
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Visualize bitmask DP separator")
    parser.add_argument("--poly4", action="store_true",
                        help="Use 4 polymers (default: 9)")
    parser.add_argument("--cascade", action="store_true",
                        help="Use compact cascade view instead of full lattice")
    parser.add_argument("-t", "--temp", type=float, default=120.0,
                        help="Temperature in C (default: 120)")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    args = parser.parse_args()

    HERE = Path(__file__).parent

    if args.poly4:
        polys = ["PS", "PVC", "LDPE", "HDPE"]
    else:
        polys = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "Nylon6", "Nylon66", "PET"]

    n = len(polys)
    print(f"Building DP data for {n} polymers at {args.temp}\u00b0C...")
    t0 = time.time()
    data = build_dp_data(polys, temperature=args.temp)
    dt = time.time() - t0

    print(f"  Selectivity lookups: {data['n_precomputed']:,} (n\u00b72^(n-1))")
    print(f"  DP states filled:    {len(data['dp'])}")
    print(f"  Computation time:    {dt:.1f}s")
    print(f"  Optimal min sel.:    {data['opt_min']:.1f}%")
    arrow = " \u2192 "
    opt_path_str = arrow.join(e["removed"] for e in data["path"])
    print(f"  Optimal path:        {opt_path_str}")

    out = args.output or str(HERE / "dp_algorithm_visual.png")

    if args.cascade:
        plot_dp_cascade(data, out)
    else:
        plot_dp_lattice(data, out)
