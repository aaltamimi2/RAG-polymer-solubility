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

from strap.solubility import get_all_solvents_selectivity, get_solubility, get_boiling_point


# ── GSK G-score loader ────────────────────────────────────────────

def _load_gsk_scores():
    """Load G-scores: actual GSK first, ML-predicted (GreenSolventDB) as fallback.

    Returns (scores, ml_keys) where *scores* is {lowercase_name: score} and
    *ml_keys* is a set of keys whose values are ML-predicted (not actual GSK).
    Actual GSK scores (154 solvents) are preferred; for solvents not in the
    GSK dataset, we fall back to GPR-predicted G-scores from GreenSolventDB
    (10,189 solvents, Datta et al. Advanced Science 2025).
    """
    import csv
    data_dir = Path(__file__).resolve().parent.parent / "data"

    # ── 1. Load actual GSK scores ──
    scores = {}
    ml_keys = set()  # track which keys are ML-predicted (not actual GSK)
    gsk_path = data_dir / "GSK_dataset.csv"
    with open(gsk_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
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
        "1,2-dimethylbenzene": "p-xylene", "1,4-dimethylbenzene": "p-xylene",
        "n-heptane": "heptane", "n-hexane": "hexane",
        "glycol": "ethylene glycol", "h2o": "water",
        "propanone": "acetone", "butanone": "methyl ethyl ketone",
        "ethylacetate": "ethyl acetate", "dimethylsulfoxide": "dimethyl sulphoxide",
        "dimethylformamide": "n,n-dimethylformamide",
        "acetylacetone": "2,4-pentanedione",
        "gvl": "y-valerolactone",
    }
    for abbr, full in _ABBREV.items():
        if full in scores and abbr not in scores:
            scores[abbr] = scores[full]

    # Map solvent registry interp_keys to GSK names
    from strap.solvent_registry import SOLVENT_REGISTRY
    for key, entry in SOLVENT_REGISTRY.items():
        gsk_name = entry.get("gsk_db")
        if gsk_name and key not in scores:
            gs = scores.get(gsk_name.lower())
            if gs is not None:
                scores[key] = gs
                for alias in entry.get("aliases", []):
                    if alias.lower() not in scores:
                        scores[alias.lower()] = gs

    # ── 2. Load GreenSolventDB 10K as fallback (ML-predicted G-scores) ──
    green_path = data_dir / "GreenSolventDB_10k.csv"
    if green_path.exists():
        pred_by_cas = {}
        with open(green_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cas = row.get("CAS", "").strip()
                try:
                    gs = float(row["G-score prediction"])
                except (ValueError, KeyError):
                    continue
                if cas:
                    pred_by_cas[cas] = gs

        # Fill gaps using CAS lookup from registry
        for key, entry in SOLVENT_REGISTRY.items():
            if key in scores:
                continue  # already have actual GSK
            cas = entry.get("cas")
            if cas and cas in pred_by_cas:
                scores[key] = pred_by_cas[cas]
                ml_keys.add(key)
                for alias in entry.get("aliases", []):
                    if alias.lower() not in scores:
                        scores[alias.lower()] = pred_by_cas[cas]
                        ml_keys.add(alias.lower())

    return scores, ml_keys


def _geomean_gscore(steps):
    """Geometric mean of G-scores across non-isolation steps.

    Follows GSK methodology (G = geometric mean of EHSW sub-scores).
    Isolation steps (no solvent) are excluded.
    """
    from math import prod
    vals = [s["gscore"] for s in steps
            if not s.get("isolation") and s.get("gscore") and s["gscore"] > 0]
    if not vals:
        return 0.0
    return prod(vals) ** (1.0 / len(vals))


# ── Article reference sequence (9 polymers) ──────────────────────

ARTICLE_POLYMERS = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "PET", "Nylon6", "Nylon66"]

ARTICLE_SEQUENCE = [
    {"target": "PS",      "solvent": "toluene",             "temp_c": 35,  "article_wt": 5.72},
    {"target": "PVC",     "solvent": "thf",                 "temp_c": 67,  "article_wt": 19.10},
    {"target": "LDPE",    "solvent": "1,2-dimethylbenzene", "temp_c": 80,  "article_wt": 3.43},
    {"target": "HDPE",    "solvent": "1,2-dimethylbenzene", "temp_c": 95,  "article_wt": 5.04},
    {"target": "PP",      "solvent": "1,2-dimethylbenzene", "temp_c": 115, "article_wt": 9.65},
    {"target": "EVOH",    "solvent": "dimethylsulfoxide",   "temp_c": 95,  "article_wt": 7.67},
    {"target": "PET",     "solvent": "gvl",                 "temp_c": 160, "article_wt": 12.45},
    {"target": "Nylon6",  "solvent": "dimethylsulfoxide",   "temp_c": 145, "article_wt": 8.41},
    {"target": "Nylon66", "solvent": None,                  "temp_c": 90,  "article_wt": 16.90},  # formic acid not in DB
]


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

def build_dp_data(polymers, temperature=120.0, temp_range=None,
                   banned_solvents=None):
    """Run bitmask DP, recording all selectivities and optimal path.

    Args:
        temp_range: list of temperatures to sweep (e.g. range(25, 165, 5)).
                    If None, uses *temperature* as a single fixed value.
        banned_solvents: set of lowercase solvent names to exclude (e.g. toxic).
    """
    n = len(polymers)
    full_mask = (1 << n) - 1
    temps = list(temp_range) if temp_range is not None else [temperature]
    _banned = {s.lower() for s in (banned_solvents or [])}

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
                # Filter out banned and solvents within 5°C of boiling point
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

    # DP: dp[mask] = (min_selectivity, last_removed_idx, came_from_mask)
    INF = float("inf")
    dp = {}

    # Initialize: remove one polymer from the full set
    for i in range(n):
        rem = full_mask ^ (1 << i)
        _, _, sel = sel_cache.get((i, full_mask), ("N/A", 0.0, 0.0))
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
            _, _, sel = sel_cache.get((i, mask), ("N/A", 0.0, 0.0))
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


# ── Safety-optimized DP ──────────────────────────────────────────

def build_dp_data_safety(polymers, temperature=120.0, min_sel=5.0,
                         temp_range=None, banned_solvents=None):
    """Run bitmask DP optimizing for GSK safety (maximize min G-score).

    At each step, picks the solvent with the highest G-score among those
    with selectivity >= *min_sel*. Falls back to best-selectivity solvent
    if no safe option meets the threshold.

    Args:
        temp_range: list of temperatures to sweep. If None, uses *temperature*.
        banned_solvents: set of lowercase solvent names to exclude (e.g. toxic).
    """
    gsk, _ = _load_gsk_scores()
    n = len(polymers)
    full_mask = (1 << n) - 1
    _banned = {s.lower() for s in (banned_solvents or [])}
    temps = list(temp_range) if temp_range is not None else [temperature]

    # Precompute: for each (target_idx, mask), pick safest viable solvent
    sel_cache = {}   # (tidx, mask) -> (solvent, temp, selectivity, gscore)
    for tidx in range(n):
        for mask in range(1, 1 << n):
            if not (mask & (1 << tidx)):
                continue
            others_mask = mask ^ (1 << tidx)
            if others_mask == 0:
                continue
            target = polymers[tidx]
            others = [polymers[i] for i in range(n) if others_mask & (1 << i)]
            best = ("N/A", temps[0], 0.0, 0.0)
            for t in temps:
                results = get_all_solvents_selectivity(target, others, t)
                if not results:
                    continue
                # Filter out banned solvents and those within 5°C of BP
                results = [r for r in results
                           if r["solvent"].lower() not in _banned
                           and ((bp := get_boiling_point(r["solvent"])) is None
                                or t <= bp - 5)]
                if not results:
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
                cand = max(viable, key=lambda r: r["gscore"])
                if cand["gscore"] > best[3] or (
                    cand["gscore"] == best[3] and cand["selectivity"] > best[2]
                ):
                    best = (cand["solvent"], t,
                            cand["selectivity"], cand["gscore"])
            sel_cache[(tidx, mask)] = best

    # DP: maximize min G-score along path
    INF = float("inf")
    dp = {}

    for i in range(n):
        rem = full_mask ^ (1 << i)
        _, _, sel, gs = sel_cache.get((i, full_mask), ("N/A", 0.0, 0.0, 0.0))
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
            _, _, sel, gs = sel_cache.get((i, mask), ("N/A", 0.0, 0.0, 0.0))
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
        solv, tmp, sel, gs = sel_cache.get((ridx, came), ("N/A", 0.0, 0.0, 0.0))
        path.append({
            "from_mask": came, "to_mask": cur,
            "removed_idx": ridx, "removed": polymers[ridx],
            "solvent": solv, "temperature": tmp, "selectivity": sel,
            "gscore": gs,
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


# ── Top-K path extraction ─────────────────────────────────────────

def extract_top_k_paths(data, k=10):
    """Extract top-K separation paths using forward beam DP.

    Operates on the plot script's sel_cache format: (solvent, selectivity).
    Returns a list of K paths, each a list of edge dicts matching the
    existing path format used by plot_dp_lattice.
    """
    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]
    INF = float("inf")

    # beam[mask] = [(min_sel_so_far, [removal_indices])]
    beam = {full_mask: [(INF, [])]}

    for mask in range(full_mask, 0, -1):
        if mask not in beam:
            continue
        popcount = bin(mask).count("1")

        if popcount == 1:
            idx = next(i for i in range(n) if mask & (1 << i))
            for min_sel, seq in beam[mask]:
                if 0 not in beam:
                    beam[0] = []
                beam[0].append((min_sel, seq + [idx]))
                beam[0].sort(key=lambda x: x[0], reverse=True)
                if len(beam[0]) > k:
                    beam[0] = beam[0][:k]
            continue

        for min_sel, seq in beam[mask]:
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                child = mask ^ (1 << i)
                cache_key = (i, mask)
                if cache_key not in sel_cache:
                    continue
                _, _, sel = sel_cache[cache_key]
                new_min = min(min_sel, sel) if seq else sel

                if child not in beam:
                    beam[child] = []
                beam[child].append((new_min, seq + [i]))
                beam[child].sort(key=lambda x: x[0], reverse=True)
                if len(beam[child]) > k:
                    beam[child] = beam[child][:k]

    # Convert to path-edge-dict format
    results = []
    for min_sel, idx_seq in beam.get(0, []):
        path = []
        remaining = full_mask
        for polymer_idx in idx_seq:
            child = remaining ^ (1 << polymer_idx)
            solv, tmp, sel = sel_cache.get((polymer_idx, remaining), ("N/A", 0.0, 0.0))
            is_last = (bin(remaining).count("1") == 1)
            path.append({
                "from_mask": remaining,
                "to_mask": child,
                "removed_idx": polymer_idx,
                "removed": polymers[polymer_idx],
                "solvent": solv if not is_last else "N/A",
                "temperature": tmp if not is_last else 0.0,
                "selectivity": sel if not is_last else 0.0,
            })
            remaining = child
        results.append({"path": path, "min_sel": min_sel})

    return results


# ── Full enumeration: all n! sequences ────────────────────────────

def enumerate_all_sequences(data):
    """Enumerate ALL n! separation sequences and their min selectivities.

    Returns:
        edge_best: dict mapping (from_mask, to_mask) to best min_selectivity
                   of any complete path passing through that edge.
        all_min_sels: sorted list of (min_sel, sequence_str) for all n! paths.
        stats: dict with summary statistics.
    """
    from itertools import permutations

    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]

    # For each edge, track the best min_selectivity of any path through it
    edge_best = {}  # (from_mask, to_mask) -> best min_sel
    all_min_sels = []
    count = 0

    for perm in permutations(range(n)):
        remaining = full_mask
        path_edges = []
        min_sel = float("inf")

        for polymer_idx in perm:
            child = remaining ^ (1 << polymer_idx)
            # Last polymer: isolation step, no selectivity
            if bin(remaining).count("1") == 1:
                path_edges.append((remaining, child))
                remaining = child
                continue
            key = (polymer_idx, remaining)
            _, _, sel = sel_cache.get(key, ("N/A", 0.0, 0.0))
            min_sel = min(min_sel, sel)
            path_edges.append((remaining, child))
            remaining = child

        # Record this path's min_sel for all its edges
        for edge in path_edges:
            if edge not in edge_best or min_sel > edge_best[edge]:
                edge_best[edge] = min_sel

        seq_str = " → ".join(polymers[i] for i in perm)
        all_min_sels.append((min_sel, seq_str))
        count += 1

    all_min_sels.sort(key=lambda x: x[0], reverse=True)

    sels = [s for s, _ in all_min_sels]
    stats = {
        "total": count,
        "best": sels[0],
        "worst": sels[-1],
        "median": sels[count // 2],
        "mean": sum(sels) / count,
        "p90": sels[int(count * 0.10)],  # 90th percentile (top 10%)
        "p10": sels[int(count * 0.90)],  # 10th percentile (bottom 10%)
    }

    return edge_best, all_min_sels, stats


def top_k_unique_from_enumeration(data, all_min_sels, k=10):
    """Extract top-K paths with UNIQUE min-selectivity values.

    Picks one representative sequence per distinct min_sel score from
    the sorted enumeration results. Returns path-edge-dict format.
    """
    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]

    seen_scores = set()
    results = []

    for score, seq_str in all_min_sels:
        # Round to 1 decimal to group near-identical scores
        rounded = round(score, 1)
        if rounded in seen_scores:
            continue
        seen_scores.add(rounded)

        # Parse sequence string back to indices
        names = [s.strip() for s in seq_str.split("→")]
        idx_seq = [polymers.index(name) for name in names]

        # Build path edges
        path = []
        remaining = full_mask
        for polymer_idx in idx_seq:
            child = remaining ^ (1 << polymer_idx)
            solv, tmp, sel = sel_cache.get((polymer_idx, remaining), ("N/A", 0.0, 0.0))
            is_last = (bin(remaining).count("1") == 1)
            path.append({
                "from_mask": remaining,
                "to_mask": child,
                "removed_idx": polymer_idx,
                "removed": polymers[polymer_idx],
                "solvent": solv if not is_last else "N/A",
                "temperature": tmp if not is_last else 0.0,
                "selectivity": sel if not is_last else 0.0,
            })
            remaining = child
        results.append({"path": path, "min_sel": score})

        if len(results) >= k:
            break

    return results


def top_k_unique_safety_from_enumeration(data, all_scores, k=10):
    """Extract top-K paths with UNIQUE min G-score values (safety mode).

    Uses safety sel_cache format: (solvent, selectivity, gscore).
    """
    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]

    seen_scores = set()
    results = []

    for min_gs, min_sel, seq_str in all_scores:
        rounded = round(min_gs, 1)
        if rounded in seen_scores:
            continue
        seen_scores.add(rounded)

        names = [s.strip() for s in seq_str.split("→")]
        idx_seq = [polymers.index(name) for name in names]

        path = []
        remaining = full_mask
        for polymer_idx in idx_seq:
            child = remaining ^ (1 << polymer_idx)
            entry = sel_cache.get((polymer_idx, remaining), ("N/A", 0.0, 0.0, 0.0))
            solv, tmp, sel, gs = entry
            is_last = (bin(remaining).count("1") == 1)
            path.append({
                "from_mask": remaining,
                "to_mask": child,
                "removed_idx": polymer_idx,
                "removed": polymers[polymer_idx],
                "solvent": solv if not is_last else "N/A",
                "temperature": tmp if not is_last else 0.0,
                "selectivity": sel if not is_last else 0.0,
                "gscore": gs if not is_last else 0.0,
            })
            remaining = child
        results.append({"path": path, "min_gs": min_gs})

        if len(results) >= k:
            break

    return results


def enumerate_all_sequences_safety(data):
    """Enumerate ALL n! sequences tracking min G-score (safety metric).

    Uses the safety sel_cache format: (solvent, selectivity, gscore).

    Returns:
        edge_best: dict mapping (from_mask, to_mask) to best min_gscore
                   of any complete path passing through that edge.
        all_scores: sorted list of (min_gscore, min_sel, sequence_str).
        stats: dict with summary statistics.
    """
    from itertools import permutations

    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]

    edge_best = {}
    all_scores = []
    count = 0

    for perm in permutations(range(n)):
        remaining = full_mask
        path_edges = []
        min_gs = float("inf")
        min_sel = float("inf")

        for polymer_idx in perm:
            child = remaining ^ (1 << polymer_idx)
            if bin(remaining).count("1") == 1:
                path_edges.append((remaining, child))
                remaining = child
                continue
            key = (polymer_idx, remaining)
            entry = sel_cache.get(key, ("N/A", 0.0, 0.0, 0.0))
            _, _, sel, gs = entry
            min_gs = min(min_gs, gs)
            min_sel = min(min_sel, sel)
            path_edges.append((remaining, child))
            remaining = child

        for edge in path_edges:
            if edge not in edge_best or min_gs > edge_best[edge]:
                edge_best[edge] = min_gs

        seq_str = " → ".join(polymers[i] for i in perm)
        all_scores.append((min_gs, min_sel, seq_str))
        count += 1

    all_scores.sort(key=lambda x: x[0], reverse=True)

    gs_vals = [g for g, _, _ in all_scores]
    stats = {
        "total": count,
        "best": gs_vals[0],
        "worst": gs_vals[-1],
        "median": gs_vals[count // 2],
        "mean": sum(gs_vals) / count,
        "p90": gs_vals[int(count * 0.10)],
        "p10": gs_vals[int(count * 0.90)],
    }

    return edge_best, all_scores, stats


# ── Article sequence evaluation & ranking ────────────────────────

def evaluate_article_exact(temperature=120.0):
    """Evaluate the article's sequence using its own (solvent, temp) pairs.

    Returns:
        steps: list of per-step dicts with target, solvent, temp, selectivity,
               gscore, article_wt, and whether the step is N/A.
        min_selectivity: minimum selectivity across available steps.
        min_gscore: minimum G-score across available steps.
    """
    gsk, _ = _load_gsk_scores()
    remaining = list(ARTICLE_POLYMERS)
    steps = []

    for step in ARTICLE_SEQUENCE:
        target = step["target"]
        solvent = step["solvent"]
        temp = step["temp_c"]
        article_wt = step["article_wt"]

        others = [p for p in remaining if p != target]

        if not others:
            # Last polymer: isolation step
            steps.append({
                "target": target, "solvent": solvent, "temp_c": temp,
                "selectivity": None, "gscore": None,
                "article_wt": article_wt, "is_na": False, "is_last": True,
            })
            remaining.remove(target)
            continue

        if solvent is None:
            # Solvent not in DB — flag as N/A
            steps.append({
                "target": target, "solvent": "N/A", "temp_c": temp,
                "selectivity": None, "gscore": None,
                "article_wt": article_wt, "is_na": True, "is_last": False,
            })
            remaining.remove(target)
            continue

        # Query solubility for target and all others at article's conditions
        target_sol = get_solubility(target, solvent, temp)
        if target_sol is None:
            target_sol = 0.0

        max_other_sol = 0.0
        for other in others:
            other_sol = get_solubility(other, solvent, temp)
            if other_sol is not None and other_sol > max_other_sol:
                max_other_sol = other_sol

        selectivity = target_sol - max_other_sol
        gscore = gsk.get(solvent.lower(), 0.0)

        steps.append({
            "target": target, "solvent": solvent, "temp_c": temp,
            "selectivity": selectivity, "gscore": gscore,
            "article_wt": article_wt, "is_na": False, "is_last": False,
        })
        remaining.remove(target)

    # Compute minimums over available (non-N/A, non-last) steps
    valid_sels = [s["selectivity"] for s in steps
                  if not s["is_na"] and not s["is_last"] and s["selectivity"] is not None]
    valid_gs = [s["gscore"] for s in steps
                if not s["is_na"] and not s["is_last"] and s["gscore"] is not None]

    min_selectivity = min(valid_sels) if valid_sels else 0.0
    min_gscore = min(valid_gs) if valid_gs else 0.0

    return steps, min_selectivity, min_gscore


def rank_in_enumeration(all_values, target_value):
    """Find rank of target_value in a descending-sorted list of values.

    Args:
        all_values: list of (value, ...) tuples sorted descending by value[0],
                    OR list of plain floats sorted descending.
        target_value: the value to rank.

    Returns:
        dict with rank, total, percentile, better_count, tied_count.
    """
    total = len(all_values)
    better = 0
    tied = 0
    for entry in all_values:
        val = entry[0] if isinstance(entry, (tuple, list)) else entry
        if val > target_value + 1e-9:
            better += 1
        elif abs(val - target_value) < 1e-9:
            tied += 1
        else:
            break  # sorted descending, no more better/tied

    # Continue counting tied past the break
    for entry in all_values[better + tied:]:
        val = entry[0] if isinstance(entry, (tuple, list)) else entry
        if abs(val - target_value) < 1e-9:
            tied += 1
        else:
            break

    rank = better + 1  # 1-indexed
    percentile = 100.0 * (1.0 - better / total) if total > 0 else 0.0

    return {
        "rank": rank, "total": total,
        "percentile": percentile,
        "better_count": better, "tied_count": tied,
    }


def _walk_article_ordering_through_cache(sel_cache, polymers):
    """Walk ARTICLE_SEQUENCE ordering through the system's sel_cache.

    Uses the article's polymer ORDER but the system's best solvents (from
    sel_cache) to compute the min-selectivity for that ordering.

    Returns:
        steps: list of per-step dicts with system's solvent and selectivity.
        min_sel: minimum selectivity along the path.
    """
    n = len(polymers)
    full_mask = (1 << n) - 1
    remaining = full_mask
    steps = []
    min_sel = float("inf")

    for step in ARTICLE_SEQUENCE:
        target = step["target"]
        tidx = polymers.index(target)

        if bin(remaining).count("1") == 1:
            # Isolation step
            steps.append({
                "target": target, "solvent": "(isolation)",
                "selectivity": None, "is_last": True,
            })
            remaining = remaining ^ (1 << tidx)
            continue

        key = (tidx, remaining)
        solv, tmp, sel = sel_cache.get(key, ("N/A", 0.0, 0.0))
        min_sel = min(min_sel, sel)
        steps.append({
            "target": target, "solvent": solv, "temperature": tmp,
            "selectivity": sel, "is_last": False,
        })
        remaining = remaining ^ (1 << tidx)

    return steps, min_sel


def extract_top_k_paths_safety(data, k=10):
    """Extract top-K safety-optimized paths using forward beam DP.

    Uses the safety sel_cache format: (solvent, selectivity, gscore).
    """
    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]
    INF = float("inf")

    beam = {full_mask: [(INF, [])]}

    for mask in range(full_mask, 0, -1):
        if mask not in beam:
            continue
        popcount = bin(mask).count("1")

        if popcount == 1:
            idx = next(i for i in range(n) if mask & (1 << i))
            for min_gs, seq in beam[mask]:
                if 0 not in beam:
                    beam[0] = []
                beam[0].append((min_gs, seq + [idx]))
                beam[0].sort(key=lambda x: x[0], reverse=True)
                if len(beam[0]) > k:
                    beam[0] = beam[0][:k]
            continue

        for min_gs, seq in beam[mask]:
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                child = mask ^ (1 << i)
                cache_key = (i, mask)
                if cache_key not in sel_cache:
                    continue
                _, _, sel, gs = sel_cache[cache_key]
                new_min = min(min_gs, gs) if seq else gs

                if child not in beam:
                    beam[child] = []
                beam[child].append((new_min, seq + [i]))
                beam[child].sort(key=lambda x: x[0], reverse=True)
                if len(beam[child]) > k:
                    beam[child] = beam[child][:k]

    results = []
    for min_gs, idx_seq in beam.get(0, []):
        path = []
        remaining = full_mask
        for polymer_idx in idx_seq:
            child = remaining ^ (1 << polymer_idx)
            entry = sel_cache.get((polymer_idx, remaining), ("N/A", 0.0, 0.0, 0.0))
            solv, tmp, sel, gs = entry
            is_last = (bin(remaining).count("1") == 1)
            path.append({
                "from_mask": remaining,
                "to_mask": child,
                "removed_idx": polymer_idx,
                "removed": polymers[polymer_idx],
                "solvent": solv if not is_last else "N/A",
                "temperature": tmp if not is_last else 0.0,
                "selectivity": sel if not is_last else 0.0,
                "gscore": gs if not is_last else 0.0,
            })
            remaining = child
        results.append({"path": path, "min_gs": min_gs})

    return results


def plot_dp_lattice_all(data, edge_best, all_min_sels, stats, output_path,
                        top_k_results=None, mode="selectivity"):
    """Plot DP lattice with edges shaded by best-path-through-edge metric.

    Each edge is colored by the best min_selectivity (or min G-score in
    safety mode) achievable by any complete sequence passing through that
    edge (green=high, red=low). Optionally overlays top-K paths.

    Args:
        mode: "selectivity" or "safety". Changes titles, labels, colorbar.
    """
    from math import comb, factorial
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]

    max_level_count = max(comb(n, kk) for kk in range(n + 1))

    # ── Layout ──
    lvl_y = 1.6
    node_sp = max(0.28, 14.0 / max_level_count)
    r_norm, r_opt, r_end = 0.10, 0.40, 0.52
    node_fs = 20
    title_fs, sub_fs, level_fs = 42, 26, 22
    result_fs = 20
    legend_fs = 20
    save_dpi = 300

    positions = {}
    for level in range(n + 1):
        nodes = sorted(m for m in range(1 << n) if bin(m).count("1") == level)
        cnt = len(nodes)
        for i, mask in enumerate(nodes):
            x = (i - (cnt - 1) / 2) * node_sp
            y = level * lvl_y
            positions[mask] = (x, y)

    # ── Colormap normalization from edge_best values ──
    best_vals = [v for v in edge_best.values() if v != float("inf")]
    if best_vals:
        vmin = min(best_vals)
        vmax = max(best_vals)
    else:
        vmin, vmax = 0, 100
    if vmin == vmax:
        vmin -= 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    # ── Top-K path data (optional overlay) ──
    k = len(top_k_results) if top_k_results else 0
    _RANK_COLORS = [
        "#0173b2",  # rank 1: strong blue
        "#de8f05",  # rank 2: orange
        "#029e73",  # rank 3: teal
        "#cc78bc",  # rank 4: pink
        "#949494",  # rank 5: gray
    ]
    _MUTED = "#b0b0b0"

    topk_edges = set()
    topk_nodes = set()
    if top_k_results:
        for rank, result in enumerate(top_k_results):
            for edge in result["path"]:
                fm, tm = edge["from_mask"], edge["to_mask"]
                topk_edges.add((fm, tm))
                topk_nodes.add(fm)
                topk_nodes.add(tm)

    # ── Figure ──
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad_x, pad_y = 3.5, 3.0
    fig_w = max(14, max(xs) - min(xs) + 2 * pad_x)
    n_table_lines = min(k, 10) if top_k_results else 10
    fig_h = max(ys) - min(ys) + 2 * pad_y + 8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y - 6, max(ys) + pad_y + 2)
    ax.set_facecolor("white")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ──
    is_safety = (mode == "safety")
    metric_name = "G-Score (Safety)" if is_safety else "Min-Selectivity"
    metric_short = "G-Score" if is_safety else "Selectivity"
    top = max(ys) + pad_y + 0.8
    n_factorial = factorial(n)
    title_str = (f"ALL {n_factorial:,} SEQUENCES \u2014 {metric_name.upper()} LANDSCAPE"
                 if is_safety else
                 f"ALL {n_factorial:,} SEPARATION SEQUENCES (BITMASK DP)")
    ax.text(0, top + 0.4, title_str,
            ha="center", va="center", fontsize=title_fs, fontweight="bold",
            color="#1a252f")
    sub_str = (f"{n} polymers  |  Edges colored by best-path min {metric_short}  |  "
               f"Dark = high, Light = low")
    ax.text(0, top - 0.3, sub_str,
            ha="center", va="center", fontsize=sub_fs, color="#34495e")

    # ── All edges: colored by best-path-through-edge ──
    segments, seg_colors = [], []
    for level in range(1, n + 1):
        for mask in range(1 << n):
            if bin(mask).count("1") != level:
                continue
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                child = mask ^ (1 << i)
                if (mask, child) in topk_edges:
                    continue
                x1, y1 = positions[mask]
                x2, y2 = positions[child]
                segments.append([(x1, y1), (x2, y2)])
                best_min = edge_best.get((mask, child), 0.0)
                if best_min == float("inf"):
                    best_min = vmax
                seg_colors.append(cmap(norm(best_min)))

    edge_lw = 0.8
    edge_alpha = 0.45
    lc = LineCollection(segments, colors=seg_colors,
                        linewidths=edge_lw, alpha=edge_alpha, zorder=1)
    ax.add_collection(lc)

    # ── Top-K path overlay ──
    if top_k_results:
        # Draw glow layer for rank 1 first (underneath)
        for edge in top_k_results[0]["path"]:
            fm, tm = edge["from_mask"], edge["to_mask"]
            x1, y1 = positions[fm]
            x2, y2 = positions[tm]
            r1 = r_end if (fm == full_mask or fm == 0) else r_opt
            r2 = r_end if (tm == full_mask or tm == 0) else r_opt
            gap1, gap2 = r1 + 0.04, r2 + 0.04
            ax.annotate("", xy=(x2, y2 + gap2), xytext=(x1, y1 - gap1),
                         arrowprops=dict(arrowstyle="-", lw=10, color=_RANK_COLORS[0],
                                         mutation_scale=22, alpha=0.18),
                         zorder=1.5)

        for rank, result in enumerate(top_k_results):
            path = result["path"]
            if rank == 0:
                color = _RANK_COLORS[0]
                lw = 5.0
                ls = "-"
            elif rank < 5:
                color = _RANK_COLORS[min(rank, len(_RANK_COLORS) - 1)]
                lw = 3.5
                ls = "-"
            else:
                color = _MUTED
                lw = 2.0
                ls = "--"

            for edge in path:
                fm, tm = edge["from_mask"], edge["to_mask"]
                x1, y1 = positions[fm]
                x2, y2 = positions[tm]
                r1 = r_end if (fm == full_mask or fm == 0) else r_opt
                r2 = r_end if (tm == full_mask or tm == 0) else r_opt
                gap1, gap2 = r1 + 0.04, r2 + 0.04
                ax.annotate("", xy=(x2, y2 + gap2), xytext=(x1, y1 - gap1),
                             arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                             mutation_scale=22, linestyle=ls),
                             zorder=2 + (k - rank))

        # Rank 1 labels
        for edge in top_k_results[0]["path"]:
            fm, tm = edge["from_mask"], edge["to_mask"]
            x1, y1 = positions[fm]
            x2, y2 = positions[tm]
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dx = x2 - x1
            off_x = 2.0 if dx >= 0 else -2.0
            ha_align = "left" if off_x > 0 else "right"
            sel = edge["selectivity"]
            c = _RANK_COLORS[0]
            is_iso = sel == 0.0 and edge["solvent"] == "N/A"
            if is_iso:
                label = f"\u2212{edge['removed']}\n(isolated)"
            elif is_safety:
                gs = edge.get("gscore", 0.0)
                label = f"\u2212{edge['removed']}\n{edge['solvent']} (G:{gs:.1f})"
            else:
                label = f"\u2212{edge['removed']}\n{edge['solvent']} ({sel:.1f}%)"
            ax.text(mx + off_x, my, label, ha=ha_align, va="center",
                    fontsize=20, color=c, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none",
                              alpha=0.92),
                    zorder=6)

    # ── Nodes ──
    for mask, (x, y) in positions.items():
        is_topk = mask in topk_nodes
        is_full = mask == full_mask
        is_empty = mask == 0

        if is_full:
            fc, ec, tc = "#3498db", "#1a252f", "white"
            r = r_end
        elif is_empty:
            fc, ec, tc = "#2ecc71", "#1a252f", "white"
            r = r_end
        elif is_topk:
            fc, ec, tc = "#ffeaa7", "#e67e22", "#2c3e50"
            r = r_opt
        else:
            fc, ec = "#f0f1f3", "#ced4da"
            tc = "#6c757d"
            r = r_norm

        lw = 3.0 if (is_topk or is_full or is_empty) else 0.6
        circle = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                             linewidth=lw, zorder=4)
        ax.add_patch(circle)

        if is_full:
            ax.text(x, y, "FULL", ha="center", va="center",
                    fontsize=node_fs, fontweight="bold", color=tc, zorder=5)
        elif is_empty:
            ax.text(x, y, "\u2205", ha="center", va="center",
                    fontsize=node_fs + 2, fontweight="bold", color=tc, zorder=5)

    # ── Level labels ──
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

    # ── Colorbar ──
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.10, 0.02, 0.80, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar_label = ("Best-path min G-Score through edge" if is_safety
                   else "Best-path min-selectivity through edge (%)")
    cbar.set_label(cbar_label, fontsize=legend_fs + 2, weight="bold")
    cbar.ax.tick_params(labelsize=legend_fs - 2, width=1.5, length=5)
    cbar.outline.set_linewidth(0.8)

    # ── Statistics + ranked table ──
    table_y = min(ys) - pad_y - 0.2
    line_h = 0.50
    ax.text(0, table_y,
            f"DISTRIBUTION ({stats['total']:,} sequences)",
            ha="center", va="center",
            fontsize=result_fs + 4, fontweight="bold", color="#1a252f")
    table_y -= line_h * 0.8

    unit = "" if is_safety else "%"
    stat_line = (
        f"Best: {stats['best']:.1f}{unit}  |  "
        f"Median: {stats['median']:.1f}{unit}  |  "
        f"Mean: {stats['mean']:.1f}{unit}  |  "
        f"P90: {stats['p90']:.1f}{unit}  |  "
        f"P10: {stats['p10']:.1f}{unit}  |  "
        f"Worst: {stats['worst']:.1f}{unit}"
    )
    ax.text(0, table_y, stat_line, ha="center", va="center",
            fontsize=result_fs, color="#2c3e50", family="monospace")
    table_y -= line_h

    # Top sequences — show deduplicated top_k_results when available
    unique_label = " (unique scores)" if top_k_results else ""
    hdr = (f"TOP SEQUENCES (by min G-Score{unique_label})" if is_safety
           else f"TOP SEQUENCES{unique_label}")
    ax.text(0, table_y, hdr, ha="center", va="center",
            fontsize=result_fs + 2, fontweight="bold", color="#2c3e50")
    table_y -= line_h * 0.6

    if top_k_results:
        show_n = min(10, len(top_k_results))
        for rank in range(show_n):
            result = top_k_results[rank]
            path = result["path"]
            seq_str = " → ".join(e["removed"] for e in path)
            if is_safety:
                score = result["min_gs"]
                score_str = f"{score:>5.1f}"
            else:
                score = result["min_sel"]
                score_str = f"{score:>6.1f}%"
            if score == float("inf"):
                score_str = "  inf"
            rank_num = rank + 1

            if rank < len(_RANK_COLORS):
                color = _RANK_COLORS[rank]
            else:
                color = _MUTED

            line = f"#{rank_num:<3}  {score_str}  {seq_str}"
            ax.text(-(fig_w / 2 - pad_x), table_y, line, ha="left", va="center",
                    fontsize=result_fs - 2,
                    fontweight="bold" if rank == 0 else "normal",
                    color=color, family="monospace")
            table_y -= line_h
    else:
        show_n = min(10, len(all_min_sels))
        for rank in range(show_n):
            entry = all_min_sels[rank]
            if is_safety:
                score, _, seq_str = entry
                score_str = f"{score:>5.1f}"
            else:
                score, seq_str = entry
                score_str = f"{score:>6.1f}%"
            if score == float("inf"):
                score_str = "  inf"
            s = entry[0] if entry[0] != float("inf") else vmax
            color = cmap(norm(s))

            line = f"#{rank + 1:<3}  {score_str}  {seq_str}"
            ax.text(-(fig_w / 2 - pad_x), table_y, line, ha="left", va="center",
                    fontsize=result_fs - 2,
                    fontweight="bold" if rank == 0 else "normal",
                    color=color, family="monospace")
            table_y -= line_h

    # ── Legend ──
    items = [
        mpatches.Patch(fc="#3498db", ec="black", label="Start (full mixture)"),
        mpatches.Patch(fc="#2ecc71", ec="black", label="Goal (all separated)"),
    ]
    if top_k_results:
        items.append(mpatches.Patch(fc="#ffeaa7", ec="#f39c12", label="Top-K path node"))
        for rk in range(min(k, 5)):
            items.append(mpatches.Patch(
                fc=_RANK_COLORS[rk], ec="black", label=f"Rank {rk + 1}"))
        if k > 5:
            items.append(mpatches.Patch(fc=_MUTED, ec="black", label=f"Ranks 6\u2013{k}"))
    items.append(mpatches.Patch(fc="#f5f6fa", ec="#dcdde1", label="Other state"))

    ax.legend(handles=items, loc="upper left", fontsize=legend_fs,
              frameon=True, fancybox=True, edgecolor="#bdc3c7",
              title="Legend", title_fontsize=legend_fs + 2,
              bbox_to_anchor=(0.0, 1.0))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.05)
    fig.savefig(output_path, dpi=save_dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Top-K lattice plot with continuous colormap ───────────────────

def plot_dp_lattice_topk(data, top_k_results, output_path):
    """Plot DP lattice with continuous green-red colormap and top-K highlighted paths."""
    from math import comb, factorial
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    n = data["n"]
    polymers = data["polymers"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]
    k = len(top_k_results)

    max_level_count = max(comb(n, kk) for kk in range(n + 1))
    large = max_level_count > 20

    def mask_label(mask):
        names = [polymers[i] for i in range(n) if mask & (1 << i)]
        return ", ".join(names) if names else "\u2205"

    # ── Adaptive layout constants ──
    lvl_y = 1.6
    node_sp = max(0.28, 14.0 / max_level_count)
    r_norm, r_opt, r_end = 0.08, 0.38, 0.50
    edge_lw, edge_alpha = 0.4, 0.25
    node_fs = 20
    title_fs, sub_fs, level_fs = 44, 28, 24
    result_fs = 20
    legend_fs = 20
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

    # ── Collect all selectivity values for colormap normalization ──
    all_sels = [sel for (_, sel) in sel_cache.values() if sel != 0.0]
    if all_sels:
        vmin = min(all_sels)
        vmax = max(all_sels)
    else:
        vmin, vmax = 0, 100
    if vmin == vmax:
        vmin = vmin - 1
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.RdYlGn  # red=low, yellow=mid, green=high

    # ── Top-K path data ──
    # Rank colors: Rank 1 thick blue, Ranks 2-5 distinct, Ranks 6+ muted dashed
    _RANK_COLORS = [
        "#2c7bb6",  # rank 1: strong blue
        "#d7191c",  # rank 2: red
        "#fdae61",  # rank 3: orange
        "#abd9e9",  # rank 4: light blue
        "#756bb1",  # rank 5: purple
    ]
    _MUTED = "#999999"

    topk_edges = set()  # all edges in any top-K path
    topk_nodes = set()
    rank1_edges = set()
    for rank, result in enumerate(top_k_results):
        for edge in result["path"]:
            fm, tm = edge["from_mask"], edge["to_mask"]
            topk_edges.add((fm, tm))
            topk_nodes.add(fm)
            topk_nodes.add(tm)
            if rank == 0:
                rank1_edges.add((fm, tm))

    # ── Figure sizing ──
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad_x, pad_y = 3.0, 2.5
    fig_w = max(14, max(xs) - min(xs) + 2 * pad_x)
    fig_h = max(ys) - min(ys) + 2 * pad_y + 8  # extra space for result table

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y - 6, max(ys) + pad_y + 2)
    ax.axis("off")

    # ── Title ──
    top = max(ys) + pad_y + 0.8
    ax.text(0, top + 0.4,
            f"TOP-{k} SEPARATION SEQUENCES (BITMASK DP)",
            ha="center", va="center", fontsize=title_fs, fontweight="bold",
            color="#2c3e50")
    n_lookups = data["n_precomputed"]
    n_factorial = int(np.prod(range(1, n + 1)))
    total_ops = n ** 2 * (1 << n) * 32
    ax.text(0, top - 0.3,
            f"{n} polymers | {n_lookups:,} selectivity lookups | "
            f"O(n\u00b2\u00b72\u207f) \u2248 {total_ops:,} ops | "
            f"{n_factorial:,} possible orderings",
            ha="center", va="center", fontsize=sub_fs, color="#2c3e50")

    # ── Background edges: continuous colormap ──
    segments, seg_colors = [], []
    for level in range(1, n + 1):
        for mask in range(1 << n):
            if bin(mask).count("1") != level:
                continue
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                child = mask ^ (1 << i)
                if (mask, child) in topk_edges:
                    continue
                x1, y1 = positions[mask]
                x2, y2 = positions[child]
                segments.append([(x1, y1), (x2, y2)])
                key = (i, mask)
                sel = sel_cache[key][1] if key in sel_cache else 0.0
                seg_colors.append(cmap(norm(sel)))

    lc = LineCollection(segments, colors=seg_colors,
                        linewidths=edge_lw, alpha=edge_alpha, zorder=1)
    ax.add_collection(lc)

    # ── Top-K path edges ──
    for rank, result in enumerate(top_k_results):
        path = result["path"]
        if rank == 0:
            color = _RANK_COLORS[0]
            lw = 4.0
            ls = "-"
        elif rank < 5:
            color = _RANK_COLORS[rank]
            lw = 3.0
            ls = "-"
        else:
            color = _MUTED
            lw = 1.5
            ls = "--"

        for edge in path:
            fm, tm = edge["from_mask"], edge["to_mask"]
            x1, y1 = positions[fm]
            x2, y2 = positions[tm]
            r1 = r_end if (fm == full_mask or fm == 0) else r_opt
            r2 = r_end if (tm == full_mask or tm == 0) else r_opt
            gap1, gap2 = r1 + 0.04, r2 + 0.04
            ax.annotate("", xy=(x2, y2 + gap2), xytext=(x1, y1 - gap1),
                         arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                         mutation_scale=20, linestyle=ls),
                         zorder=2 + (k - rank))

    # ── Rank 1 edge labels ──
    rank1_path = top_k_results[0]["path"]
    for edge in rank1_path:
        fm, tm = edge["from_mask"], edge["to_mask"]
        x1, y1 = positions[fm]
        x2, y2 = positions[tm]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx = x2 - x1
        off = 2.0
        off_x = off if dx >= 0 else -off
        ha_align = "left" if off_x > 0 else "right"
        sel = edge["selectivity"]
        c = _RANK_COLORS[0]
        is_iso = sel == 0.0 and edge["solvent"] == "N/A"
        if is_iso:
            label = f"\u2212{edge['removed']}\n(isolated)"
        else:
            label = f"\u2212{edge['removed']}\n{edge['solvent']} ({sel:.1f}%)"
        ax.text(mx + off_x, my, label, ha=ha_align, va="center",
                fontsize=22, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c,
                          alpha=0.92, lw=1.2),
                zorder=6)

    # ── Nodes ──
    for mask, (x, y) in positions.items():
        is_topk = mask in topk_nodes
        is_full = mask == full_mask
        is_empty = mask == 0

        if is_full:
            fc, ec, tc = "#3498db", "#2c3e50", "white"
            r = r_end
        elif is_empty:
            fc, ec, tc = "#2ecc71", "#2c3e50", "white"
            r = r_end
        elif is_topk:
            fc, ec, tc = "#ffeaa7", "#f39c12", "#2c3e50"
            r = r_opt
        else:
            fc, ec = "#f5f6fa", "#dcdde1"
            tc = "#95a5a6"
            r = r_norm

        lw = 3.0 if (is_topk or is_full or is_empty) else 0.3
        circle = plt.Circle((x, y), r, facecolor=fc, edgecolor=ec,
                             linewidth=lw, zorder=4)
        ax.add_patch(circle)

        if is_full:
            ax.text(x, y, "FULL", ha="center", va="center",
                    fontsize=node_fs, fontweight="bold", color=tc, zorder=5)
        elif is_empty:
            ax.text(x, y, "\u2205", ha="center", va="center",
                    fontsize=node_fs + 2, fontweight="bold", color=tc, zorder=5)

    # ── Level labels ──
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

    # ── Colorbar ──
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Selectivity (%)", fontsize=legend_fs)
    cbar.ax.tick_params(labelsize=legend_fs - 4)

    # ── Top-K results table (below lattice) ──
    table_y = min(ys) - pad_y - 0.2
    line_h = 0.45
    ax.text(0, table_y, "RANKED SEQUENCES", ha="center", va="center",
            fontsize=result_fs + 4, fontweight="bold", color="#2c3e50")
    table_y -= line_h * 0.8

    # Header
    hdr = f"{'Rank':>4}  {'Min Sel':>8}  {'Sequence'}"
    ax.text(-(fig_w / 2 - pad_x), table_y, hdr, ha="left", va="center",
            fontsize=result_fs - 2, fontweight="bold", color="#2c3e50",
            family="monospace")
    table_y -= line_h * 0.6

    for rank, result in enumerate(top_k_results):
        path = result["path"]
        seq_str = " \u2192 ".join(e["removed"] for e in path)
        min_sel = result["min_sel"]
        if min_sel == float("inf"):
            min_sel = 0.0
        rank_num = rank + 1

        if rank < len(_RANK_COLORS):
            color = _RANK_COLORS[rank]
        else:
            color = _MUTED

        line = f"#{rank_num:<3}  {min_sel:>7.1f}%  {seq_str}"
        ax.text(-(fig_w / 2 - pad_x), table_y, line, ha="left", va="center",
                fontsize=result_fs - 2, fontweight="bold" if rank == 0 else "normal",
                color=color, family="monospace")
        table_y -= line_h

    # ── Legend ──
    items = [
        mpatches.Patch(fc="#3498db", ec="black", label="Start (full mixture)"),
        mpatches.Patch(fc="#ffeaa7", ec="#f39c12", label="Top-K path node"),
        mpatches.Patch(fc="#f5f6fa", ec="#dcdde1", label="Other state"),
        mpatches.Patch(fc="#2ecc71", ec="black", label="Goal (all separated)"),
    ]
    for rank in range(min(k, 5)):
        c = _RANK_COLORS[rank]
        items.append(mpatches.Patch(fc=c, ec="black", label=f"Rank {rank + 1}"))
    if k > 5:
        items.append(mpatches.Patch(fc=_MUTED, ec="black", label=f"Ranks 6\u2013{k}"))

    ax.legend(handles=items, loc="upper left", fontsize=legend_fs,
              frameon=True, fancybox=True, edgecolor="#bdc3c7",
              title="Legend", title_fontsize=legend_fs + 2,
              bbox_to_anchor=(0.0, 1.0))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.05)
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


# ── Top-K step-cascade comparison plot ────────────────────────────

_SHORT_SOLVENT = {
    "1,2-dimethylbenzene": "o-xylene", "dimethylsulfoxide": "DMSO",
    "dimethylformamide": "DMF", "propyleneglycol": "propylene glycol",
    "methylacetate": "methyl acetate", "diphenylether": "diphenyl ether",
    "ethylacetate": "ethyl acetate",
}


def _abbrev_solvent(name):
    """Shorten solvent name for cell display (max ~18 chars)."""
    short = _SHORT_SOLVENT.get(name, name)
    if len(short) > 18:
        short = short[:16] + ".."
    return short


def plot_ordering_feasibility(all_min_sels, output_path,
                              article_min_sel=None):
    """Histogram of min-selectivity across all n! orderings.

    Shows how many orderings are feasible (min-sel > 0%) vs infeasible,
    and where the article's ordering falls.
    """
    scores = [s for s, _ in all_min_sels]
    total = len(scores)
    feasible = sum(1 for s in scores if s > 0)
    infeasible = total - feasible

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")

    # Histogram
    bins = np.linspace(min(scores) - 1, max(scores) + 1, 80)
    n_vals, bin_edges, patches = ax.hist(scores, bins=bins, edgecolor="white",
                                          linewidth=0.5)

    # Color bars: green if bin center > 0, red otherwise
    for patch, left_edge in zip(patches, bin_edges[:-1]):
        center = left_edge + (bin_edges[1] - bin_edges[0]) / 2
        patch.set_facecolor("#27ae60" if center > 0 else "#e74c3c")

    # Zero line
    ax.axvline(0, color="#2c3e50", linewidth=2, linestyle="--", zorder=5)
    ax.text(0.3, ax.get_ylim()[1] * 0.95, "0% threshold",
            fontsize=9, color="#2c3e50", va="top")

    # Article marker
    if article_min_sel is not None:
        ax.axvline(article_min_sel, color="#e67e22", linewidth=2.5,
                   linestyle="-", zorder=6)
        ax.text(article_min_sel + 0.3, ax.get_ylim()[1] * 0.85,
                f"Article ordering\n({article_min_sel:.1f}%)",
                fontsize=9, fontweight="bold", color="#e67e22", va="top")

    # Annotations
    pct_feasible = 100 * feasible / total
    ax.text(0.98, 0.95,
            f"Feasible: {feasible:,} / {total:,} ({pct_feasible:.1f}%)\n"
            f"Infeasible: {infeasible:,} / {total:,} ({100 - pct_feasible:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#bdc3c7"))

    ax.set_xlabel("Min selectivity along ordering (%)", fontsize=11)
    ax.set_ylabel("Number of orderings", fontsize=11)
    ax.set_title(f"Polymer Separation Ordering Feasibility "
                 f"({total:,} orderings, {len(all_min_sels[0][1].split('→'))} polymers)",
                 fontsize=13, fontweight="bold", color="#2c3e50")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_top_k_comparison(data, all_min_sels, output_path, k=10,
                          article_steps=None, target_sels=None):
    """Grid plot comparing top-K sequences side-by-side.

    Rows = top-K sequences + optional article row.
    Columns = separation steps 1..n.
    Each cell shows polymer removed, solvent, temperature, and selectivity.

    Args:
        target_sels: optional list of target min-sel values to pick (closest
                     unique match for each). Overrides *k* when provided.
    """
    gsk, gsk_ml = _load_gsk_scores()
    polymers = data["polymers"]
    n = data["n"]
    full_mask = data["full_mask"]
    sel_cache = data["sel_cache"]

    # ── Deduplicate: one representative per unique min-sel (1 d.p.) ──
    unique_all = []
    seen_scores = set()
    for min_sel, seq_str in all_min_sels:
        key = round(min_sel, 1)
        if key in seen_scores:
            continue
        seen_scores.add(key)
        unique_all.append((min_sel, seq_str))

    # ── Select sequences ──
    if target_sels is not None:
        # Pick the closest unique sequence for each target
        unique = []
        used = set()
        for tgt in target_sels:
            best = min(unique_all, key=lambda x: abs(x[0] - tgt))
            if id(best) not in used:
                used.add(id(best))
                unique.append(best)
    else:
        unique = unique_all[:k]

    # ── Parse top-K sequences into step details ──
    rows = []  # list of (label, min_sel, steps_list)
    for rank_i, (min_sel, seq_str) in enumerate(unique):
        names = [s.strip() for s in seq_str.split("→")]
        idx_seq = [polymers.index(nm) for nm in names]
        remaining = full_mask
        steps = []
        for polymer_idx in idx_seq:
            is_isolation = bin(remaining).count("1") == 1
            if is_isolation:
                steps.append({
                    "polymer": polymers[polymer_idx], "solvent": None,
                    "temp": None, "selectivity": None, "gscore": None,
                    "gs_ml": False, "isolation": True,
                })
            else:
                solv, tmp, sel = sel_cache.get(
                    (polymer_idx, remaining), ("N/A", 0.0, 0.0))
                gs = gsk.get(solv.lower()) if solv != "N/A" else None
                is_ml = solv.lower() in gsk_ml if solv != "N/A" else False
                steps.append({
                    "polymer": polymers[polymer_idx], "solvent": solv,
                    "temp": tmp, "selectivity": sel, "gscore": gs,
                    "gs_ml": is_ml, "isolation": False,
                })
            remaining = remaining ^ (1 << polymer_idx)
        geo_gs = _geomean_gscore(steps)
        min_gs = min((s["gscore"] for s in steps
                      if not s.get("isolation") and s.get("gscore")),
                     default=0.0)
        label = f"#{rank_i + 1}\n{min_sel:.1f}%\nG\u0305={geo_gs:.1f} (min {min_gs:.1f})"
        rows.append((label, min_sel, steps))

    # ── Article row ──
    has_article = article_steps is not None
    if has_article:
        art_row_steps = []
        art_min = float("inf")
        for s in article_steps:
            if s["is_last"]:
                art_row_steps.append({
                    "polymer": s["target"], "solvent": None,
                    "temp": None, "selectivity": None, "gscore": None,
                    "gs_ml": False, "isolation": True,
                })
            else:
                sel_val = s.get("selectivity")
                art_solv = s.get("solvent", "N/A")
                gs = gsk.get(art_solv.lower()) if art_solv != "N/A" else None
                is_ml = art_solv.lower() in gsk_ml if art_solv != "N/A" else False
                art_row_steps.append({
                    "polymer": s["target"],
                    "solvent": art_solv,
                    "temp": s.get("temp_c"),
                    "selectivity": sel_val,
                    "gscore": gs,
                    "gs_ml": is_ml, "isolation": False,
                })
                if sel_val is not None:
                    art_min = min(art_min, sel_val)
        art_geo_gs = _geomean_gscore(art_row_steps)
        art_min_gs = min((s["gscore"] for s in art_row_steps
                          if not s.get("isolation") and s.get("gscore")),
                         default=0.0)
        art_label = f"Article\n{art_min:.1f}%\nG\u0305={art_geo_gs:.1f} (min {art_min_gs:.1f})"
        rows.append((art_label, art_min, art_row_steps))

    n_steps = n
    n_rows = len(rows)

    # ── Layout ──
    cell_w, cell_h = 2.2, 1.3
    left_margin = 2.2
    top_margin = 0.9
    bot_margin = 1.0
    right_margin = 0.5
    fig_w = left_margin + n_steps * cell_w + right_margin
    fig_h = top_margin + n_rows * cell_h + bot_margin

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Title ──
    ax.text(fig_w / 2, fig_h - 0.15,
            f"Separation Sequences (by min selectivity)",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color="#2c3e50")

    # ── Column headers ──
    for col in range(n_steps):
        cx = left_margin + col * cell_w + cell_w / 2
        cy = fig_h - top_margin + 0.15
        header = f"Step {col + 1}" if col < n_steps - 1 else "Isolation"
        ax.text(cx, cy, header, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#2c3e50")

    # ── Draw cells ──
    for row_i, (label, _min_sel, steps) in enumerate(rows):
        y_top = fig_h - top_margin - row_i * cell_h
        is_article_row = has_article and row_i == n_rows - 1

        # Row label
        ax.text(left_margin - 0.15, y_top - cell_h / 2, label,
                ha="right", va="center", fontsize=8, fontweight="bold",
                color="#e74c3c" if is_article_row else "#2c3e50")

        for col, step in enumerate(steps):
            x_left = left_margin + col * cell_w
            sel_val = step["selectivity"]
            is_iso = step["isolation"]
            bg = "#95a5a6" if is_iso else sel_color(sel_val)

            rect = mpatches.FancyBboxPatch(
                (x_left + 0.05, y_top - cell_h + 0.05),
                cell_w - 0.10, cell_h - 0.10,
                boxstyle="round,pad=0.04", facecolor=bg,
                edgecolor="#e74c3c" if is_article_row else "#7f8c8d",
                linewidth=2.5 if is_article_row else 0.8)
            ax.add_patch(rect)

            # Text color: white on dark cells
            txt_col = "white" if (sel_val is not None and sel_val < 10) else "#2c3e50"
            if is_iso:
                txt_col = "white"

            cx = x_left + cell_w / 2
            cy_mid = y_top - cell_h / 2

            # Line 1: polymer name (bold)
            ax.text(cx, cy_mid + 0.35, step["polymer"],
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color=txt_col)

            # Line 2: solvent
            if is_iso:
                solv_txt = "(isolation)"
            else:
                solv_txt = _abbrev_solvent(step["solvent"] or "N/A")
            ax.text(cx, cy_mid + 0.12, solv_txt,
                    ha="center", va="center", fontsize=8, color=txt_col)

            # Line 3: temp | sel
            if is_iso:
                detail = ""
            elif sel_val is not None:
                temp_str = f"{step['temp']:.0f}" if step["temp"] else "?"
                detail = f"{temp_str}°C | {sel_val:.1f}%"
            else:
                detail = "N/A"
            ax.text(cx, cy_mid - 0.10, detail,
                    ha="center", va="center", fontsize=8, color=txt_col)

            # Line 4: G-score (* = ML-predicted)
            gs = step.get("gscore")
            if is_iso:
                gs_txt = ""
            elif gs is not None:
                star = "*" if step.get("gs_ml") else ""
                gs_txt = f"G={gs:.1f}{star}"
            else:
                gs_txt = "G=n/a"
            ax.text(cx, cy_mid - 0.32, gs_txt,
                    ha="center", va="center", fontsize=7,
                    fontstyle="italic", color=txt_col)

    # ── Legend ──
    legend_items = [
        mpatches.Patch(fc="#27ae60", ec="black", label="Sel > 30%"),
        mpatches.Patch(fc="#f39c12", ec="black", label="Sel 10\u201330%"),
        mpatches.Patch(fc="#e67e22", ec="black", label="Sel 0\u201310%"),
        mpatches.Patch(fc="#e74c3c", ec="black", label="Sel < 0%"),
        mpatches.Patch(fc="#95a5a6", ec="black", label="N/A / isolation"),
    ]
    ax.legend(handles=legend_items, loc="lower center",
              ncol=5, fontsize=8, frameon=True, fancybox=True,
              edgecolor="#bdc3c7", bbox_to_anchor=(0.5, -0.01),
              bbox_transform=ax.transAxes)

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
    parser.add_argument("--top-k", type=int, default=0,
                        help="Extract and plot top-K sequences (e.g., --top-k 10)")
    parser.add_argument("--all", action="store_true",
                        help="Enumerate all n! sequences; shade edges by best-path metric")
    parser.add_argument("-t", "--temp", type=float, default=120.0,
                        help="Temperature in C (default: 120)")
    parser.add_argument("-o", "--output", default=None, help="Output path")
    parser.add_argument("--rank-article", action="store_true",
                        help="Rank the reference article's 8-polymer sequence against all permutations")
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

    if args.all:
        from math import factorial

        # ── Figure 1: Selectivity landscape ──
        print(f"\n{'='*60}")
        print(f"FIGURE 1: SELECTIVITY LANDSCAPE")
        print(f"{'='*60}")
        print(f"Enumerating all {factorial(n):,} sequences (selectivity)...")
        t1 = time.time()
        edge_best, all_min_sels, stats = enumerate_all_sequences(data)
        dt2 = time.time() - t1
        print(f"  Enumeration time: {dt2:.2f}s")
        print(f"  Best min sel:     {stats['best']:.1f}%")
        print(f"  Median min sel:   {stats['median']:.1f}%")
        print(f"  Mean min sel:     {stats['mean']:.1f}%")
        print(f"  Worst min sel:    {stats['worst']:.1f}%")

        top_k_k = max(args.top_k, 10)
        top_k_results = top_k_unique_from_enumeration(data, all_min_sels, k=top_k_k)

        print(f"\nTop {len(top_k_results)} sequences (unique min-selectivity):")
        for rank, result in enumerate(top_k_results, 1):
            path = result["path"]
            seq_str = arrow.join(e["removed"] for e in path)
            min_sel = result["min_sel"]
            if min_sel == float("inf"):
                min_sel = 0.0
            print(f"  #{rank:>2}: min_sel={min_sel:>6.1f}%  {seq_str}")

        out_sel = args.output or str(HERE / "dp_lattice_sweep" / f"dp_lattice_n{n}_all_selectivity.png")
        plot_dp_lattice_all(data, edge_best, all_min_sels, stats, out_sel,
                            top_k_results=top_k_results, mode="selectivity")

        # ── Figure 2: Safety landscape ──
        print(f"\n{'='*60}")
        print(f"FIGURE 2: SAFETY (G-SCORE) LANDSCAPE")
        print(f"{'='*60}")
        print(f"Building safety-optimized DP data...")
        t2 = time.time()
        data_safety = build_dp_data_safety(polys, temperature=args.temp)
        dt3 = time.time() - t2
        print(f"  Safety cache build: {dt3:.1f}s")

        print(f"Enumerating all {factorial(n):,} sequences (safety)...")
        t3 = time.time()
        edge_best_s, all_scores_s, stats_s = enumerate_all_sequences_safety(data_safety)
        dt4 = time.time() - t3
        print(f"  Enumeration time: {dt4:.2f}s")
        print(f"  Best min G-score: {stats_s['best']:.1f}")
        print(f"  Median min G:     {stats_s['median']:.1f}")
        print(f"  Mean min G:       {stats_s['mean']:.1f}")
        print(f"  Worst min G:      {stats_s['worst']:.1f}")

        top_k_safety = top_k_unique_safety_from_enumeration(
            data_safety, all_scores_s, k=top_k_k,
        )

        print(f"\nTop {len(top_k_safety)} sequences (unique min G-score):")
        for rank, result in enumerate(top_k_safety, 1):
            path = result["path"]
            seq_str = arrow.join(e["removed"] for e in path)
            min_gs = result["min_gs"]
            if min_gs == float("inf"):
                min_gs = 0.0
            sels = [e["selectivity"] for e in path if e["selectivity"] != 0.0]
            min_sel = min(sels) if sels else 0.0
            print(f"  #{rank:>2}: min_G={min_gs:>4.1f}  min_sel={min_sel:>6.1f}%  {seq_str}")

        out_safe = str(HERE / "dp_lattice_sweep" / f"dp_lattice_n{n}_all_safety.png")
        plot_dp_lattice_all(data_safety, edge_best_s, all_scores_s, stats_s,
                            out_safe, top_k_results=top_k_safety, mode="safety")

    elif args.top_k > 0:
        print(f"\nExtracting top-{args.top_k} sequences...")
        t1 = time.time()
        top_k_results = extract_top_k_paths(data, k=args.top_k)
        dt2 = time.time() - t1
        print(f"  Beam DP time: {dt2:.2f}s")
        print(f"  Sequences found: {len(top_k_results)}\n")

        for rank, result in enumerate(top_k_results, 1):
            path = result["path"]
            seq_str = arrow.join(e["removed"] for e in path)
            min_sel = result["min_sel"]
            if min_sel == float("inf"):
                min_sel = 0.0
            print(f"  #{rank:>2}: min_sel={min_sel:>6.1f}%  {seq_str}")

        out = args.output or str(HERE / "dp_lattice_sweep" / f"dp_lattice_n{n}_topk.png")
        plot_dp_lattice_topk(data, top_k_results, out)
    elif not args.rank_article:
        if args.cascade:
            out = args.output or str(HERE / "dp_algorithm_visual.png")
            plot_dp_cascade(data, out)
        else:
            out = args.output or str(HERE / "dp_algorithm_visual.png")
            plot_dp_lattice(data, out)

    # ── Article sequence ranking ─────────────────────────────────
    if args.rank_article:
        from math import factorial

        polys_art = list(ARTICLE_POLYMERS)
        n_art = len(polys_art)
        n_perm = factorial(n_art)

        print(f"\n{'='*60}")
        print(f"ARTICLE SEQUENCE RANKING ({n_art} polymers)")
        print(f"{'='*60}")

        # Build selectivity DP for the 9 article polymers (sweep 25-160°C)
        _trange = list(range(25, 165, 5))
        _banned = {"benzene"}
        print(f"\nBuilding DP data for {n_art} polymers @ {_trange[0]}-{_trange[-1]}\u00b0C "
              f"({len(_trange)} temps, excluding {_banned})...")
        t_a = time.time()
        data_art = build_dp_data(polys_art, temp_range=_trange,
                                 banned_solvents=_banned)
        dt_a = time.time() - t_a
        print(f"  DP build time: {dt_a:.1f}s")
        print(f"  System optimal min-sel: {data_art['opt_min']:.1f}%")

        # Enumerate all 9! selectivity sequences
        print(f"\nEnumerating all {n_perm:,} sequences (selectivity)...")
        t_e = time.time()
        _, all_min_sels_art, stats_art = enumerate_all_sequences(data_art)
        dt_e = time.time() - t_e
        print(f"  Enumeration time: {dt_e:.2f}s")

        # Walk article ordering through system's sel_cache
        sys_steps, sys_min_sel = _walk_article_ordering_through_cache(
            data_art["sel_cache"], polys_art)
        sel_rank = rank_in_enumeration(all_min_sels_art, sys_min_sel)

        # Evaluate article's exact (solvent, temp) pairs
        art_steps, art_min_sel, art_min_gs = evaluate_article_exact(
            temperature=args.temp)

        # Build safety DP and enumerate
        print(f"Building safety DP for {n_art} polymers @ {_trange[0]}-{_trange[-1]}\u00b0C "
              f"(excluding {_banned})...")
        t_s = time.time()
        data_arts = build_dp_data_safety(polys_art, temp_range=_trange,
                                         banned_solvents=_banned)
        dt_s = time.time() - t_s
        print(f"  Safety DP build time: {dt_s:.1f}s")

        print(f"Enumerating all {n_perm:,} sequences (safety)...")
        t_se = time.time()
        _, all_scores_art, stats_arts = enumerate_all_sequences_safety(data_arts)
        dt_se = time.time() - t_se
        print(f"  Enumeration time: {dt_se:.2f}s")

        # Walk article ordering through safety cache for ordering-based G-rank
        n_art_full = (1 << n_art) - 1
        remaining_s = n_art_full
        sys_min_gs = float("inf")
        for step in ARTICLE_SEQUENCE:
            tidx = polys_art.index(step["target"])
            if bin(remaining_s).count("1") == 1:
                remaining_s = remaining_s ^ (1 << tidx)
                continue
            entry = data_arts["sel_cache"].get((tidx, remaining_s), ("N/A", 0.0, 0.0, 0.0))
            _, _, _, gs = entry
            sys_min_gs = min(sys_min_gs, gs)
            remaining_s = remaining_s ^ (1 << tidx)

        gs_rank = rank_in_enumeration(all_scores_art, sys_min_gs)

        # ── Print report ──
        print(f"\n{'='*60}")
        print(f"ORDERING RANK (best solvents @ {_trange[0]}-{_trange[-1]}\u00b0C):")
        print(f"{'='*60}")
        print(f"  Article ordering min-sel: {sys_min_sel:>6.1f}%  "
              f"\u2192  Rank {sel_rank['rank']:,} / {sel_rank['total']:,} "
              f"({sel_rank['percentile']:.1f}th percentile)")
        print(f"  System optimal min-sel:   {data_art['opt_min']:>6.1f}%  \u2192  Rank 1")
        print(f"  Median min-sel:           {stats_art['median']:>6.1f}%")

        # Step-by-step table
        _col_sys = "System-Solvent (T\u00b0C)"
        _col_art = "Article-Solvent (T\u00b0C)"
        hdr = (f"  {'Step':>4}  {'Target':<8}  {_col_sys:<28} {'Sel%':>7}  "
               f"  {_col_art:<24} {'Sel%':>7}  {'Art wt%':>7}")
        sep = "  " + "\u2500" * (len(hdr) - 2)
        print(f"\n{hdr}")
        print(sep)
        for i, (ss, arts) in enumerate(zip(sys_steps, art_steps), 1):
            # System side
            if ss["is_last"]:
                sys_solv_str = "(isolation)"
                sys_sel_str = "\u2014"
            else:
                sys_solv_str = f"{ss['solvent']} ({ss['temperature']:.0f}\u00b0C)"
                sys_sel_str = f"{ss['selectivity']:.1f}"

            # Article side
            if arts["is_last"]:
                art_solv_str = "(isolation)"
                art_sel_str = "\u2014"
            elif arts["is_na"]:
                art_solv_str = f"{arts['solvent']} ({arts['temp_c']}\u00b0C)"
                art_sel_str = "N/A"
            else:
                art_solv_str = f"{arts['solvent']} ({arts['temp_c']}\u00b0C)"
                art_sel_str = f"{arts['selectivity']:.1f}"

            art_wt_str = f"{arts['article_wt']:.2f}"

            print(f"  {i:>4}  {ss['target']:<8}  {sys_solv_str:<28} {sys_sel_str:>7}  "
                  f"  {art_solv_str:<24} {art_sel_str:>7}  {art_wt_str:>7}")

        print(f"\n{'='*60}")
        print(f"SAFETY RANK (G-score):")
        print(f"{'='*60}")
        print(f"  Article ordering min-G:   {sys_min_gs:>6.1f}  "
              f"\u2192  Rank {gs_rank['rank']:,} / {gs_rank['total']:,} "
              f"({gs_rank['percentile']:.1f}th percentile)")
        print(f"  System optimal min-G:     {stats_arts['best']:>6.1f}  \u2192  Rank 1")
        print(f"  Median min-G:             {stats_arts['median']:>6.1f}")

        # Article's exact eval min-G (from its own solvents)
        print(f"\n  Article exact eval (own solvents/temps):")
        print(f"    Min selectivity: {art_min_sel:.1f}%")
        print(f"    Min G-score:     {art_min_gs:.1f}")

        print(f"\n{'='*60}")

        # ── Ordering feasibility histogram ──
        out_feas = str(HERE / "dp_lattice_sweep" / "ordering_feasibility.png")
        plot_ordering_feasibility(all_min_sels_art, out_feas,
                                  article_min_sel=sys_min_sel)

        # ── Top-5 comparison plot ──
        out_cmp = str(HERE / "dp_lattice_sweep" / "top_k_comparison.png")
        plot_top_k_comparison(data_art, all_min_sels_art, out_cmp, k=5,
                              article_steps=None)

        # ── Selectivity spread comparison plot (with article row) ──
        out_spread = str(HERE / "dp_lattice_sweep" / "selectivity_spread.png")
        plot_top_k_comparison(data_art, all_min_sels_art, out_spread,
                              target_sels=[15.0, 10.0, 5.0, 0.0],
                              article_steps=art_steps)
