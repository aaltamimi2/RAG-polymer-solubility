"""Visualize the Branch and Bound separation algorithm.

Shows the search tree with explored/pruned branches, priority queue
exploration order, and bound comparisons. Uses 4 polymers for clarity.
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
import heapq

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


# ── Simulate B&B with full exploration log ────────────────────────

class TreeNode:
    """Node in the B&B search tree."""
    __slots__ = ("seq", "remaining", "min_sel", "sel_here", "solvent",
                 "status", "order", "children", "polymer_removed")

    def __init__(self, seq, remaining, min_sel):
        self.seq = tuple(seq)
        self.remaining = frozenset(remaining)
        self.min_sel = min_sel
        self.sel_here = None      # selectivity of last removal step
        self.solvent = None       # solvent used for last removal
        self.status = "pending"   # "expanded", "pruned", "complete", "pending"
        self.order = None         # exploration order
        self.children = []        # list of TreeNode
        self.polymer_removed = None


def simulate_bb(polymers, temperature=120.0):
    """Run B&B, building an explicit search tree with exploration info."""
    n = len(polymers)

    # Build the tree nodes indexed by sequence tuple
    node_map = {}  # tuple(seq) -> TreeNode

    root = TreeNode([], set(range(n)), float("inf"))
    root.status = "expanded"
    root.order = 0
    node_map[()] = root

    # Priority queue: (-min_sel, seq_list, remaining_set)
    pq = [(0.0, [], set(range(n)))]
    best_min = -float("inf")
    best_seq = None
    order = 0

    while pq:
        neg_min, seq, remaining = heapq.heappop(pq)
        cur_min = -neg_min if seq else float("inf")
        order += 1

        seq_key = tuple(seq)

        # Pruning
        if cur_min < best_min:
            node = node_map.get(seq_key)
            if node:
                node.status = "pruned"
                node.order = order
            continue

        # Complete solution
        if not remaining:
            node = node_map.get(seq_key)
            if node:
                node.status = "complete"
                node.order = order
            if cur_min > best_min:
                best_min = cur_min
                best_seq = list(seq)
            continue

        # Expand: create children
        node = node_map.get(seq_key)
        if node:
            node.status = "expanded"
            node.order = order

        for pidx in sorted(remaining):
            others = remaining - {pidx}
            target = polymers[pidx]
            other_names = [polymers[i] for i in sorted(others)]

            if other_names:
                results = get_all_solvents_selectivity(target, other_names, temperature)
                sel = results[0]["selectivity"] if results else 0.0
                solvent = results[0]["solvent"] if results else "N/A"
            else:
                sel = 100.0  # Last polymer: trivially isolated
                solvent = "(isolated)"

            new_min = min(cur_min, sel) if seq else sel
            child_key = tuple(seq + [pidx])

            child = TreeNode(seq + [pidx], others, new_min)
            child.sel_here = sel
            child.solvent = solvent
            child.polymer_removed = polymers[pidx]

            node_map[child_key] = child
            if node:
                node.children.append(child)

            if new_min >= best_min:
                heapq.heappush(pq, (-new_min, seq + [pidx], others))
            else:
                child.status = "pruned"

    return root, node_map, best_seq, best_min, order


# ── Tree layout algorithm ─────────────────────────────────────────

def layout_tree(root, x_start=0, x_gap=2.4, y_gap=3.0):
    """Assign (x, y) positions to tree nodes using a simple recursive layout."""
    positions = {}
    _next_x = [x_start]

    def assign_positions(node, depth):
        if not node.children:
            # Leaf node
            x = _next_x[0]
            _next_x[0] += x_gap
            y = -depth * y_gap
            positions[node.seq] = (x, y)
            return

        # Recurse on children first
        for child in node.children:
            assign_positions(child, depth + 1)

        # Center parent above children
        child_xs = [positions[c.seq][0] for c in node.children]
        x = (min(child_xs) + max(child_xs)) / 2
        y = -depth * y_gap
        positions[node.seq] = (x, y)

    assign_positions(root, 0)
    return positions


# ── Main plot ─────────────────────────────────────────────────────

def plot_bb_tree(root, node_map, polymers, best_seq, best_min, n_explored,
                 output_path):
    n = len(polymers)

    # Compute optimal path keys (from best_seq)
    opt_keys = set()
    if best_seq:
        for k in range(len(best_seq) + 1):
            opt_keys.add(tuple(best_seq[:k]))

    # Layout
    positions = layout_tree(root, x_start=0, x_gap=1.8, y_gap=2.8)

    # Centering: shift so tree is centered at x=0
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    cx = (min(all_x) + max(all_x)) / 2
    positions = {k: (x - cx, y) for k, (x, y) in positions.items()}
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]

    pad_x, pad_y = 2.5, 2.0
    fig_w = max(14, max(all_x) - min(all_x) + 2 * pad_x)
    fig_h = max(all_y) - min(all_y) + 2 * pad_y + 4.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(min(all_x) - pad_x, max(all_x) + pad_x)
    ax.set_ylim(min(all_y) - pad_y - 1.5, max(all_y) + pad_y + 3)
    ax.axis("off")

    # ── Title ──
    top = max(all_y) + pad_y + 1.5
    ax.text(0, top + 0.4,
            "BRANCH AND BOUND SEPARATOR",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="#2c3e50")
    n_factorial = int(np.prod(range(1, n + 1)))
    n_pruned = sum(1 for nd in node_map.values() if nd.status == "pruned")
    n_complete = sum(1 for nd in node_map.values() if nd.status == "complete")
    ax.text(0, top - 0.2,
            f"{n} polymers | O(n!) = {n_factorial} worst case | "
            f"{n_explored} nodes explored | {n_pruned} pruned | "
            f"{n_complete} complete solutions found",
            ha="center", va="center", fontsize=15, color="#2c3e50")
    ax.text(0, top - 0.7,
            "Best-first search (priority queue) · Prunes branches below current best",
            ha="center", va="center", fontsize=13, color="#7f8c8d",
            style="italic")

    # ── Draw edges ──
    def draw_edges(node):
        if node.seq not in positions:
            return
        px, py = positions[node.seq]
        for child in node.children:
            if child.seq not in positions:
                continue
            cx_pos, cy = positions[child.seq]
            is_opt = (node.seq in opt_keys and child.seq in opt_keys)
            is_pruned = child.status == "pruned"

            if is_opt:
                c = "#2ecc71"
                lw = 3.0
                alpha = 1.0
            elif is_pruned:
                c = "#e74c3c"
                lw = 1.2
                alpha = 0.5
            else:
                c = "#3498db"
                lw = 1.5
                alpha = 0.7

            ax.plot([px, cx_pos], [py - 0.4, cy + 0.55], "-",
                    color=c, linewidth=lw, alpha=alpha, zorder=1)

            # Edge label: polymer removed + selectivity
            mx = (px + cx_pos) / 2
            my = (py - 0.4 + cy + 0.55) / 2
            if child.polymer_removed and child.sel_here is not None:
                sel_val = child.sel_here
                solv = child.solvent or ""
                if solv == "(isolated)":
                    elabel = f"−{child.polymer_removed}\n(isolated)"
                else:
                    elabel = f"−{child.polymer_removed}\n{sel_val:.1f}%"
                ec_color = sel_color(sel_val) if not is_pruned else "#bdc3c7"
                fs = 11 if is_opt else 10
                fw = "bold" if is_opt else "normal"
                text_alpha = 1.0 if not is_pruned else 0.5
                ax.text(mx + 0.15, my, elabel, ha="left", va="center",
                        fontsize=fs, fontweight=fw, color=ec_color,
                        alpha=text_alpha, zorder=6)

            # Draw children recursively
            draw_edges(child)

    draw_edges(root)

    # ── Draw nodes ──
    def draw_nodes(node):
        if node.seq not in positions:
            return
        x, y = positions[node.seq]
        is_opt = node.seq in opt_keys
        is_root = len(node.seq) == 0
        is_complete = node.status == "complete"
        is_pruned = node.status == "pruned"

        # Node appearance
        if is_root:
            fc = "#3498db"
            ec = "#2c3e50"
            tc = "white"
        elif is_opt and is_complete:
            fc = "#2ecc71"
            ec = "#2c3e50"
            tc = "white"
        elif is_opt:
            fc = "#ffeaa7"
            ec = "#27ae60"
            tc = "#2c3e50"
        elif is_pruned:
            fc = "#fadbd8"
            ec = "#e74c3c"
            tc = "#95a5a6"
        elif is_complete:
            fc = "#d5f5e3"
            ec = "#27ae60"
            tc = "#2c3e50"
        else:
            fc = "#ebf5fb"
            ec = "#3498db"
            tc = "#2c3e50"

        lw = 2.5 if (is_opt or is_root) else (1.5 if is_pruned else 1.2)

        # Box for the node
        box_w = 2.0
        box_h = 1.1
        box = FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                              boxstyle="round,pad=0.1",
                              facecolor=fc, edgecolor=ec,
                              linewidth=lw, zorder=4)
        ax.add_patch(box)

        # Node label
        if is_root:
            remaining_names = [polymers[i] for i in sorted(node.remaining)]
            label = ", ".join(remaining_names)
            fs = 10
        elif not node.remaining:
            label = "∅"
            fs = 16
        else:
            remaining_names = [polymers[i] for i in sorted(node.remaining)]
            label = ", ".join(remaining_names)
            fs = 10

        ax.text(x, y + 0.05, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if is_opt else "normal",
                color=tc, zorder=5)

        # Min selectivity bound below node
        if node.min_sel is not None and node.min_sel < float("inf"):
            bound_text = f"bound: {node.min_sel:.1f}%"
            ax.text(x, y - box_h / 2 - 0.15, bound_text,
                    ha="center", va="top", fontsize=10,
                    color="#636e72", style="italic", zorder=5)

        # Exploration order badge (top-left corner)
        if node.order is not None and node.order > 0:
            badge_x = x - box_w / 2 - 0.1
            badge_y = y + box_h / 2 + 0.1
            badge_r = 0.32
            badge_fc = "#2c3e50" if not is_pruned else "#e74c3c"
            ax.add_patch(plt.Circle((badge_x, badge_y), badge_r,
                                    facecolor=badge_fc, edgecolor="white",
                                    linewidth=1.0, zorder=7))
            ax.text(badge_x, badge_y, str(node.order),
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=8)

        # Pruning X marker
        if is_pruned:
            ax.text(x + box_w / 2 + 0.15, y, "✗",
                    ha="left", va="center", fontsize=18,
                    color="#e74c3c", fontweight="bold", zorder=6)

        # Recurse
        for child in node.children:
            draw_nodes(child)

    draw_nodes(root)

    # ── Result box at bottom ──
    if best_seq:
        seq_str = " → ".join(polymers[i] for i in best_seq)
        ry = min(all_y) - pad_y
        rw = max(all_x) - min(all_x) + 3
        result = FancyBboxPatch((-rw / 2, ry - 0.35), rw, 0.7,
                                 boxstyle="round,pad=0.12",
                                 facecolor="#2ecc71", edgecolor="#2c3e50",
                                 linewidth=2.5)
        ax.add_patch(result)
        ax.text(0, ry,
                f"OPTIMAL: {seq_str}  |  "
                f"Min selectivity: {best_min:.1f}%",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="white")

    # ── Legend ──
    items = [
        mpatches.Patch(fc="#3498db", ec="black", label="Root (full mixture)"),
        mpatches.Patch(fc="#ebf5fb", ec="#3498db", label="Explored node"),
        mpatches.Patch(fc="#ffeaa7", ec="#27ae60", label="Optimal path"),
        mpatches.Patch(fc="#d5f5e3", ec="#27ae60", label="Complete solution"),
        mpatches.Patch(fc="#2ecc71", ec="black", label="Best solution"),
        mpatches.Patch(fc="#fadbd8", ec="#e74c3c", label="Pruned (bound < best)"),
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

    print("Simulating Branch & Bound for 4 polymers...")
    root, node_map, best_seq, best_min, n_explored = simulate_bb(
        poly4, temperature=120.0
    )

    print(f"  Nodes explored: {n_explored}")
    print(f"  Best min selectivity: {best_min:.1f}%")
    if best_seq:
        print(f"  Best sequence: {' → '.join(poly4[i] for i in best_seq)}")

    n_expanded = sum(1 for n in node_map.values() if n.status == "expanded")
    n_pruned = sum(1 for n in node_map.values() if n.status == "pruned")
    n_complete = sum(1 for n in node_map.values() if n.status == "complete")
    print(f"  Expanded: {n_expanded}, Pruned: {n_pruned}, Complete: {n_complete}")

    out = HERE / "bb_algorithm_visual.png"
    plot_bb_tree(root, node_map, poly4, best_seq, best_min, n_explored, str(out))
