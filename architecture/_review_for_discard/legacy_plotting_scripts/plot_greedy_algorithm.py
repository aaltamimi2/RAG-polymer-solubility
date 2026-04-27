"""Visualize the greedy separation algorithm decision process.

Same visual style as the separation tree diagrams from advanced_separation.py.
Shows the candidate evaluation and selection at each step.
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


# ── Rebuild greedy data with full candidate info ──────────────────────

def build_greedy_with_candidates(polymer_list, temperature=120.0):
    """Run greedy algorithm, recording ALL candidates at each step."""
    remaining = list(polymer_list)
    steps = []

    while len(remaining) > 1:
        candidates = []
        for target in remaining:
            others = [p for p in remaining if p != target]
            results = get_all_solvents_selectivity(target, others, temperature)
            best = results[0] if results else {"solvent": "N/A", "selectivity": 0}
            candidates.append({
                "polymer": target,
                "solvent": best["solvent"],
                "selectivity": best["selectivity"],
            })

        # Sort by selectivity descending
        candidates.sort(key=lambda c: c["selectivity"], reverse=True)
        winner = candidates[0]

        steps.append({
            "remaining": remaining.copy(),
            "candidates": candidates,
            "selected": winner,
        })
        remaining.remove(winner["polymer"])

    steps.append({
        "remaining": remaining.copy(),
        "candidates": [],
        "selected": {"polymer": remaining[0], "solvent": "(isolated)", "selectivity": None},
    })
    return steps


# ── Color helpers (matching separation tree style) ────────────────────

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


def sel_color_light(sel):
    if sel is None:
        return "#ecf0f1"
    if sel > 30:
        return "#d5f5e3"
    if sel > 10:
        return "#fef9e7"
    if sel > 0:
        return "#fdebd0"
    return "#fadbd8"


# ── Main plot ─────────────────────────────────────────────────────────

def plot_greedy_algorithm(steps, polymer_list, temperature, output_path):
    n_steps = len(steps) - 1  # last step is isolation, no candidates

    # Layout: each step gets a row with candidate bars + selection arrow
    row_height = 2.2
    fig_width = 14
    fig_height = 2.5 + n_steps * row_height + 1.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-1.5, n_steps * row_height + 3.0)
    ax.axis("off")

    # ── Title ──
    top_y = n_steps * row_height + 2.2
    ax.text(6.75, top_y + 0.3,
            "GREEDY SEPARATION ALGORITHM",
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="#2c3e50")
    ax.text(6.75, top_y - 0.15,
            f"{len(polymer_list)} polymers | {temperature}°C | "
            f"O(n²) ≈ {len(polymer_list)**2} evaluations vs "
            f"{len(polymer_list)}! = {int(np.prod(range(1, len(polymer_list)+1))):,} exhaustive",
            ha="center", va="center", fontsize=15, color="#7f8c8d")

    # ── Starting mixture bar ──
    mix_y = n_steps * row_height + 0.8
    mix_box = FancyBboxPatch((1.0, mix_y - 0.3), 11.5, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor="#3498db", edgecolor="#2c3e50",
                              linewidth=2)
    ax.add_patch(mix_box)
    ax.text(6.75, mix_y,
            f'STARTING MIXTURE:  {", ".join(polymer_list)}',
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white")

    # ── Steps ──
    for step_idx in range(n_steps):
        step = steps[step_idx]
        candidates = step["candidates"]
        selected = step["selected"]
        remaining = step["remaining"]
        n_cands = len(candidates)

        y_base = (n_steps - 1 - step_idx) * row_height + 0.3

        # Central down-arrow from previous level
        arrow_top = y_base + row_height - 0.15
        arrow_bot = y_base + row_height - 0.65
        ax.annotate("", xy=(6.75, arrow_bot), xytext=(6.75, arrow_top),
                     arrowprops=dict(arrowstyle="-|>", lw=2.5,
                                     color="#2c3e50", mutation_scale=20))

        # Step number badge
        badge_x = 0.3
        badge_y = y_base + 0.85
        ax.add_patch(plt.Circle((badge_x, badge_y), 0.35,
                                facecolor="#2c3e50", edgecolor="white",
                                linewidth=2, zorder=5))
        ax.text(badge_x, badge_y, str(step_idx + 1),
                ha="center", va="center", fontsize=16,
                fontweight="bold", color="white", zorder=6)

        # "Evaluate N candidates" label
        ax.text(1.1, badge_y + 0.15,
                f"Evaluate {n_cands} candidates",
                ha="left", va="center", fontsize=14, color="#2c3e50",
                style="italic")
        # Split remaining polymers across two lines to avoid overlapping bars
        rem_names = remaining
        mid = (len(rem_names) + 1) // 2
        line1 = ", ".join(rem_names[:mid])
        line2 = ", ".join(rem_names[mid:])
        rem_text = f"{{{line1},\n  {line2}}}" if line2 else f"{{{line1}}}"
        ax.text(1.1, badge_y - 0.25,
                f"Remaining: {rem_text}",
                ha="left", va="center", fontsize=13, color="#2c3e50",
                linespacing=1.3)

        # Candidate horizontal bar chart
        bar_left = 4.2
        bar_width_max = 5.0
        bar_height = 0.28
        max_sel = max(c["selectivity"] for c in candidates) if candidates else 1
        max_sel = max(max_sel, 1)  # avoid div by zero

        # Show top candidates (limit to avoid overcrowding)
        show_n = min(n_cands, 5)
        bar_spacing = min(0.38, (row_height - 0.8) / show_n)

        for i in range(show_n):
            c = candidates[i]
            is_winner = (c["polymer"] == selected["polymer"])
            sel = c["selectivity"]
            bw = max(0.05, (sel / max_sel) * bar_width_max) if sel > 0 else 0.15
            by = y_base + (show_n - 1 - i) * bar_spacing + 0.15

            color = sel_color(sel)
            alpha = 1.0 if is_winner else 0.4
            lw = 2.5 if is_winner else 0.8
            ec = "#2c3e50" if is_winner else "#bdc3c7"

            # Bar
            bar = FancyBboxPatch((bar_left, by - bar_height / 2), bw, bar_height,
                                  boxstyle="round,pad=0.02",
                                  facecolor=color, edgecolor=ec,
                                  linewidth=lw, alpha=alpha, zorder=3)
            ax.add_patch(bar)

            # Polymer label (left of bar)
            fw = "bold" if is_winner else "normal"
            fs = 14 if is_winner else 12
            ax.text(bar_left - 0.1, by,
                    c["polymer"], ha="right", va="center",
                    fontsize=fs, fontweight=fw,
                    color="#2c3e50" if is_winner else "#95a5a6")

            # Selectivity + solvent (right of bar)
            sel_text = f'{sel:.1f}%  ({c["solvent"]})'
            ax.text(bar_left + bw + 0.15, by,
                    sel_text, ha="left", va="center",
                    fontsize=12 if is_winner else 11,
                    fontweight="bold" if is_winner else "normal",
                    color="#2c3e50" if is_winner else "#aab7b8")

            # Winner marker
            if is_winner:
                ax.text(bar_left + bw + 0.15, by + 0.18,
                        "← SELECTED (highest selectivity)",
                        ha="left", va="center", fontsize=11,
                        color=color, fontweight="bold", style="italic")

        # Truncation note
        if n_cands > show_n:
            trunc_y = y_base + 0.05
            ax.text(bar_left + 1.0, trunc_y,
                    f"... +{n_cands - show_n} more candidates (all < {candidates[show_n - 1]['selectivity']:.1f}%)",
                    ha="left", va="center", fontsize=11, color="#bdc3c7",
                    style="italic")

    # ── Final isolation bar ──
    final_y = -0.5
    ax.annotate("", xy=(6.75, final_y + 0.35), xytext=(6.75, final_y + 0.85),
                arrowprops=dict(arrowstyle="-|>", lw=2.5,
                                color="#2c3e50", mutation_scale=20))
    final_box = FancyBboxPatch((2.5, final_y - 0.3), 8.5, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor="#2ecc71", edgecolor="#2c3e50",
                                linewidth=2.5)
    ax.add_patch(final_box)
    last_polymer = steps[-1]["selected"]["polymer"]
    ax.text(6.75, final_y,
            f"STEP {n_steps + 1}: {last_polymer} ISOLATED  →  ALL POLYMERS SEPARATED",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color="white")

    # ── Legend ──
    legend_items = [
        mpatches.Patch(facecolor="#27ae60", edgecolor="black", label="Excellent (>30%)"),
        mpatches.Patch(facecolor="#f39c12", edgecolor="black", label="Good (10–30%)"),
        mpatches.Patch(facecolor="#e67e22", edgecolor="black", label="Marginal (0–10%)"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="black", label="Poor (<0%)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=13,
              frameon=True, fancybox=True, edgecolor="#bdc3c7",
              title="Selectivity Quality", title_fontsize=14,
              bbox_to_anchor=(1.0, 1.0))

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HERE = Path(__file__).parent

    poly9 = ["PS", "PVC", "LDPE", "HDPE", "PP", "EVOH", "Nylon6", "Nylon66", "PET"]

    print("Building greedy data for 9 polymers...")
    steps = build_greedy_with_candidates(poly9, temperature=120.0)

    out = HERE / "greedy_algorithm_visual.png"
    plot_greedy_algorithm(steps, poly9, 120.0, str(out))
