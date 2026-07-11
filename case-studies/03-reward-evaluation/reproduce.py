"""Reward / evaluation substrate — scored across the query-complexity spectrum.

Renders the publication figure for the STRAP reward model:

  Panel A  reward decomposition across a simple->complex query suite (real,
           engine-backed results scored with zero API calls).
  Panel B  reward discrimination: on one complex separation query, the reward
           model ranks the engine result above a greedy single-candidate
           version and collapses physically-infeasible / fabricated variants
           via the physical-validity gate — the property an RL loop relies on.

USAGE
    python case-studies/03-reward-evaluation/reproduce.py
"""

from __future__ import annotations

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
import numpy as np  # noqa: E402

import casestudy_style as style  # noqa: E402
from strap.eval import default_reward_model  # noqa: E402
from strap.eval.query_suite import SUITE, ablations_for_separation  # noqa: E402

FIGURES = _HERE / "figures"
DATA = _HERE / "data"

COMPONENT_ORDER = ["physical_validity", "grounding", "richness", "completeness", "efficiency"]
COMPONENT_LABEL = {
    "physical_validity": "Physical validity (gate)",
    "grounding": "Grounding",
    "richness": "Richness",
    "completeness": "Completeness",
    "efficiency": "Efficiency",
}
COMPONENT_COLOR = dict(zip(COMPONENT_ORDER, [
    style.SERIES_COLORS[2], style.SERIES_COLORS[0], style.SERIES_COLORS[1],
    style.SERIES_COLORS[3], style.SERIES_COLORS[4],
]))
COMPLEXITY_LABEL = {"simple": "Simple", "moderate": "Moderate",
                    "complex": "Complex", "very_complex": "Very complex"}


def score_suite(model):
    rows = []
    for q in SUITE:
        result = model.score(q.produce())
        comp = {c.name: (c.score if c.applicable else None) for c in result.components}
        rows.append({
            "key": q.key, "complexity": q.complexity, "query": q.query,
            "reward": result.scalar, "feasible": result.feasible, "components": comp,
        })
    return rows


def score_ablations(model):
    labels = {"engine_result": "Engine result", "greedy_top1": "Greedy (top-1 only)",
              "infeasible_above_bp": "Infeasible (above BP)", "fabricated_claim": "Fabricated claim"}
    out = []
    for name, ep in ablations_for_separation().items():
        r = model.score(ep)
        out.append({"variant": name, "label": labels.get(name, name),
                    "reward": r.scalar, "feasible": r.feasible,
                    "physical": (r.component("physical_validity").score if r.component("physical_validity") else None)})
    # engine, greedy, infeasible, fabricated order
    order = ["engine_result", "greedy_top1", "infeasible_above_bp", "fabricated_claim"]
    out.sort(key=lambda d: order.index(d["variant"]))
    return out


def render(suite_rows, ablation_rows) -> None:
    style.apply_style()
    fig = plt.figure(figsize=(18, 6.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.32)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    # ---- Panel A: component decomposition across complexity ----
    # Component scores and the composite reward all live in [0,1] -> one axis.
    n = len(suite_rows)
    x = np.arange(n)
    width = 0.15
    for i, comp in enumerate(COMPONENT_ORDER):
        vals = [row["components"].get(comp) for row in suite_rows]
        plotted = [v if v is not None else 0.0 for v in vals]
        bars = ax_a.bar(x + (i - 2) * width, plotted, width,
                        color=COMPONENT_COLOR[comp], label=COMPONENT_LABEL[comp],
                        edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if v is None:  # "not applicable" bars hatched faint
                bar.set_alpha(0.18)
                bar.set_hatch("//")
    ax_a.plot(x, [row["reward"] for row in suite_rows], "o-", color=style.INK,
              lw=2.4, ms=9, zorder=6, label="Composite reward")

    ax_a.set_ylim(0, 1.08)
    ax_a.set_ylabel("Score / composite reward")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [f"{COMPLEXITY_LABEL[r['complexity']]}\n{r['key'].replace('_', chr(10))}" for r in suite_rows],
        fontsize=8,
    )
    ax_a.set_xlim(-0.55, n - 0.45)
    ax_a.set_title("Reward decomposition across query complexity", fontsize=12.5)
    ax_a.legend(loc="lower center", ncol=3, fontsize=7.8, framealpha=0.95)

    # ---- Panel B: discrimination on ablations ----
    labels = [r["label"] for r in ablation_rows]
    rewards = [r["reward"] for r in ablation_rows]
    colors = [style.ACCENT_CHEAPEST if r["feasible"] else style.ACCENT_OPTIMAL for r in ablation_rows]
    y = np.arange(len(labels))[::-1]
    ax_b.barh(y, rewards, color=colors, edgecolor="white", linewidth=0.6)
    for yi, r in zip(y, ablation_rows):
        tag = "" if r["feasible"] else "  ✗ gated infeasible"
        ax_b.text(r["reward"] + 0.02, yi, f"{r['reward']:.2f}{tag}", va="center", fontsize=8.5,
                  color=style.INK if r["feasible"] else style.ACCENT_OPTIMAL)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=9)
    ax_b.set_xlim(0, 1.15)
    ax_b.set_xlabel("Reward")
    ax_b.set_title("Reward discrimination\n(one complex separation query)", fontsize=12.5)

    fig.suptitle("STRAP reward substrate: physically-gated, decomposed reward for learning & evaluation",
                 fontsize=14.5, fontweight="bold", y=1.03, color=style.INK)
    style.caption(fig, "All scores computed from structured results + the deterministic v10 engines "
                       "(solubility, boiling points) — zero API calls. Green = physically feasible, "
                       "red = gated by the physical-validity check.")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "reward_evaluation.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "reward_evaluation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    model = default_reward_model()
    suite_rows = score_suite(model)
    ablation_rows = score_ablations(model)

    render(suite_rows, ablation_rows)

    DATA.mkdir(exist_ok=True)
    (DATA / "scores.json").write_text(json.dumps(
        {"suite": suite_rows, "ablations": ablation_rows}, indent=2))
    print(json.dumps({
        "suite": [{"key": r["key"], "complexity": r["complexity"],
                   "reward": r["reward"], "feasible": r["feasible"]} for r in suite_rows],
        "ablations": [{"label": r["label"], "reward": r["reward"], "feasible": r["feasible"]}
                      for r in ablation_rows],
        "figure": "case-studies/03-reward-evaluation/figures/reward_evaluation.png",
    }, indent=2))


if __name__ == "__main__":
    main()
