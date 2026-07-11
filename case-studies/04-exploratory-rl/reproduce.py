"""Exploratory RL on the STRAP harness — best-of-N + a contextual bandit that
learns how much exploration each query class deserves.

WHAT THIS SHOWS
  Panel A  Best-of-N anatomy on one 4-polymer query: N diverse candidates
           (temperature / pool-depth / solvent-exclusion variants) scored by
           the physically-gated reward; the greedy default is just candidate 0.
  Panel B  A Thompson-sampling contextual bandit choosing the exploration
           breadth N per query-complexity context, learning online from the
           reward substrate. Rolling mean reward vs fixed and random baselines
           on the identical query stream.
  Panel C  What the bandit learned: arm-selection share per context.

Everything runs on the deterministic engines + the strap.eval reward — zero
API calls; engine results are memoized so the 200-round simulation is cheap.

USAGE
    python case-studies/04-exploratory-rl/reproduce.py
"""

from __future__ import annotations

import json
import random
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
from strap.learn import ThompsonBandit, explore_separation  # noqa: E402

FIGURES = _HERE / "figures"
DATA = _HERE / "data"

ARMS = [1, 2, 4, 6]          # exploration breadth N
ROUNDS = 200
ROLLING = 25
SPEC_SEED = 0                 # fixed candidate specs -> stationary environment

QUERY_POOLS: dict[str, list[list[str]]] = {
    "2 polymers": [
        ["LDPE", "PET"], ["HDPE", "PS"], ["PP", "PVC"], ["EVOH", "PET"], ["LDPE", "EVOH"],
    ],
    "3 polymers": [
        ["LDPE", "PP", "PS"], ["HDPE", "EVOH", "PET"], ["PS", "PVC", "PET"],
        ["LDPE", "HDPE", "PP"], ["EVOH", "PC", "PET"],
    ],
    "4 polymers": [
        ["LDPE", "PP", "PS", "PET"], ["HDPE", "EVOH", "PS", "PVC"],
        ["LDPE", "EVOH", "PET", "PP"], ["PS", "PVC", "PC", "PET"],
    ],
}


def episode_reward(polymers: list[str], n: int, model) -> float:
    outcome = explore_separation(polymers, n=n, temperature_ceiling_c=140.0,
                                 seed=SPEC_SEED, reward_model=model)
    chosen = outcome.best_feasible or outcome.best
    return chosen.reward.scalar


def run_bandit_simulation(model):
    """One shared query stream; four policies scored on identical rounds."""
    rng = random.Random(2026)
    stream = []
    contexts = list(QUERY_POOLS)
    for _ in range(ROUNDS):
        ctx = rng.choice(contexts)
        stream.append((ctx, rng.choice(QUERY_POOLS[ctx])))

    # Pre-score every (query, arm) once (memoized engine underneath).
    reward_table: dict[tuple[str, int], float] = {}
    for ctx, polymers in {(c, tuple(p)) for c, p in stream}:
        for arm in ARMS:
            reward_table[(",".join(polymers), arm)] = episode_reward(list(polymers), arm, model)
        print(f"  scored {ctx}: {','.join(polymers)}")

    def reward_of(polymers: list[str], arm: int) -> float:
        return reward_table[(",".join(polymers), arm)]

    bandit = ThompsonBandit(arms=ARMS, seed=7)
    rand_rng = random.Random(99)
    trajectories = {"bandit": [], "fixed N=1": [], "fixed N=6": [], "random N": []}
    for ctx, polymers in stream:
        arm = bandit.select(ctx)
        reward = reward_of(polymers, arm)
        bandit.update(ctx, arm, reward)
        trajectories["bandit"].append(reward)
        trajectories["fixed N=1"].append(reward_of(polymers, 1))
        trajectories["fixed N=6"].append(reward_of(polymers, 6))
        trajectories["random N"].append(reward_of(polymers, rand_rng.choice(ARMS)))

    arm_share = {
        ctx: {arm: bandit.pulls(ctx, arm) for arm in ARMS} for ctx in QUERY_POOLS
    }
    learned = {ctx: bandit.best_arm(ctx) for ctx in QUERY_POOLS}
    return stream, trajectories, arm_share, learned, bandit


def render(anatomy, trajectories, arm_share, learned) -> None:
    style.apply_style()
    fig = plt.figure(figsize=(19, 6.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.0], wspace=0.28)
    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, i]) for i in range(3))

    # ---- Panel A: best-of-N anatomy ----
    labels = [c["label"] for c in anatomy["candidates"]]
    rewards = [c["reward"] for c in anatomy["candidates"]]
    colors = [style.ACCENT_CHEAPEST if i == 0 else style.SERIES_COLORS[0]
              for i in range(len(rewards))]
    order = np.argsort(rewards)[::-1]
    y = np.arange(len(rewards))[::-1]
    ax_a.barh(y, [rewards[i] for i in order],
              color=[("#b8c2cc" if order[j] != 0 else style.SERIES_COLORS[1]) for j in range(len(order))],
              edgecolor="white")
    for yi, idx in zip(y, order):
        marker = "  ← greedy default" if idx == 0 else ""
        ax_a.text(rewards[idx] + 0.01, yi, f"{rewards[idx]:.3f}{marker}", va="center", fontsize=8)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([labels[i] for i in order], fontsize=8)
    ax_a.set_xlim(0, 1.12)
    ax_a.set_xlabel("Reward")
    ax_a.set_title(f"Best-of-{len(rewards)} anatomy\n{anatomy['query_polymers']}"
                   f"  ·  {anatomy['solvent_diversity']} unique solvents surfaced", fontsize=11.5)

    # ---- Panel B: learning curves ----
    palette = {"bandit": style.INK, "fixed N=1": style.SERIES_COLORS[1],
               "fixed N=6": style.SERIES_COLORS[0], "random N": "#999999"}
    for name, series in trajectories.items():
        arr = np.array(series, dtype=float)
        rolling = np.convolve(arr, np.ones(ROLLING) / ROLLING, mode="valid")
        ax_b.plot(np.arange(len(rolling)) + ROLLING, rolling,
                  lw=2.4 if name == "bandit" else 1.5,
                  color=palette[name],
                  label=f"{name} (mean {arr.mean():.3f})",
                  zorder=5 if name == "bandit" else 3)
    ax_b.set_xlabel("Round")
    ax_b.set_ylabel(f"Reward (rolling mean, window {ROLLING})")
    ax_b.set_title("Contextual bandit learns exploration breadth online", fontsize=12)
    ax_b.legend(loc="lower right", fontsize=8.5)

    # ---- Panel C: learned policy ----
    contexts = list(arm_share)
    x = np.arange(len(contexts))
    bottom = np.zeros(len(contexts))
    for i, arm in enumerate(ARMS):
        shares = []
        for ctx in contexts:
            total = sum(arm_share[ctx].values()) or 1
            shares.append(arm_share[ctx][arm] / total)
        ax_c.bar(x, shares, 0.6, bottom=bottom, label=f"N={arm}",
                 color=style.SERIES_COLORS[i % len(style.SERIES_COLORS)],
                 edgecolor="white", linewidth=0.5)
        bottom += np.array(shares)
    for xi, ctx in zip(x, contexts):
        ax_c.text(xi, 1.03, f"learned: N={learned[ctx]}", ha="center", fontsize=9,
                  color=style.INK, fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(contexts, fontsize=9.5)
    ax_c.set_ylim(0, 1.12)
    ax_c.set_ylabel("Arm selection share")
    ax_c.set_title("Learned exploration policy by context", fontsize=12)
    ax_c.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8.5)

    fig.suptitle("Reinforcement learning on the STRAP harness: physically-gated reward → "
                 "best-of-N exploration → learned exploration policy",
                 fontsize=14.5, fontweight="bold", y=1.04, color=style.INK)
    style.caption(fig, "Thompson-sampling contextual bandit over exploration breadth N; reward from "
                       "strap.eval (physical gate + grounding + separation quality + richness + "
                       "completeness + efficiency). Deterministic engines, zero API calls.")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "exploratory_rl.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "exploratory_rl.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    model = default_reward_model()

    # Panel A data: anatomy of one exploration
    outcome = explore_separation(["LDPE", "EVOH", "PET", "PP"], n=6,
                                 temperature_ceiling_c=140.0, seed=SPEC_SEED,
                                 reward_model=model)
    anatomy = {
        "query_polymers": "LDPE / EVOH / PET / PP",
        "solvent_diversity": outcome.solvent_diversity,
        "candidates": [
            {"label": c.episode.label, "reward": c.reward.scalar,
             "feasible": c.reward.feasible,
             "min_selectivity": c.episode.result.get("min_selectivity")}
            for c in outcome.candidates
        ],
    }

    print("running bandit simulation ...")
    stream, trajectories, arm_share, learned, bandit = run_bandit_simulation(model)

    render(anatomy, trajectories, arm_share, learned)

    DATA.mkdir(exist_ok=True)
    (DATA / "results.json").write_text(json.dumps({
        "anatomy": anatomy,
        "learned_policy": {k: int(v) for k, v in learned.items()},
        "mean_rewards": {k: float(np.mean(v)) for k, v in trajectories.items()},
        "arm_share": {c: {str(a): n for a, n in arms.items()} for c, arms in arm_share.items()},
        "bandit_state": bandit.to_dict(),
        "rounds": ROUNDS,
        "arms": ARMS,
    }, indent=2))

    print(json.dumps({
        "learned_policy": {k: int(v) for k, v in learned.items()},
        "mean_rewards": {k: round(float(np.mean(v)), 4) for k, v in trajectories.items()},
        "figure": "case-studies/04-exploratory-rl/figures/exploratory_rl.png",
    }, indent=2))


if __name__ == "__main__":
    main()
