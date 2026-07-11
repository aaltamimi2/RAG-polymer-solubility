"""Offline RL on last night's live agent traces — zero new API calls.

Last night's live multistage stress runs are logged trajectories from a fixed
policy: the offline / off-policy RL setting. The expensive part (the model
deciding routes and shortlisting solvents) is already paid for in the traces;
the deterministic v10 engines the reward model and best-of-N run on are free.
So we can reward-label the real runs, evaluate the harness fixes off-policy,
warm-start a bandit from logged outcomes, and run trajectory-rooted
counterfactual exploration — all without generating a single new model call.

WHAT THIS SHOWS
  Panel A  Offline policy evaluation. Every logged run reward-labeled by the
           physically-gated reward model. The runs share one query under
           evolving harness config, so the optimization-outcome reward curve
           measures whether the fixes actually raised the reward the harness
           earns (infeasible ~0.1 -> feasible 1.0 when the candidate-admission
           fix landed).
  Panel B  Trajectory-rooted counterfactual. For the real query, the harness's
           actual single greedy separation pass (logged) vs deterministic
           best-of-N over the same query. The gap is reward left on the table,
           recoverable for free — the engine exploration needs no API.
  Panel C  Offline -> online. The N=1 arm's Thompson posterior seeded from the
           5 real logged rewards (vs the cold uniform prior a fresh bandit
           starts from). Real logged experience initializes the policy.

The data source is a committed cache harvested once from our OWN LangSmith logs
(no model inference); see architecture/harvest_trace_rl_cache.py. Everything
here runs offline against that cache.

USAGE
    python case-studies/05-offline-rl-from-traces/reproduce.py
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
from strap.eval.trace_ingest import episodes_from_trace_cache, load_trace_cache  # noqa: E402
from strap.eval.trace_rl import (  # noqa: E402
    score_runs,
    trajectory_rooted_exploration,
    warm_start_bandit,
)

DATA = _HERE / "data"
FIGURES = _HERE / "figures"
CACHE = DATA / "trace_rl_cache.json"
ARMS_N = [1, 2, 4, 6]


def _beta_pdf(alpha: float, beta: float, x: np.ndarray) -> np.ndarray:
    from math import lgamma

    log_b = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
    return np.exp((alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x) - log_b)


def render(runs, cfs, pooled_alpha, pooled_beta, logged_rewards) -> None:
    style.apply_style()
    fig = plt.figure(figsize=(18, 5.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.30)
    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, i]) for i in range(3))

    opt_color, sep_color = style.SERIES_COLORS[0], style.SERIES_COLORS[1]

    # ---- Panel A: offline fix-evaluation curve ----
    orders = [r.order for r in runs]
    opt = [r.optimization_reward for r in runs]
    sep = [r.separation_reward for r in runs]
    ax_a.plot(orders, opt, "-o", color=opt_color, lw=2.2, ms=8, label="optimization outcome", zorder=5)
    ax_a.plot(orders, sep, "-o", color=sep_color, lw=2.0, ms=7, label="separation result", zorder=4)
    # mark where the candidate-admission fix made optimization feasible
    fix_x = 4.5
    ax_a.axvline(fix_x, color="#888888", ls="--", lw=1.2, zorder=1)
    ax_a.text(fix_x + 0.05, 0.17, "baseline-fallback fix\n→ optimization feasible",
              fontsize=8, color="#555555", va="bottom")
    for x, y in zip(orders, opt):
        ax_a.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                      ha="center", fontsize=8, color=opt_color, fontweight="bold")
    ax_a.set_xticks(orders)
    ax_a.set_xticklabels([f"run {o}" for o in orders])
    ax_a.set_ylim(0, 1.02)
    ax_a.set_ylabel("Reward (physically-gated model)")
    ax_a.set_title("A · Offline policy evaluation of the harness fixes", fontsize=11.5)
    ax_a.legend(loc="center right", fontsize=8.5)

    # ---- Panel B: trajectory-rooted counterfactual ----
    # best-of-N ceiling is the same query across runs; take the shared curve.
    ceiling = cfs[0].explored  # {N: reward}
    n_vals = sorted(ceiling)
    ax_b.plot(n_vals, [ceiling[n] for n in n_vals], "-D", color=style.SERIES_COLORS[2],
              lw=2.2, ms=8, label="deterministic best-of-N (engine)", zorder=5)
    logged_x = -0.9
    lo, hi = min(logged_rewards), max(logged_rewards)
    ax_b.axhspan(lo, hi, color=sep_color, alpha=0.18, zorder=1)
    ax_b.axhline(np.mean(logged_rewards), color=sep_color, lw=2.0, ls="-",
                 label="harness's real single pass (logged)", zorder=4)
    for r in logged_rewards:
        ax_b.plot(logged_x, r, "o", color=sep_color, ms=6, alpha=0.8, zorder=5)
    ax_b.annotate("", xy=(logged_x + 0.35, max(ceiling.values())),
                  xytext=(logged_x + 0.35, np.mean(logged_rewards)),
                  arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.3))
    ax_b.text(logged_x + 0.5, (max(ceiling.values()) + np.mean(logged_rewards)) / 2,
              "headroom\n(free, no API)", fontsize=8, color="#444444", va="center")
    ax_b.set_xticks([logged_x] + n_vals)
    ax_b.set_xticklabels(["logged\nsingle pass"] + [f"N={n}" for n in n_vals])
    ax_b.set_xlim(logged_x - 0.5, max(n_vals) + 0.4)
    ax_b.set_ylim(0.5, 1.02)
    ax_b.set_ylabel("Separation reward")
    ax_b.set_title("B · Trajectory-rooted best-of-N (zero new calls)", fontsize=11.5)
    ax_b.legend(loc="lower right", fontsize=8)

    # ---- Panel C: warm-start posterior ----
    x = np.linspace(0.001, 0.999, 400)
    ax_c.plot(x, _beta_pdf(1.0, 1.0, x), color="#999999", lw=1.8, ls="--",
              label="cold prior  Beta(1, 1)")
    ax_c.plot(x, _beta_pdf(pooled_alpha, pooled_beta, x), color=opt_color, lw=2.4,
              label=f"warm-started  Beta({pooled_alpha:.1f}, {pooled_beta:.1f})")
    ax_c.fill_between(x, _beta_pdf(pooled_alpha, pooled_beta, x), color=opt_color, alpha=0.12)
    for r in logged_rewards:
        ax_c.axvline(r, color=sep_color, lw=1.0, alpha=0.5, ymax=0.12)
    mean = pooled_alpha / (pooled_alpha + pooled_beta)
    ax_c.axvline(mean, color=opt_color, lw=1.4, ls=":")
    ax_c.annotate(f"posterior mean {mean:.2f}", xy=(mean, _beta_pdf(pooled_alpha, pooled_beta, np.array([mean]))[0]),
                  xytext=(0.06, ax_c.get_ylim()[1] * 0.62), fontsize=8, color=opt_color,
                  arrowprops=dict(arrowstyle="->", color=opt_color, lw=1.0))
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("Reward")
    ax_c.set_ylabel("Posterior density")
    ax_c.set_title("C · Warm-start the N=1 arm from logged rewards", fontsize=11.5)
    ax_c.legend(loc="upper left", fontsize=8.5)

    fig.suptitle("Offline RL on logged agent traces: reward-label real runs → evaluate fixes → "
                 "warm-start & explore, with zero new API calls",
                 fontsize=14, fontweight="bold", y=1.03, color=style.INK)
    style.caption(fig, "Trajectories harvested once from our own LangSmith logs (no model inference); "
                       "all analysis runs offline against the committed cache using the physically-gated "
                       "strap.eval reward and strap.learn best-of-N / Thompson bandit.")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "offline_rl_from_traces.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "offline_rl_from_traces.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cache = load_trace_cache(CACHE)
    model = default_reward_model()

    runs = score_runs(cache, model)
    cfs = trajectory_rooted_exploration(cache, arms_n=ARMS_N, model=model)

    # pooled warm-start posterior for the figure: all logged separation rewards
    # into one N=1 arm (the module keeps per-context for real use).
    logged_rewards = [
        model.score(te.episode).scalar
        for te in episodes_from_trace_cache(cache)
        if te.agent == "separation-engineer"
    ]
    pooled_alpha = 1.0 + sum(logged_rewards)
    pooled_beta = 1.0 + sum(1.0 - r for r in logged_rewards)
    ws, _ = warm_start_bandit(cache, arms=ARMS_N, logged_arm=1, model=model)

    render(runs, cfs, pooled_alpha, pooled_beta, logged_rewards)

    results = {
        "fix_evaluation": [
            {"order": r.order, "label": r.label, "config_note": r.config_note,
             "optimization_reward": round(r.optimization_reward, 4) if r.optimization_reward is not None else None,
             "separation_reward": round(r.separation_reward, 4) if r.separation_reward is not None else None,
             "components": r.per_agent_components}
            for r in runs
        ],
        "trajectory_rooted": [
            {"run": cf.run_label, "polymers": cf.polymers, "logged_reward": cf.logged_reward,
             "best_of_n": cf.explored, "best_n": cf.best_n, "headroom": cf.headroom}
            for cf in cfs
        ],
        "warm_start": {
            "logged_arm": ws.logged_arm, "n_samples": ws.n_samples, "samples": ws.samples,
            "pooled_posterior": {"alpha": round(pooled_alpha, 4), "beta": round(pooled_beta, 4),
                                 "mean": round(pooled_alpha / (pooled_alpha + pooled_beta), 4)},
        },
        "no_api_calls": True,
    }
    (DATA / "results.json").write_text(json.dumps(results, indent=2))

    print(json.dumps({
        "fix_evaluation_optimization": {r.label: round(r.optimization_reward, 3) for r in runs},
        "trajectory_headroom": {cf.run_label: cf.headroom for cf in cfs},
        "warm_start_posterior_mean": round(pooled_alpha / (pooled_alpha + pooled_beta), 3),
        "figure": "case-studies/05-offline-rl-from-traces/figures/offline_rl_from_traces.png",
    }, indent=2))


if __name__ == "__main__":
    main()
