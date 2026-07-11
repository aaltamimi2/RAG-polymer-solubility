"""Offline RL analyses on logged agent traces — zero model inference.

Given a harvested trace cache (:mod:`strap.eval.trace_ingest`), this module
runs the three offline-RL analyses that last night's live multistage runs make
possible without any new API calls:

1. :func:`score_runs` — reward-label every real trajectory (offline policy
   evaluation). Because the runs share one query under evolving harness config,
   the reward curve across runs measures whether the fixes actually raised the
   reward the harness earns.
2. :func:`warm_start_bandit` — seed a contextual bandit's *logged* arm from the
   real (context, reward) samples, so online play begins with the value of the
   arm the harness actually pulled already characterized (offline → online).
3. :func:`trajectory_rooted_exploration` — the counterfactual the deterministic
   engine makes free: root at each trace's real query and ask whether best-of-N
   over the separation decision would have beaten the harness's actual single
   greedy pass. The upstream cost is already paid (it is in the log); only the
   free engine exploration is new.

All three consume the physically-gated reward model, so the same anti-reward-
hacking guarantees hold on logged data as on live rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from strap.eval.reward import RewardModel, default_reward_model
from strap.eval.trace_ingest import TraceEpisode, episodes_from_trace_cache


# ---------------------------------------------------------------------------
# 1. Reward-label the real trajectories (offline policy evaluation)
# ---------------------------------------------------------------------------

@dataclass
class ScoredRun:
    order: int
    label: str
    config_note: str
    per_agent: dict[str, float]           # agent -> composite reward
    per_agent_components: dict[str, dict[str, float]]
    feasible: dict[str, bool]

    @property
    def optimization_reward(self) -> float | None:
        return self.per_agent.get("optimization-engineer")

    @property
    def separation_reward(self) -> float | None:
        return self.per_agent.get("separation-engineer")

    @property
    def mean_reward(self) -> float:
        vals = list(self.per_agent.values())
        return sum(vals) / len(vals) if vals else 0.0


def score_runs(cache: dict, model: RewardModel | None = None) -> list[ScoredRun]:
    """Reward-label each logged run, grouped by run and agent."""
    model = model or default_reward_model()
    episodes = episodes_from_trace_cache(cache)
    by_run: dict[tuple[int, str], dict[str, Any]] = {}
    for te in episodes:
        key = (te.order, te.run_label)
        bucket = by_run.setdefault(key, {"config_note": te.config_note,
                                         "per_agent": {}, "components": {}, "feasible": {}})
        rr = model.score(te.episode)
        bucket["per_agent"][te.agent] = rr.scalar
        bucket["components"][te.agent] = {
            c.name: round(c.score, 4) for c in rr.components if c.applicable
        }
        bucket["feasible"][te.agent] = rr.feasible
    runs = [
        ScoredRun(order=order, label=label, config_note=data["config_note"],
                  per_agent=data["per_agent"], per_agent_components=data["components"],
                  feasible=data["feasible"])
        for (order, label), data in by_run.items()
    ]
    runs.sort(key=lambda r: r.order)
    return runs


# ---------------------------------------------------------------------------
# 2. Warm-start a bandit from logged outcomes (offline -> online)
# ---------------------------------------------------------------------------

def context_of(polymers: Any) -> str:
    """The bandit context used by case study 04: query complexity = #polymers."""
    n = len(polymers) if isinstance(polymers, (list, tuple)) else 0
    return f"{n} polymers"


@dataclass
class WarmStart:
    logged_arm: Any
    samples: list[tuple[str, float]]       # (context, reward) seeded into the logged arm
    bandit_state: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return len(self.samples)


def warm_start_bandit(
    cache: dict,
    *,
    arms: list,
    logged_arm,
    agent: str = "separation-engineer",
    model: RewardModel | None = None,
    seed: int = 7,
):
    """Seed the *logged* arm of a fresh ThompsonBandit from the real trace
    rewards for ``agent``, returning the warm-started bandit + a record.

    Honest limitation: the logs only ever exercised one arm (the harness did a
    single greedy pass, i.e. N=1), so only that arm is warm-started from data;
    the other arms keep their uniform prior until online play. This is exactly
    the offline→online setting — real logged experience initializes the policy
    where data exists.
    """
    from strap.learn import ThompsonBandit

    model = model or default_reward_model()
    if logged_arm not in arms:
        raise ValueError(f"logged_arm {logged_arm} not in arms {arms}")

    bandit = ThompsonBandit(arms=arms, seed=seed)
    samples: list[tuple[str, float]] = []
    for te in episodes_from_trace_cache(cache):
        if te.agent != agent:
            continue
        ctx = context_of(te.episode.context.get("polymers"))
        reward = model.score(te.episode).scalar
        bandit.update(ctx, logged_arm, reward)
        samples.append((ctx, round(reward, 4)))
    return WarmStart(logged_arm=logged_arm, samples=samples, bandit_state=bandit.to_dict()), bandit


# ---------------------------------------------------------------------------
# 3. Trajectory-rooted counterfactual exploration (free, engine-only)
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryCounterfactual:
    run_label: str
    polymers: list[str]
    logged_reward: float                   # the harness's real single-pass separation reward
    explored: dict[int, float] = field(default_factory=dict)  # N -> best-feasible reward
    solvent_diversity: dict[int, int] = field(default_factory=dict)

    @property
    def best_n(self) -> int | None:
        return max(self.explored, key=self.explored.get) if self.explored else None

    @property
    def headroom(self) -> float | None:
        if not self.explored:
            return None
        return round(max(self.explored.values()) - self.logged_reward, 4)


def trajectory_rooted_exploration(
    cache: dict,
    *,
    arms_n: list[int],
    temperature_ceiling_c: float = 140.0,
    model: RewardModel | None = None,
    seed: int = 0,
) -> list[TrajectoryCounterfactual]:
    """For each logged run, compare the harness's real single-pass separation
    reward against deterministic best-of-N over the same query — the free
    counterfactual the engine allows. The logged upstream cost is already paid;
    only the engine exploration is new (zero API)."""
    from strap.learn import explore_separation

    model = model or default_reward_model()
    out: list[TrajectoryCounterfactual] = []
    # cache best-of-N per polymer set (the runs share one query, so this also
    # avoids recomputing the identical exploration five times).
    explore_cache: dict[tuple[tuple[str, ...], int], Any] = {}
    for te in episodes_from_trace_cache(cache):
        if te.agent != "separation-engineer":
            continue
        polymers = [str(p).upper() for p in (te.episode.context.get("polymers") or [])]
        if not polymers:
            continue
        logged_reward = model.score(te.episode).scalar
        cf = TrajectoryCounterfactual(run_label=te.run_label, polymers=polymers,
                                      logged_reward=round(logged_reward, 4))
        for n in arms_n:
            key = (tuple(polymers), n)
            if key not in explore_cache:
                explore_cache[key] = explore_separation(
                    polymers, n=n, temperature_ceiling_c=temperature_ceiling_c,
                    seed=seed, reward_model=model)
            outcome = explore_cache[key]
            chosen = outcome.best_feasible or outcome.best
            cf.explored[n] = round(chosen.reward.scalar, 4) if chosen else 0.0
            cf.solvent_diversity[n] = outcome.solvent_diversity
        out.append(cf)
    out.sort(key=lambda c: c.run_label)
    return out
