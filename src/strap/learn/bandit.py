"""Contextual bandits over harness decisions (RL option 2).

Small, dependency-free learners for discrete harness decisions (exploration
breadth, solver rung, candidate-pool depth, ...). Rewards are expected in
[0, 1] — exactly what :func:`strap.eval.RewardModel.reward_fn` produces.

Two policies with one interface:

- :class:`UCB1Bandit` — deterministic optimism (UCB1) per context bucket.
- :class:`ThompsonBandit` — Beta-Bernoulli Thompson sampling; a [0,1] reward
  updates the posterior with fractional pseudo-counts (alpha += r,
  beta += 1 - r), the standard unbiased treatment of bounded rewards.

Both are serializable (``to_dict``/``from_dict``) so a learned policy can be
persisted by the harness and reloaded across sessions.

Usage::

    bandit = ThompsonBandit(arms=[1, 2, 4, 8], seed=7)
    arm = bandit.select(context="polymers=4")
    episode = run_harness_with(arm)
    bandit.update("polymers=4", arm, reward_fn(episode))
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence


@dataclass
class _ArmStats:
    pulls: int = 0
    reward_sum: float = 0.0
    alpha: float = 1.0  # Beta prior
    beta: float = 1.0

    @property
    def mean(self) -> float:
        return self.reward_sum / self.pulls if self.pulls else 0.0

    def to_dict(self) -> dict[str, float]:
        return {"pulls": self.pulls, "reward_sum": self.reward_sum,
                "alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_ArmStats":
        return cls(int(data["pulls"]), float(data["reward_sum"]),
                   float(data["alpha"]), float(data["beta"]))


@dataclass
class _BanditBase:
    arms: Sequence[Hashable]
    seed: int | None = None
    _stats: dict[Hashable, dict[Hashable, _ArmStats]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.arms:
            raise ValueError("bandit needs at least one arm")
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be unique")
        self._rng = random.Random(self.seed)

    # -- shared plumbing -----------------------------------------------------

    def _context_stats(self, context: Hashable) -> dict[Hashable, _ArmStats]:
        return self._stats.setdefault(context, {arm: _ArmStats() for arm in self.arms})

    def update(self, context: Hashable, arm: Hashable, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward must be in [0, 1], got {reward}")
        stats = self._context_stats(context)[arm]
        stats.pulls += 1
        stats.reward_sum += reward
        stats.alpha += reward
        stats.beta += 1.0 - reward

    def mean_reward(self, context: Hashable, arm: Hashable) -> float:
        return self._context_stats(context)[arm].mean

    def pulls(self, context: Hashable, arm: Hashable) -> int:
        return self._context_stats(context)[arm].pulls

    def best_arm(self, context: Hashable) -> Hashable:
        """Greedy (exploitation-only) arm for a context — the learned policy."""
        stats = self._context_stats(context)
        return max(self.arms, key=lambda arm: (stats[arm].mean, -self.arms.index(arm)))

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": type(self).__name__,
            "arms": list(self.arms),
            "seed": self.seed,
            "stats": {
                str(ctx): {str(arm): s.to_dict() for arm, s in arms.items()}
                for ctx, arms in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_BanditBase":
        bandit = cls(arms=list(data["arms"]), seed=data.get("seed"))
        arm_by_str = {str(arm): arm for arm in bandit.arms}
        for ctx, arms in (data.get("stats") or {}).items():
            bucket = bandit._context_stats(ctx)
            for arm_str, s in arms.items():
                if arm_str in arm_by_str:
                    bucket[arm_by_str[arm_str]] = _ArmStats.from_dict(s)
        return bandit


class UCB1Bandit(_BanditBase):
    """UCB1 per context: pull each arm once, then argmax mean + c*sqrt(2 ln t / n)."""

    exploration_c: float = 1.0

    def select(self, context: Hashable) -> Hashable:
        stats = self._context_stats(context)
        for arm in self.arms:  # play every arm once first
            if stats[arm].pulls == 0:
                return arm
        total = sum(s.pulls for s in stats.values())
        def ucb(arm: Hashable) -> float:
            s = stats[arm]
            return s.mean + self.exploration_c * math.sqrt(2.0 * math.log(total) / s.pulls)
        return max(self.arms, key=ucb)


class ThompsonBandit(_BanditBase):
    """Beta-Bernoulli Thompson sampling with fractional updates for [0,1] rewards."""

    def select(self, context: Hashable) -> Hashable:
        stats = self._context_stats(context)
        draws = {arm: self._rng.betavariate(stats[arm].alpha, stats[arm].beta) for arm in self.arms}
        return max(self.arms, key=lambda arm: draws[arm])
