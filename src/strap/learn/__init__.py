"""STRAP learning layer: RL-style decision-making on top of strap.eval.

- :mod:`strap.learn.bandit` — contextual bandits (UCB1, Thompson) for discrete
  harness decisions, rewards in [0,1] straight from ``RewardModel.reward_fn``.
- :mod:`strap.learn.best_of_n` — best-of-N exploratory analysis with a
  physically-gated reward ranking.
"""

from strap.learn.bandit import ThompsonBandit, UCB1Bandit
from strap.learn.best_of_n import (
    Candidate,
    ExplorationOutcome,
    best_of_n,
    clear_engine_cache,
    explore_separation,
    separation_candidate_specs,
)

__all__ = [
    "ThompsonBandit",
    "UCB1Bandit",
    "Candidate",
    "ExplorationOutcome",
    "best_of_n",
    "clear_engine_cache",
    "explore_separation",
    "separation_candidate_specs",
]
