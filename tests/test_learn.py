"""Tests for the learning layer (strap.learn): bandits + best-of-N."""

from __future__ import annotations

import random

import pytest

from strap.learn import (
    ThompsonBandit,
    UCB1Bandit,
    best_of_n,
    explore_separation,
    separation_candidate_specs,
)
from strap.eval import Episode, default_reward_model


# ---------------------------------------------------------------------------
# Bandits: convergence, determinism, persistence
# ---------------------------------------------------------------------------

def _simulate(bandit, true_means: dict, rounds: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    for _ in range(rounds):
        arm = bandit.select("ctx")
        # bounded stochastic reward around the arm's true mean
        reward = min(1.0, max(0.0, true_means[arm] + rng.uniform(-0.05, 0.05)))
        bandit.update("ctx", arm, reward)


class TestBandits:
    @pytest.mark.parametrize("cls", [UCB1Bandit, ThompsonBandit])
    def test_converges_to_best_arm(self, cls):
        bandit = cls(arms=[1, 2, 4, 8], seed=13)
        _simulate(bandit, {1: 0.55, 2: 0.65, 4: 0.9, 8: 0.7}, rounds=400)
        assert bandit.best_arm("ctx") == 4
        # the best arm should dominate pulls after convergence
        assert bandit.pulls("ctx", 4) > max(bandit.pulls("ctx", a) for a in (1, 2, 8))

    def test_contexts_learned_independently(self):
        bandit = ThompsonBandit(arms=["a", "b"], seed=5)
        rng = random.Random(1)
        for _ in range(300):
            for ctx, best in (("easy", "a"), ("hard", "b")):
                arm = bandit.select(ctx)
                mean = 0.9 if arm == best else 0.3
                bandit.update(ctx, arm, min(1.0, max(0.0, mean + rng.uniform(-0.05, 0.05))))
        assert bandit.best_arm("easy") == "a"
        assert bandit.best_arm("hard") == "b"

    def test_thompson_seed_determinism(self):
        a = ThompsonBandit(arms=[1, 2], seed=42)
        b = ThompsonBandit(arms=[1, 2], seed=42)
        picks_a = [a.select("c") for _ in range(10)]
        picks_b = [b.select("c") for _ in range(10)]
        assert picks_a == picks_b

    def test_ucb_plays_every_arm_first(self):
        bandit = UCB1Bandit(arms=[1, 2, 3])
        first = []
        for _ in range(3):
            arm = bandit.select("c")
            first.append(arm)
            bandit.update("c", arm, 0.5)
        assert sorted(first) == [1, 2, 3]

    def test_reward_bounds_enforced(self):
        bandit = UCB1Bandit(arms=[1])
        with pytest.raises(ValueError):
            bandit.update("c", 1, 1.5)

    def test_serialization_roundtrip(self):
        bandit = ThompsonBandit(arms=[1, 2, 4], seed=9)
        _simulate(bandit, {1: 0.4, 2: 0.8, 4: 0.6}, rounds=120)
        restored = ThompsonBandit.from_dict(bandit.to_dict())
        assert restored.best_arm("ctx") == bandit.best_arm("ctx")
        assert restored.pulls("ctx", 2) == bandit.pulls("ctx", 2)

    def test_duplicate_arms_rejected(self):
        with pytest.raises(ValueError):
            UCB1Bandit(arms=[1, 1])


# ---------------------------------------------------------------------------
# Best-of-N: generic + separation exploration
# ---------------------------------------------------------------------------

class TestBestOfN:
    def test_generic_best_of_n_ranks_by_reward(self):
        model = default_reward_model()

        def produce(spec):
            # higher min_selectivity -> higher separation-quality reward
            return Episode("sep", {
                "polymers": ["LDPE", "PP"], "best_sequence": ["LDPE", "PP"],
                "min_selectivity": spec["min_sel"],
                "steps": [{"polymer": "LDPE", "solvent": "toluene",
                           "temperature_c": 100.0, "selectivity_pct": spec["min_sel"]}],
            })

        outcome = best_of_n([{"min_sel": 5.0}, {"min_sel": 25.0}, {"min_sel": 15.0}],
                            produce, model)
        assert outcome.best.spec["min_sel"] == 25.0
        rewards = [c.reward.scalar for c in outcome.candidates]
        assert rewards == sorted(rewards, reverse=True)

    def test_best_feasible_skips_gated_candidates(self):
        model = default_reward_model()

        def produce(spec):
            return Episode("sep", {
                "polymers": ["LDPE"], "best_sequence": ["LDPE"],
                "min_selectivity": spec["min_sel"],
                "steps": [{"polymer": "LDPE", "solvent": "toluene",
                           "temperature_c": spec["temp"], "selectivity_pct": spec["min_sel"]}],
            })

        # infeasible candidate has a huge min_sel but runs above BP
        outcome = best_of_n(
            [{"min_sel": 90.0, "temp": 250.0}, {"min_sel": 20.0, "temp": 100.0}],
            produce, model)
        assert outcome.best_feasible is not None
        assert outcome.best_feasible.spec["temp"] == 100.0
        assert outcome.best_feasible.reward.feasible

    def test_spec_generation_deterministic_and_diverse(self):
        a = separation_candidate_specs(["LDPE", "PP", "PS"], n=6, seed=3)
        b = separation_candidate_specs(["LDPE", "PP", "PS"], n=6, seed=3)
        assert a == b
        assert len({tuple(sorted(s.items())) for s in map(
            lambda d: {**d, "excluded": tuple(d["excluded"])}, a)}) == 6
        # spec 0 is the greedy default at the ceiling
        assert a[0]["temperature"] == 140.0 and a[0]["excluded"] == ()

    def test_explore_separation_end_to_end(self):
        outcome = explore_separation(["LDPE", "PP", "PS"], n=4,
                                     temperature_ceiling_c=140.0, seed=7)
        assert len(outcome.candidates) == 4
        assert outcome.best.reward.feasible
        assert 0.0 <= outcome.best.reward.scalar <= 1.0
        # exploration surfaces more unique solvents than a single greedy run
        greedy = explore_separation(["LDPE", "PP", "PS"], n=1,
                                    temperature_ceiling_c=140.0, seed=7)
        assert outcome.solvent_diversity >= greedy.solvent_diversity
        # best-of-N never loses to best-of-1 on the same reward basis
        assert outcome.best.reward.scalar >= greedy.candidates[0].reward.scalar - 0.06

    def test_reward_fn_composes_with_bandit(self):
        """The intended RL wiring: bandit picks N, explorer produces, reward updates."""
        bandit = ThompsonBandit(arms=[1, 2], seed=11)
        model = default_reward_model()
        for _ in range(4):
            arm = bandit.select("polymers=3")
            outcome = explore_separation(["LDPE", "PP", "PS"], n=arm,
                                         temperature_ceiling_c=140.0, seed=arm)
            bandit.update("polymers=3", arm, outcome.best.reward.scalar)
        assert sum(bandit.pulls("polymers=3", a) for a in (1, 2)) == 4
