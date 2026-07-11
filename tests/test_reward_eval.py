"""Tests for the reward / evaluation substrate (strap.eval)."""

from __future__ import annotations

import pytest

from strap.eval import (
    CompletenessScorer,
    Episode,
    GroundingScorer,
    PhysicalValidityScorer,
    RewardModel,
    RichnessScorer,
    default_reward_model,
)


def _route(steps, **extra):
    return {"polymers": [s["polymer"] for s in steps], "best_sequence": [s["polymer"] for s in steps],
            "steps": steps, **extra}


# ---------------------------------------------------------------------------
# Physical-validity gate
# ---------------------------------------------------------------------------

class TestPhysicalGate:
    def test_feasible_route_passes(self):
        model = default_reward_model()
        ep = Episode("sep LDPE/PP", _route([
            {"polymer": "LDPE", "solvent": "toluene", "temperature_c": 100.0, "selectivity_pct": 40.0},
        ]))
        result = model.score(ep)
        assert result.feasible
        assert result.component("physical_validity").score == 1.0

    def test_above_boiling_point_gates_reward(self):
        model = default_reward_model()
        # toluene BP ~110.6 C; 200 C is a hard atmospheric violation.
        ep = Episode("sep LDPE", _route([
            {"polymer": "LDPE", "solvent": "toluene", "temperature_c": 200.0, "selectivity_pct": 40.0},
        ]))
        result = model.score(ep)
        assert result.feasible is False
        assert result.component("physical_validity").score == 0.0
        # gated reward is heavily penalized regardless of other dimensions
        assert result.scalar < 0.2

    def test_gate_penalty_configurable_to_hard_zero(self):
        model = default_reward_model(gate_penalty=0.0)
        ep = Episode("sep", _route([
            {"polymer": "LDPE", "solvent": "toluene", "temperature_c": 250.0},
        ]))
        assert model.score(ep).scalar == 0.0

    def test_no_steps_makes_gate_inapplicable(self):
        scorer = PhysicalValidityScorer()
        comp = scorer.score(Episode("lookup", {"solubility_pct": 5.0}))
        assert comp.applicable is False


# ---------------------------------------------------------------------------
# Grounding (anti-hallucination)
# ---------------------------------------------------------------------------

class TestGrounding:
    def test_point_claim_matching_engine_scores_high(self):
        # engine value for LDPE/toluene/100 is ~12.8%
        ep = Episode("solubility of LDPE in toluene at 100C",
                     {"polymer_name": "LDPE", "solvent_name": "toluene",
                      "temperature_c": 100.0, "solubility_pct": 12.77},
                     context={"polymer": "LDPE", "solvent": "toluene", "temperature_c": 100.0})
        comp = GroundingScorer().score(ep)
        assert comp.score > 0.9

    def test_fabricated_point_claim_scores_low(self):
        ep = Episode("solubility of LDPE in toluene at 100C",
                     {"polymer_name": "LDPE", "solvent_name": "toluene",
                      "temperature_c": 100.0, "solubility_pct": 99.0},  # wildly wrong
                     context={"polymer": "LDPE", "solvent": "toluene", "temperature_c": 100.0})
        comp = GroundingScorer().score(ep)
        assert comp.score < 0.5

    def test_fabricated_dissolution_claim_flagged(self):
        # polyolefin "dissolving" in water is chemically false
        ep = Episode("sep", _route([
            {"polymer": "LDPE", "solvent": "water", "temperature_c": 90.0, "selectivity_pct": 95.0},
        ]))
        comp = GroundingScorer().score(ep)
        assert comp.score < 1.0


# ---------------------------------------------------------------------------
# Richness (exploration)
# ---------------------------------------------------------------------------

class TestRichness:
    def test_more_candidates_scores_higher(self):
        scorer = RichnessScorer(target_candidates_per_polymer=3)
        thin = Episode("sep", {"polymer_solvent_candidates": {"LDPE": [{"solvent": "toluene"}]}})
        rich = Episode("sep", {"polymer_solvent_candidates": {
            "LDPE": [{"solvent": "toluene"}, {"solvent": "decalin"}, {"solvent": "xylene"}]}})
        assert scorer.score(rich).score > scorer.score(thin).score

    def test_richness_inapplicable_on_bare_lookup(self):
        comp = RichnessScorer().score(Episode("lookup", {"solubility_pct": 5.0}))
        assert comp.applicable is False

    def test_frontier_points_reward_richness(self):
        scorer = RichnessScorer(target_frontier_points=4)
        one = Episode("pareto", {"frontier": [{"x": 1}]})
        four = Episode("pareto", {"frontier": [{"x": i} for i in range(4)]})
        assert scorer.score(four).score > scorer.score(one).score


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

class TestCompleteness:
    def test_all_polymers_and_metrics_covered(self):
        ep = Episode("cost and gwp for LDPE and PP",
                     {"polymers": ["LDPE", "PP"], "total_cost": 1.0, "gwp_kg": 2.0},
                     context={"polymers": ["LDPE", "PP"]})
        comp = CompletenessScorer().score(ep)
        assert comp.score == 1.0

    def test_missing_polymer_lowers_score(self):
        ep = Episode("separate LDPE, PP and PS",
                     {"polymers": ["LDPE", "PP"]},  # PS missing
                     context={"polymers": ["LDPE", "PP", "PS"]})
        comp = CompletenessScorer().score(ep)
        assert comp.score < 1.0


# ---------------------------------------------------------------------------
# RewardModel composition + RL plug-in
# ---------------------------------------------------------------------------

class TestRewardModel:
    def test_scalar_in_unit_interval(self):
        model = default_reward_model()
        ep = Episode("sep", _route([
            {"polymer": "LDPE", "solvent": "toluene", "temperature_c": 100.0, "selectivity_pct": 30.0},
        ]))
        r = model.score(ep)
        assert 0.0 <= r.scalar <= 1.0

    def test_reward_fn_returns_float(self):
        fn = default_reward_model().reward_fn()
        ep = Episode("lookup", {"polymer_name": "LDPE", "solvent_name": "toluene",
                                "temperature_c": 100.0, "solubility_pct": 12.77},
                     context={"polymer": "LDPE", "solvent": "toluene", "temperature_c": 100.0})
        value = fn(ep)
        assert isinstance(value, float) and 0.0 <= value <= 1.0

    def test_rank_orders_best_first(self):
        model = default_reward_model()
        good = Episode("sep", _route([
            {"polymer": "LDPE", "solvent": "toluene", "temperature_c": 100.0, "selectivity_pct": 40.0}],
            polymer_solvent_candidates={"LDPE": [{"solvent": "toluene"}, {"solvent": "decalin"}, {"solvent": "xylene"}]}))
        infeasible = Episode("sep", _route([
            {"polymer": "LDPE", "solvent": "toluene", "temperature_c": 250.0, "selectivity_pct": 40.0}]))
        ranked = model.rank([infeasible, good])
        assert ranked[0][0] is good
        assert ranked[0][1].scalar > ranked[1][1].scalar

    def test_gate_penalty_validated(self):
        with pytest.raises(ValueError):
            RewardModel([], gate_penalty=2.0)


# ---------------------------------------------------------------------------
# End-to-end on the real engine-backed suite
# ---------------------------------------------------------------------------

class TestSuiteIntegration:
    def test_every_suite_query_scores_in_range_and_feasible(self):
        from strap.eval.query_suite import SUITE

        model = default_reward_model()
        for q in SUITE:
            r = model.score(q.produce())
            assert 0.0 <= r.scalar <= 1.0, q.key
            assert r.feasible, q.key  # all engine-produced results are physical

    def test_ablations_rank_engine_over_degraded(self):
        from strap.eval.query_suite import ablations_for_separation

        model = default_reward_model()
        scored = {name: model.score(ep).scalar for name, ep in ablations_for_separation().items()}
        assert scored["engine_result"] >= scored["greedy_top1"]
        assert scored["greedy_top1"] > scored["infeasible_above_bp"]
        assert scored["engine_result"] > scored["fabricated_claim"]
        # physical/fabrication failures are gated infeasible
        infeasible = model.score(ablations_for_separation()["infeasible_above_bp"])
        assert infeasible.feasible is False
