"""Offline RL on logged traces: ingestion, the optimization-outcome reward fix,
and the three trace analyses (fix-evaluation, warm-start, trajectory-rooted
best-of-N). All offline, zero API — same as the substrate it extends."""

from pathlib import Path

import pytest

from strap.eval import default_reward_model
from strap.eval.reward import Episode
from strap.eval.scorers import OptimizationOutcomeScorer
from strap.eval.trace_ingest import episodes_from_trace_cache, load_trace_cache
from strap.eval.trace_rl import (
    context_of,
    score_runs,
    trajectory_rooted_exploration,
    warm_start_bandit,
)

_CACHE = Path(__file__).resolve().parents[1] / "case-studies" / "05-offline-rl-from-traces" / "data" / "trace_rl_cache.json"


def _synthetic_cache() -> dict:
    """Two runs of the same query: one infeasible optimization, one feasible."""
    sep = {
        "agent": "separation-engineer", "schema_version": "1.0",
        "polymers": ["PE", "EVOH"], "best_sequence": ["EVOH", "PE"],
        "steps": [
            {"polymer": "EVOH", "solvent": "Ethylene glycol", "temperature_c": 120.0, "selectivity_pct": 31.0},
            {"polymer": "PE", "solvent": "Toluene", "temperature_c": 105.0, "selectivity_pct": 20.0},
        ],
        "polymer_solvent_candidates": {"PE": [{"solvent": "Toluene"}], "EVOH": [{"solvent": "Ethylene glycol"}]},
        "min_selectivity_pct": 20.0,
    }
    opt_infeasible = {
        "agent": "optimization-engineer", "schema_version": "1.3",
        "analysis_type": "infeasible", "failure_reason": "all_shortlisted_sims_failed",
    }
    opt_feasible = {
        "agent": "optimization-engineer", "schema_version": "1.5",
        "analysis_type": "pareto_front", "n_points_feasible": 1,
        "points": [{"point_id": 1, "total_cost": 2559000, "emissions": 1444.0, "wash1_selection": ["EVOH-Ethylene Glycol"]}],
    }
    return {
        "schema": "dissolve.trace_rl.v1",
        "runs": [
            {"run_id": "a", "label": "run_infeasible", "config_note": "pre-fix", "order": 1,
             "query": "Separate PE/EVOH then Pareto on cost vs emissions.",
             "structured_results": {"separation-engineer": sep, "optimization-engineer": opt_infeasible},
             "ledger_by_agent": {"separation-engineer": {"prompt_tokens": 200000, "output_tokens": 8000, "llm_calls": 10}},
             "tool_calls_by_agent": {"separation-engineer": 11, "optimization-engineer": 2}, "final_answer": "infeasible"},
            {"run_id": "b", "label": "run_feasible", "config_note": "post-fix", "order": 2,
             "query": "Separate PE/EVOH then Pareto on cost vs emissions.",
             "structured_results": {"separation-engineer": sep, "optimization-engineer": opt_feasible},
             "ledger_by_agent": {"separation-engineer": {"prompt_tokens": 130000, "output_tokens": 6000, "llm_calls": 8}},
             "tool_calls_by_agent": {"separation-engineer": 12, "optimization-engineer": 2}, "final_answer": "pareto"},
        ],
    }


class TestOptimizationOutcomeScorer:
    def _score(self, result):
        return OptimizationOutcomeScorer().score(Episode(query="q", result=result))

    def test_infeasible_scores_low(self):
        comp = self._score({"analysis_type": "infeasible", "failure_reason": "x", "points": []})
        assert comp.applicable
        assert comp.score == pytest.approx(0.1)

    def test_feasible_frontier_scores_high(self):
        comp = self._score({"analysis_type": "pareto_front",
                            "points": [{"total_cost": 1, "emissions": 2}]})
        assert comp.applicable
        assert comp.score == pytest.approx(1.0)

    def test_non_applicable_on_separation_result(self):
        comp = self._score({"best_sequence": ["PE"], "steps": [{"polymer": "PE", "solvent": "toluene"}]})
        assert comp.applicable is False

    def test_feasible_beats_infeasible_in_full_model(self):
        model = default_reward_model()
        infeasible = model.score(Episode("q", {"analysis_type": "infeasible", "points": [], "polymers": ["PE"]}))
        feasible = model.score(Episode("q", {"analysis_type": "pareto_front", "polymers": ["PE"],
                                             "points": [{"total_cost": 1, "emissions": 2}]}))
        assert feasible.scalar > infeasible.scalar


class TestIngestion:
    def test_synthetic_cache_parses_both_agents(self):
        eps = episodes_from_trace_cache(_synthetic_cache())
        agents = {(te.run_label, te.agent) for te in eps}
        assert ("run_infeasible", "optimization-engineer") in agents
        assert ("run_feasible", "separation-engineer") in agents

    def test_separation_result_shape(self):
        eps = episodes_from_trace_cache(_synthetic_cache())
        sep = next(te for te in eps if te.agent == "separation-engineer")
        r = sep.episode.result
        assert r["best_sequence"] == ["EVOH", "PE"]
        assert r["min_selectivity"] == pytest.approx(20.0)
        assert len(r["steps"]) == 2
        assert sep.episode.ledger["tool_calls"] == 11

    def test_infeasible_optimization_flagged(self):
        eps = episodes_from_trace_cache(_synthetic_cache())
        opt = next(te for te in eps if te.agent == "optimization-engineer" and te.run_label == "run_infeasible")
        assert opt.episode.result["infeasible"] is True
        assert opt.episode.result["no_data"] is True


class TestTraceRL:
    def test_fix_evaluation_feasible_optimization_outscores_infeasible(self):
        runs = score_runs(_synthetic_cache())
        infeasible = next(r for r in runs if r.label == "run_infeasible")
        feasible = next(r for r in runs if r.label == "run_feasible")
        assert feasible.optimization_reward > infeasible.optimization_reward

    def test_warm_start_seeds_only_logged_arm(self):
        ws, bandit = warm_start_bandit(_synthetic_cache(), arms=[1, 2, 4], logged_arm=1)
        assert ws.n_samples == 2  # one separation episode per run
        assert bandit.pulls("2 polymers", 1) == 2
        assert bandit.pulls("2 polymers", 2) == 0  # unlogged arms stay at prior

    def test_warm_start_rejects_arm_not_in_set(self):
        with pytest.raises(ValueError):
            warm_start_bandit(_synthetic_cache(), arms=[1, 2], logged_arm=9)

    def test_context_of_counts_polymers(self):
        assert context_of(["PE", "EVOH"]) == "2 polymers"
        assert context_of(["PE", "EVOH", "PET"]) == "3 polymers"

    def test_trajectory_rooted_reports_headroom(self):
        cfs = trajectory_rooted_exploration(_synthetic_cache(), arms_n=[1, 2])
        assert cfs
        cf = cfs[0]
        assert cf.polymers == ["PE", "EVOH"]
        assert set(cf.explored) == {1, 2}
        assert cf.headroom is not None
        assert cf.best_n in (1, 2)


@pytest.mark.skipif(not _CACHE.exists(), reason="harvested trace cache not committed")
class TestCommittedCache:
    def test_committed_cache_scores_and_orders(self):
        cache = load_trace_cache(_CACHE)
        runs = score_runs(cache)
        assert len(runs) == 5
        # the feasible runs (5,6) must outscore the infeasible ones (2,3,4) on optimization
        infeasible = [r.optimization_reward for r in runs if r.order in (2, 3, 4)]
        feasible = [r.optimization_reward for r in runs if r.order in (5, 6)]
        assert min(feasible) > max(infeasible)

    def test_committed_cache_trajectory_headroom_positive(self):
        cache = load_trace_cache(_CACHE)
        cfs = trajectory_rooted_exploration(cache, arms_n=[1, 2, 4])
        # every logged single-pass separation left non-negative headroom vs best-of-N
        assert all(cf.headroom is not None and cf.headroom >= 0 for cf in cfs)
