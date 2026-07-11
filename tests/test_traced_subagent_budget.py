"""Guarded subagent invocation: per-dispatch budgets, step-budget failure
containment, and model-visible handoff contracts.

Live-run regression (2026-07-07 multistage stress test): the optimization
engineer was told a handoff payload was "attached", could not see it (state is
model-invisible), searched the real filesystem for it for 26 tool calls, and
then GraphRecursionError crashed the whole run. These tests pin the three
fixes: seed_guard_state per dispatch, bounded recursion with a graceful
failure string, and the RUNTIME-ATTACHED HANDOFF context block.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from langgraph.errors import GraphRecursionError

from strap.traced_subagent_middleware import (
    _ainvoke_subagent_guarded,
    _handoff_context_block,
    _invoke_subagent_guarded,
    _subagent_recursion_limit,
    _subagent_step_budget_failure,
)


class TestGuardedInvoke:
    def test_success_passes_bounded_recursion_config(self):
        sub = MagicMock()
        sub.invoke.return_value = {"messages": []}
        out = _invoke_subagent_guarded(sub, {"messages": []}, "separation-engineer")
        assert out == {"messages": []}
        config = sub.invoke.call_args.args[1]
        assert config["recursion_limit"] == _subagent_recursion_limit()

    def test_recursion_error_returns_failure_string_not_exception(self):
        sub = MagicMock()
        sub.invoke.side_effect = GraphRecursionError("boom")
        out = _invoke_subagent_guarded(sub, {"messages": []}, "optimization-engineer")
        assert isinstance(out, str)
        assert "optimization-engineer" in out
        assert "step budget" in out
        # actionable guidance for the orchestrator, not just an error dump
        assert "re-dispatch" in out.lower()

    def test_async_recursion_error_returns_failure_string(self):
        sub = MagicMock()
        sub.ainvoke = AsyncMock(side_effect=GraphRecursionError("boom"))
        out = asyncio.run(
            _ainvoke_subagent_guarded(sub, {"messages": []}, "biosteam-analyst")
        )
        assert isinstance(out, str)
        assert "biosteam-analyst" in out

    def test_each_dispatch_gets_a_fresh_guard_budget(self):
        from strap.guardrails import _guard_state

        sub = MagicMock()
        sub.invoke.return_value = {"messages": []}
        _invoke_subagent_guarded(sub, {"messages": []}, "separation-engineer")
        state = _guard_state.get()
        state.iterations = 99  # pretend the previous specialist spent its budget
        _invoke_subagent_guarded(sub, {"messages": []}, "optimization-engineer")
        assert _guard_state.get().iterations == 0

    def test_recursion_limit_env_override(self, monkeypatch):
        monkeypatch.setenv("DISSOLVE_SUBAGENT_RECURSION_LIMIT", "77")
        assert _subagent_recursion_limit() == 77
        monkeypatch.setenv("DISSOLVE_SUBAGENT_RECURSION_LIMIT", "banana")
        assert _subagent_recursion_limit() == 120
        monkeypatch.setenv("DISSOLVE_SUBAGENT_RECURSION_LIMIT", "3")
        assert _subagent_recursion_limit() == 20  # floor


class TestHandoffContextBlock:
    def _record(self, **overrides):
        base = {
            "handoff_id": "h_abc123",
            "contract": "optimization.stage_candidates.v1",
            "producer": "separation-engineer",
            "payload": {"candidate_pairs": [], "feed_composition": {"PE": 0.6}},
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_names_contract_and_injection_mechanism(self):
        block = _handoff_context_block(self._record())
        assert "h_abc123" in block
        assert "optimization.stage_candidates.v1" in block
        assert "separation-engineer" in block
        assert "stage_candidates_json" in block
        assert "candidate_pairs" in block  # payload key digest grounds the model

    def test_forbids_the_observed_failure_modes(self):
        block = _handoff_context_block(self._record()).lower()
        assert "never search the filesystem" in block
        assert "never reconstruct" in block

    def test_empty_payload_still_renders(self):
        block = _handoff_context_block(self._record(payload={}))
        assert "(empty)" in block


class TestFailureMessage:
    def test_mentions_budget_and_names_subagent(self):
        text = _subagent_step_budget_failure("optimization-engineer")
        assert "optimization-engineer" in text
        assert str(_subagent_recursion_limit()) in text
