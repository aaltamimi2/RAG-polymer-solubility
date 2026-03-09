from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, ToolMessage

import architecture.test_harness as harness
from architecture.test_harness import (
    TestQuery as HarnessTestQuery,
    clear_timeout_snapshot,
    extract_metrics,
    fetch_langsmith_trace,
    load_timeout_snapshot,
    run_query,
)


class _FakeClient:
    def __init__(self, root_runs, trace_runs_by_id):
        self._root_runs = root_runs
        self._trace_runs_by_id = trace_runs_by_id

    def list_runs(self, **kwargs):
        trace_id = kwargs.get("trace_id")
        if trace_id is not None:
            return list(self._trace_runs_by_id.get(str(trace_id), []))
        return list(self._root_runs)


def _run(
    *,
    run_id: str,
    trace_id: str,
    name: str = "dissolve-agent",
    run_type: str = "chain",
    query: str = "",
    start_time: datetime,
    parent_run_id=None,
    error=None,
):
    return SimpleNamespace(
        id=run_id,
        trace_id=trace_id,
        name=name,
        run_type=run_type,
        inputs={"messages": [{"role": "user", "content": query}]},
        start_time=start_time,
        parent_run_id=parent_run_id,
        error=error,
    )


def test_fetch_langsmith_trace_prefers_query_match_over_latest_run():
    started_at = datetime.now(timezone.utc)
    target_query = "Find the optimal separation sequence for PS and PET."

    matching = _run(
        run_id="run-match",
        trace_id="trace-match",
        query=target_query,
        start_time=started_at + timedelta(seconds=2),
    )
    unrelated_newer = _run(
        run_id="run-other",
        trace_id="trace-other",
        query="Unrelated query",
        start_time=started_at + timedelta(seconds=4),
    )
    client = _FakeClient(
        [unrelated_newer, matching],
        {
            "trace-match": [
                matching,
                _run(
                    run_id="tool-1",
                    trace_id="trace-match",
                    name="task",
                    run_type="tool",
                    start_time=started_at + timedelta(seconds=3),
                    parent_run_id="run-match",
                ),
            ],
        },
    )

    trace = fetch_langsmith_trace(
        client,
        query=target_query,
        project="strap-agent",
        started_at=started_at,
    )

    assert trace["run_id"] == "run-match"
    assert trace["trace_id"] == "trace-match"
    assert trace["tool_run_count"] == 1


def test_fetch_langsmith_trace_falls_back_to_latest_when_no_query_match():
    started_at = datetime.now(timezone.utc)
    older = _run(
        run_id="run-old",
        trace_id="trace-old",
        query="older",
        start_time=started_at + timedelta(seconds=1),
    )
    newer = _run(
        run_id="run-new",
        trace_id="trace-new",
        query="newer",
        start_time=started_at + timedelta(seconds=3),
    )
    client = _FakeClient([older, newer], {"trace-new": [newer]})

    trace = fetch_langsmith_trace(
        client,
        query="query not present",
        project="strap-agent",
        started_at=started_at,
    )

    assert trace["run_id"] == "run-new"
    assert trace["trace_id"] == "trace-new"


def test_extract_metrics_ignores_router_guarded_task_calls():
    messages = [
        AIMessage(content="", tool_calls=[{
            "id": "task-ok",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(content="completed", tool_call_id="task-ok"),
        AIMessage(content="", tool_calls=[{
            "id": "task-blocked",
            "name": "task",
            "args": {"subagent_type": "biosteam-analyst"},
        }]),
        ToolMessage(
            content="Router guard: All routed specialists already completed successfully.",
            tool_call_id="task-blocked",
            status="error",
        ),
    ]

    metrics = extract_metrics(messages)

    assert metrics["subagents_invoked"] == ["separation-engineer"]
    assert metrics["tool_names"] == ["task"]
    assert metrics["n_tool_calls"] == 1


def test_run_query_persists_full_answer_and_preview(monkeypatch):
    long_answer = "A" * 650

    agent = MagicMock()
    agent.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "id": "task-1",
                "name": "task",
                "args": {"subagent_type": "separation-engineer"},
            }]),
            ToolMessage(content="done", tool_call_id="task-1"),
            AIMessage(content=long_answer),
        ]
    }

    monkeypatch.setattr(
        "architecture.test_harness.fetch_langsmith_trace",
        lambda *args, **kwargs: {"run_id": "run-1", "trace_id": "trace-1"},
    )

    tq = HarnessTestQuery(
        name="unit-full-answer",
        query="Find the best separation sequence.",
        pattern="sequential",
        expected_subagents=["separation-engineer"],
        recursion_limit=10,
    )

    result = run_query(agent, tq, _FakeClient([], {}), project_name="strap-agent")

    assert result.full_answer == long_answer
    assert result.answer_preview == long_answer[:500] + "..."
    assert result.final_answer_diagnostics["final_answer_length"] == len(long_answer)
    assert result.final_answer_diagnostics["last_ai_excerpt"] == long_answer[:280]


def test_run_query_uses_single_specialist_separation_fallback_when_final_ai_answer_is_blank(monkeypatch):
    agent = MagicMock()
    agent.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "id": "tc_sep",
                "name": "task",
                "args": {"subagent_type": "separation-engineer"},
            }]),
            ToolMessage(
                content=(
                    "<STRUCTURED_RESULT>"
                    '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET","PC"],'
                    '"best_sequence":["PS","PC","PET"],'
                    '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":105.0},'
                    '{"step":2,"polymer":"PC","solvent":"THF","temperature_c":60.0}],'
                    '"solvent_mapping":{"PS":"Toluene","PC":"THF"}}'
                    "</STRUCTURED_RESULT>"
                ),
                tool_call_id="tc_sep",
            ),
            AIMessage(content=""),
        ]
    }

    monkeypatch.setattr(
        "architecture.test_harness.fetch_langsmith_trace",
        lambda *args, **kwargs: {"run_id": "run-1", "trace_id": "trace-1"},
    )

    tq = HarnessTestQuery(
        name="unit-sep-fallback",
        query="Only do process design. Find the best separation sequence for PS, PET, and PC up to 120C at 1 atm.",
        pattern="single-agent",
        expected_subagents=["separation-engineer"],
        recursion_limit=10,
    )

    result = run_query(agent, tq, _FakeClient([], {}), project_name="strap-agent")

    assert "Recommended separation sequence" in result.full_answer
    assert result.final_answer_diagnostics["last_ai_origin"] == "routing_single_specialist_separation_fallback"


def test_run_query_routing_match_requires_exact_subagent_set(monkeypatch):
    agent = MagicMock()
    agent.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "id": "task-1",
                "name": "task",
                "args": {"subagent_type": "separation-engineer"},
            }]),
            ToolMessage(content="done", tool_call_id="task-1"),
            AIMessage(content="", tool_calls=[{
                "id": "task-2",
                "name": "task",
                "args": {"subagent_type": "visualization-specialist"},
            }]),
            ToolMessage(content="done", tool_call_id="task-2"),
            AIMessage(content="Answer text"),
        ]
    }

    monkeypatch.setattr(
        "architecture.test_harness.fetch_langsmith_trace",
        lambda *args, **kwargs: {"run_id": "run-1", "trace_id": "trace-1"},
    )

    tq = HarnessTestQuery(
        name="unit-strict-route",
        query="Only do process design.",
        pattern="single-agent",
        expected_subagents=["separation-engineer"],
        recursion_limit=10,
    )

    result = run_query(agent, tq, _FakeClient([], {}), project_name="strap-agent")

    assert result.actual_subagents == ["separation-engineer", "visualization-specialist"]
    assert result.routing_match is False


def test_run_query_persists_timeout_snapshot_before_trace_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(harness, "_TIMEOUT_SNAPSHOT_DIR", tmp_path)

    agent = MagicMock()
    agent.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[{
                "id": "tc_sep",
                "name": "task",
                "args": {"subagent_type": "separation-engineer"},
            }]),
            ToolMessage(content="Recovered prose answer from separation.", tool_call_id="tc_sep"),
            AIMessage(content="Recovered prose answer from separation."),
        ]
    }

    tq = HarnessTestQuery(
        name="unit-timeout-snapshot",
        query="Only do process design.",
        pattern="single-agent",
        expected_subagents=["separation-engineer"],
        recursion_limit=10,
    )

    result = run_query(
        agent,
        tq,
        None,
        project_name="strap-agent",
        thread_id="thread-snapshot",
        fetch_trace=False,
        persist_timeout_snapshot=True,
    )

    snapshot = load_timeout_snapshot("thread-snapshot")
    assert snapshot is not None
    assert snapshot.full_answer == "Recovered prose answer from separation."
    assert snapshot.actual_subagents == ["separation-engineer"]
    clear_timeout_snapshot("thread-snapshot")
    assert result.full_answer == "Recovered prose answer from separation."
