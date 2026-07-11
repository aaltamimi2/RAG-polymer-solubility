from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import app_server
from export_manager import export_manager


client = TestClient(app_server.app)


def _task_call(tool_call_id: str, subagent: str, description: str | None = None) -> dict:
    args = {"subagent_type": subagent}
    if description is not None:
        args["description"] = description
    return {
        "id": tool_call_id,
        "name": "task",
        "args": args,
    }


def _structured_result_content(agent: str) -> str:
    return (
        "<STRUCTURED_RESULT>"
        f'{{"agent":"{agent}","schema_version":"1.0","no_data":true}}'
        "</STRUCTURED_RESULT>"
    )


def test_model_alias_normalization_supports_default_and_legacy_values():
    assert app_server._normalize_model_name(None) == "google_genai:gemini-3.1-flash-lite"
    assert app_server._normalize_model_name("gemini-3.1-pro-preview") == "google_genai:gemini-3.1-pro-preview"
    assert app_server._normalize_model_name("gemini-2.5-flash") == "google_genai:gemini-3.5-flash"


def test_status_endpoint_returns_expected_shape():
    response = client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "tables_loaded" in payload
    assert "tools_available" in payload
    assert isinstance(payload["tables"], list)


def test_chat_endpoint_uses_runtime(monkeypatch):
    async def fake_chat(message: str, session_id: str | None, model: str | None):
        return app_server.ChatResponse(
            response=f"echo: {message}",
            session_id=session_id or "session-123",
            images=[],
            elapsed_time=0.12,
            iterations=3,
            model_used="google_genai:gemini-3.1-flash-lite",
        )

    monkeypatch.setattr(app_server.agent_runtime, "chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={"message": "hello", "session_id": "session-123", "model": "gemini-3.1-flash-lite"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["response"] == "echo: hello"
    assert payload["session_id"] == "session-123"
    assert payload["iterations"] == 3


def test_workflow_preview_endpoint_returns_topological_plan():
    response = client.post(
        "/api/workflow/preview",
        json={
            "query": (
                "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream "
                "using selective dissolution, propose a phthalate wash step, then run "
                "techno-economic analysis on solvent recovery."
            )
        },
    )
    assert response.status_code == 200
    payload = response.json()
    node_ids = {node["id"] for node in payload["nodes"]}
    assert payload["mode"] == "preview"
    assert "separation-engineer" in node_ids
    assert "contaminant-removal-analyst" in node_ids
    assert "biosteam-analyst" in node_ids
    assert any(edge["source"] == "separation-engineer" and edge["target"] == "contaminant-removal-analyst" for edge in payload["edges"])


def test_session_workflow_endpoint_returns_live_execution_status(monkeypatch):
    session = app_server.SessionState(
        session_id="session-workflow",
        model_name="google_genai:gemini-3.1-flash-lite",
    )
    session.last_query = (
        "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream "
        "using selective dissolution, then run techno-economic analysis on solvent recovery."
    )
    session.last_run_id = "run-123"
    session.last_trace_url = "https://smith.langchain.com/o/example/projects/p/demo/r/run-123?poll=true"
    session.workflow_messages = [
        HumanMessage(content=session.last_query),
        AIMessage(content="", tool_calls=[_task_call("tc_sep", "separation-engineer", "Plan the route.")]),
        ToolMessage(content=_structured_result_content("separation-engineer"), tool_call_id="tc_sep"),
    ]
    session.subagent_traces = {
        "tc_sep": {
            "tool_call_id": "tc_sep",
            "subagent": "separation-engineer",
            "run_id": "run-sep",
            "trace_url": "https://smith.langchain.com/o/example/projects/p/demo/r/run-sep?poll=true",
            "shared_trace_url": None,
        }
    }
    session.subagent_trace_details = {
        "tc_sep": {
            "run_id": "run-sep",
            "tool_count": 2,
            "tools": [
                {
                    "name": "analyze_selective_solubility",
                    "run_id": "tool-1",
                    "trace_url": "https://smith.langchain.com/o/example/projects/p/demo/r/tool-1?poll=true",
                    "shared_trace_url": None,
                    "status": "success",
                    "error": None,
                    "depth": 0,
                },
                {
                    "name": "plan_multiple_separation_schemes",
                    "run_id": "tool-2",
                    "trace_url": "https://smith.langchain.com/o/example/projects/p/demo/r/tool-2?poll=true",
                    "shared_trace_url": None,
                    "status": "success",
                    "error": None,
                    "depth": 0,
                },
            ],
            "error": None,
        }
    }
    monkeypatch.setitem(app_server.agent_runtime._sessions, "session-workflow", session)

    response = client.get("/api/session/session-workflow/workflow")
    assert response.status_code == 200
    payload = response.json()
    nodes = {node["id"]: node for node in payload["nodes"]}
    edges = {(edge["source"], edge["target"]): edge for edge in payload["edges"]}

    assert payload["mode"] == "live"
    assert nodes["separation-engineer"]["status"] == "completed"
    assert nodes["biosteam-analyst"]["status"] == "waiting_on_handoff"
    assert edges[("separation-engineer", "biosteam-analyst")]["status"] == "handoff_pending"
    assert payload["langsmith"]["run_id"] == "run-123"
    assert payload["langsmith"]["trace_url"] == session.last_trace_url
    assert nodes["separation-engineer"]["langsmith"]["run_id"] == "run-sep"
    assert nodes["separation-engineer"]["langsmith"]["trace_url"] == session.subagent_traces["tc_sep"]["trace_url"]
    assert nodes["separation-engineer"]["langsmith"]["tool_count"] == 2
    assert nodes["separation-engineer"]["langsmith"]["tools"][0]["name"] == "analyze_selective_solubility"


def test_ml_polymer_types_endpoint_returns_catalog():
    response = client.get("/api/ml/polymer-types")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_types"] > 0
    assert payload["total_polymers"] > 0
    assert all("type" in item and "count" in item for item in payload["polymer_types"])


def test_export_session_endpoint_returns_csv(tmp_path, monkeypatch):
    export_manager.export_dir = tmp_path
    export_manager.exports.clear()

    session = app_server.SessionState(session_id="session-export", model_name="google_genai:gemini-3.1-flash-lite")
    session.messages.extend(
        [
            {
                "timestamp": "2026-03-15T00:00:00",
                "role": "user",
                "content": "hello",
                "elapsed_time": None,
                "iterations": None,
                "images": [],
            },
            {
                "timestamp": "2026-03-15T00:00:01",
                "role": "assistant",
                "content": "world",
                "elapsed_time": 1.0,
                "iterations": 2,
                "images": ["plot.png"],
            },
        ]
    )
    monkeypatch.setitem(app_server.agent_runtime._sessions, "session-export", session)

    response = client.post("/api/export/session/session-export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "role,content" in response.text


def test_report_issue_saves_local_report_when_github_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(app_server, "REPORTS_DIR", tmp_path)

    async def fake_run_issue_reporter(request):
        raise RuntimeError("diagnosis unavailable")

    monkeypatch.setattr(app_server, "_run_issue_reporter", fake_run_issue_reporter)

    async def fake_create_issue(payload):
        raise RuntimeError("github unavailable")

    monkeypatch.setattr(app_server, "_create_github_issue", fake_create_issue)

    response = client.post(
        "/api/report-issue",
        json={
            "user_question": "Question",
            "assistant_response": "Answer",
            "user_description": "The response missed a contaminant step",
            "issue_type": "incorrect_response",
            "severity": "medium",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["local_report_path"] is not None
    assert Path(payload["local_report_path"]).exists()
    assert "diagnosis unavailable" in payload["error"]


def test_report_issue_returns_diagnosis_and_pr_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(app_server, "REPORTS_DIR", tmp_path)

    async def fake_run_issue_reporter(request):
        return SimpleNamespace(
            success=True,
            message="Issue diagnosed and PR created: https://github.com/example/repo/pull/12",
            error=None,
            diagnosis={
                "summary": "Workflow graph did not show the contaminant branch",
                "root_cause": "The graph panel ignored one planned node during render.",
                "fix_category": "simple_fix",
                "affected_files": ["frontend/src/App.js"],
                "proposed_changes": [{"file": "frontend/src/App.js", "description": "Render missing branch"}],
                "additional_notes": "Regression test recommended.",
                "confidence": 0.92,
            },
            pr_result={
                "success": True,
                "pr_url": "https://github.com/example/repo/pull/12",
                "pr_number": 12,
                "branch_name": "fix/workflow-graph",
            },
            issue_result=None,
        )

    monkeypatch.setattr(app_server, "_run_issue_reporter", fake_run_issue_reporter)

    response = client.post(
        "/api/report-issue",
        json={
            "user_question": "Question",
            "assistant_response": "Answer",
            "user_description": "The workflow graph missed the contaminant branch",
            "issue_type": "ui_bug",
            "severity": "high",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["diagnosis"]["fix_category"] == "simple_fix"
    assert payload["pr_url"] == "https://github.com/example/repo/pull/12"
    assert payload["pr_result"]["branch_name"] == "fix/workflow-graph"
    assert Path(payload["local_report_path"]).exists()
