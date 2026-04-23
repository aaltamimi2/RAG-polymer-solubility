from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from export_manager import export_manager
from strap.agent import create_dissolve_agent
from strap.database import get_connection, reload_database
from strap.langsmith_tracing import (
    get_captured_subagent_traces,
    get_langsmith_client,
    langsmith_trace,
    resolve_run_links,
    summarize_subagent_tool_runs,
    start_subagent_trace_capture,
    stop_subagent_trace_capture,
)
from strap.ml_assets import load_ml_polymer_catalog, missing_ml_assets
from strap.routing_classifier import (
    classify_query_keywords,
    plan_workflow_rules,
    select_workflow_rules,
)
from strap.routing_handoff_state import (
    _get_built_handoff_since,
    _get_missing_required_handoffs_for_consumer,
)
from strap.routing_message_state import (
    _get_active_remaining_steps,
    _get_effective_completed_task_ids,
    _get_effective_failed_task_ids,
    _get_last_human_message,
    _get_latest_dispatch_for_subagent,
    _get_ordered_plan,
    _get_step_dependencies,
    _get_task_handoff_statuses,
)
from strap.tools import get_all_tools
from strap.tools._helpers import get_plots_dir

load_dotenv(override=True)

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / "data")).resolve()
PLOTS_DIR = Path(os.environ.get("PLOTS_DIR", get_plots_dir())).resolve()
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", ROOT_DIR / "reports")).resolve()
FRONTEND_BUILD_DIR = Path(os.environ.get("FRONTEND_BUILD_DIR", ROOT_DIR / "frontend" / "build")).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ALIASES = {
    "gemini-3.1-flash-lite-preview": "google_genai:gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview": "google_genai:gemini-3.1-pro-preview",
    "gemini-3-flash-preview": "google_genai:gemini-3-flash-preview",
    # Backward-compatible aliases for existing clients/localStorage values.
    "gemini-2.5-flash-lite": "google_genai:gemini-3.1-flash-lite-preview",
    "gemini-2.5-flash": "google_genai:gemini-3-flash-preview",
    "gemini-2.5-pro": "google_genai:gemini-3.1-pro-preview",
}
PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
SUBAGENT_GRAPH_META: dict[str, dict[str, str]] = {
    "separation-engineer": {"label": "Separation", "icon": "layers", "accent": "blue"},
    "contaminant-removal-analyst": {"label": "Contaminants", "icon": "alert-triangle", "accent": "amber"},
    "biosteam-analyst": {"label": "TEA / Recovery", "icon": "calculator", "accent": "green"},
    "scholar-researcher": {"label": "Literature", "icon": "book-open", "accent": "violet"},
    "patent-researcher": {"label": "Patents", "icon": "search", "accent": "orange"},
    "rag-analyst": {"label": "RAG Synthesis", "icon": "brain", "accent": "pink"},
    "visualization-specialist": {"label": "Visualization", "icon": "bar-chart-3", "accent": "cyan"},
    "statistics-ml": {"label": "Statistics / ML", "icon": "brain", "accent": "indigo"},
    "safety-analyst": {"label": "Safety", "icon": "shield-alert", "accent": "rose"},
}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    model: str | None = "gemini-3.1-flash-lite-preview"


class ChatResponse(BaseModel):
    response: str
    session_id: str
    images: list[str] = []
    elapsed_time: float
    iterations: int
    model_used: str


class IssueReportRequest(BaseModel):
    user_question: str
    assistant_response: str
    elapsed_time: float = 0.0
    iterations: int = 0
    images: list[dict[str, str]] = []
    user_description: str
    issue_type: str = "incorrect_response"
    severity: str = "medium"
    session_id: str | None = None


class IssueReportResponse(BaseModel):
    success: bool
    message: str = ""
    error: str | None = None
    issue_url: str | None = None
    issue_number: int | None = None
    pr_url: str | None = None
    pr_number: int | None = None
    local_report_path: str | None = None
    diagnosis: dict[str, Any] | None = None
    issue_result: dict[str, Any] | None = None
    pr_result: dict[str, Any] | None = None


class WorkflowPreviewRequest(BaseModel):
    query: str


@dataclass
class SessionState:
    session_id: str
    model_name: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    workflow_messages: list[Any] = field(default_factory=list)
    last_query: str | None = None
    last_run_id: str | None = None
    last_trace_url: str | None = None
    last_shared_trace_url: str | None = None
    subagent_traces: dict[str, dict[str, Any]] = field(default_factory=dict)
    subagent_trace_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _normalize_model_name(raw_model: str | None) -> str:
    if not raw_model:
        return MODEL_ALIASES["gemini-3.1-flash-lite-preview"]
    model = raw_model.strip()
    if ":" in model:
        return model
    return MODEL_ALIASES.get(model, f"google_genai:{model}")


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


def _snapshot_plot_state() -> dict[str, float]:
    state: dict[str, float] = {}
    for path in PLOTS_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in PLOT_SUFFIXES:
            state[path.name] = path.stat().st_mtime
    return state


def _new_plot_names(before: dict[str, float]) -> list[str]:
    after = _snapshot_plot_state()
    new_names = [
        name
        for name, mtime in after.items()
        if name not in before or mtime > before[name]
    ]
    new_names.sort(key=lambda item: after[item], reverse=True)
    return new_names


def _list_plot_entries() -> list[dict[str, str]]:
    entries = []
    for path in sorted(PLOTS_DIR.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_file() and path.suffix.lower() in PLOT_SUFFIXES:
            entries.append(
                {
                    "filename": path.name,
                    "url": f"/plots/{path.name}",
                    "created": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            )
    return entries


def _tool_count() -> int:
    return len({tool.__name__ for tool in get_all_tools()})


def _table_summaries() -> list[dict[str, Any]]:
    conn = get_connection(DATA_DIR)
    tables_df = conn.execute("SHOW TABLES").fetchdf()
    tables: list[dict[str, Any]] = []
    for table_name in tables_df["name"].tolist():
        schema_df = conn.execute(f"DESCRIBE {table_name}").fetchdf()
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        tables.append(
            {
                "name": str(table_name),
                "rows": int(row_count),
                "columns": [str(col) for col in schema_df["column_name"].tolist()],
            }
        )
    return tables

def _git_revision() -> dict[str, str]:
    def _run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                args,
                cwd=ROOT_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    return {
        "branch": _run("git", "branch", "--show-current") or "unknown",
        "commit": _run("git", "rev-parse", "--short", "HEAD") or "unknown",
    }


def _graph_langsmith_metadata(
    *,
    thread_id: str | None,
    run_id: str | None = None,
    trace_url: str | None = None,
    shared_trace_url: str | None = None,
) -> dict[str, Any]:
    enabled = bool(os.getenv("LANGSMITH_API_KEY"))
    return {
        "enabled": enabled,
        "project": os.getenv("LANGSMITH_PROJECT", "strap-agent") if enabled else None,
        "thread_id": thread_id,
        "run_id": run_id,
        "trace_url": trace_url,
        "shared_trace_url": shared_trace_url,
    }


def _resolve_langsmith_run_links(run_tree: Any) -> dict[str, str | None]:
    return resolve_run_links(run_tree)


def _deterministic_allowed_rules(query_text: str) -> list[dict[str, Any]]:
    if not query_text.strip():
        return []
    keyword_matched = classify_query_keywords([HumanMessage(content=query_text)])
    matched = select_workflow_rules(
        query_text,
        keyword_matched=keyword_matched,
    )
    return plan_workflow_rules(query_text, matched)


def _compute_node_levels(nodes: list[dict[str, Any]]) -> dict[str, int]:
    dependency_map = {
        str(node["id"]): tuple(str(dep) for dep in node.get("depends_on", []) if str(dep).strip())
        for node in nodes
    }
    memo: dict[str, int] = {}

    def level_for(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        deps = dependency_map.get(node_id, ())
        if not deps:
            memo[node_id] = 0
            return 0
        memo[node_id] = 1 + max(level_for(dep) for dep in deps)
        return memo[node_id]

    for node_id in dependency_map:
        level_for(node_id)
    return memo


def _node_graph_meta(subagent: str) -> dict[str, str]:
    meta = SUBAGENT_GRAPH_META.get(subagent)
    if meta is not None:
        return meta
    return {"label": subagent.replace("-", " ").title(), "icon": "flask-conical", "accent": "slate"}


def _build_next_action(
    messages: list[Any] | None,
    ordered_plan: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not messages:
        if not ordered_plan:
            return None
        roots = [step["subagent"] for step in ordered_plan if not step.get("depends_on")]
        if not roots:
            return None
        return {"type": "dispatch", "subagents": roots, "label": "Dispatch root specialists"}

    remaining = _get_active_remaining_steps(messages, ordered_plan)
    if not remaining:
        return {"type": "complete", "label": "Workflow complete"}

    next_step = remaining[0]
    next_name = str(next_step["subagent"])
    missing_handoffs = _get_missing_required_handoffs_for_consumer(messages, ordered_plan, next_name)
    if missing_handoffs:
        producer, consumer = missing_handoffs[0]
        return {
            "type": "build_handoff",
            "producer": producer,
            "consumer": consumer,
            "label": f"Build handoff {producer} -> {consumer}",
        }

    return {
        "type": "dispatch",
        "subagents": [next_name],
        "label": f"Dispatch {next_name}",
    }


def _build_workflow_payload(
    query_text: str,
    *,
    session_id: str | None = None,
    messages: list[Any] | None = None,
    run_id: str | None = None,
    trace_url: str | None = None,
    shared_trace_url: str | None = None,
    subagent_traces: dict[str, dict[str, Any]] | None = None,
    subagent_trace_details: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    allowed_rules = _deterministic_allowed_rules(query_text)
    if messages:
        ordered_plan = _get_ordered_plan(messages, allowed_rules=allowed_rules)
    else:
        ordered_plan = _get_ordered_plan(
            [HumanMessage(content=query_text)],
            allowed_rules=allowed_rules,
        )

    completed_ids = _get_effective_completed_task_ids(messages) if messages else set()
    failed_ids = _get_effective_failed_task_ids(messages) if messages else set()
    handoff_statuses = _get_task_handoff_statuses(messages) if messages else {}
    level_map: dict[str, int] = {}

    raw_nodes: list[dict[str, Any]] = []
    for step in ordered_plan:
        subagent = str(step["subagent"])
        depends_on = tuple(str(dep) for dep in step.get("depends_on", ()) if str(dep).strip())
        latest_dispatch = _get_latest_dispatch_for_subagent(messages, subagent) if messages else None
        latest_ok_dispatch = (
            _get_latest_dispatch_for_subagent(messages, subagent, status="ok")
            if messages else None
        )
        latest_trace = None
        latest_trace_details = None
        if latest_dispatch is not None and subagent_traces:
            latest_trace = subagent_traces.get(latest_dispatch["tool_call_id"])
        if latest_dispatch is not None and subagent_trace_details:
            latest_trace_details = subagent_trace_details.get(latest_dispatch["tool_call_id"])

        if messages and latest_dispatch is not None:
            task_id = latest_dispatch["tool_call_id"]
            if task_id in failed_ids:
                status = "failed"
            elif task_id in completed_ids:
                status = "completed"
            else:
                status = "running"
        elif messages and depends_on:
            unmet_dependencies = [
                dep for dep in depends_on
                if (
                    dispatch := _get_latest_dispatch_for_subagent(messages, dep, status="ok")
                ) is None or dispatch["tool_call_id"] not in completed_ids
            ]
            if unmet_dependencies:
                status = "waiting_on_dependencies"
            elif _get_missing_required_handoffs_for_consumer(messages, ordered_plan, subagent):
                status = "waiting_on_handoff"
            else:
                status = "planned"
        else:
            status = "planned"

        meta = _node_graph_meta(subagent)
        raw_nodes.append(
            {
                "id": subagent,
                "subagent": subagent,
                "label": meta["label"],
                "icon": meta["icon"],
                "accent": meta["accent"],
                "status": status,
                "description": str(step.get("description", "")),
                "depends_on": list(depends_on),
                "step_id": str(step.get("step_id", f"advisory:{subagent}")),
                "dispatch_tool_call_id": latest_dispatch["tool_call_id"] if latest_dispatch else None,
                "handoff_status": handoff_statuses.get(latest_ok_dispatch["tool_call_id"]) if latest_ok_dispatch else None,
                "langsmith": {
                    "run_id": latest_trace.get("run_id") if latest_trace else None,
                    "trace_url": latest_trace.get("trace_url") if latest_trace else None,
                    "shared_trace_url": latest_trace.get("shared_trace_url") if latest_trace else None,
                    "tool_count": latest_trace_details.get("tool_count") if latest_trace_details else 0,
                    "total_duration_ms": latest_trace_details.get("total_duration_ms") if latest_trace_details else None,
                    "tools": latest_trace_details.get("tools") if latest_trace_details else [],
                    "tools_error": latest_trace_details.get("error") if latest_trace_details else None,
                },
            }
        )

    if raw_nodes:
        level_map = _compute_node_levels(raw_nodes)

    nodes_by_level: dict[int, list[str]] = {}
    for node in raw_nodes:
        level = level_map.get(node["id"], 0)
        node["level"] = level
        nodes_by_level.setdefault(level, []).append(node["id"])
    for level_nodes in nodes_by_level.values():
        level_nodes.sort(key=lambda node_id: next(index for index, node in enumerate(raw_nodes) if node["id"] == node_id))
    for node in raw_nodes:
        level = int(node["level"])
        node["position"] = {
            "column": level,
            "row": nodes_by_level[level].index(node["id"]),
        }

    edges: list[dict[str, Any]] = []
    for node in raw_nodes:
        consumer = node["id"]
        for producer in node["depends_on"]:
            edge_status = "planned"
            handoff_id = None
            contract = None
            if messages:
                producer_dispatch = _get_latest_dispatch_for_subagent(messages, producer, status="ok")
                producer_any_dispatch = _get_latest_dispatch_for_subagent(messages, producer)
                if producer_any_dispatch and producer_any_dispatch["tool_call_id"] in failed_ids:
                    edge_status = "blocked"
                elif producer_dispatch is None:
                    edge_status = "waiting_on_dependency"
                else:
                    built_handoff = _get_built_handoff_since(
                        messages,
                        producer=producer,
                        consumer=consumer,
                        after_task_call_id=producer_dispatch["tool_call_id"],
                    )
                    if built_handoff is not None:
                        edge_status = "handoff_ready"
                        handoff_id = built_handoff.get("handoff_id")
                        contract = built_handoff.get("contract")
                    elif producer_dispatch["tool_call_id"] in completed_ids:
                        edge_status = "handoff_pending"
                    else:
                        edge_status = "in_progress"

            edges.append(
                {
                    "id": f"{producer}->{consumer}",
                    "source": producer,
                    "target": consumer,
                    "status": edge_status,
                    "handoff_id": handoff_id,
                    "contract": contract,
                }
            )

    completed_count = sum(1 for node in raw_nodes if node["status"] == "completed")
    failed_count = sum(1 for node in raw_nodes if node["status"] == "failed")
    running_count = sum(1 for node in raw_nodes if node["status"] == "running")

    if not messages:
        workflow_state = "preview"
    elif raw_nodes and completed_count == len(raw_nodes):
        workflow_state = "complete"
    elif failed_count:
        workflow_state = "attention"
    elif running_count:
        workflow_state = "active"
    else:
        workflow_state = "planned"

    return {
        "query": query_text,
        "mode": "live" if messages else "preview",
        "workflow_state": workflow_state,
        "nodes": raw_nodes,
        "edges": edges,
        "levels": [
            {"level": level, "nodes": nodes}
            for level, nodes in sorted(nodes_by_level.items(), key=lambda item: item[0])
        ],
        "summary": {
            "total_nodes": len(raw_nodes),
            "completed_nodes": completed_count,
            "failed_nodes": failed_count,
            "running_nodes": running_count,
        },
        "next_action": _build_next_action(messages, ordered_plan),
        "langsmith": _graph_langsmith_metadata(
            thread_id=session_id,
            run_id=run_id,
            trace_url=trace_url,
            shared_trace_url=shared_trace_url,
        ),
    }


def _ensure_subagent_trace_details(session: SessionState) -> None:
    if not session.subagent_traces:
        session.subagent_trace_details = {}
        return

    for tool_call_id, trace_meta in session.subagent_traces.items():
        run_id = trace_meta.get("run_id")
        cached = session.subagent_trace_details.get(tool_call_id)
        if cached and cached.get("run_id") == run_id:
            continue
        summary = summarize_subagent_tool_runs(run_id)
        session.subagent_trace_details[tool_call_id] = {
            "run_id": run_id,
            **summary,
        }


async def _create_github_issue(payload: dict[str, Any]) -> dict[str, Any]:
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO", "aaltamimi2/RAG-polymer-solubility")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not configured")

    labels = [
        "frontend-report",
        f"severity:{payload['severity']}",
        f"type:{payload['issue_type']}",
    ]
    revision = _git_revision()
    title = f"[{payload['severity'].upper()}] {payload['issue_type']}: {payload['user_description'][:80]}"
    body = "\n".join(
        [
            "## User Report",
            payload["user_description"],
            "",
            "## User Question",
            payload.get("user_question") or "_none provided_",
            "",
            "## Assistant Response",
            payload.get("assistant_response") or "_none provided_",
            "",
            "## Runtime Metadata",
            f"- Session: `{payload.get('session_id') or 'n/a'}`",
            f"- Elapsed: `{payload.get('elapsed_time', 0.0):.2f}s`",
            f"- Iterations: `{payload.get('iterations', 0)}`",
            f"- Branch: `{revision['branch']}`",
            f"- Commit: `{revision['commit']}`",
        ]
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"https://api.github.com/repos/{repo}/issues",
            headers=headers,
            json={"title": title, "body": body, "labels": labels},
        )
        response.raise_for_status()
        issue = response.json()
    return {
        "issue_url": issue.get("html_url"),
        "issue_number": issue.get("number"),
        "issue_result": issue,
    }


def _persist_local_report(payload: dict[str, Any]) -> str:
    report_id = uuid.uuid4().hex[:12]
    path = REPORTS_DIR / f"issue_report_{report_id}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return str(path)


async def _run_issue_reporter(request: IssueReportRequest):
    from services.issue_reporter import get_issue_reporter

    reporter = get_issue_reporter()
    return await reporter.process_report(
        user_question=request.user_question,
        assistant_response=request.assistant_response,
        elapsed_time=request.elapsed_time,
        iterations=request.iterations,
        images=request.images,
        user_description=request.user_description,
        issue_type=request.issue_type,
        severity=request.severity,
        session_id=request.session_id,
    )


def _coerce_issue_report_response(result: Any, local_path: str) -> IssueReportResponse:
    diagnosis = getattr(result, "diagnosis", None)
    pr_result = getattr(result, "pr_result", None)
    issue_result = getattr(result, "issue_result", None)

    issue_url = None
    issue_number = None
    if isinstance(issue_result, dict):
        issue_url = issue_result.get("issue_url") or issue_result.get("html_url")
        issue_number = issue_result.get("issue_number") or issue_result.get("number")

    pr_url = None
    pr_number = None
    if isinstance(pr_result, dict):
        pr_url = pr_result.get("pr_url") or pr_result.get("html_url")
        pr_number = pr_result.get("pr_number") or pr_result.get("number")

    return IssueReportResponse(
        success=bool(getattr(result, "success", False)),
        message=str(getattr(result, "message", "")),
        error=getattr(result, "error", None),
        issue_url=issue_url,
        issue_number=issue_number,
        pr_url=pr_url,
        pr_number=pr_number,
        local_report_path=local_path,
        diagnosis=diagnosis,
        issue_result=issue_result,
        pr_result=pr_result,
    )


class V9AgentRuntime:
    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}
        self._sessions: dict[str, SessionState] = {}

    def _get_agent(self, model_name: str):
        agent = self._agents.get(model_name)
        if agent is None:
            logger.info("Creating DISSOLVE agent for model %s", model_name)
            agent = create_dissolve_agent(model_name=model_name, enable_persistence=True)
            self._agents[model_name] = agent
        return agent

    def warm_agent(self, model: str | None = None) -> None:
        model_name = _normalize_model_name(model)
        try:
            self._get_agent(model_name)
        except Exception:
            logger.exception("Failed to warm DISSOLVE agent for model %s", model_name)

    def get_or_create_session(self, session_id: str | None, requested_model: str) -> SessionState:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        new_session_id = session_id or uuid.uuid4().hex[:12]
        session = SessionState(session_id=new_session_id, model_name=requested_model)
        self._sessions[new_session_id] = session
        return session

    def clear_session(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def export_session(self, session_id: str) -> str:
        session = self._sessions.get(session_id)
        if session is None or not session.messages:
            raise KeyError(session_id)
        export_id = export_manager.create_export(
            data=session.messages,
            tool_name=f"conversation_{session_id}",
            columns=["timestamp", "role", "content", "elapsed_time", "iterations", "images"],
        )
        path = export_manager.get_export_path(export_id)
        if path is None:
            raise RuntimeError("Failed to create session export")
        return path

    async def chat(self, message: str, session_id: str | None, model: str | None) -> ChatResponse:
        requested_model = _normalize_model_name(model)
        session = self.get_or_create_session(session_id, requested_model)
        async with session.lock:
            return await asyncio.to_thread(self._chat_sync, session, message)

    def _chat_sync(self, session: SessionState, message: str) -> ChatResponse:
        before_plots = _snapshot_plot_state()
        agent = self._get_agent(session.model_name)
        start = time.time()
        result: dict[str, Any] | None = None
        root_run = None
        trace_capture_token = start_subagent_trace_capture()
        if langsmith_trace is not None and get_langsmith_client() is not None:
            invoke_started = False
            try:
                with langsmith_trace(
                    "DISSOLVE frontend query",
                    run_type="chain",
                    project_name=os.getenv("LANGSMITH_PROJECT", "strap-agent"),
                    inputs={
                        "message": message,
                        "session_id": session.session_id,
                        "thread_id": session.session_id,
                        "model_name": session.model_name,
                    },
                    metadata={
                        "session_id": session.session_id,
                        "thread_id": session.session_id,
                        "model_name": session.model_name,
                        "entrypoint": "app_server",
                    },
                    tags=["dissolve", "frontend", "app_server"],
                ) as traced_run:
                    root_run = traced_run
                    invoke_started = True
                    result = agent.invoke(
                        {"messages": [HumanMessage(content=message)]},
                        {"configurable": {"thread_id": session.session_id}, "recursion_limit": 150},
                    )
                    traced_run.metadata.update(
                        {
                            "iteration_count": int(result.get("iteration_count", 0) or 0),
                        }
                    )
                    traced_run.end(
                        outputs={
                            "iteration_count": int(result.get("iteration_count", 0) or 0),
                            "message_count": len(result.get("messages", []) or []),
                        }
                    )
            except Exception:
                if result is not None:
                    logger.exception("LangSmith trace finalization failed after successful agent invocation")
                elif invoke_started:
                    stop_subagent_trace_capture(trace_capture_token)
                    raise
                else:
                    logger.exception("LangSmith trace setup failed; falling back to unwrapped agent invocation")

        try:
            if result is None:
                result = agent.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    {"configurable": {"thread_id": session.session_id}, "recursion_limit": 150},
                )
            session.subagent_traces = get_captured_subagent_traces()
        finally:
            stop_subagent_trace_capture(trace_capture_token)
        elapsed = time.time() - start

        workflow_messages = result.get("messages", [])
        final_text = _extract_text(workflow_messages[-1].content) if workflow_messages else "No response generated."
        iterations = int(result.get("iteration_count", 0) or 0)
        time.sleep(0.15)
        images = _new_plot_names(before_plots)
        session.workflow_messages = list(workflow_messages)
        session.last_query = message
        session.subagent_trace_details = {}
        trace_links = _resolve_langsmith_run_links(root_run)
        session.last_run_id = trace_links["run_id"]
        session.last_trace_url = trace_links["trace_url"]
        session.last_shared_trace_url = trace_links["shared_trace_url"]

        session.messages.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "role": "user",
                "content": message,
                "elapsed_time": None,
                "iterations": None,
                "images": [],
            }
        )
        session.messages.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "role": "assistant",
                "content": final_text,
                "elapsed_time": round(elapsed, 4),
                "iterations": iterations,
                "images": images,
            }
        )
        return ChatResponse(
            response=final_text,
            session_id=session.session_id,
            images=images,
            elapsed_time=elapsed,
            iterations=iterations,
            model_used=session.model_name,
        )


agent_runtime = V9AgentRuntime()

app = FastAPI(title="DISSOLVE v9 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")


@app.on_event("startup")
async def warm_default_agent_on_startup() -> None:
    if os.getenv("DISSOLVE_WARM_DEFAULT_AGENT", "true").strip().lower() in {"0", "false", "no"}:
        logger.info("Skipping default agent warmup")
        return
    default_model = MODEL_ALIASES["gemini-3.1-flash-lite-preview"]
    logger.info("Warming default DISSOLVE agent for model %s", default_model)
    await asyncio.to_thread(agent_runtime.warm_agent, default_model)


@app.get("/api/status")
async def api_status():
    try:
        tables = _table_summaries()
        missing_files: list[str] = []
        if not any(DATA_DIR.glob("*.csv")):
            missing_files.append("No CSV files found in data directory")
        missing_files.extend(missing_ml_assets())
        return {
            "status": "ready" if tables else "limited",
            "tables_loaded": len(tables),
            "tools_available": _tool_count(),
            "tables": [table["name"] for table in tables],
            "missing_files": missing_files,
        }
    except Exception as exc:
        logger.exception("Failed to load status")
        return {
            "status": "error",
            "tables_loaded": 0,
            "tools_available": 0,
            "tables": [],
            "missing_files": [str(exc)],
        }


@app.get("/api/tables")
async def api_tables():
    return {"tables": _table_summaries()}


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    try:
        return await agent_runtime.chat(request.message, request.session_id, request.model)
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/workflow/preview")
async def api_workflow_preview(request: WorkflowPreviewRequest):
    query = request.query.strip()
    if not query:
        return {
            "query": "",
            "mode": "preview",
            "workflow_state": "empty",
            "nodes": [],
            "edges": [],
            "levels": [],
            "summary": {
                "total_nodes": 0,
                "completed_nodes": 0,
                "failed_nodes": 0,
                "running_nodes": 0,
            },
            "next_action": None,
            "langsmith": _graph_langsmith_metadata(thread_id=None),
        }
    return _build_workflow_payload(query)


@app.get("/api/session/{session_id}/workflow")
async def api_session_workflow(session_id: str):
    session = agent_runtime._sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    query = (
        _get_last_human_message(session.workflow_messages)
        if session.workflow_messages else None
    ) or session.last_query

    if not query:
        return {
            "query": "",
            "mode": "live",
            "workflow_state": "empty",
            "nodes": [],
            "edges": [],
            "levels": [],
            "summary": {
                "total_nodes": 0,
                "completed_nodes": 0,
                "failed_nodes": 0,
                "running_nodes": 0,
            },
            "next_action": None,
            "langsmith": _graph_langsmith_metadata(
                thread_id=session_id,
                run_id=session.last_run_id,
                trace_url=session.last_trace_url,
                shared_trace_url=session.last_shared_trace_url,
            ),
        }

    _ensure_subagent_trace_details(session)
    return _build_workflow_payload(
        query,
        session_id=session_id,
        messages=session.workflow_messages or None,
        run_id=session.last_run_id,
        trace_url=session.last_trace_url,
        shared_trace_url=session.last_shared_trace_url,
        subagent_traces=session.subagent_traces,
        subagent_trace_details=session.subagent_trace_details,
    )


@app.post("/api/reindex")
async def api_reindex():
    database = reload_database(DATA_DIR)
    tables = database.conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    return {"success": True, "tables_loaded": len(tables), "tables": tables}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are supported")
    destination = DATA_DIR / Path(file.filename).name
    content = await file.read()
    destination.write_bytes(content)
    reload_database(DATA_DIR)
    return {"success": True, "filename": destination.name}


@app.get("/api/plots")
async def api_plots():
    return {"plots": _list_plot_entries()}


@app.delete("/api/plots")
async def api_clear_plots():
    removed = 0
    for path in list(PLOTS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in PLOT_SUFFIXES:
            path.unlink()
            removed += 1
    return {"success": True, "removed": removed}


@app.delete("/api/session/{session_id}")
async def api_clear_session(session_id: str):
    if not agent_runtime.clear_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True}


@app.get("/api/export/{export_id}")
async def api_download_export(export_id: str):
    filepath = export_manager.get_export_path(export_id)
    if filepath is None:
        raise HTTPException(status_code=404, detail="Export not found or expired")
    return FileResponse(filepath, media_type="text/csv", filename=Path(filepath).name)


@app.post("/api/export/session/{session_id}")
async def api_export_session(session_id: str):
    try:
        filepath = agent_runtime.export_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(filepath, media_type="text/csv", filename=Path(filepath).name)


@app.post("/api/report-issue", response_model=IssueReportResponse)
async def api_report_issue(request: IssueReportRequest):
    payload = request.model_dump()
    local_path = _persist_local_report(payload)
    try:
        result = await _run_issue_reporter(request)
        response = _coerce_issue_report_response(result, local_path)
        if response.success:
            return response
        logger.warning("Issue reporter did not complete successfully: %s", response.error)
        reporter_error = response.error or "Issue reporter did not complete successfully"
    except Exception as exc:
        logger.warning("Issue reporter failed: %s", exc)
        reporter_error = str(exc)

    try:
        issue_payload = await _create_github_issue(payload)
        return IssueReportResponse(
            success=True,
            message="Issue saved locally and filed on GitHub. AI diagnosis/PR automation was unavailable.",
            error=reporter_error,
            local_report_path=local_path,
            issue_url=issue_payload["issue_url"],
            issue_number=issue_payload["issue_number"],
            issue_result=issue_payload["issue_result"],
        )
    except Exception as exc:
        logger.warning("GitHub fallback issue creation failed: %s", exc)
        return IssueReportResponse(
            success=True,
            message="Issue saved locally. AI diagnosis and GitHub automation are unavailable in this environment.",
            local_report_path=local_path,
            error=f"{reporter_error}; GitHub fallback failed: {exc}",
        )


@app.get("/api/ml/polymer-types")
async def api_ml_polymer_types():
    try:
        types, grouped = load_ml_polymer_catalog()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "total_types": len(types),
        "total_polymers": sum(len(items) for items in grouped.values()),
        "polymer_types": types,
    }


@app.get("/api/ml/polymers-by-type/{polymer_type}")
async def api_ml_polymers_by_type(polymer_type: str):
    try:
        _, grouped = load_ml_polymer_catalog()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    polymers = grouped.get(polymer_type)
    if polymers is None:
        raise HTTPException(status_code=404, detail=f"Unknown polymer type: {polymer_type}")
    return {"type": polymer_type, "count": len(polymers), "polymers": polymers}


if FRONTEND_BUILD_DIR.exists():
    static_dir = FRONTEND_BUILD_DIR / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="frontend-static")

    @app.get("/", include_in_schema=False)
    async def frontend_index():
        return FileResponse(str(FRONTEND_BUILD_DIR / "index.html"))


    @app.get("/{full_path:path}", include_in_schema=False)
    async def frontend_spa(full_path: str):
        if full_path.startswith(("api/", "plots/", "static/")):
            raise HTTPException(status_code=404, detail="Not found")
        target = FRONTEND_BUILD_DIR / full_path
        if target.exists() and target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(FRONTEND_BUILD_DIR / "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def api_root():
        return JSONResponse(
            {
                "name": "DISSOLVE v9 API",
                "frontend_build": False,
                "message": "Build frontend/ and serve the generated assets, or run the CRA dev server separately.",
            }
        )
