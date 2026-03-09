"""Structured result extraction and handoff tools."""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from .handoffs import (
    bind_handoff_scope,
    build_handoff_for_consumer,
    cleanup_handoff_scope,
    get_handoff,
    get_latest_result_handoff,
    get_current_scope,
    initialize_handoff_scope,
    list_handoff_records,
    list_result_records,
    record_to_json_envelope,
    records_to_json_envelope,
    store_agent_failure,
    store_agent_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import ToolCallRequest

logger = logging.getLogger(__name__)

_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)


def _json_error(message: str, **details: Any) -> str:
    payload: dict[str, Any] = {"ok": False, "error": message}
    if details:
        payload["details"] = details
    return json.dumps(payload, indent=2)


def _extract_structured_result(text: str) -> dict[str, Any] | None:
    """Extract and parse a <STRUCTURED_RESULT> JSON block from text."""
    match = _STRUCTURED_RESULT_RE.search(text)
    if not match:
        return None
    json_text = match.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()
    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "result_extractor: malformed JSON in <STRUCTURED_RESULT> — skipping. "
            "Error: %s  |  First 200 chars: %.200s",
            exc,
            json_text,
        )
        return None


def get_structured_results() -> dict[str, list[dict[str, Any]]]:
    """Return all stored subagent results grouped by producer."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in list_result_records():
        grouped.setdefault(record.producer, []).append(record.payload)
    return grouped


def get_structured_result(agent_name: str) -> dict[str, Any] | None:
    """Return the latest structured result payload for a specific subagent."""
    record = get_latest_result_handoff(producer=agent_name)
    return None if record is None else record.payload


def get_subagent_result(agent_name: str, strategy: str = "latest") -> str:
    """Retrieve stored handoff records for a specific subagent."""
    if strategy != "latest":
        return _json_error(
            f"Unsupported strategy '{strategy}'",
            supported_strategies=["latest"],
        )
    record = get_latest_result_handoff(producer=agent_name)
    if record is None:
        available = sorted({r.producer for r in list_handoff_records()})
        return _json_error(
            f"No structured result found for '{agent_name}'",
            available_agents=available,
        )
    return record_to_json_envelope(record)


def get_subagent_results(agent_name: str) -> str:
    """Retrieve all stored handoff records for a specific subagent."""
    records = list_result_records(producer=agent_name)
    if not records:
        available = sorted({r.producer for r in list_handoff_records()})
        return _json_error(
            f"No structured results found for '{agent_name}'",
            available_agents=available,
        )
    return records_to_json_envelope(records)


def get_all_subagent_results() -> str:
    """Retrieve all stored subagent result records."""
    records = list_result_records()
    if not records:
        return _json_error("No structured results available")
    return records_to_json_envelope(records)


def list_handoffs(
    producer: str = "",
    consumer: str = "",
    contract: str = "",
    status: str = "",
) -> str:
    """List stored handoff records filtered by metadata."""
    records = list_handoff_records(
        producer=producer or None,
        consumer=consumer or None,
        contract=contract or None,
        status=status or None,
    )
    if not records:
        return _json_error(
            "No handoffs matched the requested filters",
            producer=producer or None,
            consumer=consumer or None,
            contract=contract or None,
            status=status or None,
        )
    return records_to_json_envelope(records)


def get_handoff_details(handoff_id: str) -> str:
    """Retrieve one handoff by ID."""
    record = get_handoff(handoff_id)
    if record is None:
        return _json_error(f"Handoff '{handoff_id}' not found")
    return record_to_json_envelope(record)


def build_handoff(
    consumer: str,
    source_handoff_id: str = "",
    producer: str = "",
    strategy: str = "latest",
) -> str:
    """Build a consumer-specific handoff via a typed adapter or generic fallback."""
    try:
        record = build_handoff_for_consumer(
            consumer=consumer,
            source_handoff_id=source_handoff_id or None,
            producer=producer or None,
            strategy=strategy,
        )
    except ValueError as exc:
        logger.warning(
            "build_handoff failed for producer=%s consumer=%s source_handoff_id=%s: %s",
            producer or "<latest>",
            consumer,
            source_handoff_id or "<none>",
            exc,
        )
        return _json_error(str(exc))
    return record_to_json_envelope(record)


class StructuredResultExtractorMiddleware(AgentMiddleware):
    """Capture subagent structured results as append-only handoff records."""

    def __init__(self, artifact_root: Path | None = None) -> None:
        self._artifact_root = artifact_root
        self._scope = None
        self._user_query: str | None = None

    def before_agent(self, state, runtime) -> None:
        scope = self._initialize_scope(state)
        self._scope = scope
        logger.debug("result_extractor: initialized scope %s", scope.scope_id)

    async def abefore_agent(self, state, runtime) -> None:
        scope = self._initialize_scope(state)
        self._scope = scope
        logger.debug("result_extractor: initialized scope %s (async)", scope.scope_id)

    def after_agent(self, state, runtime) -> None:
        cleanup_handoff_scope()
        self._scope = None
        self._user_query = None

    async def aafter_agent(self, state, runtime) -> None:
        cleanup_handoff_scope()
        self._scope = None
        self._user_query = None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if get_current_scope() is None:
            if self._scope is not None:
                bind_handoff_scope(
                    self._scope,
                    artifact_root=self._artifact_root,
                    user_query=self._user_query,
                )
            else:
                self._scope = self._initialize_scope()
        result = handler(request)
        if request.tool_call.get("name") == "task":
            self._process_result(result, request.tool_call)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        if get_current_scope() is None:
            if self._scope is not None:
                bind_handoff_scope(
                    self._scope,
                    artifact_root=self._artifact_root,
                    user_query=self._user_query,
                )
            else:
                self._scope = self._initialize_scope()
        result = await handler(request)
        if request.tool_call.get("name") == "task":
            self._process_result(result, request.tool_call)
        return result

    def _initialize_scope(self, state: Any | None = None):
        if state is not None:
            self._user_query = self._extract_root_user_query(state)
        try:
            from langgraph.config import get_config

            cfg = get_config()
            configurable = cfg.get("configurable", {})
            run_id = cfg.get("run_id")
            thread_id = configurable.get("thread_id")
            return initialize_handoff_scope(
                run_id=str(run_id) if run_id is not None else None,
                thread_id=str(thread_id) if thread_id is not None else None,
                invocation_id=uuid.uuid4().hex,
                artifact_root=self._artifact_root,
                user_query=self._user_query,
            )
        except Exception:
            return initialize_handoff_scope(
                invocation_id=uuid.uuid4().hex,
                artifact_root=self._artifact_root,
                user_query=self._user_query,
            )

    @staticmethod
    def _extract_root_user_query(state: Any) -> str | None:
        messages = None
        if isinstance(state, dict):
            messages = state.get("messages")
        else:
            messages = getattr(state, "messages", None)
        if not isinstance(messages, list):
            return None

        for message in messages:
            text = StructuredResultExtractorMiddleware._message_to_user_text(message)
            if text:
                return text
        return None

    @staticmethod
    def _message_to_user_text(message: Any) -> str | None:
        if isinstance(message, HumanMessage):
            return StructuredResultExtractorMiddleware._coerce_text(message.content)
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("type") or "").lower()
            if role in {"user", "human"}:
                return StructuredResultExtractorMiddleware._coerce_text(
                    message.get("content")
                )
            return None
        if isinstance(message, tuple) and len(message) == 2:
            role, content = message
            if str(role).lower() in {"user", "human"}:
                return StructuredResultExtractorMiddleware._coerce_text(content)
        return None

    @staticmethod
    def _coerce_text(content: Any) -> str | None:
        if isinstance(content, str):
            text = content.strip()
            return text or None
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        parts.append(stripped)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        stripped = str(text).strip()
                        if stripped:
                            parts.append(stripped)
            if parts:
                return "\n".join(parts)
            return None
        if content is None:
            return None
        text = str(content).strip()
        return text or None

    def _process_result(self, result, tool_call: dict[str, Any]) -> None:
        subagent_type = tool_call.get("args", {}).get("subagent_type", "")
        task_prompt = tool_call.get("args", {}).get("description")
        if not subagent_type:
            return

        if get_current_scope() is None:
            self._initialize_scope()

        text = self._extract_text_from_result(result)
        if text is None:
            record = store_agent_failure(
                producer=subagent_type,
                error_kind="missing_tool_text",
                message=(
                    f"Subagent '{subagent_type}' returned no text content that could "
                    "be parsed for a structured result."
                ),
                source_tool_call_id=tool_call.get("id"),
                task_prompt=task_prompt,
            )
            logger.info(
                "result_extractor: stored %s for '%s' status=%s",
                record.handoff_id,
                subagent_type,
                record.status,
            )
            return

        parsed = _extract_structured_result(text)
        if parsed is None:
            record = store_agent_failure(
                producer=subagent_type,
                error_kind="missing_structured_result",
                message=(
                    f"Subagent '{subagent_type}' returned no valid "
                    "<STRUCTURED_RESULT> block."
                ),
                source_tool_call_id=tool_call.get("id"),
                raw_text=text,
                task_prompt=task_prompt,
            )
            logger.info(
                "result_extractor: stored %s for '%s' status=%s",
                record.handoff_id,
                subagent_type,
                record.status,
            )
            return

        record = store_agent_result(
            producer=subagent_type,
            payload=parsed,
            source_tool_call_id=tool_call.get("id"),
            task_prompt=task_prompt,
        )
        logger.info(
            "result_extractor: stored %s for '%s' status=%s",
            record.handoff_id,
            subagent_type,
            record.status,
        )

    @staticmethod
    def _extract_text_from_result(result) -> str | None:
        if isinstance(result, ToolMessage):
            content = result.content
            return content if isinstance(content, str) else None
        if isinstance(result, Command):
            update = result.update
            if not isinstance(update, dict):
                return None
            messages = update.get("messages", [])
            fallback_text: str | None = None
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    content = msg.content
                    if isinstance(content, str):
                        fallback_text = content
                        if _extract_structured_result(content) is not None:
                            return content
            return fallback_text
        return None
