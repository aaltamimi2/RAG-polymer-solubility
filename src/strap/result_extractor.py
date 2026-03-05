"""Structured result extractor middleware.

Intercepts task() ToolMessage responses from subagents, extracts
<STRUCTURED_RESULT> JSON blocks, and stores them in a per-invocation
thread-safe dict registry keyed by invocation ID.

The original ToolMessage content is never modified — the extracted
data is stored separately and made available via accessor functions.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain.agents.middleware.types import ToolCallRequest

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Storage: thread-safe invocation-ID-keyed registry
# ------------------------------------------------------------------

@dataclass
class _RegistryState:
    """Accumulated structured results for one orchestrator invocation."""
    results: dict[str, dict] = field(default_factory=dict)


# Thread-safe registry keyed by invocation ID.
# ContextVar stores only the immutable ID string — safe across copy_context().
# Actual mutable state lives in the module-level dict, protected by a lock.
_registry_lock = threading.Lock()
_registry: dict[str, _RegistryState] = {}
_invocation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_structured_result_invocation_id"
)


def _get_current_state() -> _RegistryState | None:
    """Get the _RegistryState for the current invocation, or None."""
    try:
        inv_id = _invocation_id.get()
    except LookupError:
        return None
    with _registry_lock:
        return _registry.get(inv_id)


# Regex to extract <STRUCTURED_RESULT>...</STRUCTURED_RESULT> blocks.
# re.DOTALL so . matches newlines; non-greedy to stop at first closing tag.
_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)


def _extract_structured_result(text: str) -> dict | None:
    """Extract and parse a <STRUCTURED_RESULT> JSON block from text."""
    match = _STRUCTURED_RESULT_RE.search(text)
    if not match:
        return None
    json_text = match.group(1).strip()
    try:
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "result_extractor: malformed JSON in <STRUCTURED_RESULT> — skipping. "
            "Error: %s  |  First 200 chars: %.200s",
            e,
            json_text,
        )
        return None


# ------------------------------------------------------------------
# Public accessor functions
# ------------------------------------------------------------------

def get_structured_results() -> dict[str, dict]:
    """Return all accumulated structured results for this invocation.

    Keys are subagent names (e.g. "separation-engineer", "safety-analyst").
    Returns an empty dict if no results yet or outside middleware context.
    """
    state = _get_current_state()
    if state is None:
        return {}
    with _registry_lock:
        return dict(state.results)


def get_structured_result(agent_name: str) -> dict | None:
    """Return the structured result for a specific subagent, or None."""
    state = _get_current_state()
    if state is None:
        return None
    with _registry_lock:
        return state.results.get(agent_name)


# ------------------------------------------------------------------
# Orchestrator tools for querying results
# ------------------------------------------------------------------

def get_subagent_result(agent_name: str) -> str:
    """Retrieve the structured JSON result from a previously-completed subagent.

    Use this after a subagent has returned its results to get the machine-readable
    structured data rather than parsing the prose response.

    Args:
        agent_name: The subagent name that was called via task(subagent_type=...).
            Known agents: separation-engineer, safety-analyst, biosteam-analyst,
            scholar-researcher, patent-researcher, rag-analyst,
            visualization-specialist, statistics-ml.

    Returns:
        JSON string of the structured result, or a descriptive error message.
    """
    result = get_structured_result(agent_name)
    if result is None:
        available = list(get_structured_results().keys())
        if not available:
            return (
                f"No structured results available yet. "
                f"Subagent '{agent_name}' has not returned in this session."
            )
        return (
            f"No structured result found for '{agent_name}'. "
            f"Available agents with results: {available}"
        )
    try:
        return json.dumps(result, indent=2)
    except (TypeError, ValueError) as e:
        return f"Error serializing result for '{agent_name}': {e}"


def get_all_subagent_results() -> str:
    """Retrieve all structured JSON results from all completed subagents.

    Returns:
        JSON string mapping agent names to their structured results.
    """
    all_results = get_structured_results()
    if not all_results:
        return "No structured results available. No subagents have completed yet."
    try:
        return json.dumps(all_results, indent=2)
    except (TypeError, ValueError) as e:
        return f"Error serializing results: {e}"


# ------------------------------------------------------------------
# Middleware class
# ------------------------------------------------------------------

class StructuredResultExtractorMiddleware(AgentMiddleware):
    """Intercepts task() ToolMessage responses and extracts structured result JSON.

    For every task() tool call:
    1. Lets the handler execute the subagent normally.
    2. Extracts the subagent name from the tool call arguments.
    3. Finds and parses the <STRUCTURED_RESULT> block in the returned text.
    4. Stores the result in a per-invocation thread-safe dict registry.
    5. Returns the original result unmodified.
    """

    def before_agent(self, state, runtime) -> None:
        try:
            from langgraph.config import get_config
            cfg = get_config()
            run_id = cfg.get("run_id")
            inv_id = str(run_id) if run_id is not None else uuid.uuid4().hex
        except Exception:
            inv_id = uuid.uuid4().hex
        _invocation_id.set(inv_id)
        with _registry_lock:
            _registry[inv_id] = _RegistryState()
        logger.debug("result_extractor: registry initialized, invocation=%s", inv_id)

    async def abefore_agent(self, state, runtime) -> None:
        try:
            from langgraph.config import get_config
            cfg = get_config()
            run_id = cfg.get("run_id")
            inv_id = str(run_id) if run_id is not None else uuid.uuid4().hex
        except Exception:
            inv_id = uuid.uuid4().hex
        _invocation_id.set(inv_id)
        with _registry_lock:
            _registry[inv_id] = _RegistryState()
        logger.debug("result_extractor: registry initialized (async), invocation=%s", inv_id)

    def after_agent(self, state, runtime) -> None:
        try:
            inv_id = _invocation_id.get()
            with _registry_lock:
                _registry.pop(inv_id, None)
            logger.debug("result_extractor: registry cleaned up, invocation=%s", inv_id)
        except LookupError:
            pass

    async def aafter_agent(self, state, runtime) -> None:
        try:
            inv_id = _invocation_id.get()
            with _registry_lock:
                _registry.pop(inv_id, None)
            logger.debug("result_extractor: registry cleaned up (async), invocation=%s", inv_id)
        except LookupError:
            pass

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = handler(request)
        if request.tool_call.get("name") == "task":
            subagent_type = request.tool_call.get("args", {}).get("subagent_type", "")
            self._process_result(result, subagent_type)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = await handler(request)
        if request.tool_call.get("name") == "task":
            subagent_type = request.tool_call.get("args", {}).get("subagent_type", "")
            self._process_result(result, subagent_type)
        return result

    def _process_result(self, result, subagent_type: str) -> None:
        if not subagent_type:
            return

        text = self._extract_text_from_result(result)
        if text is None:
            logger.debug(
                "result_extractor: could not find text for subagent '%s'",
                subagent_type,
            )
            return

        parsed = _extract_structured_result(text)
        if parsed is None:
            logger.info(
                "result_extractor: no <STRUCTURED_RESULT> block for '%s'",
                subagent_type,
            )
            return

        try:
            inv_id = _invocation_id.get()
        except LookupError:
            # Fallback: create a new invocation ID so the result is not lost
            inv_id = uuid.uuid4().hex
            _invocation_id.set(inv_id)
            with _registry_lock:
                _registry[inv_id] = _RegistryState()
            logger.warning(
                "result_extractor: created fallback invocation ID '%s' for '%s'",
                inv_id,
                subagent_type,
            )

        with _registry_lock:
            reg_state = _registry.get(inv_id)
            if reg_state is None:
                reg_state = _RegistryState()
                _registry[inv_id] = reg_state

            if subagent_type in reg_state.results:
                logger.info(
                    "result_extractor: overwriting result for '%s' (duplicate call)",
                    subagent_type,
                )
            reg_state.results[subagent_type] = parsed

        logger.info(
            "result_extractor: stored result for '%s' — keys: %s",
            subagent_type,
            list(parsed.keys()),
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
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    content = msg.content
                    if isinstance(content, str):
                        return content
        return None
