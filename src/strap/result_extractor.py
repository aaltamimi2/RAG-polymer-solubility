"""Structured result extractor middleware.

Intercepts task() ToolMessage responses from subagents, extracts
<STRUCTURED_RESULT> JSON blocks, and stores them in a per-invocation
ContextVar registry keyed by subagent name.

The original ToolMessage content is never modified — the extracted
data is stored separately and made available via accessor functions.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
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
# Storage: per-invocation ContextVar registry
# ------------------------------------------------------------------

@dataclass
class _RegistryState:
    """Accumulated structured results for one orchestrator invocation."""
    results: dict[str, dict] = field(default_factory=dict)


_registry_state: contextvars.ContextVar[_RegistryState] = contextvars.ContextVar(
    "_structured_result_registry"
)


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
    try:
        return dict(_registry_state.get().results)
    except LookupError:
        return {}


def get_structured_result(agent_name: str) -> dict | None:
    """Return the structured result for a specific subagent, or None."""
    try:
        return _registry_state.get().results.get(agent_name)
    except LookupError:
        return None


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
    4. Stores the result in a per-invocation ContextVar registry.
    5. Returns the original result unmodified.
    """

    def before_agent(self, state, runtime) -> None:
        _registry_state.set(_RegistryState())
        logger.debug("result_extractor: registry reset for new invocation")

    async def abefore_agent(self, state, runtime) -> None:
        _registry_state.set(_RegistryState())
        logger.debug("result_extractor: registry reset for new invocation (async)")

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
            state = _registry_state.get()
        except LookupError:
            state = _RegistryState()
            _registry_state.set(state)

        if subagent_type in state.results:
            logger.info(
                "result_extractor: overwriting result for '%s' (duplicate call)",
                subagent_type,
            )

        state.results[subagent_type] = parsed
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
