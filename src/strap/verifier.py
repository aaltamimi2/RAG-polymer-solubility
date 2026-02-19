"""Output verifier middleware: single reflection pass on the orchestrator's
final synthesis.  Makes one Gemini Flash call to check for unsupported claims,
internal contradictions, missing caveats, and logical gaps.  Only revises on
HIGH-confidence issues.  Max one verification per invocation."""

from __future__ import annotations

import contextvars
import json
import logging
import re
from typing import TYPE_CHECKING

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain.agents.middleware.types import ModelCallResult, ModelRequest

logger = logging.getLogger(__name__)

_verified_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_verified_flag", default=False
)

_VERIFIER_SYSTEM_PROMPT = """\
You are a scientific accuracy verifier for a polymer dissolution analysis agent.
Given a user query and the agent's response, check for:

1. UNSUPPORTED CLAIMS: Rankings, values, or recommendations not derivable from \
the data discussed in the response
2. INTERNAL CONTRADICTIONS: Conflicting statements about the same solvent/polymer
3. MISSING CAVEATS: Safety warnings omitted when toxic solvents are recommended, \
or temperature/pressure limitations not mentioned
4. LOGICAL GAPS: Conclusions that don't follow from the evidence presented

Respond with JSON only:
{"pass": true/false, "confidence": "HIGH"|"MEDIUM"|"LOW", "issues": ["..."]}

Rules:
- Set "pass": true if the response is scientifically sound and complete
- Set "confidence": "HIGH" only when you are certain the issue is real (not a \
style preference)
- Do not flag formatting, verbosity, or stylistic choices — only factual/logical \
errors
- An empty issues list with "pass": true is the most common correct response"""


def _extract_text(content) -> str:
    """Extract plain text from an AI message content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _get_user_query(messages: list) -> str:
    """Extract text from the last HumanMessage."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _parse_verdict(text: str) -> dict:
    """Parse JSON verdict from verifier response.  Fail-open on errors."""
    text = text.strip()
    # Try raw JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, TypeError):
            pass
    # Fail-open
    logger.warning(
        "verifier: could not parse verdict from response, fail-open activated — text: %.200s",
        text,
    )
    return {"pass": True}


class OutputVerifierMiddleware(AgentMiddleware):
    """Single-pass output verifier.  Checks the orchestrator's final synthesis
    for factual/logical issues and triggers one revision on HIGH-confidence
    findings."""

    def __init__(self, verifier_model: BaseChatModel) -> None:
        self._verifier_model = verifier_model

    # -- per-invocation state (ContextVar-isolated) --------------------------

    @property
    def _verified(self) -> bool:
        try:
            return _verified_flag.get()
        except LookupError:
            return False

    # -- lifecycle: reset per invocation ------------------------------------

    def before_agent(self, state, runtime):
        _verified_flag.set(False)

    async def abefore_agent(self, state, runtime):
        _verified_flag.set(False)

    # -- model-call wrapper -------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        response = handler(request)
        return self._maybe_verify(request, response, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelCallResult],
    ) -> ModelCallResult:
        response = await handler(request)
        return await self._amaybe_verify(request, response, handler)

    # -- core logic ---------------------------------------------------------

    def _maybe_verify(self, request, response, handler):
        """Check response; if final synthesis with HIGH-confidence issues,
        inject feedback and re-invoke model once."""
        if self._verified:
            return response

        # Need a ModelResponse with result to inspect
        result = getattr(response, "result", None)
        if not result:
            return response

        ai_msg = result[0]
        tool_calls = getattr(ai_msg, "tool_calls", None)
        if tool_calls:
            return response  # Agent still working — pass through

        # Final synthesis detected
        _verified_flag.set(True)
        content = _extract_text(getattr(ai_msg, "content", ""))
        if not content or len(content) < 50:
            return response

        user_query = _get_user_query(request.messages)

        try:
            verdict = self._verify(user_query, content)
        except Exception:
            logger.warning("verifier: model call failed — passing through", exc_info=True)
            return response

        if verdict.get("pass", True):
            logger.info("verifier: PASS")
            return response

        if verdict.get("confidence") != "HIGH":
            logger.info(
                "verifier: issues found but confidence=%s — passing through",
                verdict.get("confidence"),
            )
            return response

        # HIGH-confidence issues — inject feedback and re-invoke
        issues_text = "\n".join(f"- {i}" for i in verdict.get("issues", []))
        logger.warning("verifier: HIGH confidence issues found:\n%s", issues_text)

        if request.system_message is None:
            logger.warning("verifier: cannot inject feedback — no system message")
            return response

        feedback = (
            f"\n\n[VERIFICATION FEEDBACK — revise your response]\n"
            f"A quality check found these issues with your answer:\n"
            f"{issues_text}\n"
            f"Please revise your response to address these issues. "
            f"Use the tool results already in context — do not call additional tools."
        )
        new_system = append_to_system_message(request.system_message, feedback)
        return handler(request.override(system_message=new_system))

    def _verify(self, user_query: str, response_text: str) -> dict:
        """Single Gemini Flash call to check response quality."""
        msgs = [
            SystemMessage(content=_VERIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"USER QUERY:\n{user_query}\n\nAGENT RESPONSE:\n{response_text}"
            ),
        ]
        result = self._verifier_model.invoke(msgs)
        return _parse_verdict(
            result.content if isinstance(result.content, str) else str(result.content)
        )

    async def _averify(self, user_query: str, response_text: str) -> dict:
        """Async version of _verify(): uses ainvoke() to avoid blocking the event loop."""
        msgs = [
            SystemMessage(content=_VERIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=f"USER QUERY:\n{user_query}\n\nAGENT RESPONSE:\n{response_text}"
            ),
        ]
        result = await self._verifier_model.ainvoke(msgs)
        return _parse_verdict(
            result.content if isinstance(result.content, str) else str(result.content)
        )

    async def _amaybe_verify(self, request, response, handler):
        """Async version of _maybe_verify(): awaits _averify() and handler() to avoid
        blocking the event loop."""
        if self._verified:
            return response

        result = getattr(response, "result", None)
        if not result:
            return response

        ai_msg = result[0]
        tool_calls = getattr(ai_msg, "tool_calls", None)
        if tool_calls:
            return response  # Agent still working — pass through

        # Final synthesis detected
        _verified_flag.set(True)
        content = _extract_text(getattr(ai_msg, "content", ""))
        if not content or len(content) < 50:
            return response

        user_query = _get_user_query(request.messages)

        try:
            verdict = await self._averify(user_query, content)
        except Exception:
            logger.warning("verifier: model call failed — passing through", exc_info=True)
            return response

        if verdict.get("pass", True):
            logger.info("verifier: PASS")
            return response

        if verdict.get("confidence") != "HIGH":
            logger.info(
                "verifier: issues found but confidence=%s — passing through",
                verdict.get("confidence"),
            )
            return response

        # HIGH-confidence issues — inject feedback and re-invoke
        issues_text = "\n".join(f"- {i}" for i in verdict.get("issues", []))
        logger.warning("verifier: HIGH confidence issues found:\n%s", issues_text)

        if request.system_message is None:
            logger.warning("verifier: cannot inject feedback — no system message")
            return response

        feedback = (
            f"\n\n[VERIFICATION FEEDBACK — revise your response]\n"
            f"A quality check found these issues with your answer:\n"
            f"{issues_text}\n"
            f"Please revise your response to address these issues. "
            f"Use the tool results already in context — do not call additional tools."
        )
        new_system = append_to_system_message(request.system_message, feedback)
        return await handler(request.override(system_message=new_system))
