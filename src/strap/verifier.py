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
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .handoffs import validate_agent_payload
from .guardrail_checks import get_selectivity_overclaim_errors
from .solubility import get_boiling_point

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain.agents.middleware.types import ModelCallResult, ModelRequest

logger = logging.getLogger(__name__)

_verified_flag: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_verified_flag", default=False
)
_verification_revision_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_verification_revision_count", default=0
)
_STRUCTURED_RESULT_RE = re.compile(
    r"<STRUCTURED_RESULT>\s*(.*?)\s*</STRUCTURED_RESULT>",
    re.DOTALL,
)
_TEMPERATURE_LIMIT_PATTERNS = (
    re.compile(
        r"\b(?:up to|max(?:imum)?|at most|no more than|under|below)\s*(\d+(?:\.\d+)?)\s*°?\s*C\b",
        re.IGNORECASE,
    ),
)
_BOILING_POINT_CAVEAT_MARGIN_C = 10.0
_NEAR_BOILING_MARGIN_C = 5.0
_MAX_VERIFIER_REVISIONS = 2
_SELECTIVITY_OVERCLAIM_TERMS = (
    "will selectively dissolve",
    "will dissolve",
    "remains a solid",
    "remains solid",
    "remain solid",
    "undissolved solid",
    "undissolved solids",
    "physically separated from",
    "via filtration",
    "practical route exists",
    "wide, safe operating window",
    "safe and controllable",
)
_SELECTIVITY_QUALIFIER_TERMS = (
    "predicted",
    "prediction",
    "best-ranked candidate",
    "best candidate",
    "best available candidate",
    "selectivity-based",
    "based on selectivity",
    "interpolation",
    "model suggests",
    "model-predicted",
    "should be validated experimentally",
    "experimental validation",
    "should be confirmed experimentally",
    "needs experimental confirmation",
)

_VERIFIER_SYSTEM_PROMPT = """\
You are a scientific accuracy verifier for a polymer dissolution analysis agent.
Given a user query, relevant tool context, and the agent's response, check for:

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
- Do not use external chemistry knowledge that is not explicitly present in the \
tool context or the response itself
- If evidence is missing, flag unsupported certainty; do not infer the opposite \
scientific conclusion from outside knowledge
- Treat missing database coverage or absent tool evidence as uncertainty, not as \
proof that the agent's conclusion is false
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


def _get_tool_context(messages: list, limit: int = 6, max_chars: int = 4000) -> str:
    """Extract recent tool outputs to ground verification in actual evidence."""
    snippets: list[str] = []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        text = _extract_text(getattr(msg, "content", "")).strip()
        if not text:
            continue
        snippets.append(text[:800])
        if len(snippets) >= limit:
            break
    if not snippets:
        return ""
    combined = "\n\n".join(reversed(snippets))
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def _extract_user_temperature_limit_c(messages: list) -> float | None:
    limits: list[float] = []
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        text = _extract_text(msg.content)
        for pattern in _TEMPERATURE_LIMIT_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    limits.append(float(match.group(1)))
                except (TypeError, ValueError):
                    continue
    return max(limits) if limits else None


def _get_task_tool_registry(messages: list) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tool_call in tool_calls:
            if tool_call.get("name") == "task":
                registry[tool_call.get("id", "")] = tool_call
    return registry


def _extract_structured_payload_from_tool_message(
    message: ToolMessage,
    producer: str | None = None,
) -> dict | None:
    text = _extract_text(getattr(message, "content", ""))
    match = _STRUCTURED_RESULT_RE.search(text)
    if not match:
        return None

    json_text = match.group(1).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if fenced:
        json_text = fenced.group(1).strip()

    try:
        payload = json.loads(json_text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(payload, dict):
        return None

    payload_producer = producer or payload.get("agent")
    if isinstance(payload_producer, str) and payload_producer:
        if validate_agent_payload(payload_producer, payload):
            return None

    return payload


def _get_latest_validated_subagent_payload(messages: list, subagent: str) -> dict | None:
    task_calls = _get_task_tool_registry(messages)
    latest_payload: dict | None = None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_call_id = getattr(msg, "tool_call_id", "")
        tool_call = task_calls.get(tool_call_id, {})
        if tool_call.get("args", {}).get("subagent_type") != subagent:
            continue
        payload = _extract_structured_payload_from_tool_message(msg, producer=subagent)
        if payload is not None:
            latest_payload = payload
    return latest_payload


def _get_latest_subagent_payload(messages: list, subagent: str) -> dict | None:
    task_calls = _get_task_tool_registry(messages)
    latest_payload: dict | None = None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_call_id = getattr(msg, "tool_call_id", "")
        tool_call = task_calls.get(tool_call_id, {})
        if tool_call.get("args", {}).get("subagent_type") != subagent:
            continue
        payload = _extract_structured_payload_from_tool_message(msg)
        if payload is not None:
            latest_payload = payload
    return latest_payload


def _is_single_specialist_separation_context(messages: list) -> bool:
    task_calls = _get_task_tool_registry(messages)
    subagents = {
        tool_call.get("args", {}).get("subagent_type")
        for tool_call in task_calls.values()
        if tool_call.get("args", {}).get("subagent_type")
    }
    return subagents == {"separation-engineer"}


def _build_separation_verifier_fallback(messages: list) -> str | None:
    payload = (
        _get_latest_validated_subagent_payload(messages, "separation-engineer")
        or _get_latest_subagent_payload(messages, "separation-engineer")
    )
    if not isinstance(payload, dict):
        return None

    user_query = _get_user_query(messages).lower()
    polymers = [str(polymer) for polymer in payload.get("polymers", []) if str(polymer).strip()]
    supported = [str(polymer) for polymer in payload.get("supported_polymers", []) if str(polymer).strip()]
    unsupported = [str(polymer) for polymer in payload.get("unsupported_polymers", []) if str(polymer).strip()]
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    max_temp_c = _extract_user_temperature_limit_c(messages)

    if not steps:
        return None

    step = next((candidate for candidate in steps if isinstance(candidate, dict)), None)
    if step is None:
        return None

    polymer = str(step.get("polymer", "")).strip() or "the target polymer"
    solvent = str(step.get("solvent", "")).strip() or "the recommended solvent"
    temp_value = step.get("temperature_c", step.get("temp"))
    try:
        operating_temp_c = float(temp_value)
    except (TypeError, ValueError):
        operating_temp_c = None

    lines: list[str] = []
    if len(polymers) == 2 and any(term in user_query for term in ("selective", "dissolution", "dissolve", "separate")):
        lines.append(
            f"Recommended best practical partial route below the user's temperature limit: use {solvent} to target {polymer}"
            + (f" at {operating_temp_c:.1f}°C." if operating_temp_c is not None else ".")
        )
        lines.append(
            "This is a selectivity-based candidate, not a proven fully selective route. "
            "No fully selective atmospheric-pressure route is established from the current validated tool evidence, "
            "so a fully selective route is not feasible from the current data."
        )
    else:
        lines.append(
            "Recommended separation sequence from the validated specialist result:"
        )
        lines.append(
            f"- Step 1: use {solvent} to recover {polymer}"
            + (f" at {operating_temp_c:.1f}°C." if operating_temp_c is not None else ".")
        )

    if supported or unsupported:
        if supported:
            lines.append(f"Supported subset: {', '.join(supported)}.")
        if unsupported:
            lines.append(
                f"Unsupported polymers: {', '.join(unsupported)}. Their phase behavior is unknown from the current data, "
                "so they should not be assigned to a dissolved stream or solid residue without experimental confirmation."
            )

    if operating_temp_c is not None:
        boiling_point_c = get_boiling_point(solvent)
        if boiling_point_c is not None:
            if operating_temp_c < boiling_point_c:
                lines.append(
                    f"Operate at {operating_temp_c:.1f}°C, below the {boiling_point_c:.1f}°C boiling point of {solvent} at 1 atm."
                )
                if boiling_point_c - operating_temp_c <= _NEAR_BOILING_MARGIN_C:
                    lines.append(
                        "This is a narrow atmospheric-pressure operating margin and requires careful temperature control."
                    )
                elif max_temp_c is not None and max_temp_c > operating_temp_c:
                    lines.append(
                        f"The user allows up to {max_temp_c:.1f}°C, but the recommended operating temperature is lower."
                    )
            else:
                lines.append(
                    f"This step would be infeasible at atmospheric pressure because {solvent} boils at {boiling_point_c:.1f}°C."
                )

    lines.append(
        "Experimental confirmation is required before claiming complete selectivity, residue composition, or filtration behavior."
    )
    return "\n\n".join(lines)


def _get_latest_subagent_output_text(messages: list, subagent: str) -> str:
    task_calls = _get_task_tool_registry(messages)
    latest_text = ""
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_call_id = getattr(msg, "tool_call_id", "")
        tool_call = task_calls.get(tool_call_id, {})
        if tool_call.get("args", {}).get("subagent_type") != subagent:
            continue
        latest_text = _extract_text(getattr(msg, "content", ""))
    return latest_text


def _mentions_temperature(text: str, value_c: float) -> bool:
    formatted = f"{value_c:.1f}".rstrip("0").rstrip(".")
    integer = str(int(round(value_c)))
    variants = "|".join(re.escape(item) for item in {formatted, integer})
    return bool(re.search(rf"(?:{variants})\s*°?\s*c", text.lower()))


def _get_deterministic_separation_issues(messages: list, response_text: str) -> list[str]:
    payload = _get_latest_validated_subagent_payload(messages, "separation-engineer")
    if not isinstance(payload, dict):
        return []

    issues: list[str] = []
    response_lower = response_text.lower()
    subagent_output_text = _get_latest_subagent_output_text(messages, "separation-engineer")
    if subagent_output_text:
        subagent_overclaim_issues = get_selectivity_overclaim_errors(
            messages,
            subagent_output_text,
            "separation-engineer",
        )
        response_has_overclaim = any(
            term in response_lower for term in _SELECTIVITY_OVERCLAIM_TERMS
        )
        response_has_qualifier = any(
            term in response_lower for term in _SELECTIVITY_QUALIFIER_TERMS
        )
        subagent_has_qualifier = any(
            term in subagent_output_text.lower() for term in _SELECTIVITY_QUALIFIER_TERMS
        )
        if (
            response_has_overclaim
            and not response_has_qualifier
            and (subagent_overclaim_issues or subagent_has_qualifier)
        ):
            issues.append(
                "The final answer preserves an overclaim from the separation result. "
                "Keep the recommendation framed as a predicted/selectivity-based candidate "
                "and preserve the need for experimental confirmation or fuller feasibility validation."
            )

    unsupported = [
        str(polymer).strip().upper()
        for polymer in payload.get("unsupported_polymers", [])
        if str(polymer).strip()
    ]
    phase_terms = (
        "remain solid",
        "remains solid",
        "remain in the residue",
        "remain in the solid residue",
        "remains in the solid residue",
        "remain as solids",
        "undissolved solid",
        "undissolved solids",
        "stays in the residue",
        "remains in the residue",
        "insoluble",
        "dissolve",
        "soluble",
        "precipitate",
    )
    uncertainty_terms = (
        "unknown",
        "cannot determine",
        "cannot conclude",
        "may ",
        "could ",
        "might ",
    )
    for polymer in unsupported:
        for sentence in re.split(r"[.\n]+", response_lower):
            if polymer.lower() not in sentence:
                continue
            if any(term in sentence for term in phase_terms) and not any(
                term in sentence for term in uncertainty_terms
            ):
                issues.append(
                    f"The answer treats unsupported polymer {polymer} as having known phase behavior. "
                    "State that its behavior is unknown without additional data."
                )
                break

    user_max_temp_c = _extract_user_temperature_limit_c(messages)
    steps = payload.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            solvent = str(step.get("solvent", "")).strip()
            if not solvent:
                continue
            temp_value = step.get("temperature_c", step.get("temp"))
            try:
                operating_temp_c = float(temp_value)
            except (TypeError, ValueError):
                continue
            boiling_point_c = get_boiling_point(solvent)
            if boiling_point_c is None:
                continue

            if (
                user_max_temp_c is not None
                and user_max_temp_c >= boiling_point_c - _BOILING_POINT_CAVEAT_MARGIN_C
            ):
                if not (
                    _mentions_temperature(response_text, operating_temp_c)
                    and any(
                        phrase in response_lower
                        for phrase in ("boiling point", "at 1 atm", "at atmospheric pressure")
                    )
                ):
                    issues.append(
                        f"The answer must explicitly say that {solvent} runs at {operating_temp_c:.1f}C, "
                        f"below its {boiling_point_c:.1f}C boiling point at 1 atm, rather than implying "
                        "operation at the user's maximum temperature."
                    )

            bp_margin_c = boiling_point_c - operating_temp_c
            if 0 < bp_margin_c <= _NEAR_BOILING_MARGIN_C:
                if not any(
                    phrase in response_lower
                    for phrase in (
                        "narrow atmospheric-pressure operating margin",
                        "narrow operating margin",
                        "narrow temperature margin",
                        "close to its boiling point",
                        "near its boiling point",
                        "careful temperature control",
                        "tight temperature control",
                    )
                ):
                    issues.append(
                        f"The answer must explicitly say that {solvent} at {operating_temp_c:.1f}C is only "
                        f"{bp_margin_c:.1f}C below its boiling point and therefore has a narrow atmospheric-pressure "
                        "operating margin requiring careful temperature control."
                    )

    return issues


def _merge_verdicts(primary: dict, extra_issues: list[str]) -> dict:
    if not extra_issues:
        return primary
    issues = list(primary.get("issues", [])) if isinstance(primary.get("issues"), list) else []
    for issue in extra_issues:
        if issue not in issues:
            issues.append(issue)
    return {"pass": False, "confidence": "HIGH", "issues": issues}


def _should_use_separation_fallback(messages: list, verdict: dict) -> bool:
    if verdict.get("confidence") != "HIGH":
        return False
    if not _is_single_specialist_separation_context(messages):
        return False
    return (
        _get_latest_validated_subagent_payload(messages, "separation-engineer") is not None
        or _get_latest_subagent_payload(messages, "separation-engineer") is not None
    )


def _parse_verdict(text: str) -> dict:
    """Parse JSON verdict from verifier response.  Fail-open on errors."""
    text = _extract_text(text).strip()
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
    # Try the first JSON object embedded in plain text
    m = re.search(r"(\{.*\})", text, re.DOTALL)
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

    @property
    def _revision_count(self) -> int:
        try:
            return _verification_revision_count.get()
        except LookupError:
            return 0

    # -- lifecycle: reset per invocation ------------------------------------

    def before_agent(self, state, runtime):
        _verified_flag.set(False)
        _verification_revision_count.set(0)

    async def abefore_agent(self, state, runtime):
        _verified_flag.set(False)
        _verification_revision_count.set(0)

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

        content = _extract_text(getattr(ai_msg, "content", ""))
        if not content or len(content) < 50:
            _verified_flag.set(True)
            return response

        user_query = _get_user_query(request.messages)
        tool_context = _get_tool_context(request.messages)
        deterministic_issues = _get_deterministic_separation_issues(
            request.messages,
            content,
        )

        verdict = {"pass": True, "confidence": "LOW", "issues": []}
        use_model_verifier = not _is_single_specialist_separation_context(request.messages)
        if not deterministic_issues and use_model_verifier:
            try:
                verdict = self._verify(user_query, tool_context, content)
            except Exception:
                logger.warning("verifier: model call failed — passing through", exc_info=True)
                _verified_flag.set(True)
                return response
        verdict = _merge_verdicts(verdict, deterministic_issues)

        if verdict.get("pass", True):
            logger.info("verifier: PASS")
            _verified_flag.set(True)
            return response

        if verdict.get("confidence") != "HIGH":
            logger.info(
                "verifier: issues found but confidence=%s — passing through",
                verdict.get("confidence"),
            )
            _verified_flag.set(True)
            return response

        if _should_use_separation_fallback(request.messages, verdict):
            fallback = _build_separation_verifier_fallback(request.messages)
            if fallback:
                logger.warning(
                    "verifier: replacing high-risk single-specialist separation answer with deterministic fallback revision_count=%d",
                    self._revision_count,
                )
                _verified_flag.set(True)
                return ModelResponse(result=[AIMessage(
                    content=fallback,
                    additional_kwargs={
                        "strap_origin": "verifier_separation_fallback",
                        "strap_route_type": "single_specialist_separation",
                    },
                )])

        if self._revision_count >= _MAX_VERIFIER_REVISIONS:
            logger.warning(
                "verifier: revision limit reached (%d) — returning latest response",
                _MAX_VERIFIER_REVISIONS,
            )
            _verified_flag.set(True)
            return response

        # HIGH-confidence issues — inject feedback and re-invoke
        issues_text = "\n".join(f"- {i}" for i in verdict.get("issues", []))
        logger.warning("verifier: HIGH confidence issues found:\n%s", issues_text)

        if request.system_message is None:
            logger.warning("verifier: cannot inject feedback — no system message")
            _verified_flag.set(True)
            return response

        feedback = (
            f"\n\n[VERIFICATION FEEDBACK — revise your response]\n"
            f"A quality check found these issues with your answer:\n"
            f"{issues_text}\n"
            f"Please revise your response to address these issues. "
            f"Use the tool results already in context — do not call additional tools."
        )
        if any(
            phrase in issues_text.lower()
            for phrase in (
                "selectivity-based candidate",
                "preserves an overclaim",
                "experimental confirmation",
                "not present in or derivable from the provided tool context",
            )
        ):
            feedback += (
                "\nFor selectivity-ranking-only separation results: do not invent numeric "
                "selectivity values, do not say the comparison polymer will remain solid or "
                "can be cleanly filtered with certainty, and frame the recommendation as a "
                "predicted candidate that requires experimental confirmation."
            )
        new_system = append_to_system_message(request.system_message, feedback)
        _verification_revision_count.set(self._revision_count + 1)
        revised_response = handler(request.override(system_message=new_system))
        return self._maybe_verify(request.override(system_message=new_system), revised_response, handler)

    def _verify(self, user_query: str, tool_context: str, response_text: str) -> dict:
        """Single Gemini Flash call to check response quality."""
        msgs = [
            SystemMessage(content=_VERIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"USER QUERY:\n{user_query}\n\n"
                    f"TOOL CONTEXT:\n{tool_context or '[no recent tool context available]'}\n\n"
                    f"AGENT RESPONSE:\n{response_text}"
                )
            ),
        ]
        result = self._verifier_model.invoke(msgs)
        return _parse_verdict(result.content)

    async def _averify(self, user_query: str, tool_context: str, response_text: str) -> dict:
        """Async version of _verify(): uses ainvoke() to avoid blocking the event loop."""
        msgs = [
            SystemMessage(content=_VERIFIER_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"USER QUERY:\n{user_query}\n\n"
                    f"TOOL CONTEXT:\n{tool_context or '[no recent tool context available]'}\n\n"
                    f"AGENT RESPONSE:\n{response_text}"
                )
            ),
        ]
        result = await self._verifier_model.ainvoke(msgs)
        return _parse_verdict(result.content)

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

        content = _extract_text(getattr(ai_msg, "content", ""))
        if not content or len(content) < 50:
            _verified_flag.set(True)
            return response

        user_query = _get_user_query(request.messages)
        tool_context = _get_tool_context(request.messages)
        deterministic_issues = _get_deterministic_separation_issues(
            request.messages,
            content,
        )

        verdict = {"pass": True, "confidence": "LOW", "issues": []}
        use_model_verifier = not _is_single_specialist_separation_context(request.messages)
        if not deterministic_issues and use_model_verifier:
            try:
                verdict = await self._averify(user_query, tool_context, content)
            except Exception:
                logger.warning("verifier: model call failed — passing through", exc_info=True)
                _verified_flag.set(True)
                return response
        verdict = _merge_verdicts(verdict, deterministic_issues)

        if verdict.get("pass", True):
            logger.info("verifier: PASS")
            _verified_flag.set(True)
            return response

        if verdict.get("confidence") != "HIGH":
            logger.info(
                "verifier: issues found but confidence=%s — passing through",
                verdict.get("confidence"),
            )
            _verified_flag.set(True)
            return response

        if _should_use_separation_fallback(request.messages, verdict):
            fallback = _build_separation_verifier_fallback(request.messages)
            if fallback:
                logger.warning(
                    "verifier: replacing high-risk single-specialist separation answer with deterministic fallback (async) revision_count=%d",
                    self._revision_count,
                )
                _verified_flag.set(True)
                return ModelResponse(result=[AIMessage(
                    content=fallback,
                    additional_kwargs={
                        "strap_origin": "verifier_separation_fallback",
                        "strap_route_type": "single_specialist_separation",
                    },
                )])

        if self._revision_count >= _MAX_VERIFIER_REVISIONS:
            logger.warning(
                "verifier: revision limit reached (%d) — returning latest response",
                _MAX_VERIFIER_REVISIONS,
            )
            _verified_flag.set(True)
            return response

        # HIGH-confidence issues — inject feedback and re-invoke
        issues_text = "\n".join(f"- {i}" for i in verdict.get("issues", []))
        logger.warning("verifier: HIGH confidence issues found:\n%s", issues_text)

        if request.system_message is None:
            logger.warning("verifier: cannot inject feedback — no system message")
            _verified_flag.set(True)
            return response

        feedback = (
            f"\n\n[VERIFICATION FEEDBACK — revise your response]\n"
            f"A quality check found these issues with your answer:\n"
            f"{issues_text}\n"
            f"Please revise your response to address these issues. "
            f"Use the tool results already in context — do not call additional tools."
        )
        if any(
            phrase in issues_text.lower()
            for phrase in (
                "selectivity-based candidate",
                "preserves an overclaim",
                "experimental confirmation",
                "not present in or derivable from the provided tool context",
            )
        ):
            feedback += (
                "\nFor selectivity-ranking-only separation results: do not invent numeric "
                "selectivity values, do not say the comparison polymer will remain solid or "
                "can be cleanly filtered with certainty, and frame the recommendation as a "
                "predicted candidate that requires experimental confirmation."
            )
        new_system = append_to_system_message(request.system_message, feedback)
        _verification_revision_count.set(self._revision_count + 1)
        revised_response = await handler(request.override(system_message=new_system))
        return await self._amaybe_verify(
            request.override(system_message=new_system),
            revised_response,
            handler,
        )
