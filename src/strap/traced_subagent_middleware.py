"""Custom subagent middleware that emits explicit LangSmith child traces per task()."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.subagents import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    TASK_SYSTEM_PROMPT,
    TASK_TOOL_DESCRIPTION,
    CompiledSubAgent,
    SubAgent,
    _EXCLUDED_STATE_KEYS,
    _get_subagents,
)

from .guardrails import seed_guard_state
from .langsmith_tracing import langsmith_trace, record_subagent_trace, tracing_enabled
from .handoff_store import get_handoff

logger = logging.getLogger(__name__)


def _subagent_recursion_limit() -> int:
    """Step budget for one subagent invocation (middleware nodes inflate steps ~6x per turn)."""
    try:
        return max(20, int(os.getenv("DISSOLVE_SUBAGENT_RECURSION_LIMIT", "120")))
    except (TypeError, ValueError):
        return 120


def _subagent_step_budget_failure(subagent_type: str) -> str:
    return (
        f"Subagent '{subagent_type}' was terminated after exhausting its step budget "
        f"({_subagent_recursion_limit()} graph steps) without producing a final answer. "
        "Do not re-dispatch it with the same instructions. If a validated handoff is "
        "attached, its payload is injected into the specialist's domain tool call "
        "automatically — re-dispatch with a short, tool-first instruction that names the "
        "single tool to call, or synthesize the final answer from the results already "
        "gathered."
    )


def _handoff_context_block(record) -> str:
    """Render the runtime-attached handoff contract into model-visible text.

    The payload itself rides in the subagent's runtime state; the consumer's
    guardrails inject it into the domain tool call (e.g. ``stage_candidates_json``).
    Without this block the specialist is told a payload is "attached" that it can
    never see, and models go hunting for it (filesystem, prose reconstruction).
    """
    payload_keys = ", ".join(sorted(record.payload.keys())[:12]) if record.payload else "(empty)"
    return (
        "\n\n--- RUNTIME-ATTACHED HANDOFF ---\n"
        f"handoff_id: {record.handoff_id} | contract: {record.contract} | "
        f"producer: {record.producer}\n"
        f"payload keys: {payload_keys}\n"
        "The validated payload is attached to your runtime state and is not printed "
        "here. When you call your domain tool, the runtime injects the payload into "
        "the tool arguments automatically (for optimization tools: "
        "`stage_candidates_json`). Call your domain tool directly. Never reconstruct "
        "the payload from prose, never ask for it, and never search the filesystem "
        "for it."
    )


def _invoke_subagent_guarded(subagent: Runnable, subagent_state: dict, subagent_type: str) -> dict | str:
    """Invoke a subagent with a fresh guard budget and a bounded step count.

    seed_guard_state() must run here — in the parent context of the subagent's
    graph — because ContextVar writes inside graph nodes do not survive to
    sibling nodes; only mutations of this pre-seeded object do. Returns the
    result state, or a failure string when the subagent exhausts its step
    budget (so one runaway specialist degrades the turn instead of crashing
    the whole run).
    """
    seed_guard_state()
    config = {"recursion_limit": _subagent_recursion_limit()}
    try:
        return cast(dict, subagent.invoke(subagent_state, config))
    except GraphRecursionError:
        logger.warning(
            "task(%s): subagent exhausted recursion limit %d; returning failure to orchestrator",
            subagent_type,
            _subagent_recursion_limit(),
        )
        return _subagent_step_budget_failure(subagent_type)


async def _ainvoke_subagent_guarded(subagent: Runnable, subagent_state: dict, subagent_type: str) -> dict | str:
    seed_guard_state()
    config = {"recursion_limit": _subagent_recursion_limit()}
    try:
        return cast(dict, await subagent.ainvoke(subagent_state, config))
    except GraphRecursionError:
        logger.warning(
            "task(%s): subagent exhausted recursion limit %d; returning failure to orchestrator",
            subagent_type,
            _subagent_recursion_limit(),
        )
        return _subagent_step_budget_failure(subagent_type)


def _create_traced_task_tool(
    *,
    default_model: str | BaseChatModel,
    default_tools: Sequence[BaseTool | Callable | dict[str, Any]],
    default_middleware: list[AgentMiddleware] | None,
    default_interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    subagents: list[SubAgent | CompiledSubAgent],
    general_purpose_agent: bool,
    task_description: str | None = None,
) -> BaseTool:
    subagent_graphs, subagent_descriptions = _get_subagents(
        default_model=default_model,
        default_tools=default_tools,
        default_middleware=default_middleware,
        default_interrupt_on=default_interrupt_on,
        subagents=subagents,
        general_purpose_agent=general_purpose_agent,
    )
    subagent_description_str = "\n".join(subagent_descriptions)

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        if "messages" not in result:
            raise ValueError(
                "CompiledSubAgent must return a state containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )

        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}
        message_text = result["messages"][-1].text.rstrip() if result["messages"][-1].text else ""
        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(message_text, tool_call_id=tool_call_id)],
            }
        )

    def _validate_and_prepare_state(
        subagent_type: str,
        description: str,
        runtime: ToolRuntime,
        handoff_id: str | None = None,
    ) -> tuple[Runnable, dict]:
        subagent = subagent_graphs[subagent_type]
        subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
        handoff_additional_kwargs: dict[str, Any] = {}
        handoff_text = str(handoff_id or "").strip()
        if handoff_text:
            record = get_handoff(handoff_text)
            if record is not None:
                subagent_state["strap_handoff_id"] = record.handoff_id
                subagent_state["strap_handoff_contract"] = record.contract
                subagent_state["strap_handoff_payload"] = dict(record.payload)
                subagent_state["strap_handoff_task_prompt"] = record.task_prompt
                subagent_state["strap_handoff_producer"] = record.producer
                subagent_state["strap_handoff_consumer"] = record.consumer
                handoff_additional_kwargs = {
                    "strap_handoff_id": record.handoff_id,
                    "strap_handoff_contract": record.contract,
                    "strap_handoff_task_prompt": record.task_prompt or "",
                    "strap_handoff_producer": record.producer,
                    "strap_handoff_consumer": record.consumer,
                }
                description = f"{description}{_handoff_context_block(record)}"
        subagent_state["messages"] = [
            HumanMessage(content=description, additional_kwargs=handoff_additional_kwargs)
        ]
        return subagent, subagent_state

    if task_description is None:
        task_description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        task_description = task_description.format(available_agents=subagent_description_str)

    def _trace_subagent_sync(subagent_type: str, description: str, runtime: ToolRuntime, subagent: Runnable, subagent_state: dict) -> dict | str:
        if not tracing_enabled() or langsmith_trace is None:
            return _invoke_subagent_guarded(subagent, subagent_state, subagent_type)

        with langsmith_trace(
            f"Subagent: {subagent_type}",
            run_type="chain",
            project_name=None,
            inputs={
                "subagent_type": subagent_type,
                "description": description,
                "tool_call_id": runtime.tool_call_id,
            },
            metadata={
                "subagent_type": subagent_type,
                "tool_call_id": runtime.tool_call_id,
                "entrypoint": "task_tool",
            },
            tags=["dissolve", "subagent", subagent_type],
        ) as traced_run:
            result = _invoke_subagent_guarded(subagent, subagent_state, subagent_type)
            if isinstance(result, str):
                traced_run.end(outputs={"error": "subagent_step_budget_exhausted"})
            else:
                traced_run.end(
                    outputs={
                        "message_count": len(result.get("messages", []) or []),
                    }
                )
        record_subagent_trace(
            tool_call_id=runtime.tool_call_id,
            subagent_type=subagent_type,
            description=description,
            run_tree=traced_run,
        )
        return result

    async def _trace_subagent_async(subagent_type: str, description: str, runtime: ToolRuntime, subagent: Runnable, subagent_state: dict) -> dict | str:
        if not tracing_enabled() or langsmith_trace is None:
            return await _ainvoke_subagent_guarded(subagent, subagent_state, subagent_type)

        with langsmith_trace(
            f"Subagent: {subagent_type}",
            run_type="chain",
            project_name=None,
            inputs={
                "subagent_type": subagent_type,
                "description": description,
                "tool_call_id": runtime.tool_call_id,
            },
            metadata={
                "subagent_type": subagent_type,
                "tool_call_id": runtime.tool_call_id,
                "entrypoint": "task_tool",
            },
            tags=["dissolve", "subagent", subagent_type],
        ) as traced_run:
            result = await _ainvoke_subagent_guarded(subagent, subagent_state, subagent_type)
            if isinstance(result, str):
                traced_run.end(outputs={"error": "subagent_step_budget_exhausted"})
            else:
                traced_run.end(
                    outputs={
                        "message_count": len(result.get("messages", []) or []),
                    }
                )
        record_subagent_trace(
            tool_call_id=runtime.tool_call_id,
            subagent_type=subagent_type,
            description=description,
            run_tree=traced_run,
        )
        return result

    def task(
        description: Annotated[
            str,
            "A detailed description of the task for the subagent to perform autonomously. Include all necessary context and specify the expected output format.",
        ],
        subagent_type: Annotated[str, "The type of subagent to use. Must be one of the available agent types listed in the tool description."],
        handoff_id: Annotated[str | None, "Optional validated upstream handoff ID to attach structurally to the subagent runtime state."] = None,
        runtime: ToolRuntime = None,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime, handoff_id)
        result = _trace_subagent_sync(subagent_type, description, runtime, subagent, subagent_state)
        if isinstance(result, str):
            return result
        if not runtime.tool_call_id:
            raise ValueError("Tool call ID is required for subagent invocation")
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: Annotated[
            str,
            "A detailed description of the task for the subagent to perform autonomously. Include all necessary context and specify the expected output format.",
        ],
        subagent_type: Annotated[str, "The type of subagent to use. Must be one of the available agent types listed in the tool description."],
        handoff_id: Annotated[str | None, "Optional validated upstream handoff ID to attach structurally to the subagent runtime state."] = None,
        runtime: ToolRuntime = None,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        subagent, subagent_state = _validate_and_prepare_state(subagent_type, description, runtime, handoff_id)
        result = await _trace_subagent_async(subagent_type, description, runtime, subagent, subagent_state)
        if isinstance(result, str):
            return result
        if not runtime.tool_call_id:
            raise ValueError("Tool call ID is required for subagent invocation")
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=task_description,
    )


class TracedSubAgentMiddleware(AgentMiddleware):
    """Subagent middleware that emits explicit child traces per task() call."""

    def __init__(
        self,
        *,
        default_model: str | BaseChatModel,
        default_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
        default_middleware: list[AgentMiddleware] | None = None,
        default_interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
        subagents: list[SubAgent | CompiledSubAgent] | None = None,
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        general_purpose_agent: bool = True,
        task_description: str | None = None,
    ) -> None:
        super().__init__()

        subagent_descriptions = []
        if general_purpose_agent:
            subagent_descriptions.append(f"- general-purpose: {DEFAULT_GENERAL_PURPOSE_DESCRIPTION}")
        subagent_descriptions.extend(f"- {agent_['name']}: {agent_['description']}" for agent_ in subagents or [])

        if system_prompt is not None and subagent_descriptions:
            agents_section = "\n\nAvailable subagent types:\n" + "\n".join(subagent_descriptions)
            self.system_prompt = system_prompt + agents_section
        else:
            self.system_prompt = system_prompt

        task_tool = _create_traced_task_tool(
            default_model=default_model,
            default_tools=default_tools or [],
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
            task_description=task_description,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)

