"""Tests for SubagentGuardMiddleware."""
import json
import pytest
from unittest.mock import MagicMock
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage


def _make_model_response(content="ok", tool_calls=None, input_tokens=1000, output_tokens=500):
    """Build a mock ModelResponse with usage_metadata."""
    ai_msg = AIMessage(content=content, tool_calls=tool_calls or [])
    ai_msg.usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    resp = MagicMock()
    resp.result = [ai_msg]
    return resp


class TestIterationLimit:
    def test_stops_at_max_iterations(self):
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(max_iterations=2)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        handler = MagicMock(return_value=_make_model_response())
        mw.wrap_model_call(req, handler)
        mw.wrap_model_call(req, handler)
        mw.wrap_model_call(req, handler)  # 3rd call: budget trip grants the final synthesis pass
        result = mw.wrap_model_call(req, MagicMock())
        assert isinstance(result, AIMessage)
        assert "Max iterations" in result.content

    def test_resets_on_before_agent(self):
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(max_iterations=2)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        handler = MagicMock(return_value=_make_model_response())
        mw.wrap_model_call(req, handler)
        mw.wrap_model_call(req, handler)
        mw.before_agent(None, None)  # reset
        result = mw.wrap_model_call(req, handler)
        # After reset iteration count is 1, which is <= max_iterations=2, so no limit
        if isinstance(result, AIMessage):
            assert "Max iterations" not in result.content

    def test_exact_limit_boundary(self):
        """Call exactly max_iterations times; the (max+1)th call triggers the limit."""
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(max_iterations=3, token_budget=10_000_000)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        handler = MagicMock(return_value=_make_model_response())
        for _ in range(3):
            mw.wrap_model_call(req, handler)
        mw.wrap_model_call(req, handler)  # budget trip grants the final synthesis pass
        result = mw.wrap_model_call(req, MagicMock())
        assert isinstance(result, AIMessage)
        assert "Max iterations" in result.content


class TestTokenBudget:
    def test_enforced_after_expensive_call(self):
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(token_budget=100_000)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        handler = MagicMock(
            return_value=_make_model_response(input_tokens=80_000, output_tokens=30_000)
        )
        first = mw.wrap_model_call(req, handler)
        # over budget but the response has no tool calls: it already is the final answer
        assert not isinstance(first, AIMessage)
        mw.wrap_model_call(req, handler)  # pre-call trip grants the final synthesis pass
        result = mw.wrap_model_call(req, MagicMock())
        assert isinstance(result, AIMessage)
        assert "Token budget" in result.content

    def test_counts_output_tokens(self):
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(token_budget=5000)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        handler = MagicMock(
            return_value=_make_model_response(input_tokens=2000, output_tokens=4000)
        )
        first = mw.wrap_model_call(req, handler)
        # over budget but the response has no tool calls: it already is the final answer
        assert not isinstance(first, AIMessage)
        mw.wrap_model_call(req, handler)  # pre-call trip grants the final synthesis pass
        result = mw.wrap_model_call(req, MagicMock())
        assert isinstance(result, AIMessage)
        assert "Token budget" in result.content

    def test_accumulates_across_calls(self):
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(token_budget=3000)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        handler = MagicMock(
            return_value=_make_model_response(input_tokens=1000, output_tokens=500)
        )
        mw.wrap_model_call(req, handler)
        mw.wrap_model_call(req, handler)   # crosses the budget; passes through (no tool calls)
        mw.wrap_model_call(req, handler)   # pre-call trip grants the final synthesis pass
        result = mw.wrap_model_call(req, MagicMock())
        assert isinstance(result, AIMessage)
        assert "Token budget" in result.content

    def test_budget_not_exceeded_when_under(self):
        """Under-budget calls should pass through normally."""
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(token_budget=1_000_000, max_tool_calls=999)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        response = _make_model_response(input_tokens=100, output_tokens=50)
        handler = MagicMock(return_value=response)
        result = mw.wrap_model_call(req, handler)
        # Should not be a limit AIMessage — returns the response (or the response itself)
        if isinstance(result, AIMessage):
            assert "Token budget" not in result.content

    def test_counts_anthropic_response_metadata_usage(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(token_budget=5000, max_tool_calls=999)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        ai_msg = AIMessage(
            content="ok",
            response_metadata={
                "usage": {
                    "input_tokens": 4500,
                    "output_tokens": 700,
                }
            },
        )
        resp = MagicMock()
        resp.result = [ai_msg]

        first = mw.wrap_model_call(req, MagicMock(return_value=resp))
        # over budget, no tool calls: passes through as the final answer
        assert first is resp
        mw.wrap_model_call(req, MagicMock(return_value=resp))  # synthesis grant
        result = mw.wrap_model_call(req, MagicMock(return_value=resp))
        assert isinstance(result, AIMessage)
        assert "Token budget" in result.content

    def test_counts_anthropic_raw_cache_usage_tokens(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(token_budget=5000, max_tool_calls=999)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        ai_msg = AIMessage(
            content="ok",
            response_metadata={
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 4200,
                    "cache_creation_input_tokens": 300,
                }
            },
        )
        resp = MagicMock()
        resp.result = [ai_msg]

        first = mw.wrap_model_call(req, MagicMock(return_value=resp))
        # over budget, no tool calls: passes through as the final answer
        assert first is resp
        mw.wrap_model_call(req, MagicMock(return_value=resp))  # synthesis grant
        result = mw.wrap_model_call(req, MagicMock(return_value=resp))
        assert isinstance(result, AIMessage)
        assert "Token budget" in result.content


class TestSeparationSupportDirective:
    def test_injects_support_coverage_warning_for_unsupported_polymers(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="separation-engineer")
        mw.before_agent(None, None)

        request = MagicMock()
        request.messages = [
            HumanMessage(
                content="Find the optimal separation sequence for PS, PMMA, and PET at up to 120C."
            )
        ]
        request.system_message = SystemMessage(content="You are a separation engineering specialist.")
        request.override = MagicMock(side_effect=lambda **kwargs: request)

        updated = mw._inject_separation_support_directive(request)

        assert updated is request
        assert request.override.called
        content = request.override.call_args.kwargs["system_message"].content
        if isinstance(content, list):
            system_text = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            system_text = str(content)
        assert "[SUPPORT COVERAGE]" in system_text
        assert "Requested polymers inferred from the task: PS, PMMA, PET" in system_text
        assert "Supported by local interpolation data: PS, PET" in system_text
        assert "Unsupported polymers: PMMA" in system_text

    def test_injects_temperature_limit_warning_for_bounded_queries(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="separation-engineer")
        mw.before_agent(None, None)

        request = MagicMock()
        request.messages = [
            HumanMessage(
                content="Find the optimal separation sequence for PS and PET up to 120C."
            )
        ]
        request.system_message = SystemMessage(content="You are a separation engineering specialist.")
        request.override = MagicMock(side_effect=lambda **kwargs: request)

        updated = mw._inject_separation_temperature_bound_directive(request)

        assert updated is request
        assert request.override.called
        content = request.override.call_args.kwargs["system_message"].content
        if isinstance(content, list):
            system_text = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        else:
            system_text = str(content)
        assert "[TEMPERATURE LIMIT]" in system_text
        assert "upper temperature bound of 120.0C" in system_text
        assert "actual recommended temperature for each step" in system_text
        assert "step must run below that solvent's boiling point at 1 atm" in system_text


class TestToolCallLimit:
    def test_strips_tool_calls_at_limit(self):
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(max_tool_calls=1)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        tc = {"name": "some_tool", "id": "1", "args": {}}
        handler = MagicMock(return_value=_make_model_response(tool_calls=[tc]))
        result = mw.wrap_model_call(req, handler)
        assert isinstance(result, AIMessage)
        assert "Tool call budget" in result.content or "tool" in result.content.lower()

    def test_free_tools_not_counted(self):
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(max_tool_calls=1, free_tools={"think"}, token_budget=10_000_000)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        tc = {"name": "think", "id": "1", "args": {}}
        handler = MagicMock(return_value=_make_model_response(tool_calls=[tc]))
        result = mw.wrap_model_call(req, handler)
        # "think" is free — should NOT trigger limit
        if isinstance(result, AIMessage):
            assert "Tool call budget" not in result.content

    def test_tool_call_limit_message_content(self):
        """The limit message should preserve any existing LLM text."""
        from strap.guardrails import SubagentGuardMiddleware
        mw = SubagentGuardMiddleware(max_tool_calls=1)
        mw.before_agent(None, None)
        req = MagicMock()
        req.messages = []
        req.system_message = None
        tc = {"name": "search", "id": "1", "args": {}}
        handler = MagicMock(
            return_value=_make_model_response(content="partial answer", tool_calls=[tc])
        )
        result = mw.wrap_model_call(req, handler)
        assert isinstance(result, AIMessage)
        # Existing content should be preserved before the limit message
        assert "partial answer" in result.content
        assert "LIMIT" in result.content

    def test_after_model_repairs_budget_limited_safety_answer_once(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="safety-analyst",
            synthesis_tools={"compare_pubchem_safety"},
        )
        mw.before_agent(None, None)

        state = {
            "messages": [
                ToolMessage(content="safety batch complete", tool_call_id="tool1", name="compare_pubchem_safety"),
                AIMessage(content="[LIMIT] Tool call budget exhausted. Synthesize your findings now."),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert update["jump_to"] == "model"
        assert "Do not call any more tools" in update["messages"][0].content
        assert "<STRUCTURED_RESULT>" in update["messages"][0].content

        second = mw.after_model(state, None)
        assert second is None

    def test_after_model_does_not_repair_budget_limit_for_other_agents(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="separation-engineer")
        mw.before_agent(None, None)

        state = {
            "messages": [
                AIMessage(content="[LIMIT] Tool call budget exhausted. Synthesize your findings now."),
            ]
        }

        assert mw.after_model(state, None) is None

    def test_wrap_tool_call_blocks_late_separation_write_todos(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="separation-engineer")
        mw.before_agent(None, None)

        request = MagicMock()
        request.tool_call = {
            "id": "todo_sep",
            "name": "write_todos",
            "args": {"todos": [{"content": "refine route", "status": "in_progress"}]},
        }
        request.state = {
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "id": "plan1",
                    "name": "plan_sequential_separation",
                    "args": {"polymers": "PS,PET"},
                }]),
                ToolMessage(content="sequence built", tool_call_id="plan1"),
            ]
        }

        blocked = mw.wrap_tool_call(request, MagicMock())

        assert isinstance(blocked, ToolMessage)
        assert blocked.status == "error"
        assert "todo rewriting is blocked" in blocked.content

    def test_wrap_tool_call_allows_initial_separation_write_todos(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="separation-engineer")
        mw.before_agent(None, None)

        request = MagicMock()
        request.tool_call = {
            "id": "todo_sep_init",
            "name": "write_todos",
            "args": {"todos": [{"content": "plan route", "status": "in_progress"}]},
        }
        request.state = {"messages": []}

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="todo_sep_init"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        handler.assert_called_once()


class TestStructuredResultRepair:
    def test_after_model_requires_real_separation_analysis_before_route_recommendation(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="separation-engineer")
        mw.before_agent(None, None)

        state = {
            "messages": [
                AIMessage(content="", tool_calls=[{
                    "id": "list_poly",
                    "name": "list_available_polymers",
                    "args": {},
                }]),
                ToolMessage(content='{"display":"ok","data":{"success":true,"tool_name":"list_available_polymers"}}', tool_call_id="list_poly"),
                AIMessage(
                    content=(
                        "PS can be separated from PET.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                        '"best_sequence":["PS","PET"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":100.0}],'
                        '"solvent_mapping":{"PS":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert update["jump_to"] == "model"
        assert "without running a substantive separation analysis tool" in update["messages"][0].content

    def test_after_model_reprompts_when_synthesis_finalizes_without_contract(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"plan_sequential_separation"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                ToolMessage(content="sequence found", tool_call_id="tool1", name="plan_sequential_separation"),
                AIMessage(content="Here is the best sequence in prose only."),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert update["jump_to"] == "model"
        repair_message = update["messages"][0]
        assert isinstance(repair_message, HumanMessage)
        assert "<STRUCTURED_RESULT>" in repair_message.content

    def test_after_model_accepts_valid_structured_result(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"plan_sequential_separation"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                ToolMessage(content="sequence found", tool_call_id="tool1", name="plan_sequential_separation"),
                AIMessage(
                    content=(
                        "Final answer.\n<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS"],'
                        '"best_sequence":["PS"],"steps":[],"solvent_mapping":{"PS":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS"],"solvent_mapping":{"PS":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        assert mw.after_model(state, None) is None

    def test_after_model_repairs_only_once(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="safety-analyst",
            synthesis_tools={"compare_pubchem_safety"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {"messages": [AIMessage(content="Still missing the JSON block.")]}

        first = mw.after_model(state, None)
        second = mw.after_model(state, None)

        assert first is not None
        assert second is None

    def test_wrap_model_call_detects_synthesis_tool_even_after_followup_think_and_repairs_empty_separation_final(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"find_optimal_separation_sequence"},
        )
        mw.before_agent(None, None)

        request = MagicMock()
        request.messages = [
            AIMessage(content="", tool_calls=[{
                "id": "seq1",
                "name": "find_optimal_separation_sequence",
                "args": {"polymers": "LDPE,HDPE,PP"},
            }]),
            ToolMessage(content="sequence built", tool_call_id="seq1"),
            AIMessage(content="", tool_calls=[{
                "id": "think1",
                "name": "think",
                "args": {"thought": "need final synthesis"},
            }]),
            ToolMessage(content="reflection", tool_call_id="think1"),
        ]
        request.system_message = SystemMessage(content="base system")
        request.override = MagicMock(side_effect=lambda **kwargs: request)

        response = _make_model_response(content="")
        handler = MagicMock(return_value=response)

        result = mw.wrap_model_call(request, handler)

        assert result is response
        assert mw._state.synthesis_tool_seen is True

        state = {"messages": [*request.messages, AIMessage(content="")]}
        update = mw.after_model(state, None)

        assert update is not None
        assert update["jump_to"] == "model"
        assert "Rewrite the full final answer now" in update["messages"][0].content
        assert "<STRUCTURED_RESULT>" in update["messages"][0].content

    def test_after_model_repairs_separation_step_above_boiling_point(self):
        from strap.guardrails import SubagentGuardMiddleware
        from strap.solubility import get_boiling_point

        bp = get_boiling_point("Toluene")
        assert bp is not None

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"plan_sequential_separation"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                ToolMessage(content="sequence found", tool_call_id="tool1", name="plan_sequential_separation"),
                AIMessage(
                    content=(
                        "Final answer.\n<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PE","PP"],'
                        '"best_sequence":["PE","PP"],'
                        f'"steps":[{{"step":1,"polymer":"PE","solvent":"Toluene","temperature_c":{bp + 5:.1f}}}],'
                        '"solvent_mapping":{"PE":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PE","PP"],"solvent_mapping":{"PE":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "boiling point" in update["messages"][0].content
        assert "no feasible atmospheric-pressure sequence exists" in update["messages"][0].content

    def test_after_model_repairs_optimal_but_infeasible_contradiction(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"plan_sequential_separation"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                ToolMessage(content="sequence found", tool_call_id="tool1", name="plan_sequential_separation"),
                AIMessage(
                    content=(
                        "The optimal separation sequence is PS -> PET -> PMMA. "
                        "However, this route is infeasible at atmospheric pressure and would require pressurization.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET","PMMA"],'
                        '"best_sequence":["PS","PET","PMMA"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":60.0}],'
                        '"solvent_mapping":{"PS":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET","PMMA"],"solvent_mapping":{"PS":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "Do not present an infeasible route as the best or optimal executable sequence" in update["messages"][0].content

    def test_after_model_repairs_missing_bounded_temperature_caveat(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"plan_sequential_separation"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                HumanMessage(
                    content="Find the optimal separation sequence for PS and PET up to 120C."
                ),
                ToolMessage(content="sequence found", tool_call_id="tool1", name="plan_sequential_separation"),
                AIMessage(
                    content=(
                        "The recommended route uses Toluene at 75C for PS extraction, then isolates PET.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                        '"best_sequence":["PS","PET"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                        '"solvent_mapping":{"PS":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "stays below the solvent boiling point at 1 atm" in update["messages"][0].content
        assert "do not imply operation at the user's maximum temperature" in update["messages"][0].content

    def test_after_model_accepts_explicit_bounded_temperature_caveat(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"plan_sequential_separation"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                HumanMessage(
                    content="Find the optimal separation sequence for PS and PET up to 120C."
                ),
                ToolMessage(content="sequence found", tool_call_id="tool1", name="plan_sequential_separation"),
                AIMessage(
                    content=(
                        "Although the user allows up to 120C, the PS extraction step should run at 75C "
                        "with Toluene so it stays below Toluene's boiling point at 1 atm. PET remains for the next step.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                        '"best_sequence":["PS","PET"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                        '"solvent_mapping":{"PS":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        assert mw.after_model(state, None) is None

    def test_after_model_repairs_near_boiling_operation_without_margin_caveat(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"find_optimal_separation_sequence"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                HumanMessage(
                    content="Find the optimal separation sequence for PS and PET up to 120C."
                ),
                ToolMessage(content="sequence found", tool_call_id="tool1", name="find_optimal_separation_sequence"),
                AIMessage(
                    content=(
                        "Although the user allows up to 120C, the PS extraction step should run at 85C "
                        "with Tetrahydropyran so it stays below Tetrahydropyran's boiling point at 1 atm.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                        '"best_sequence":["PS","PET"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Tetrahydropyran","temperature_c":85.0}],'
                        '"solvent_mapping":{"PS":"Tetrahydropyran"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Tetrahydropyran"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "narrow atmospheric-pressure operating margin" in update["messages"][0].content
        assert "careful temperature control" in update["messages"][0].content

    def test_after_model_accepts_near_boiling_operation_with_margin_caveat(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"find_optimal_separation_sequence"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                HumanMessage(
                    content="Find the optimal separation sequence for PS and PET up to 120C."
                ),
                ToolMessage(content="sequence found", tool_call_id="tool1", name="find_optimal_separation_sequence"),
                AIMessage(
                    content=(
                        "Although the user allows up to 120C, the PS extraction step should run at 85C "
                        "with Tetrahydropyran so it stays below Tetrahydropyran's boiling point at 1 atm. "
                        "This is a narrow atmospheric-pressure operating margin and requires careful temperature control.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                        '"best_sequence":["PS","PET"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Tetrahydropyran","temperature_c":85.0}],'
                        '"solvent_mapping":{"PS":"Tetrahydropyran"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Tetrahydropyran"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        assert mw.after_model(state, None) is None

    def test_after_model_repairs_unsupported_polymer_purity_claim(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"find_optimal_separation_sequence"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        supported_envelope = json.dumps({
            "display": "supported polymers",
            "data": {
                "tool_name": "get_supported_polymers_and_solvents",
                "success": True,
                "polymers": ["PS", "PET"],
            },
        })

        state = {
            "messages": [
                ToolMessage(
                    content=supported_envelope,
                    tool_call_id="supported1",
                    name="get_supported_polymers_and_solvents",
                ),
                ToolMessage(
                    content="sequence found",
                    tool_call_id="tool1",
                    name="find_optimal_separation_sequence",
                ),
                AIMessage(
                    content=(
                        "PMMA is not in the current database, so the analysis focuses on the supported subset. "
                        "Step 1 dissolves PS, and Step 2 leaves a purified PET residue.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PMMA","PET"],'
                        '"best_sequence":["PS","PET","PMMA"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":60.0}],'
                        '"solvent_mapping":{"PS":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET","PMMA"],"solvent_mapping":{"PS":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "supported subset" in update["messages"][0].content
        assert "purified" in update["messages"][0].content
        assert "supported_polymers" in update["messages"][0].content

    def test_after_model_repairs_unsupported_polymer_phase_claim(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"find_optimal_separation_sequence"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        supported_envelope = json.dumps({
            "display": "supported polymers",
            "data": {
                "tool_name": "get_supported_polymers_and_solvents",
                "success": True,
                "polymers": ["PS", "PET"],
            },
        })

        state = {
            "messages": [
                ToolMessage(
                    content=supported_envelope,
                    tool_call_id="supported1",
                    name="get_supported_polymers_and_solvents",
                ),
                ToolMessage(
                    content="sequence found",
                    tool_call_id="tool1",
                    name="find_optimal_separation_sequence",
                ),
                AIMessage(
                    content=(
                        "PMMA is not supported by the database. At 65C, PS dissolves in THF while PMMA remains solid "
                        "and PET remains solid.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PMMA","PET"],'
                        '"best_sequence":["PS","PET","PMMA"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"THF","temperature_c":65.0}],'
                        '"solvent_mapping":{"PS":"THF"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PET","PMMA"],"solvent_mapping":{"PS":"THF"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "Do not assert whether an unsupported polymer dissolves" in update["messages"][0].content

    def test_after_model_repairs_unsupported_polymer_phase_claim_without_support_tool(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"plan_sequential_separation"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                ToolMessage(
                    content="sequence found",
                    tool_call_id="tool1",
                    name="plan_sequential_separation",
                ),
                AIMessage(
                    content=(
                        "The best route is to dissolve PS first while PMMA remains solid and PET remains solid.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PMMA","PET"],'
                        '"best_sequence":["PS","PMMA","PET"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"THF","temperature_c":65.0}],'
                        '"solvent_mapping":{"PS":"THF"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PMMA","PET"],"solvent_mapping":{"PS":"THF"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "supported subset" in update["messages"][0].content

    def test_after_model_repairs_selectivity_only_overclaim(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(
            agent_name="separation-engineer",
            synthesis_tools={"analyze_selective_solubility_enhanced"},
        )
        mw.before_agent(None, None)
        mw._state.synthesis_tool_seen = True

        state = {
            "messages": [
                ToolMessage(
                    content='{"display":"ok","data":{"tool_name":"analyze_selective_solubility_enhanced","success":true}}',
                    tool_call_id="tool1",
                    name="analyze_selective_solubility_enhanced",
                ),
                AIMessage(
                    content=(
                        "A practical route exists. Toluene will selectively dissolve PS while PVC remains a solid, "
                        "and it offers a wide, safe operating window.\n"
                        "<STRUCTURED_RESULT>"
                        '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PVC"],'
                        '"best_sequence":["PS","PVC"],'
                        '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":57.5}],'
                        '"solvent_mapping":{"PS":"Toluene"},'
                        '"top_k_sequences":[{"rank":1,"sequence":["PS","PVC"],"solvent_mapping":{"PS":"Toluene"}}]}'
                        "</STRUCTURED_RESULT>"
                    )
                ),
            ]
        }

        update = mw.after_model(state, None)

        assert update is not None
        assert "predicted/selectivity-based candidate" in update["messages"][0].content
        assert "experimental confirmation" in update["messages"][0].content


class TestBioSteamDuplicateBatchGuard:
    @staticmethod
    def _prior_multi_polymer_messages(*, solvents, energy_case="C1", allocation_method="value"):
        tool_call_id = "bio_prev"
        polymers_json = json.dumps([
            {"polymer": f"P{i}", "solvent": solvent}
            for i, solvent in enumerate(solvents, start=1)
        ])
        envelope = json.dumps({
            "display": "ok",
            "data": {
                "tool_name": "run_biosteam_multi_polymer",
                "success": True,
                "energy_case": energy_case,
                "allocation_method": allocation_method,
                "per_polymer": [
                    {"polymer": f"P{i}", "solvent": solvent, "success": True}
                    for i, solvent in enumerate(solvents, start=1)
                ],
            },
        })
        return [
            AIMessage(content="", tool_calls=[{
                "id": tool_call_id,
                "name": "run_biosteam_multi_polymer",
                "args": {
                    "polymers_json": polymers_json,
                    "energy_case": energy_case,
                    "allocation_method": allocation_method,
                },
            }]),
            ToolMessage(content=envelope, tool_call_id=tool_call_id, name="run_biosteam_multi_polymer"),
        ]

    def test_blocks_overlapping_duplicate_multi_polymer_batch(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="biosteam-analyst")
        request = MagicMock()
        request.state = {
            "messages": self._prior_multi_polymer_messages(solvents=["Tetrahydrofuran", "Ethanol"])
        }
        request.tool_call = {
            "id": "bio_new",
            "name": "run_biosteam_multi_polymer",
            "args": {
                "polymers_json": json.dumps([
                    {"polymer": "PE", "solvent": "THF"},
                    {"polymer": "PP", "solvent": "Cyclohexane"},
                ]),
                "energy_case": "C1",
                "allocation_method": "value",
            },
        }

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "Overlapping solvents: Tetrahydrofuran" in result.content
        assert "Cyclohexane" in result.content
        handler.assert_not_called()

    def test_allows_non_overlapping_multi_polymer_batch(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="biosteam-analyst")
        request = MagicMock()
        request.state = {
            "messages": self._prior_multi_polymer_messages(solvents=["Toluene", "Ethanol"])
        }
        request.tool_call = {
            "id": "bio_new",
            "name": "run_biosteam_multi_polymer",
            "args": {
                "polymers_json": json.dumps([
                    {"polymer": "PE", "solvent": "Cyclohexane"},
                    {"polymer": "PP", "solvent": "Heptane"},
                ]),
                "energy_case": "C1",
                "allocation_method": "value",
            },
        }

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="bio_new"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        handler.assert_called_once()

    def test_allows_same_solvents_when_energy_case_differs(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="biosteam-analyst")
        request = MagicMock()
        request.state = {
            "messages": self._prior_multi_polymer_messages(
                solvents=["Tetrahydrofuran"],
                energy_case="C1",
            )
        }
        request.tool_call = {
            "id": "bio_new",
            "name": "run_biosteam_multi_polymer",
            "args": {
                "polymers_json": json.dumps([
                    {"polymer": "PE", "solvent": "THF"},
                ]),
                "energy_case": "C2",
                "allocation_method": "value",
            },
        }

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="bio_new"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        handler.assert_called_once()


class TestVisualizationToolDirectiveGuard:
    def test_injects_required_visualization_tool_into_system_prompt(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="visualization-specialist")
        request = MagicMock()
        request.messages = [
            HumanMessage(
                content=(
                    "The user specifically requested a selectivity heatmap. "
                    "Required tool: create_selectivity_heatmap"
                )
            )
        ]
        request.system_message = SystemMessage(content="base system")
        request.override = MagicMock(side_effect=lambda **kwargs: MagicMock(**{**request.__dict__, **kwargs}))

        captured = {}

        def handler(req):
            captured["system_message"] = req.system_message
            return _make_model_response()

        mw.before_agent(None, None)
        mw.wrap_model_call(request, handler)

        rendered = str(captured["system_message"].content)
        assert "create_selectivity_heatmap" in rendered
        assert "[REQUIRED TOOL]" in rendered

    def test_restricts_visualization_tool_list_to_required_tool(self):
        from strap.guardrails import SubagentGuardMiddleware

        class DummyTool:
            def __init__(self, name):
                self.name = name

        mw = SubagentGuardMiddleware(agent_name="visualization-specialist")
        request = MagicMock()
        request.messages = [
            HumanMessage(
                content=(
                    "The user specifically requested a selectivity heatmap. "
                    "Required tool: create_selectivity_heatmap"
                )
            )
        ]
        request.system_message = SystemMessage(content="base system")
        request.tools = [
            DummyTool("create_selectivity_heatmap"),
            DummyTool("create_separation_tree_plot"),
            DummyTool("think"),
        ]
        request.override = MagicMock(
            side_effect=lambda **kwargs: MagicMock(**{**request.__dict__, **kwargs})
        )

        captured = {}

        def handler(req):
            captured["tool_names"] = [tool.name for tool in req.tools]
            return _make_model_response()

        mw.before_agent(None, None)
        mw.wrap_model_call(request, handler)

        assert captured["tool_names"] == ["create_selectivity_heatmap", "think"]

    def test_blocks_visualization_tool_that_conflicts_with_explicit_task_directive(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="visualization-specialist")
        request = MagicMock()
        request.state = {
            "messages": [
                HumanMessage(
                    content=(
                        "The user specifically requested a selectivity heatmap. "
                        "Required tool: create_selectivity_heatmap"
                    )
                )
            ]
        }
        request.tool_call = {
            "id": "viz_wrong",
            "name": "create_separation_tree_plot",
            "args": {"polymers": "PS,PMMA,PET"},
        }

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "`create_selectivity_heatmap`" in result.content
        handler.assert_not_called()

    def test_allows_visualization_tool_that_matches_explicit_task_directive(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="visualization-specialist")
        request = MagicMock()
        request.state = {
            "messages": [
                HumanMessage(
                    content=(
                        "The user specifically requested a selectivity heatmap. "
                        "Required tool: create_selectivity_heatmap"
                    )
                )
            ]
        }
        request.tool_call = {
            "id": "viz_right",
            "name": "create_selectivity_heatmap",
            "args": {"polymers": "PS,PMMA,PET"},
        }

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="viz_right"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        handler.assert_called_once()

    def test_repairs_visualization_final_answer_that_skips_required_tool(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="visualization-specialist")
        mw.before_agent(None, None)
        messages = [
            HumanMessage(
                content=(
                    "Plot the Pareto slices. Required tool: plot_optimization_pareto_slices. "
                    "Call `plot_optimization_pareto_slices(source_handoff_id=\"h_plot\")`."
                )
            ),
            AIMessage(
                content=(
                    "Plots created.\n<STRUCTURED_RESULT>{\"agent\":\"visualization-specialist\","
                    "\"schema_version\":\"1.0\",\"plot_type\":\"optimization_pareto_slices\","
                    "\"plot_paths\":[\"plots/fake.png\"],\"format\":\"png\"}</STRUCTURED_RESULT>"
                )
            ),
        ]

        result = mw.after_model({"messages": messages}, None)

        assert result is not None
        assert result["jump_to"] == "model"
        assert "plot_optimization_pareto_slices" in result["messages"][0].content
        assert "Do not invent plot paths" in result["messages"][0].content

    def test_does_not_repair_visualization_after_required_tool_completed(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="visualization-specialist")
        mw.before_agent(None, None)
        messages = [
            HumanMessage(content="Plot slices. Required tool: plot_optimization_pareto_slices"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "viz_tool",
                        "name": "plot_optimization_pareto_slices",
                        "args": {"source_handoff_id": "h_plot"},
                    }
                ],
            ),
            ToolMessage(content="ok", tool_call_id="viz_tool", name="plot_optimization_pareto_slices"),
            AIMessage(
                content=(
                    "Plots created.\n<STRUCTURED_RESULT>{\"agent\":\"visualization-specialist\","
                    "\"schema_version\":\"1.0\",\"plot_type\":\"optimization_pareto_slices\","
                    "\"plot_paths\":[\"/tmp/plot.png\"],\"format\":\"png\"}</STRUCTURED_RESULT>"
                )
            ),
        ]

        assert mw.after_model({"messages": messages}, None) is None


class TestOptimizationPreflight:
    def test_wrap_tool_call_repairs_optimization_args_from_attached_handoff(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="optimization-engineer")
        mw.before_agent(None, None)

        handoff_payload = {
            "schema_version": "1.1",
            "workflow_scope": "multi_stage",
            "stages": [{"stage_id": "candidate_pool_pe", "candidate_pairs": []}],
            "feed_composition": {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9},
            "feed_capacity_tpy": 8000.0,
        }
        request = ToolCallRequest(
            tool_call={
                "id": "opt_fix",
                "name": "run_waste_management_pareto",
                "args": {
                    "x_metric": "total_cost",
                    "y_metric": "circularity",
                    "stage_candidates_json": '{"broken": true} trailing prose',
                },
            },
            tool=None,
            state={
                "strap_handoff_contract": "optimization.stage_candidates.v1",
                "strap_handoff_payload": handoff_payload,
                "messages": [HumanMessage(content="Optimize this routed feed.")],
            },
            runtime=MagicMock(),
        )

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="opt_fix"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        handler.assert_called_once()

        repaired_request = handler.call_args.args[0]
        repaired_args = repaired_request.tool_call["args"]
        assert repaired_args["stage_candidates_json"] == handoff_payload
        assert repaired_args["feed_composition_json"] == {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9}
        assert repaired_args["feed"] == 8000.0
        assert repaired_request.state["strap_optimization_preflight"][-1]["repairs"]

    def test_wrap_tool_call_recovers_handoff_from_message_additional_kwargs(self):
        from strap.guardrails import SubagentGuardMiddleware
        from strap.handoff_store import cleanup_handoff_scope, initialize_handoff_scope, store_agent_result, store_derived_handoff

        mw = SubagentGuardMiddleware(agent_name="optimization-engineer")
        mw.before_agent(None, None)

        handoff_payload = {
            "schema_version": "1.1",
            "workflow_scope": "multi_stage",
            "route_id": "h_test_route",
            "constraint_mode": "hard",
            "fallback_policy": "fail_closed",
            "operating_constraints": {"temperature_max_c": 80.0, "pressure": "unspecified"},
            "stages": [{"stage_id": "candidate_pool_pe", "candidate_pairs": []}],
            "candidate_pairs": [],
            "polymer_solvent_filters": {"PE": ["Cyclohexane"]},
            "candidate_solvents": ["Cyclohexane"],
            "feed_composition": {"PE": 0.6, "EVOH": 0.4},
            "feed_capacity_tpy": 8000.0,
        }
        initialize_handoff_scope(run_id="guardrail-msg-handoff", invocation_id="guardrail-msg-handoff")
        parent = store_agent_result(
            producer="separation-engineer",
            payload={
                "agent": "separation-engineer",
                "schema_version": "1.0",
                "polymers": ["PE", "EVOH"],
                "best_sequence": ["PE", "EVOH"],
                "steps": [],
                "solvent_mapping": {"PE": "Cyclohexane"},
                "top_k_sequences": [],
            },
            task_prompt="Upstream separation.",
        )
        record = store_derived_handoff(
            producer="separation-engineer",
            consumer="optimization-engineer",
            contract="optimization.stage_candidates.v1",
            payload=handoff_payload,
            parent_handoff_id=parent.handoff_id,
            task_prompt="Use the attached handoff.",
        )

        request = ToolCallRequest(
            tool_call={
                "id": "opt_msg_fix",
                "name": "run_waste_management_optimization",
                "args": {"objective": "max_profit"},
            },
            tool=None,
            state={
                "messages": [
                    HumanMessage(
                        content="Run the routed optimization.",
                        additional_kwargs={
                            "strap_handoff_id": record.handoff_id,
                            "strap_handoff_contract": "optimization.stage_candidates.v1",
                        },
                    )
                ],
            },
            runtime=MagicMock(),
        )

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="opt_msg_fix"))
        try:
            result = mw.wrap_tool_call(request, handler)
        finally:
            cleanup_handoff_scope()

        assert result.content == "ok"
        handler.assert_called_once()
        repaired_request = handler.call_args.args[0]
        repaired_args = repaired_request.tool_call["args"]
        assert repaired_args["stage_candidates_json"] == handoff_payload
        assert repaired_args["feed_composition_json"] == {"PE": 0.6, "EVOH": 0.4}
        assert repaired_args["feed"] == 8000.0

    def test_wrap_tool_call_infers_feed_inputs_from_query_text(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="optimization-engineer")
        mw.before_agent(None, None)

        request = ToolCallRequest(
            tool_call={
                "id": "opt_infer",
                "name": "run_waste_management_optimization",
                "args": {"objective": "max_profit"},
            },
            tool=None,
            state={
                "messages": [
                    HumanMessage(
                        content=(
                            "Optimize waste management for an 8000 t/y multilayer feed composed of "
                            "40% PE, 40% PET, and 20% EVOH."
                        )
                    )
                ]
            },
            runtime=MagicMock(),
        )

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="opt_infer"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        handler.assert_called_once()
        repaired_request = handler.call_args.args[0]
        repaired_args = repaired_request.tool_call["args"]
        assert repaired_args["feed"] == 8000.0
        assert repaired_args["feed_composition_json"] == {"PE": 0.4, "PET": 0.4, "EVOH": 0.2}

    def test_wrap_tool_call_blocks_optimization_without_feed_inputs(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="optimization-engineer")
        mw.before_agent(None, None)

        request = ToolCallRequest(
            tool_call={
                "id": "opt_missing",
                "name": "run_waste_management_optimization",
                "args": {"objective": "max_profit"},
            },
            tool=None,
            state={"messages": [HumanMessage(content="Optimize waste management for this feed.")]},
            runtime=MagicMock(),
        )

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "missing `feed`" in result.content
        handler.assert_not_called()

    def test_wrap_tool_call_blocks_single_pareto_for_multi_slice_query(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="optimization-engineer")
        mw.before_agent(None, None)

        request = ToolCallRequest(
            tool_call={
                "id": "opt_single_wrong",
                "name": "run_waste_management_pareto",
                "args": {
                    "feed": 8000,
                    "feed_composition_json": {"LDPE": 0.2, "EVOH": 0.6, "PET": 0.2},
                    "x_metric": "total_cost",
                    "y_metric": "circularity",
                },
            },
            tool=None,
            state={
                "messages": [
                    HumanMessage(
                        content=(
                            "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year, "
                            "run Pareto slices for fixed feed compositions: 20/60/20, "
                            "34/33/33, and 5/5/90."
                        )
                    )
                ]
            },
            runtime=MagicMock(),
        )

        handler = MagicMock()
        result = mw.wrap_tool_call(request, handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "run_waste_management_pareto_slices" in result.content
        handler.assert_not_called()

    def test_wrap_tool_call_injects_composition_slices_for_multi_slice_tool(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="optimization-engineer")
        mw.before_agent(None, None)

        request = ToolCallRequest(
            tool_call={
                "id": "opt_slices",
                "name": "run_waste_management_pareto_slices",
                "args": {"x_metric": "total_cost", "y_metric": "circularity"},
            },
            tool=None,
            state={
                "messages": [
                    HumanMessage(
                        content=(
                            "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year, "
                            "run Pareto slices for fixed feed compositions: 20/60/20 and 5/5/90."
                        )
                    )
                ]
            },
            runtime=MagicMock(),
        )

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="opt_slices"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        handler.assert_called_once()
        repaired_args = handler.call_args.args[0].tool_call["args"]
        assert repaired_args["feed"] == 8000.0
        assert repaired_args["composition_slices_json"] == [
            {"LDPE": 0.2, "EVOH": 0.6, "PET": 0.2},
            {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9},
        ]

    def test_wrap_tool_call_replaces_model_quoted_composition_slice_keys(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(agent_name="optimization-engineer")
        mw.before_agent(None, None)

        request = ToolCallRequest(
            tool_call={
                "id": "opt_slices_quoted",
                "name": "run_waste_management_pareto_slices",
                "args": {
                    "feed": 8000,
                    "composition_slices_json": [
                        {"feed_composition": {'"PE"': 0.2, '"EVOH"': 0.6, '"PET"': 0.2}},
                        {"feed_composition": {'"PE"': 0.05, '"EVOH"': 0.05, '"PET"': 0.9}},
                    ],
                },
            },
            tool=None,
            state={
                "messages": [
                    HumanMessage(
                        content=(
                            "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year, "
                            "run Pareto slices for fixed feed compositions: 20/60/20 and 5/5/90."
                        )
                    )
                ]
            },
            runtime=MagicMock(),
        )

        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="opt_slices_quoted"))
        result = mw.wrap_tool_call(request, handler)

        assert result.content == "ok"
        repaired_args = handler.call_args.args[0].tool_call["args"]
        assert repaired_args["composition_slices_json"] == [
            {"LDPE": 0.2, "EVOH": 0.6, "PET": 0.2},
            {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9},
        ]


def test_separation_support_scope_rejects_non_polymer_unsupported_tokens():
    from strap.guardrail_checks import get_separation_support_scope_errors

    message = AIMessage(
        content=(
            "<STRUCTURED_RESULT>"
            '{"agent":"separation-engineer","schema_version":"1.0",'
            '"polymers":["LDPE","EVOH","PET"],'
            '"supported_polymers":["LDPE","EVOH","PET"],'
            '"unsupported_polymers":["EACHCANDIDATE"],'
            '"best_sequence":["LDPE","EVOH","PET"],'
            '"steps":[],"solvent_mapping":{},"top_k_sequences":[{"rank":1}]}'
            "</STRUCTURED_RESULT>"
        )
    )

    errors = get_separation_support_scope_errors([], message, "separation-engineer")

    assert any("unsupported_polymers" in error and "EACHCANDIDATE" in error for error in errors)


def test_wrap_tool_call_repairs_separation_top_k_solvents_from_user_query():
    from strap.guardrails import SubagentGuardMiddleware

    mw = SubagentGuardMiddleware(agent_name="separation-engineer")
    mw.before_agent(None, None)
    request = ToolCallRequest(
        tool_call={
            "id": "sep_topk",
            "name": "plan_sequential_separation",
            "args": {"polymers": "LDPE,EVOH,PET", "top_k_solvents": 5},
        },
        tool=None,
        state={
            "messages": [
                HumanMessage(
                    content=(
                        "Have the separation engineer propose the top 8 solvent candidates per polymer "
                        "for LDPE/EVOH/PET."
                    )
                )
            ]
        },
        runtime=MagicMock(),
    )

    handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="sep_topk"))
    result = mw.wrap_tool_call(request, handler)

    assert result.content == "ok"
    repaired_request = handler.call_args.args[0]
    assert repaired_request.tool_call["args"]["top_k_solvents"] == 8


class TestNodeContextIsolation:
    """langgraph executes each graph node in a copied context, so ContextVar
    writes inside a node (before_agent, wrap_model_call) are discarded when the
    node finishes. Budgets therefore only accumulate when the task tool seeds
    the guard state in the subagent's parent context (seed_guard_state) and the
    nodes mutate that shared object in place. Live-run regression: the
    2026-07-07 multistage stress test saw a specialist make 30 model calls and
    26 billable tool calls without tripping max_iterations=25 / max_tool_calls=10.
    """

    @staticmethod
    def _run_isolated(fn, *args):
        import contextvars

        return contextvars.copy_context().run(fn, *args)

    def _request(self):
        req = MagicMock()
        req.messages = []
        req.system_message = None
        return req

    def test_unseeded_budgets_do_not_accumulate_across_nodes(self):
        """Documents the failure mode: per-node context copies zero the counters."""
        import contextvars

        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(max_iterations=2)
        handler = MagicMock(return_value=_make_model_response())

        def scenario():
            # before_agent's reset lives and dies with its own node context
            self._run_isolated(mw.before_agent, None, None)
            return [
                self._run_isolated(mw.wrap_model_call, self._request(), handler)
                for _ in range(5)
            ]

        # A virgin context: nothing seeded, exactly like an unseeded agent run.
        outs = contextvars.Context().run(scenario)
        assert not any(
            isinstance(out, AIMessage) and "Max iterations" in str(out.content)
            for out in outs
        )

    def test_seeded_budgets_trip_across_isolated_nodes(self):
        """seed_guard_state() in the parent context makes the caps bind again."""
        from strap.guardrails import SubagentGuardMiddleware, seed_guard_state

        mw = SubagentGuardMiddleware(max_iterations=2)
        seed_guard_state()  # what task() now does immediately before subagent.invoke
        handler = MagicMock(return_value=_make_model_response())
        self._run_isolated(mw.wrap_model_call, self._request(), handler)
        self._run_isolated(mw.wrap_model_call, self._request(), handler)
        self._run_isolated(mw.wrap_model_call, self._request(), handler)  # synthesis grant
        result = self._run_isolated(mw.wrap_model_call, self._request(), MagicMock())
        assert isinstance(result, AIMessage)
        assert "Max iterations" in result.content

    def test_seeded_token_budget_trips_across_isolated_nodes(self):
        from strap.guardrails import SubagentGuardMiddleware, seed_guard_state

        mw = SubagentGuardMiddleware(token_budget=100_000, max_iterations=50)
        seed_guard_state()
        handler = MagicMock(
            return_value=_make_model_response(input_tokens=60_000, output_tokens=1_000)
        )
        first = self._run_isolated(mw.wrap_model_call, self._request(), handler)
        # Under budget: the handler's response passes through untouched.
        assert not isinstance(first, AIMessage)
        # Second call crosses the budget; with no tool calls it passes through
        # as the final answer. The third call trips pre-call and spends the
        # synthesis grant; the fourth is the hard stop.
        second = self._run_isolated(mw.wrap_model_call, self._request(), handler)
        assert not isinstance(second, AIMessage)
        self._run_isolated(mw.wrap_model_call, self._request(), handler)
        result = self._run_isolated(mw.wrap_model_call, self._request(), MagicMock())
        assert isinstance(result, AIMessage)
        assert "token budget" in str(result.content).lower()


class TestBudgetFinalSynthesis:
    """A budget trip grants exactly one tool-free synthesis call so the spent
    budget becomes an answer instead of a bare '[LIMIT]' string (which the
    orchestrator can only respond to by re-dispatching from scratch)."""

    def _request(self, system_message=None):
        req = MagicMock()
        req.messages = []
        req.system_message = system_message
        return req

    def test_first_trip_grants_tool_free_synthesis_call(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(max_iterations=0)
        mw.before_agent(None, None)
        req = self._request()
        synth_resp = _make_model_response(content="final synthesis")
        handler = MagicMock(return_value=synth_resp)

        out = mw.wrap_model_call(req, handler)

        assert out is synth_resp
        assert handler.call_count == 1
        req.override.assert_called_once()
        assert req.override.call_args.kwargs["tools"] == []
        # the grant is single-use: the next trip hard-stops
        result = mw.wrap_model_call(req, MagicMock())
        assert isinstance(result, AIMessage)
        assert "Max iterations" in result.content

    def test_synthesis_directive_appended_to_system_message(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(max_iterations=0)
        mw.before_agent(None, None)
        req = self._request(system_message=SystemMessage(content="You are a specialist."))
        handler = MagicMock(return_value=_make_model_response(content="done"))

        mw.wrap_model_call(req, handler)

        new_system = req.override.call_args.kwargs["system_message"]
        assert "FINAL model call" in str(new_system.content)
        assert "tools are disabled" in str(new_system.content)

    def test_post_call_token_trip_with_pending_tool_calls_gets_synthesis(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(token_budget=1000, max_iterations=50)
        mw.before_agent(None, None)
        req = self._request()
        over_budget = _make_model_response(
            tool_calls=[{"name": "expensive_tool", "args": {}, "id": "tc1"}],
            input_tokens=900,
            output_tokens=200,
        )
        synth = _make_model_response(content="wrapped up from gathered results")
        handler = MagicMock(side_effect=[over_budget, synth])

        out = mw.wrap_model_call(req, handler)

        # the pending expensive tool call is dropped; the synthesis answer wins
        assert out is synth
        assert handler.call_count == 2

    def test_synthesis_provider_failure_falls_back_to_hard_stop(self):
        from strap.guardrails import SubagentGuardMiddleware

        mw = SubagentGuardMiddleware(max_iterations=0)
        mw.before_agent(None, None)
        req = self._request()
        handler = MagicMock(side_effect=RuntimeError("provider 400"))

        out = mw.wrap_model_call(req, handler)

        assert isinstance(out, AIMessage)
        assert "Max iterations" in out.content
