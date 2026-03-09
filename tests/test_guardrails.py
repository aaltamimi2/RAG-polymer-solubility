"""Tests for SubagentGuardMiddleware."""
import json
import pytest
from unittest.mock import MagicMock
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
        result = mw.wrap_model_call(req, handler)
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
        result = mw.wrap_model_call(req, handler)
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
        result = mw.wrap_model_call(req, handler)
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
