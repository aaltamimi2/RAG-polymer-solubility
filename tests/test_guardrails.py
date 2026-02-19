"""Tests for SubagentGuardMiddleware."""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage


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
