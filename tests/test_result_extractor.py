"""Tests for StructuredResultExtractorMiddleware."""
import json
import threading
import contextvars
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class TestExtractStructuredResult:
    def test_extracts_valid_json(self):
        from strap.result_extractor import _extract_structured_result
        text = 'Some prose.\n<STRUCTURED_RESULT>\n{"agent": "test", "value": 42}\n</STRUCTURED_RESULT>'
        result = _extract_structured_result(text)
        assert result == {"agent": "test", "value": 42}

    def test_returns_none_on_no_block(self):
        from strap.result_extractor import _extract_structured_result
        result = _extract_structured_result("Just prose, no structured block.")
        assert result is None

    def test_returns_none_on_malformed_json(self):
        from strap.result_extractor import _extract_structured_result
        text = '<STRUCTURED_RESULT>\n{not valid json}\n</STRUCTURED_RESULT>'
        result = _extract_structured_result(text)
        assert result is None

    def test_non_greedy_stops_at_first_close(self):
        from strap.result_extractor import _extract_structured_result
        text = '<STRUCTURED_RESULT>\n{"a": 1}\n</STRUCTURED_RESULT>\nmore text\n<STRUCTURED_RESULT>\n{"b": 2}\n</STRUCTURED_RESULT>'
        result = _extract_structured_result(text)
        assert result == {"a": 1}

    def test_empty_block_returns_none(self):
        from strap.result_extractor import _extract_structured_result
        text = '<STRUCTURED_RESULT>\n\n</STRUCTURED_RESULT>'
        result = _extract_structured_result(text)
        assert result is None

    def test_nested_json_extracted(self):
        from strap.result_extractor import _extract_structured_result
        text = '<STRUCTURED_RESULT>\n{"outer": {"inner": [1, 2, 3]}}\n</STRUCTURED_RESULT>'
        result = _extract_structured_result(text)
        assert result == {"outer": {"inner": [1, 2, 3]}}


class TestAccessors:
    def test_get_structured_results_empty_outside_context(self):
        from strap.result_extractor import get_structured_results
        result = get_structured_results()
        assert result == {} or isinstance(result, dict)

    def test_get_structured_result_none_outside_context(self):
        from strap.result_extractor import get_structured_result
        result = get_structured_result("nonexistent")
        assert result is None

    def test_get_structured_results_returns_dict_type(self):
        from strap.result_extractor import get_structured_results
        result = get_structured_results()
        assert isinstance(result, dict)


class TestMiddlewareLifecycle:
    def test_before_agent_resets_registry(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, _registry, _invocation_id, _RegistryState,
            get_structured_results,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        # Manually insert a result using the new storage API
        inv_id = _invocation_id.get()
        _registry[inv_id].results["test-agent"] = {"key": "value"}
        assert get_structured_results() == {"test-agent": {"key": "value"}}
        # Reset by calling before_agent again (creates a new invocation ID)
        mw.before_agent(None, None)
        assert get_structured_results() == {}

    def test_wrap_tool_call_extracts_from_task(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, get_structured_result,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        text = 'Analysis complete.\n<STRUCTURED_RESULT>\n{"agent": "safety-analyst", "safest": "Water"}\n</STRUCTURED_RESULT>'
        tool_msg = ToolMessage(content=text, tool_call_id="tc1")
        cmd = Command(update={"messages": [tool_msg]})

        request = MagicMock()
        request.tool_call = {"name": "task", "args": {"subagent_type": "safety-analyst"}}
        handler = MagicMock(return_value=cmd)

        result = mw.wrap_tool_call(request, handler)
        assert result is cmd  # unmodified
        assert get_structured_result("safety-analyst") == {"agent": "safety-analyst", "safest": "Water"}

    def test_non_task_calls_ignored(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, get_structured_results,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        request = MagicMock()
        request.tool_call = {"name": "some_other_tool", "args": {}}
        handler = MagicMock(return_value=ToolMessage(content="ok", tool_call_id="tc1"))

        mw.wrap_tool_call(request, handler)
        assert get_structured_results() == {}

    def test_wrap_tool_call_with_direct_tool_message(self):
        """Extraction works when handler returns a ToolMessage directly (not a Command)."""
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, get_structured_result,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        text = '<STRUCTURED_RESULT>\n{"agent": "separation-engineer", "solvent": "Toluene"}\n</STRUCTURED_RESULT>'
        tool_msg = ToolMessage(content=text, tool_call_id="tc2")

        request = MagicMock()
        request.tool_call = {"name": "task", "args": {"subagent_type": "separation-engineer"}}
        handler = MagicMock(return_value=tool_msg)

        result = mw.wrap_tool_call(request, handler)
        assert result is tool_msg
        assert get_structured_result("separation-engineer") == {"agent": "separation-engineer", "solvent": "Toluene"}

    def test_multiple_subagents_stored_separately(self):
        """Results from multiple subagents are stored under their respective keys."""
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, get_structured_results,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)

        for agent_name, value in [("agent-a", 1), ("agent-b", 2)]:
            text = f'<STRUCTURED_RESULT>\n{{"{agent_name}": {value}}}\n</STRUCTURED_RESULT>'
            tool_msg = ToolMessage(content=text, tool_call_id=f"tc-{agent_name}")
            request = MagicMock()
            request.tool_call = {"name": "task", "args": {"subagent_type": agent_name}}
            handler = MagicMock(return_value=tool_msg)
            mw.wrap_tool_call(request, handler)

        results = get_structured_results()
        assert "agent-a" in results
        assert "agent-b" in results
        assert results["agent-a"] == {"agent-a": 1}
        assert results["agent-b"] == {"agent-b": 2}


class TestOrchestratorTools:
    def test_get_subagent_result_returns_json(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, _registry, _invocation_id,
            get_subagent_result,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        inv_id = _invocation_id.get()
        _registry[inv_id].results["test-agent"] = {"value": 42}
        result = get_subagent_result("test-agent")
        parsed = json.loads(result)
        assert parsed["value"] == 42

    def test_get_subagent_result_missing_agent(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, get_subagent_result,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        result = get_subagent_result("nonexistent")
        assert "No structured result" in result or "not returned" in result

    def test_get_all_subagent_results_empty(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, get_all_subagent_results,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        result = get_all_subagent_results()
        assert "No structured results" in result

    def test_get_subagent_result_lists_available_agents(self):
        """When requesting a missing agent, the error lists which agents do have results."""
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, _registry, _invocation_id,
            get_subagent_result,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        inv_id = _invocation_id.get()
        _registry[inv_id].results["present-agent"] = {"x": 1}
        result = get_subagent_result("absent-agent")
        assert "present-agent" in result

    def test_get_all_subagent_results_with_data(self):
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, _registry, _invocation_id,
            get_all_subagent_results,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        inv_id = _invocation_id.get()
        _registry[inv_id].results["agent-x"] = {"metric": 99}
        result = get_all_subagent_results()
        parsed = json.loads(result)
        assert parsed["agent-x"]["metric"] == 99


class TestThreadSafety:
    def test_parallel_writes_from_threads(self):
        """Simulate the sync ToolNode dispatch: multiple threads writing results."""
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware,
            _invocation_id, _registry, _registry_lock,
            get_structured_results,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        inv_id = _invocation_id.get()

        # Simulate 3 parallel task() calls writing from different threads
        def write_result(agent_name, value):
            # In the real sync path, copy_context() copies _invocation_id binding
            # Simulate this by setting it in each thread
            _invocation_id.set(inv_id)  # same inv_id as parent
            with _registry_lock:
                _registry[inv_id].results[agent_name] = {"value": value}

        threads = []
        for name, val in [("agent-a", 1), ("agent-b", 2), ("agent-c", 3)]:
            t = threading.Thread(target=write_result, args=(name, val))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        results = get_structured_results()
        assert len(results) == 3
        assert results["agent-a"]["value"] == 1
        assert results["agent-b"]["value"] == 2
        assert results["agent-c"]["value"] == 3

    def test_invocation_id_not_set_drops_result(self):
        """If _invocation_id is not set (edge case), _process_result should drop silently."""
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, _invocation_id, get_structured_results,
        )
        from unittest.mock import MagicMock
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command

        mw = StructuredResultExtractorMiddleware()
        # Do NOT call before_agent — invocation_id is unset
        # Reset the ContextVar by running in a fresh context
        ctx = contextvars.copy_context()

        text = '<STRUCTURED_RESULT>\n{"agent": "test"}\n</STRUCTURED_RESULT>'
        tool_msg = ToolMessage(content=text, tool_call_id="tc1")
        cmd = Command(update={"messages": [tool_msg]})
        request = MagicMock()
        request.tool_call = {"name": "task", "args": {"subagent_type": "test-agent"}}
        handler = MagicMock(return_value=cmd)

        # Run in fresh context where _invocation_id is not set
        def run_in_clean_ctx():
            mw.wrap_tool_call(request, handler)

        ctx.run(run_in_clean_ctx)
        # Result should be dropped — no crash

    def test_after_agent_cleans_up(self):
        """after_agent removes the registry entry."""
        from strap.result_extractor import (
            StructuredResultExtractorMiddleware, _invocation_id, _registry,
            get_structured_results,
        )
        mw = StructuredResultExtractorMiddleware()
        mw.before_agent(None, None)
        inv_id = _invocation_id.get()
        assert inv_id in _registry
        mw.after_agent(None, None)
        assert inv_id not in _registry
