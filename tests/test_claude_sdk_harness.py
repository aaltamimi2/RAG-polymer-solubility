"""Tests for the optional Claude Agent SDK harness."""

from __future__ import annotations

import json
from types import SimpleNamespace


def test_claude_model_selection_replaces_non_claude_alias(monkeypatch):
    from strap.claude_sdk_harness.models import resolve_claude_model_selection

    monkeypatch.setenv("DISSOLVE_CLAUDE_SONNET_MODEL", "claude-test-sonnet")

    selection = resolve_claude_model_selection("gemini-pro")

    assert selection.alias == "claude-sonnet"
    assert selection.sdk_model == "claude-test-sonnet"
    assert selection.provider_model_id == "anthropic:claude-test-sonnet"
    assert selection.previous_alias == "gemini-pro"
    assert selection.notice


def test_tool_name_map_translates_legacy_to_mcp():
    from strap.claude_sdk_harness.tool_catalog import ToolNameMap

    tool_map = ToolNameMap()

    assert (
        tool_map.mcp_name("predict_solubility_range")
        == "mcp__dissolve_solubility__predict_solubility_range"
    )
    assert (
        tool_map.legacy_name("mcp__dissolve_solubility__plot_solubility_vs_temperature")
        == "plot_solubility_vs_temperature"
    )
    assert (
        tool_map.mcp_name("run_waste_management_pareto")
        == "mcp__dissolve_optimization__run_waste_management_pareto"
    )


def test_tool_name_map_fails_closed_for_unknown_tool():
    import pytest

    from strap.claude_sdk_harness.tool_catalog import ToolNameMap

    with pytest.raises(KeyError):
        ToolNameMap().mcp_name("not_a_tool")


def test_normalize_tool_call_names_accepts_mcp_strings_and_dicts():
    from strap.claude_sdk_harness.tool_catalog import ToolNameMap, normalize_tool_call_names

    tool_map = ToolNameMap()
    calls = [
        "mcp__dissolve_solubility__predict_solubility_range",
        {"name": "mcp__dissolve_safety__get_solvent_safety_card"},
    ]

    assert normalize_tool_call_names(calls, tool_map) == [
        "predict_solubility_range",
        "get_solvent_safety_card",
    ]


def test_optimization_intent_exposes_optimizer_tools_before_solubility():
    from strap.claude_sdk_harness.tool_catalog import ToolNameMap, infer_intent

    tool_map = ToolNameMap()
    query = (
        "Optimize waste management for 8000 tonnes/year PE/EVOH. "
        "Restrict candidate solvents and save the Pareto front."
    )
    intent = infer_intent(query)
    allowed = tool_map.allowed_for_intent(intent)

    assert intent == "optimization"
    assert tool_map.mcp_name("run_waste_management_optimization") in allowed
    assert tool_map.mcp_name("run_waste_management_pareto") in allowed
    assert tool_map.mcp_name("plot_optimization_pareto_front") in allowed


def test_session_bridge_uses_dissolve_session_paths(monkeypatch, tmp_path):
    from strap.claude_sdk_harness.sessions import bridge_is_resumable, bridge_path, load_bridge, save_bridge

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))

    saved = save_bridge("thread-a", {"claude_session_id": "abc", "cwd": str(tmp_path)})

    assert bridge_path("thread-a") == tmp_path / "thread-a" / "claude_sdk_session.json"
    assert saved["schema_version"] == "1.0"
    assert load_bridge("thread-a")["claude_session_id"] == "abc"
    assert bridge_is_resumable(load_bridge("thread-a"), cwd=tmp_path)
    assert not bridge_is_resumable(load_bridge("thread-a"), cwd=tmp_path / "other")


def test_mcp_wrapper_returns_error_envelope_for_missing_required_arg():
    import asyncio

    from strap.claude_sdk_harness.mcp_server import call_mcp_tool

    result = asyncio.run(call_mcp_tool("predict_solubility_range", {"polymer_name": "EVOH"}))
    text = result["content"][0]["text"]
    parsed = json.loads(text)

    assert result["is_error"] is True
    assert parsed["data"]["tool_name"] == "predict_solubility_range"
    assert parsed["data"]["error_code"] == "wrapper_exception"


def test_mcp_schema_keeps_optional_fields_optional():
    from strap.claude_sdk_harness import mcp_server

    schema = mcp_server._schema_for_tool("plot_solubility_vs_temperature")

    assert schema["required"] == ["polymers", "solvents"]
    assert "temperature_max" in schema["properties"]
    assert "output_dir" in schema["properties"]

    pareto_schema = mcp_server._schema_for_tool("run_waste_management_pareto")
    assert pareto_schema["required"] == ["feed"]
    assert "feed_composition_json" in pareto_schema["properties"]
    assert "polymer_solvent_filters_json" in pareto_schema["properties"]


def test_mcp_wrapper_calls_waste_pareto_with_filters(monkeypatch):
    import asyncio

    import strap.tools.waste_optimization as waste_optimization
    from strap.claude_sdk_harness.mcp_server import call_mcp_tool

    captured = {}

    def fake_pareto(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "display": "pareto ok",
                "data": {
                    "success": True,
                    "tool_name": "run_waste_management_pareto",
                    "analysis_type": "pareto_front",
                },
            }
        )

    monkeypatch.setattr(waste_optimization, "run_waste_management_pareto", fake_pareto)

    result = asyncio.run(
        call_mcp_tool(
            "run_waste_management_pareto",
            {
                "feed": 8000,
                "feed_composition_json": {"PE": 0.6, "EVOH": 0.4},
                "polymer_solvent_filters_json": {
                    "PE": ["Toluene", "Heptane"],
                    "EVOH": ["Pyridazine", "Ethylene Glycol"],
                },
                "scenario": "A",
                "y_metric": "circularity",
                "min_active_washes": 1,
                "max_active_washes": 2,
            },
        )
    )

    assert result.get("is_error") is not True
    assert captured["feed"] == 8000
    assert captured["feed_composition_json"] == {"PE": 0.6, "EVOH": 0.4}
    assert captured["polymer_solvent_filters_json"]["PE"] == ["Toluene", "Heptane"]
    assert captured["min_active_washes"] == 1
    assert captured["max_active_washes"] == 2


def test_runner_preserves_direct_fast_path_without_anthropic_key(monkeypatch, tmp_path):
    from strap.claude_sdk_harness.runner import ClaudeSdkRunner

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))

    runner = ClaudeSdkRunner(thread_id="thread-fast", cwd=tmp_path)
    result = runner.run_turn("what are good solvents for dissolving EVOH")

    assert result.origin == "direct_tool_fast_path"
    assert result.additional_kwargs["claude_model_calls"] == 0
    assert result.additional_kwargs["strap_tool_name"] == "list_available_solvents"
    assert result.mcp_tool_calls == []
    assert result.legacy_tool_calls == ["list_available_solvents"]
    assert "Solvents with solubility data for EVOH" in result.content


def test_runner_missing_key_bridge_records_error_code(monkeypatch, tmp_path):
    from strap.claude_sdk_harness.runner import ClaudeSdkRunner
    from strap.claude_sdk_harness.sessions import load_bridge

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))

    runner = ClaudeSdkRunner(thread_id="thread-missing-key", cwd=tmp_path)
    result = runner.run_turn("summarize the science context")
    bridge = load_bridge("thread-missing-key")

    assert result.result_subtype == "missing_key"
    assert bridge["last_error_code"] == "missing_anthropic_api_key"
    assert bridge["claude_session_id"] is None


def test_runner_reports_missing_sdk_without_raw_import_error(monkeypatch, tmp_path):
    from strap.claude_sdk_harness import runner as runner_module
    from strap.claude_sdk_harness.runner import ClaudeSdkRunner
    from strap.claude_sdk_harness.sessions import load_bridge

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))

    def fake_build_options(*args, **kwargs):
        raise runner_module.ClaudeSdkUnavailableError("claude-agent-sdk is not installed")

    monkeypatch.setattr(runner_module, "build_options", fake_build_options)

    runner = ClaudeSdkRunner(thread_id="thread-no-sdk", cwd=tmp_path)
    result = runner.run_turn("summarize the science context")
    bridge = load_bridge("thread-no-sdk")

    assert result.result_subtype == "sdk_unavailable"
    assert "claude-agent-sdk is not installed" in result.content
    assert bridge["last_error_code"] == "claude_agent_sdk_unavailable"
    assert bridge["claude_session_id"] is None


def test_documented_hook_events_match_sdk_0_1_68_contract():
    from strap.claude_sdk_harness.hooks import DOCUMENTED_HOOK_EVENTS

    assert set(DOCUMENTED_HOOK_EVENTS) == {
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "UserPromptSubmit",
        "Stop",
        "SubagentStart",
        "SubagentStop",
        "PreCompact",
        "Notification",
        "PermissionRequest",
    }


def test_permission_mode_registry_includes_locked_down_mode():
    from strap.claude_sdk_harness.options import DISSOLVE_SCIENCE_PERMISSION_MODE, VALID_PERMISSION_MODES

    assert DISSOLVE_SCIENCE_PERMISSION_MODE == "dontAsk"
    assert DISSOLVE_SCIENCE_PERMISSION_MODE in VALID_PERMISSION_MODES


def test_build_options_constructs_with_real_sdk_when_installed(tmp_path):
    import pytest

    pytest.importorskip("claude_agent_sdk")
    from strap.claude_sdk_harness.options import build_options
    from strap.claude_sdk_harness.tool_catalog import ToolNameMap

    tool_map = ToolNameMap()
    options = build_options(
        sdk_model="claude-test",
        allowed_tools=[tool_map.mcp_name("list_available_solvents")],
        cwd=tmp_path,
    )

    assert options.permission_mode == "dontAsk"
    assert options.system_prompt and not isinstance(options.system_prompt, dict)


def test_pretook_guard_blocks_unapproved_tool():
    from strap.claude_sdk_harness.hooks import HookDiagnostics, guard_tool_name
    from strap.claude_sdk_harness.tool_catalog import ToolNameMap

    diagnostics = HookDiagnostics()
    ok = guard_tool_name(
        "Bash",
        allowed_tools=[ToolNameMap().mcp_name("list_available_solvents")],
        diagnostics=diagnostics,
        active_intent="solubility_lookup",
    )

    assert ok is False
    assert diagnostics.blocked_tools[0]["tool_name"] == "Bash"


def test_subagent_conversion_defers_unexposed_real_groups_without_silent_drop():
    from strap.claude_sdk_harness.agents import build_agent_definitions

    agents = build_agent_definitions()

    assert "separation-engineer" in agents
    separation = agents["separation-engineer"]
    if isinstance(separation, dict):
        assert "Deferred SDK groups" in separation["description"]
        assert all(tool.startswith("mcp__") for tool in separation["tools"])


def test_harness_status_includes_cost_and_tool_search(monkeypatch, tmp_path):
    from strap.claude_sdk_harness.cli_adapter import format_harness_status
    from strap.claude_sdk_harness.sessions import save_bridge

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    save_bridge("thread-status", {"last_cost_usd": 0.0123, "cwd": str(tmp_path)})

    status = format_harness_status(
        harness="claude_sdk",
        thread_id="thread-status",
        model_alias="claude-sonnet",
        model_name="anthropic:claude-test",
        cwd=tmp_path,
    )

    assert "Tool search:" in status
    assert "Last cost: $0.012300" in status


def test_cli_harness_defaults_to_langchain():
    from strap import agent as agent_module

    assert agent_module._resolve_cli_harness(None) == "langchain"
    assert agent_module._resolve_cli_harness("claude-sdk") == "claude_sdk"


def test_dissolve_claude_entrypoint_injects_harness_arg():
    from strap.claude_sdk_harness.cli import _argv_with_claude_harness

    assert _argv_with_claude_harness(["dissolve-claude", "--mode", "auto"]) == [
        "dissolve-claude",
        "--harness",
        "claude_sdk",
        "--mode",
        "auto",
    ]
    assert _argv_with_claude_harness(["dissolve-claude", "--harness", "langchain"]) == [
        "dissolve-claude",
        "--harness",
        "langchain",
    ]


def test_claude_sdk_query_result_message_extraction():
    from strap.claude_sdk_harness.messages import extract_session_id, is_result_message, result_text

    message = SimpleNamespace(
        __class__=SimpleNamespace(__name__="ResultMessage"),
        result="answer",
        session_id="session-1",
    )

    assert is_result_message(message)
    assert result_text(message) == "answer"
    assert extract_session_id(message) == "session-1"
