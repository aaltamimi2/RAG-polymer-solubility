"""Tests for prompt-building helpers."""

from strap.prompts import FILE_IO_DIRECTIVE, THINK_DIRECTIVE, build_system_prompt


def test_build_system_prompt_includes_routing_table_and_handoff_protocol():
    prompt = build_system_prompt("ROUTING TABLE HERE")

    assert "ROUTING TABLE HERE" in prompt
    assert "## Handoff protocol" in prompt
    assert "build_handoff(...)" in prompt


def test_directives_have_expected_sections():
    assert "## REFLECTION" in THINK_DIRECTIVE
    assert "## FILE I/O" in FILE_IO_DIRECTIVE
