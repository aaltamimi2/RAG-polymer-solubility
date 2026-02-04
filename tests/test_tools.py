"""Smoke tests for STRAP/DISSOLVE tool loading and core tools."""

from strap.tools import get_core_tools, get_all_tools


def test_core_tools_load():
    """Core tools (always loaded) should import without error."""
    tools = get_core_tools()
    assert len(tools) >= 10
    names = [t.__name__ for t in tools]
    assert "list_tables" in names
    assert "list_available_solvents" in names or "list_available_polymers" in names


def test_all_tools_load():
    """All 96 tools should load (even if some vendor deps are missing)."""
    tools = get_all_tools()
    assert len(tools) >= 48  # Allow some to fail if optional deps missing


def test_database_query_tool(conn):
    """list_tables should return a string mentioning our tables."""
    from strap.tools.database_query import list_tables
    result = list_tables()
    assert isinstance(result, str)
    assert "common_solvents_database" in result.lower() or "table" in result.lower()


def test_solvent_properties_tool(conn):
    """list_solvent_properties should return data."""
    from strap.tools.solvent_properties import list_solvent_properties
    result = list_solvent_properties()
    assert isinstance(result, str)
    assert len(result) > 50


def test_adaptive_separation_tool(conn):
    """find_optimal_separation_conditions should return a string."""
    from strap.tools.adaptive_separation import find_optimal_separation_conditions
    result = find_optimal_separation_conditions("LDPE", "HDPE,PP")
    assert isinstance(result, str)
