"""Smoke tests for STRAP/DISSOLVE tool loading and core tools."""

import json
from types import SimpleNamespace

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
    parsed = json.loads(result)
    assert parsed["data"]["tool_name"] == "list_tables"
    assert "common_solvents_database" in parsed["display"].lower() or "table" in parsed["display"].lower()


def test_solvent_properties_tool(conn):
    """get_solvent_properties should return data."""
    from strap.tools.solvent_properties import get_solvent_properties
    result = get_solvent_properties("toluene")
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["data"]["tool_name"] == "get_solvent_properties"
    assert parsed["data"]["success"] is True
    assert len(parsed["display"]) > 50


def test_adaptive_separation_tool(conn):
    """find_optimal_separation_conditions should return a string."""
    from strap.tools.adaptive_separation import find_optimal_separation_conditions
    result = find_optimal_separation_conditions("LDPE", "HDPE,PP")
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["data"]["tool_name"] == "find_optimal_separation_conditions"


def test_run_biosteam_batch_returns_partial_results_for_large_screen(monkeypatch):
    import time
    from strap.tools import biosteam_tea_lca as mod

    monkeypatch.setattr(mod, "run_single_simulation", object())
    monkeypatch.setattr(
        mod,
        "_expand_solvents",
        lambda solvents, target_plastic: [
            "sec-Butyl Acetate",
            "Isobutyl Acetate",
            "Heptane",
            "Toluene",
            "Xylene",
            "Cyclohexane",
            "Hexane",
            "Dodecane",
            "o-Xylene",
            "p-Xylene",
            "Methylcyclohexane",
            "Acetone",
            "Ethanol",
        ],
    )
    monkeypatch.setattr(mod, "_expand_energy_cases", lambda raw: ["C1"])

    call_solvents = []

    def fake_build_batch_configs(**kwargs):
        return [
            {"solvent": solvent, "energy_case": "C1"}
            for solvent in kwargs["solvents"]
        ]

    def fake_run_batch_simulations(configs, max_parallel=3, timeout_per_sim=120):
        call_solvents.append([cfg["solvent"] for cfg in configs])
        results = []
        for cfg in configs:
            if cfg["solvent"] in {"Heptane", "Toluene", "Xylene", "Cyclohexane", "Hexane"}:
                results.append(
                    {
                        "success": True,
                        "solvent": cfg["solvent"],
                        "energy_case": "C1",
                        "tea": {"msp_usd_per_kg": float(len(results) + 1), "tci_usd": 1_000_000.0},
                        "lca": {"gwp_kg_co2e_per_kg": 2.0},
                    }
                )
            else:
                results.append(
                    {
                        "success": False,
                        "solvent": cfg["solvent"],
                        "energy_case": "C1",
                        "error": "Simulation timed out after 45s",
                    }
                )
        return results

    monkeypatch.setattr(mod, "build_batch_configs", fake_build_batch_configs)
    monkeypatch.setattr(mod, "run_batch_simulations", fake_run_batch_simulations)
    monotonic_values = iter([0.0, 0.0, 35.0, 70.0, 98.0])
    monkeypatch.setattr(time, "monotonic", lambda: next(monotonic_values))

    raw = mod.run_biosteam_batch("all_pe", target_plastic="PE", energy_cases="C1")
    parsed = json.loads(raw)

    assert parsed["data"]["tool_name"] == "run_biosteam_batch"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["partial"] is True
    assert parsed["data"]["attempted"] < parsed["data"]["total_configs"]
    assert len(parsed["data"]["results"]) >= 5
    assert call_solvents[0][:2] == ["Heptane", "Toluene"]
    assert "Xylene" in call_solvents[0][2]


def test_cli_main_renders_markdown_answer_without_nameerror(monkeypatch):
    import sys

    from rich.markdown import Markdown

    from strap import agent as agent_module

    printed = []
    user_inputs = iter(["What polymers are available?", "quit"])

    class DummyLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeAgent:
        def invoke(self, payload, config):
            return {
                "messages": [
                    SimpleNamespace(type="ai", content="**Test answer**"),
                ]
            }

    def fake_print(self, *args, **kwargs):
        printed.extend(args)

    monkeypatch.setattr(agent_module, "_show_startup_animation", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_module, "create_dissolve_agent", lambda **kwargs: FakeAgent())
    monkeypatch.setattr("rich.live.Live", DummyLive)
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt="": next(user_inputs))
    monkeypatch.setattr("rich.console.Console.print", fake_print)
    monkeypatch.setattr(sys, "argv", ["dissolve", "--no-persist"])

    agent_module.main()

    assert any(isinstance(item, Markdown) for item in printed)
