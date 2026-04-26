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
    assert "plot_solubility_vs_temperature" in names


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


def test_list_available_solvents_can_filter_by_polymer(conn):
    from strap.tools.listing import list_available_solvents

    result = list_available_solvents(polymer="LDPE", limit=5)
    parsed = json.loads(result)

    assert parsed["data"]["tool_name"] == "list_available_solvents"
    assert parsed["data"]["polymer"] == "LDPE"
    assert len(parsed["data"]["solvents"]) <= 5
    assert "not a selectivity" in parsed["display"]


def test_list_available_solvents_excludes_quarantined_pair(conn):
    from strap.tools.listing import list_available_solvents

    result = list_available_solvents(polymer="EVOH", limit=20)
    parsed = json.loads(result)

    assert parsed["data"]["polymer"] == "EVOH"
    assert "triethylamine" not in parsed["data"]["solvents"]
    assert "triethylamine" not in parsed["display"].lower()


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


def test_run_biosteam_simulation_reports_boiling_margin(monkeypatch):
    from strap.tools import biosteam_tea_lca as mod

    def fake_run_single_simulation(config):
        return {
            "success": True,
            "solvent": config["solvent"],
            "target_plastic": config["target_plastic"],
            "energy_case": config["energy_case"],
            "tea": {
                "msp_usd_per_kg": 1.23,
                "tci_usd": 12_300_000,
                "aoc_usd_per_yr": 456_000,
            },
            "lca": {"gwp_kg_co2e_per_kg": 0.98},
            "operations": {"total_energy_mj_per_kg": 4.5},
            "runtime_seconds": 0.1,
        }

    monkeypatch.setattr(mod, "run_single_simulation", fake_run_single_simulation)

    raw = mod.run_biosteam_simulation(
        solvent="Cyclohexane",
        target_plastic="LDPE",
        energy_case="C1",
        processing_capacity=8000,
        target_plastic_percent=60,
        dissolution_temp_c=79.7,
    )
    parsed = json.loads(raw)

    conditions = parsed["data"]["process_conditions"]
    assert conditions["dissolution_temp_c"] == 79.7
    assert conditions["boiling_point_c"] > 79.7
    assert 0 < conditions["boiling_margin_c"] < 5
    assert conditions["near_boiling_point"] is True
    assert "narrow atmospheric-pressure margin" in parsed["display"]


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


def test_cli_model_registry_resolves_aliases(monkeypatch):
    from strap import agent as agent_module

    monkeypatch.setenv("DISSOLVE_CLAUDE_SONNET_MODEL", "anthropic:test-sonnet")

    alias, spec = agent_module._resolve_cli_model("gemini-pro")
    assert alias == "gemini-pro"
    assert spec["model"] == "google_genai:gemini-3.1-pro-preview"

    alias, spec = agent_module._resolve_cli_model("claude-sonnet")
    assert alias == "claude-sonnet"
    assert spec["model"] == "anthropic:test-sonnet"

    alias, spec = agent_module._resolve_cli_model("anthropic:custom-model")
    assert alias == "anthropic:custom-model"
    assert spec["env_var"] == "ANTHROPIC_API_KEY"


def test_cli_energy_case_clarification_detection():
    from strap import agent as agent_module

    needs_case = (
        "Estimate CAPEX/OPEX/GWP for LDPE recovered with Cyclohexane "
        "at 79.7 C and 8000 tonnes/year feed capacity."
    )
    explicit_case = needs_case + " Use energy case C1."
    compare_cases = "Compare PE recovery in Toluene across all energy cases."
    non_biosteam = "Find the best separation route for LDPE and EVOH."

    assert agent_module._query_needs_energy_case_clarification(needs_case) is True
    assert agent_module._query_needs_energy_case_clarification(explicit_case) is False
    assert agent_module._query_needs_energy_case_clarification(compare_cases) is False
    assert agent_module._query_needs_energy_case_clarification(non_biosteam) is False


def test_cli_energy_case_clarification_append():
    from strap import agent as agent_module

    query = "Estimate MSP and GWP for LDPE with Cyclohexane."
    basis_rows = agent_module._biosteam_run_basis_rows(query)
    clarified = agent_module._append_energy_case_clarification(
        query,
        {
            "value": "C2",
            "description": "Grid + AMCOR (no on-site utilities)",
        },
        basis_rows,
    )

    assert query in clarified
    assert "energy case C2" in clarified
    assert "Grid + AMCOR" in clarified
    assert "processing_capacity = 20,000 MT/yr" in clarified
    assert "target_plastic_percent = 60 wt%" in clarified
    assert "precipitation_temp_c = 25 C" in clarified
    assert "State these defaults in the final answer" in clarified


def test_cli_biosteam_run_basis_marks_user_values():
    from strap import agent as agent_module

    query = (
        "Estimate CAPEX/OPEX/GWP for LDPE recovered with Cyclohexane at 79.7 C, "
        "8000 tonnes/year total feed capacity, 60 wt% LDPE in the feed."
    )
    rows = agent_module._biosteam_run_basis_rows(query)
    source_by_parameter = {row["parameter"]: row["source"] for row in rows}
    value_by_parameter = {row["parameter"]: row["value"] for row in rows}

    assert source_by_parameter["Metrics"] == "user"
    assert value_by_parameter["Metrics"] == "TCI/CAPEX, AOC/OPEX, GWP"
    assert source_by_parameter["Capacity"] == "user"
    assert source_by_parameter["Target fraction"] == "user"
    assert source_by_parameter["Precipitation temp"] == "default"


def test_cli_biosteam_editable_settings_include_energy_case_and_defaults():
    from strap import agent as agent_module

    query = "Estimate MSP and GWP for LDPE recovered with Cyclohexane."
    settings = agent_module._biosteam_initial_run_settings(query)
    rows = agent_module._biosteam_settings_to_rows(settings)
    labels = [row["setting"] for row in rows]

    assert labels == [
        "Energy case",
        "Metrics",
        "Capacity",
        "Target fraction",
        "Precipitation temp",
        "Continue",
    ]
    assert rows[0]["value"].startswith("C1 - ")
    assert rows[2]["source"] == "default"

    settings["energy_case"]["value"] = "C3"
    settings["energy_case"]["description"] = "Grid + Boiler (boiler but no turbogenerator)"
    settings["energy_case"]["source"] = "override"
    settings["capacity"]["value"] = "8,000 MT/yr"
    settings["capacity"]["source"] = "override"
    clarified = agent_module._append_biosteam_run_settings_clarification(query, settings)

    assert "energy case C3" in clarified
    assert "Capacity = 8,000 MT/yr (override)" in clarified
    assert "Target fraction = 60 wt% target plastic in feed (default)" in clarified


def test_cli_interaction_mode_resolves_aliases(monkeypatch):
    from strap import agent as agent_module

    monkeypatch.delenv("DISSOLVE_CLI_MODE", raising=False)

    assert agent_module._resolve_cli_interaction_mode(None) == "review"
    assert agent_module._resolve_cli_interaction_mode("auto") == "auto"
    assert agent_module._resolve_cli_interaction_mode("fast") == "auto"
    assert agent_module._resolve_cli_interaction_mode("hitl") == "review"

    monkeypatch.setenv("DISSOLVE_CLI_MODE", "auto")
    assert agent_module._resolve_cli_interaction_mode(None) == "auto"


def test_cli_mode_auto_skips_biosteam_clarification(monkeypatch):
    import sys

    from strap import agent as agent_module

    invoked_messages = []
    user_inputs = iter([
        "/mode auto",
        "Estimate CAPEX/OPEX/GWP for LDPE recovered with Cyclohexane.",
        "quit",
    ])

    class DummyLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeAgent:
        def invoke(self, payload, config):
            invoked_messages.append(payload["messages"][-1]["content"])
            return {"messages": [SimpleNamespace(type="ai", content="ok")]}

    monkeypatch.setattr(agent_module, "_show_startup_animation", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_module, "create_dissolve_agent", lambda **kwargs: FakeAgent())
    monkeypatch.setattr("rich.live.Live", DummyLive)
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt="": next(user_inputs))
    monkeypatch.setattr("rich.console.Console.print", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["dissolve", "--no-persist"])

    agent_module.main()

    assert invoked_messages == ["Estimate CAPEX/OPEX/GWP for LDPE recovered with Cyclohexane."]


def test_cli_session_context_survives_restart_for_followup(tmp_path, monkeypatch):
    import sys

    from strap import agent as agent_module

    invoked_messages = []

    class DummyLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeAgent:
        def invoke(self, payload, config):
            invoked_messages.append(payload["messages"][-1]["content"])
            return {"messages": [SimpleNamespace(type="ai", content="ok")]}

    monkeypatch.setenv("DISSOLVE_SESSION_DIR", str(tmp_path))
    monkeypatch.setattr(agent_module, "_show_startup_animation", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_module, "create_dissolve_agent", lambda **kwargs: FakeAgent())
    monkeypatch.setattr("rich.live.Live", DummyLive)
    monkeypatch.setattr("rich.console.Console.print", lambda self, *args, **kwargs: None)

    first_inputs = iter([
        (
            "For a mixed plastic feedstock of 8000 tonnes/year composed of "
            "34% LDPE, 33% EVOH, and 33% PET under scenario A, estimate CAPEX/OPEX/GWP."
        ),
        "quit",
    ])
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt="": next(first_inputs))
    monkeypatch.setattr(sys, "argv", ["dissolve", "--no-persist", "--mode", "auto", "--session", "case-a"])
    agent_module.main()

    second_inputs = iter(["Now run it under C2 and focus on GWP.", "quit"])
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt="": next(second_inputs))
    monkeypatch.setattr(sys, "argv", ["dissolve", "--no-persist", "--mode", "auto", "--session", "case-a"])
    agent_module.main()

    assert invoked_messages[0].startswith("For a mixed plastic feedstock")
    assert "Session context" in invoked_messages[1]
    assert "capacity=8,000 MT/yr" in invoked_messages[1]
    assert "LDPE=34%" in invoked_messages[1]
    assert "User request:\nNow run it under C2" in invoked_messages[1]


def test_cli_model_command_switches_model(monkeypatch):
    import sys

    from strap import agent as agent_module

    created_models = []
    user_inputs = iter(["/model current", "/model gemini-pro", "quit"])

    class DummyLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeAgent:
        def invoke(self, payload, config):
            return {"messages": [SimpleNamespace(type="ai", content="ok")]}

    def fake_create_agent(**kwargs):
        created_models.append(kwargs.get("model_name"))
        return FakeAgent()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setattr(agent_module, "_show_startup_animation", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_module, "create_dissolve_agent", fake_create_agent)
    monkeypatch.setattr("rich.live.Live", DummyLive)
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt="": next(user_inputs))
    monkeypatch.setattr("rich.console.Console.print", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["dissolve", "--no-persist"])

    agent_module.main()

    assert created_models == [
        "google_genai:gemini-3.1-flash-lite-preview",
        "google_genai:gemini-3.1-pro-preview",
    ]


def test_cli_model_command_falls_back_to_table_without_tty(monkeypatch):
    import sys

    from strap import agent as agent_module

    created_models = []
    printed = []
    user_inputs = iter(["/model", "quit"])

    class DummyLive:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeAgent:
        def invoke(self, payload, config):
            return {"messages": [SimpleNamespace(type="ai", content="ok")]}

    def fake_create_agent(**kwargs):
        created_models.append(kwargs.get("model_name"))
        return FakeAgent()

    def fake_print(self, *args, **kwargs):
        printed.extend(args)

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(agent_module, "_show_startup_animation", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_module, "create_dissolve_agent", fake_create_agent)
    monkeypatch.setattr("rich.live.Live", DummyLive)
    monkeypatch.setattr("rich.console.Console.input", lambda self, prompt="": next(user_inputs))
    monkeypatch.setattr("rich.console.Console.print", fake_print)
    monkeypatch.setattr(sys, "argv", ["dissolve", "--no-persist"])

    agent_module.main()

    assert created_models == ["google_genai:gemini-3.1-flash-lite-preview"]
    assert any("/model to open the selector" in str(item) for item in printed)
