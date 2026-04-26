"""Tests for deterministic direct-tool fast paths."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage


def test_direct_fast_path_solubility_range_bypasses_model_handler():
    from strap.direct_fast_path import DirectToolFastPathMiddleware

    middleware = DirectToolFastPathMiddleware()
    request = SimpleNamespace(
        messages=[HumanMessage(content="what is the solubility of EVOH in DMF from room temp to 80C")]
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="model answer")]))

    response = middleware.wrap_model_call(request, handler)

    handler.assert_not_called()
    assert response.result[0].additional_kwargs["strap_origin"] == "direct_tool_fast_path"
    assert response.result[0].additional_kwargs["strap_tool_name"] == "predict_solubility_range"
    assert response.result[0].additional_kwargs["strap_route_decision"]["mode"] == "direct_tool"
    assert response.result[0].additional_kwargs["strap_artifacts"][0]["type"] == "solubility_table"
    assert "EVOH in dimethylformamide" in response.result[0].content
    assert "| 80.0 | 6.63 |" in response.result[0].content


def test_direct_fast_path_multi_solvent_context_lookup_combines_results():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Process: solvent_candidates=Cyclohexane, Hexane, Heptane, Dodecane\n\n"
        "User request:\n"
        "what is the solubility of LDPE in each of these solvents up to 80C"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "predict_solubility_range"
    assert result.route_decision["model_call_budget"] == 0
    assert result.artifacts[0]["type"] == "solubility_table"
    assert len(result.data["results"]) == 4
    assert "LDPE in cyclohexane" in result.display
    assert "LDPE in dodecane" in result.display


def test_direct_fast_path_plot_followup_preserves_dmf_alias_and_temperature_override(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last solubility lookup: polymer=EVOH; solvents=N,N-Dimethylformamide; temperature_range=25-153 C\n\n"
        "User request:\n"
        f'no just plot from 25C to 90C and save to "{tmp_path}"'
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "plot_solubility_vs_temperature"
    assert result.route_decision["mode"] == "artifact_transform"
    assert result.artifacts[0]["type"] == "plot_artifact"
    assert result.data["temperature_min_c"] == 25.0
    assert result.data["temperature_max_c"] == 90.0
    assert result.data["solvents"] == ["dimethylformamide"]
    assert "benzene" not in result.display.lower()
    assert result.data["plot_filepath"].startswith(str(tmp_path))


def test_direct_fast_path_plot_parses_multi_polymer_multi_solvent_request(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    result = try_direct_tool_fast_path(
        'Plot the solubility of LDPE/EVOH/PET in dodecane, o-xylene, and toluene '
        f'from 25 to 100C save to "{tmp_path}"'
    )

    assert result is not None
    assert result.tool_name == "plot_solubility_vs_temperature"
    assert result.data["polymers"] == ["LDPE", "EVOH", "PET"]
    assert result.data["solvents"] == ["dodecane", "1,2-dimethylbenzene", "toluene"]
    assert result.data["temperature_min_c"] == 25.0
    assert result.data["temperature_max_c"] == 100.0
    assert result.data["plot_filepath"].startswith(str(tmp_path))
    assert "Polymers: LDPE, EVOH, PET" in result.display
    assert "Solvents: Dodecane, o-Xylene, Toluene" in result.display


def test_direct_fast_path_replot_inherits_previous_output_directory(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    prior_path = tmp_path / "previous_plot.png"
    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last plot artifact: plot_type=solubility_vs_temperature; polymers=LDPE, EVOH, PET; "
        "solvents=dodecane, 1,2-dimethylbenzene, toluene; temperature_range=25-150 C; "
        f"path={prior_path}; output_dir={tmp_path}\n\n"
        "User request:\n"
        "no plot solubility of LDPE/EVOH/PET in dodecane, o-xylene, and toluene from 25 to 100C"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.data["temperature_max_c"] == 100.0
    assert result.data["plot_filepath"].startswith(str(tmp_path))
    assert "150.0C" not in result.display


def test_direct_fast_path_does_not_replot_solubility_for_dp_state_map(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last plot artifact: plot_type=solubility_vs_temperature; polymers=LDPE, EVOH, PET; "
        "solvents=dodecane, 1,2-dimethylbenzene, toluene; temperature_range=25-100 C; "
        f"path={tmp_path / 'prior.png'}; output_dir={tmp_path}\n\n"
        "User request:\n"
        f'Generate a dynamic-programming separation state map for LDPE/EVOH/PET at 100C and save it to "{tmp_path}". '
        "Only include separation/selectivity visuals, not cost, greenness, or optimization plots."
    )

    result = try_direct_tool_fast_path(query)

    assert result is None


def test_direct_fast_path_normalizes_wrapped_wsl_output_path():
    from strap.direct_fast_path import _extract_output_path_args

    args = _extract_output_path_args(
        'save to "\\\\wsl.localhost\\Ubuntu-20.04\\home\\aaltami\n'
        '  mi2\\langchain-STRAP-v9-contaminants\\docs\\case_studies\\plots"'
    )

    assert args == {
        "output_dir": "/home/aaltamimi2/langchain-STRAP-v9-contaminants/docs/case_studies/plots"
    }


def test_direct_fast_path_plots_each_prior_solvent_candidate_without_separation_routing(tmp_path):
    from strap.direct_fast_path import DirectToolFastPathMiddleware

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Feedstock: polymers=LDPE, EVOH, PET\n"
        "- Process: solvent_candidates=N,N-Dimethylformamide, Dimethyl sulfoxide, Isopropylamine, Triethylamine, Methanol, Ethanol, Acetone, 1-Propanol, 2-Propanol, Ethylene glycol, Tetrahydrofuran (THF), Methyl ethyl ketone\n\n"
        "User request:\n"
        f'plot the solubility of each of these solvents in EVOH as a function of temperature up to 100C and save to "{tmp_path}"'
    )
    middleware = DirectToolFastPathMiddleware()
    request = SimpleNamespace(messages=[HumanMessage(content=query)])
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="model answer")]))

    response = middleware.wrap_model_call(request, handler)

    handler.assert_not_called()
    message = response.result[0]
    assert message.additional_kwargs["strap_origin"] == "direct_tool_fast_path"
    assert message.additional_kwargs["strap_tool_name"] == "plot_solubility_vs_temperature"
    assert message.additional_kwargs["strap_route_decision"]["mode"] == "artifact_transform"
    assert "Polymers: EVOH" in message.content
    assert "Temperature range: 25.0C - 100.0C" in message.content
    assert "dimethylformamide" in message.additional_kwargs["strap_artifacts"][0]["entities"]["solvents"]
    assert "butanone" in message.additional_kwargs["strap_artifacts"][0]["entities"]["solvents"]


def test_direct_fast_path_plots_top_n_prior_solvent_candidates(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last solvent candidates: polymer=EVOH; solvents=dimethylformamide, dimethylsulfoxide, isopropylamine, triethylamine, methanol\n\n"
        "User request:\n"
        f'plot the top 4 of those solvents only up to 100C and save to "{tmp_path}"'
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "plot_solubility_vs_temperature"
    assert result.route_decision["mode"] == "artifact_transform"
    assert result.data["temperature_max_c"] == 100.0
    assert result.data["solvents"] == [
        "dimethylformamide",
        "dimethylsulfoxide",
        "isopropylamine",
        "methanol",
    ]
    assert "triethylamine" not in result.data["solvents"]
    assert "triethylamine" not in result.display.lower()


def test_direct_fast_path_prefers_candidate_artifact_for_top_n_when_lookup_exists(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last solvent candidates: polymer=EVOH; solvents=dimethylformamide, dimethylsulfoxide, isopropylamine, triethylamine, methanol\n"
        "- Last solubility lookup: polymer=EVOH; solvents=dimethylformamide; temperature_range=25-80 C\n\n"
        "User request:\n"
        f'plot the top 4 of those solvents only up to 100C and save to "{tmp_path}"'
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.data["solvents"] == [
        "dimethylformamide",
        "dimethylsulfoxide",
        "isopropylamine",
        "methanol",
    ]
    assert "triethylamine" not in result.data["solvents"]


def test_direct_fast_path_values_followup_from_last_plot():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last plot artifact: plot_type=solubility_vs_temperature; polymer=EVOH; solvents=dimethylformamide, dimethylsulfoxide; temperature_range=25-100 C; path=/tmp/evoh.png\n\n"
        "User request:\n"
        "give the exact values for those curves from 25C to 80C"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "predict_solubility_range"
    assert result.route_decision["mode"] == "direct_tool"
    assert len(result.data["results"]) == 2
    assert result.data["results"][0]["t_end_c"] == 80.0
    assert "EVOH in dimethylformamide" in result.display


def test_direct_fast_path_classifies_current_request_not_context():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last plot artifact: plot_type=solubility_vs_temperature; polymer=EVOH; solvents=dimethylformamide, dimethylsulfoxide; temperature_range=25-100 C; path=/tmp/evoh.png\n\n"
        "User request:\n"
        "what is the solubility of EVOH in DMF at 80C"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "predict_solubility"
    assert result.route_decision["intent"] == "solubility_lookup"
    assert "EVOH in dimethylformamide at 80.0 C" in result.display


def test_direct_fast_path_solubility_lookup_not_plot_update_with_prior_plot():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last solvent candidates: polymer=LDPE; solvents=cyclohexane, dodecane, hexane, n-heptane\n"
        "- Last plot artifact: plot_type=solubility_vs_temperature; polymer=EVOH; solvents=dimethylformamide, dimethylsulfoxide; temperature_range=25-100 C; path=/tmp/evoh.png\n\n"
        "User request:\n"
        "what is the solubility of LDPE in each of these solvents up to 80C"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "predict_solubility_range"
    assert result.route_decision["intent"] == "solubility_lookup"
    assert len(result.data["results"]) == 4
    assert result.data["results"][0]["t_start_c"] == 25.0
    assert result.data["results"][0]["t_end_c"] == 80.0


def test_direct_fast_path_lists_new_polymer_solvents_despite_plot_context():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last plot artifact: plot_type=solubility_vs_temperature; polymer=EVOH; solvents=dimethylformamide, dimethylsulfoxide; temperature_range=25-100 C; path=/tmp/evoh.png\n\n"
        "User request:\n"
        "what are good solvents for dissolving LDPE in the same feedstock"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "list_available_solvents"
    assert result.route_decision["intent"] == "solvent_candidate_lookup"
    assert result.artifacts[0]["entities"]["polymer"] == "LDPE"


def test_direct_fast_path_safety_compare_plural_cards():
    from strap.direct_fast_path import try_direct_tool_fast_path

    result = try_direct_tool_fast_path(
        "compare safety cards for dimethylformamide and dimethylsulfoxide at 90C"
    )

    assert result is not None
    assert result.tool_name == "compare_solvent_safety_cards"
    assert result.route_decision["intent"] == "safety_lookup"
    assert "Dimethylformamide" in result.display


def test_direct_fast_path_falls_through_for_complex_routed_request():
    from strap.direct_fast_path import DirectToolFastPathMiddleware

    middleware = DirectToolFastPathMiddleware()
    request = SimpleNamespace(
        messages=[HumanMessage(content="rank solvents for separating LDPE from EVOH and PET")]
    )
    handler = MagicMock(return_value=ModelResponse(result=[AIMessage(content="model answer")]))

    response = middleware.wrap_model_call(request, handler)

    handler.assert_called_once_with(request)
    assert response.result[0].content == "model answer"
