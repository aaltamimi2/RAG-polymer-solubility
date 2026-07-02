"""Tests for deterministic direct-tool fast paths."""

from __future__ import annotations

import json
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


def test_direct_fast_path_handles_routine_multilayer_solvent_candidates(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
        "identify solvents that are promising for dissolving any one of the components "
        f"below 100 deg C. Save any structured output to {tmp_path}."
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "routine_solvent_candidate_lookup"
    assert result.route_decision["model_call_budget"] == 0
    assert result.data["used_hsp_or_statistics_ml"] is False
    assert result.data["temperature_max_c"] == 100.0
    assert result.data["polymers"] == ["LDPE", "EVOH", "PET"]
    assert result.artifacts[0]["type"] == "solvent_candidate_table"
    assert (tmp_path / "routine_solvent_candidates.json").exists()
    assert "Method: direct solubility database lookup" in result.display


def test_direct_fast_path_handles_selectively_worded_routine_candidate_query():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
        "identify solvents that are promising for selectively dissolving any one "
        "of these components below 100 °C."
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "routine_solvent_candidate_lookup"
    assert result.route_decision["model_call_budget"] == 0
    assert result.data["used_hsp_or_statistics_ml"] is False
    assert result.data["temperature_max_c"] == 100.0


def test_direct_fast_path_handles_paraphrased_routine_candidate_queries():
    from strap.direct_fast_path import try_direct_tool_fast_path

    queries = [
        "Which solvents could dissolve one component of an LDPE/EVOH/PET multilayer below 100 C?",
        "Give candidate solvents for LDPE/EVOH/PET below 100 C.",
        "Find solvents for any polymer in an LDPE EVOH PET feedstock at or below 100C.",
    ]

    for query in queries:
        result = try_direct_tool_fast_path(query)

        assert result is not None, query
        assert result.tool_name == "routine_solvent_candidate_lookup"
        assert result.route_decision["model_call_budget"] == 0


def test_direct_fast_path_threshold_lookup_scans_all_solvents_by_default():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = "What are other solvents that have solubility of at least 10% for LDPE at 100C"

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "solubility_threshold_lookup"
    assert result.route_decision["model_call_budget"] == 0
    assert result.data["scope"] == "all supported solvents"
    solvents = [row["solvent"] for row in result.data["results"]]
    assert len(solvents) == 12
    assert "toluene" in solvents
    assert "dodecane" in solvents
    assert "dimethylsulfoxide" not in solvents
    assert all(row["solubility_pct"] >= 10.0 for row in result.data["results"])


def test_direct_fast_path_threshold_lookup_can_limit_to_prior_candidate_table():
    from strap.direct_fast_path import try_direct_tool_fast_path

    context = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Artifacts: solvent_candidate_table: polymers=LDPE,EVOH,PET; "
        "solvent_candidates=cyclohexane, hexane, n-heptane, thf, thp, dimethylsulfoxide\n\n"
        "User request:\n"
        "Among these solvents, which have solubility of at least 10% for LDPE at 100C?"
    )

    result = try_direct_tool_fast_path(context)

    assert result is not None
    assert result.tool_name == "solubility_threshold_lookup"
    assert result.data["scope"] == "prior listed solvents"
    solvents = [row["solvent"] for row in result.data["results"]]
    assert solvents == ["cyclohexane", "hexane", "n-heptane", "thf", "thp"]


def test_direct_fast_path_threshold_followup_preserves_threshold_at_lower_temperature():
    from strap.direct_fast_path import try_direct_tool_fast_path

    context = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Artifacts: solvent_candidate_table: polymer=LDPE; "
        "solvents=cyclohexane, hexane, n-heptane, thf, thp, isopropylamine, "
        "2,3-dihydropyran, toluene, 1,4-dimethylbenzene, benzene, "
        "1,2-dimethylbenzene, dodecane; temperature_c=100; min_solubility_pct=10\n\n"
        "User request:\n"
        "Which of these solvents still works at 80C or below (higher than 10% solubility)"
    )

    result = try_direct_tool_fast_path(context)

    assert result is not None
    assert result.tool_name == "solubility_threshold_lookup"
    assert result.route_decision["model_call_budget"] == 0
    assert result.data["scope"] == "prior listed solvents"
    assert result.data["temperature_c"] == 80.0
    assert result.data["min_solubility_pct"] == 10.0
    assert result.data["results"] == []
    assert "No solvents met the threshold" in result.display


def test_direct_fast_path_does_not_swallow_sequence_or_selectivity_workflows():
    from strap.direct_fast_path import try_direct_tool_fast_path

    queries = [
        "Generate the best separation sequence for LDPE/EVOH/PET below 100 C.",
        "Rank solvent selectivity for LDPE over PET below 100 C.",
        "Compare solvents for separating EVOH from LDPE and PET below 100 C.",
    ]

    for query in queries:
        result = try_direct_tool_fast_path(query)

        assert result is None, query


def test_direct_fast_path_handles_routine_multilayer_temperature_spellings():
    from strap.direct_fast_path import try_direct_tool_fast_path

    variants = {
        "below 100 C": 100.0,
        "below 100C": 100.0,
        "below 100 °C": 100.0,
        "under 100 C": 100.0,
        "up to 100 C": 100.0,
        "below 212 F": 100.0,
        "below 212 fahrenehit": 100.0,
        "below 212 degrees Fahrenheit": 100.0,
        "under 373.15 K": 100.0,
        "up to 373.15 kelvn": 100.0,
        "up to 373.15 degrees Kelvin": 100.0,
        "below 100 celcius": 100.0,
        "below 100 degrees Celsius": 100.0,
    }
    for variant, expected_c in variants.items():
        query = (
            "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
            "identify solvents that are promising for dissolving any one of the components "
            f"{variant}."
        )

        result = try_direct_tool_fast_path(query)

        assert result is not None, variant
        assert result.tool_name == "routine_solvent_candidate_lookup"
        assert result.route_decision["model_call_budget"] == 0
        assert result.data["used_hsp_or_statistics_ml"] is False
        assert abs(result.data["temperature_max_c"] - expected_c) < 1e-6


def test_direct_fast_path_repairs_wrapped_unquoted_output_path(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    wrapped_path = (
        f"{tmp_path}/case-studies/case-1/01-ldpe-evoh-p\n"
        "  et\n"
        "    -solubility/json."
    )
    query = (
        "For a multilayer mixed plastic feedstock containing LDPE, EVOH, and PET, "
        "identify solvents that are promising for dissolving any one of the components "
        f"below 100 deg C. Save any structured output to {wrapped_path}"
    )

    result = try_direct_tool_fast_path(query)

    expected_dir = tmp_path / "case-studies" / "case-1" / "01-ldpe-evoh-pet-solubility" / "json"
    assert result is not None
    assert result.tool_name == "routine_solvent_candidate_lookup"
    assert result.data["structured_output_path"] == str(expected_dir / "routine_solvent_candidates.json")
    assert (expected_dir / "routine_solvent_candidates.json").exists()


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


def test_direct_fast_path_does_not_treat_generic_result_plot_as_solubility(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Feedstock: capacity=8,000 MT/yr; composition=EVOH=40%, PE=60%\n"
        "- Process: scenario=A; solvent_candidates=Toluene, Heptane, Pyridazine, Ethylene glycol\n\n"
        "User request:\n"
        f'plot this result and save to "{tmp_path}"'
    )

    assert try_direct_tool_fast_path(query) is None


def test_direct_fast_path_plots_last_optimization_point_result(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    payload = {
        "analysis_type": "point_optimum",
        "scenario": "A",
        "feed_composition": {"PE": 0.6, "EVOH": 0.4},
        "profit": 12_136_242.52,
        "total_cost": 6_953_357.48,
        "emissions": 9_211.0833,
        "circularity_score": 0.6408,
        "stage1_tech": ["st1"],
        "stage2_tech": ["st2"],
        "stage3_tech": ["lf"],
        "optimal_washes": ["PE-o-Xylene @ 143.5C", "EVOH-Dimethyl sulfoxide @ 145C"],
    }
    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Feedstock: capacity=8,000 MT/yr; composition=EVOH=40%, PE=60%\n"
        "- Process: scenario=A; solvent_candidates=Toluene, Heptane, Pyridazine, Ethylene glycol\n"
        "- Last optimization result: artifact_id=artifact_opt; analysis_type=point_optimum; "
        f"payload_json={json.dumps(payload, separators=(',', ':'), sort_keys=True)}\n\n"
        "User request:\n"
        f'plot this result and save to "{tmp_path}"'
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "plot_optimization_point_result"
    assert result.route_decision["intent"] == "optimization_plot"
    assert result.data["plot_paths"][0].startswith(str(tmp_path))
    assert result.artifacts[0]["entities"]["plot_type"] == "optimization_point_result"
    assert "Optimization Point Result Plot Created" in result.display


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


def test_direct_fast_path_prefers_latest_displayed_candidates_over_broad_session_pool(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Feedstock: polymers=LDPE, EVOH, PET\n"
        "- Process: solvent_candidates=Glycol, Dodecane, Dimethyl sulfoxide, N,N-Dimethylformamide, Triethylamine; output_dir="
        f"{tmp_path}\n"
        "- Last solvent candidates: polymers=LDPE, EVOH, PET; solvents=Dodecane, Dimethyl sulfoxide, N,N-Dimethylformamide\n\n"
        "User request:\n"
        "can you plot the temperature dependent solubility of each of these polymers in each of these solvents over these temperatures"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "plot_solubility_vs_temperature"
    assert result.data["solvents"] == ["dodecane", "dimethylsulfoxide", "dimethylformamide"]
    assert "glycol" not in result.data["solvents"]
    assert "triethylamine" not in result.display.lower()
    assert result.data["plot_filepath"].startswith(str(tmp_path))


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


def test_direct_fast_path_combined_separation_and_solubility_plot_from_session_context(tmp_path):
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Feedstock: polymers=LDPE, EVOH, PET\n"
        f"- Process: solvent_candidates=1,4-Dimethylbenzene, Dimethylsulfoxide; output_dir={tmp_path}; dissolution_temp_c=90\n\n"
        "User request:\n"
        "okay can you plot this separation approach. also plot the solubility of these polymers "
        "in each of these solvents from room temperature to 25C"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "multi_tool_fast_path"
    assert [child["tool_name"] for child in result.data["results"]] == [
        "create_separation_tree_plot",
        "plot_solubility_vs_temperature",
    ]
    artifact_types = {artifact["type"] for artifact in result.artifacts}
    assert {"separation_tree_plot", "plot_artifact"} <= artifact_types
    assert (tmp_path / "separation_sequence_rank1.png").is_file()
    assert any(path.name.endswith("_solubility_vs_temp.png") for path in tmp_path.iterdir())


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
    assert "Dimethyl Formamide" in result.display
    assert "GVL" not in result.display


def test_direct_fast_path_safest_followup_uses_prior_polymer_candidates():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Session context (compact; use only to resolve follow-ups and do not restate unless relevant):\n"
        "- Last solvent candidates: polymers=HDPE,EVOH,PET,PVC; "
        "solvents=cyclohexane, hexane, n-heptane, thf, thp, isopropylamine, "
        "dimethylsulfoxide, dimethylformamide, methanol, ethanol, ch2cl2, propanone; "
        "rows=HDPE:cyclohexane, HDPE:hexane, EVOH:isopropylamine, "
        "EVOH:dimethylsulfoxide, EVOH:dimethylformamide, EVOH:methanol, EVOH:ethanol, "
        "PET:ch2cl2, PET:dimethylformamide, PVC:propanone; temperature_range=25-100 C\n\n"
        "User request:\n"
        "which of these solvents for EVOH is safest"
    )

    result = try_direct_tool_fast_path(query)

    assert result is not None
    assert result.tool_name == "compare_solvent_safety_cards"
    assert result.route_decision["intent"] == "safety_lookup"
    assert "Available Solvents Summary" not in result.display
    assert "Dimethyl Sulfoxide" in result.display or "DMSO" in result.display
    assert "Dimethyl Formamide" in result.display or "DMF" in result.display


def test_direct_fast_path_does_not_route_hsp_handle_query_to_safety():
    from strap.direct_fast_path import try_direct_tool_fast_path

    result = try_direct_tool_fast_path(
        "Can the HSP model handle GVL for PET, or is GVL unsupported? Do not substitute another solvent."
    )

    assert result is None


def test_direct_fast_path_does_not_route_hsp_compatibility_query_to_safety():
    from strap.direct_fast_path import try_direct_tool_fast_path

    result = try_direct_tool_fast_path(
        "Use HSP to predict LDPE compatibility with dodecane, and flag ambiguity."
    )

    assert result is None


def test_fast_path_domain_conflict_guard_does_not_block_plain_safety():
    from strap.direct_fast_path import _has_fast_path_domain_conflict

    assert _has_fast_path_domain_conflict("How should I safely heat toluene to 110 C?", "safety_lookup") is False
    assert _has_fast_path_domain_conflict("Can the HSP model handle GVL for PET?", "safety_lookup") is True


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


def test_direct_fast_path_falls_through_for_pareto_workflow():
    from strap.direct_fast_path import try_direct_tool_fast_path

    query = (
        "Run a cost-vs-circularity Pareto landscape for 8000 tonnes/year composed of "
        "25% PP, 25% PS, 25% PVC, and 25% PC. Use shortlisted solvent candidates: "
        "PP: Toluene and Cyclohexane; PVC: Dimethyl sulfoxide. Plot the resulting landscape."
    )

    assert try_direct_tool_fast_path(query) is None
