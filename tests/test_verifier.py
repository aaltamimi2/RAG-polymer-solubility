import logging
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def test_parse_verdict_handles_raw_json():
    from strap.verifier import _parse_verdict

    verdict = _parse_verdict('{"pass": true, "confidence": "LOW", "issues": []}')

    assert verdict["pass"] is True
    assert verdict["confidence"] == "LOW"


def test_parse_verdict_handles_fenced_json():
    from strap.verifier import _parse_verdict

    verdict = _parse_verdict(
        "```json\n{\"pass\": false, \"confidence\": \"HIGH\", \"issues\": [\"bad ranking\"]}\n```"
    )

    assert verdict["pass"] is False
    assert verdict["confidence"] == "HIGH"
    assert verdict["issues"] == ["bad ranking"]


def test_parse_verdict_handles_gemini_content_list_with_fenced_json():
    from strap.verifier import _parse_verdict

    verdict = _parse_verdict(
        [
            {
                "type": "text",
                "text": (
                    "```json\n"
                    "{\"pass\": false, \"confidence\": \"HIGH\", "
                    "\"issues\": [\"unsupported solvent selectivity claim\"]}\n"
                    "```"
                ),
            }
        ]
    )

    assert verdict["pass"] is False
    assert verdict["confidence"] == "HIGH"
    assert verdict["issues"] == ["unsupported solvent selectivity claim"]


def test_parse_verdict_fail_open_on_unparseable_text(caplog):
    from strap.verifier import _parse_verdict

    with caplog.at_level(logging.WARNING):
        verdict = _parse_verdict("not valid json")

    assert verdict == {"pass": True}
    assert "fail-open activated" in caplog.text


def test_get_tool_context_uses_recent_tool_messages():
    from strap.verifier import _get_tool_context

    context = _get_tool_context(
        [
            HumanMessage(content="Query"),
            ToolMessage(content="tool output one", tool_call_id="1"),
            AIMessage(content="intermediate"),
            ToolMessage(content="tool output two", tool_call_id="2"),
        ]
    )

    assert "tool output one" in context
    assert "tool output two" in context


def test_verify_includes_tool_context_in_verifier_prompt():
    from strap.verifier import OutputVerifierMiddleware

    verifier_model = MagicMock()
    verifier_model.invoke.return_value = AIMessage(
        content='{"pass": true, "confidence": "LOW", "issues": []}'
    )
    middleware = OutputVerifierMiddleware(verifier_model=verifier_model)

    middleware._verify(
        "Find the best separation sequence and create a heatmap.",
        "Database result: PMMA not found.",
        "Final answer text.",
    )

    messages = verifier_model.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage)
    assert "Do not use external chemistry knowledge" in messages[0].content
    assert "TOOL CONTEXT:\nDatabase result: PMMA not found." in messages[1].content


def test_deterministic_verifier_flags_unsupported_polymer_phase_claim():
    from strap.verifier import _get_deterministic_separation_issues

    messages = [
        HumanMessage(content="Find the optimal separation sequence for PS, PMMA, and PET up to 120C."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PMMA","PET"],'
                '"supported_polymers":["PS","PET"],"unsupported_polymers":["PMMA"],'
                '"best_sequence":["PS","PET"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
    ]

    issues = _get_deterministic_separation_issues(
        messages,
        (
            "PMMA is unsupported by the database, but it remains as an undissolved solid in the residue "
            "while PS dissolves in Toluene."
        ),
    )

    assert any("unsupported polymer PMMA" in issue for issue in issues)


def test_deterministic_verifier_flags_unsupported_polymer_residue_expectation():
    from strap.verifier import _get_deterministic_separation_issues

    messages = [
        HumanMessage(content="Find the optimal separation sequence for PS, PMMA, and PET up to 120C."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PMMA","PET"],'
                '"supported_polymers":["PS","PET"],"unsupported_polymers":["PMMA"],'
                '"best_sequence":["PS","PET"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":105.0}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
    ]

    issues = _get_deterministic_separation_issues(
        messages,
        "PMMA is not modeled, but it is expected to remain in the solid residue with PET.",
    )

    assert any("unsupported polymer PMMA" in issue for issue in issues)


def test_deterministic_verifier_flags_missing_boiling_point_caveat():
    from strap.verifier import _get_deterministic_separation_issues

    messages = [
        HumanMessage(content="Find the optimal separation sequence for PS and PET up to 120C."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                '"best_sequence":["PS","PET"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
    ]

    issues = _get_deterministic_separation_issues(
        messages,
        "Use Toluene at 75C to dissolve PS, then isolate PET.",
    )

    assert any("below its 110.6C boiling point at 1 atm" in issue for issue in issues)


def test_deterministic_verifier_flags_selectivity_overclaim_drift():
    from strap.verifier import _get_deterministic_separation_issues

    messages = [
        HumanMessage(content="Below 90C at atmospheric pressure, can you selectively separate PS from PVC by dissolution?"),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "Based on selectivity-ranking analysis, Toluene is the best predicted/selectivity-based "
                "candidate around 57.5C, but this should be confirmed experimentally before claiming a fully "
                "validated PS/PVC route.\n"
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PVC"],'
                '"best_sequence":["PS","PVC"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":57.5}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PVC"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
            name="task",
        ),
    ]

    issues = _get_deterministic_separation_issues(
        messages,
        (
            "A practical route exists: Toluene will selectively dissolve PS while PVC remains a solid, "
            "and it offers a wide, safe operating window."
        ),
    )

    assert any("preserves an overclaim" in issue for issue in issues)


def test_build_separation_verifier_fallback_for_selectivity_case():
    from strap.verifier import _build_separation_verifier_fallback

    messages = [
        HumanMessage(content="Below 90C at atmospheric pressure, can you selectively separate PS from PVC by dissolution?"),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "Based on selectivity-ranking analysis, Toluene is a predicted/selectivity-based candidate "
                "that still needs experimental confirmation.\n"
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PVC"],'
                '"best_sequence":["PS","PVC"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":57.5}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PVC"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
    ]

    fallback = _build_separation_verifier_fallback(messages)

    assert fallback is not None
    assert "selectivity-based candidate" in fallback
    assert "No fully selective atmospheric-pressure route is established" in fallback
    assert "below the" in fallback


def test_maybe_verify_uses_separation_fallback_for_high_risk_single_specialist_answer():
    from strap.verifier import OutputVerifierMiddleware

    verifier_model = MagicMock()
    verifier_model.invoke.return_value = AIMessage(
        content='{"pass": false, "confidence": "HIGH", "issues": ["unsupported detail"]}'
    )
    middleware = OutputVerifierMiddleware(verifier_model=verifier_model)
    middleware.before_agent(None, None)

    request = MagicMock()
    request.messages = [
        HumanMessage(content="Below 90C at atmospheric pressure, can you selectively separate PS from PVC by dissolution?"),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "Based on selectivity-ranking analysis, Toluene is a predicted/selectivity-based candidate "
                "that still needs experimental confirmation.\n"
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PVC"],'
                '"best_sequence":["PS","PVC"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":57.5}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PVC"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
    ]
    request.system_message = SystemMessage(content="system")
    request.override = MagicMock(side_effect=lambda **kwargs: request)

    response = MagicMock()
    response.result = [AIMessage(content="Toluene will selectively dissolve PS while PVC remains solid with full certainty.")]
    handler = MagicMock(return_value=response)

    result = middleware._maybe_verify(request, response, handler)

    assert handler.call_count == 0
    assert "selectivity-based candidate" in result.result[0].content
    assert result.result[0].additional_kwargs["strap_origin"] == "verifier_separation_fallback"


def test_single_specialist_separation_skips_model_verifier_when_no_deterministic_issue():
    from strap.verifier import OutputVerifierMiddleware

    verifier_model = MagicMock()
    middleware = OutputVerifierMiddleware(verifier_model=verifier_model)
    middleware.before_agent(None, None)

    request = MagicMock()
    request.messages = [
        HumanMessage(content="Only do process design. Find the best separation sequence for PS and PET up to 120C."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PET"],'
                '"best_sequence":["PS","PET"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
    ]
    request.system_message = SystemMessage(content="system")
    request.override = MagicMock(side_effect=lambda **kwargs: request)

    response = MagicMock()
    response.result = [AIMessage(content="Use Toluene at 75C to dissolve PS, then isolate PET while staying below Toluene's boiling point at 1 atm.")]
    handler = MagicMock(return_value=response)

    result = middleware._maybe_verify(request, response, handler)

    verifier_model.invoke.assert_not_called()
    assert result is response


def test_direct_solvent_lookup_skips_model_verifier_without_specialists():
    from strap.verifier import OutputVerifierMiddleware

    verifier_model = MagicMock()
    middleware = OutputVerifierMiddleware(verifier_model=verifier_model)
    middleware.before_agent(None, None)

    request = MagicMock()
    request.messages = [
        HumanMessage(content="i have an LDPE/EVOH/PET feedstock. what are good solvents for dissolving LDPE?"),
        AIMessage(content="", tool_calls=[{"id": "solv", "name": "list_available_solvents", "args": {}}]),
        ToolMessage(content="LDPE solvents include cyclohexane and dodecane.", tool_call_id="solv"),
    ]
    request.system_message = SystemMessage(content="system")
    request.override = MagicMock(side_effect=lambda **kwargs: request)

    response = MagicMock()
    response.result = [
        AIMessage(
            content=(
                "Good LDPE solvent candidates from the available data include cyclohexane "
                "and dodecane; validate experimentally before process design."
            )
        )
    ]
    handler = MagicMock(return_value=response)

    result = middleware._maybe_verify(request, response, handler)

    verifier_model.invoke.assert_not_called()
    handler.assert_not_called()
    assert result is response


def test_direct_route_metadata_skips_verifier():
    from strap.verifier import OutputVerifierMiddleware

    verifier_model = MagicMock()
    middleware = OutputVerifierMiddleware(verifier_model=verifier_model)
    middleware.before_agent(None, None)

    request = MagicMock()
    request.messages = [HumanMessage(content="plot the top 4 of those solvents up to 100C")]
    request.system_message = SystemMessage(content="system")
    request.override = MagicMock(side_effect=lambda **kwargs: request)

    response = MagicMock()
    response.result = [
        AIMessage(
            content="The solubility plot has been generated and saved to /tmp/evoh.png.",
            additional_kwargs={
                "strap_origin": "direct_tool_fast_path",
                "strap_route_decision": {
                    "mode": "artifact_transform",
                    "intent": "solubility_plot",
                    "model_call_budget": 0,
                },
            },
        )
    ]
    handler = MagicMock(return_value=response)

    result = middleware._maybe_verify(request, response, handler)

    verifier_model.invoke.assert_not_called()
    handler.assert_not_called()
    assert result is response


def test_maybe_verify_uses_deterministic_separation_issues_before_model():
    from strap.verifier import OutputVerifierMiddleware

    verifier_model = MagicMock()
    verifier_model.invoke.return_value = AIMessage(
        content='{"pass": true, "confidence": "LOW", "issues": []}'
    )
    middleware = OutputVerifierMiddleware(verifier_model=verifier_model)
    middleware.before_agent(None, None)

    request = MagicMock()
    request.messages = [
        HumanMessage(content="Find the optimal separation sequence for PS, PMMA, and PET up to 120C."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PMMA","PET"],'
                '"supported_polymers":["PS","PET"],"unsupported_polymers":["PMMA"],'
                '"best_sequence":["PS","PET"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"Toluene","temperature_c":75.0}],'
                '"solvent_mapping":{"PS":"Toluene"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PET"],"solvent_mapping":{"PS":"Toluene"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
    ]
    request.system_message = SystemMessage(content="system")
    request.override = MagicMock(side_effect=lambda **kwargs: request)

    response = MagicMock()
    response.result = [
        AIMessage(
            content=(
                "PMMA is unsupported, but it remains as an undissolved solid in the residue."
            )
        )
    ]
    handler = MagicMock(return_value=response)

    result = middleware._maybe_verify(request, response, handler)

    verifier_model.invoke.assert_not_called()
    assert handler.call_count == 0
    assert "Unsupported polymers: PMMA" in result.result[0].content


def test_maybe_verify_allows_second_revision_for_persistent_high_confidence_issue():
    from strap.verifier import OutputVerifierMiddleware

    verifier_model = MagicMock()
    verifier_model.invoke.return_value = AIMessage(
        content='{"pass": true, "confidence": "LOW", "issues": []}'
    )
    middleware = OutputVerifierMiddleware(verifier_model=verifier_model)
    middleware.before_agent(None, None)

    request = MagicMock()
    request.messages = [
        HumanMessage(content="Below 90C at atmospheric pressure, can you selectively separate PS from PVC by dissolution?"),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "Based on selectivity-ranking analysis, THF is a predicted/selectivity-based candidate "
                "that still needs experimental confirmation.\n"
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["PS","PVC"],'
                '"best_sequence":["PS","PVC"],'
                '"steps":[{"step":1,"polymer":"PS","solvent":"THF","temperature_c":60.0}],'
                '"solvent_mapping":{"PS":"THF"},'
                '"top_k_sequences":[{"rank":1,"sequence":["PS","PVC"],"solvent_mapping":{"PS":"THF"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
            name="task",
        ),
    ]
    request.system_message = SystemMessage(content="system")
    request.override = MagicMock(side_effect=lambda **kwargs: request)

    initial_response = MagicMock()
    initial_response.result = [
        AIMessage(content="THF will selectively dissolve PS while PVC remains solid, with selectivity 10.9.")
    ]
    revised_bad_response = MagicMock()
    revised_bad_response.result = [
        AIMessage(content="A practical route exists: THF will dissolve PS and leave PVC solid.")
    ]
    revised_good_response = MagicMock()
    revised_good_response.result = [
        AIMessage(content="THF is the best predicted/selectivity-based candidate at 60C, but this result should be confirmed experimentally before claiming a fully validated PS/PVC route.")
    ]
    handler = MagicMock(side_effect=[revised_bad_response, revised_good_response])

    result = middleware._maybe_verify(request, initial_response, handler)

    assert handler.call_count == 0
    assert "selectivity-based candidate" in result.result[0].content
    verifier_model.invoke.assert_not_called()


def test_deterministic_verifier_flags_missing_optimization_filter_fallback_disclosure():
    from strap.verifier import _get_deterministic_optimization_issues

    messages = [
        HumanMessage(content="Use the separation shortlist, then optimize the waste pathway."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["LDPE","EVOH"],'
                '"steps":[{"step":1,"polymer":"LDPE","solvent":"Cyclohexane"},'
                '{"step":2,"polymer":"EVOH","solvent":"isopropylamine"}],'
                '"solvent_mapping":{"LDPE":"Cyclohexane","EVOH":"isopropylamine"},'
                '"top_solvents":["Cyclohexane","isopropylamine"],'
                '"top_k_sequences":[]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
        AIMessage(content="", tool_calls=[{
            "id": "tc_opt",
            "name": "task",
            "args": {"subagent_type": "optimization-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"optimization-engineer","schema_version":"1.1",'
                '"profit":18660000,"emissions":6814,"ce_score":0.83,'
                '"optimal_washes":["PE-Cyclohexane","EVOH-gamma-butyrolactone"],'
                '"solvent_filter_status":"partially_applied_with_fallback",'
                '"requested_solvent_filters":{"global":["Cyclohexane","isopropylamine"],'
                '"PE":["Cyclohexane"],"EVOH":["isopropylamine"]},'
                '"applied_solvent_filters":{"PE":["Cyclohexane"]},'
                '"solvent_filter_warnings":["No EVOH solvent overlap between upstream shortlist and the shared optimization catalog; falling back to the full EVOH candidate set."]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_opt",
        ),
    ]

    issues = _get_deterministic_optimization_issues(
        messages,
        "The optimizer used the upstream shortlist and selected Cyclohexane plus gamma-butyrolactone.",
    )

    assert any("fell back to the broader candidate set" in issue for issue in issues)


def test_deterministic_verifier_allows_explicit_optimization_filter_fallback_disclosure():
    from strap.verifier import _get_deterministic_optimization_issues

    messages = [
        HumanMessage(content="Use the separation shortlist, then optimize the waste pathway."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_opt",
            "name": "task",
            "args": {"subagent_type": "optimization-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"optimization-engineer","schema_version":"1.1",'
                '"profit":18660000,"emissions":6814,"ce_score":0.83,'
                '"optimal_washes":["PE-Cyclohexane","EVOH-gamma-butyrolactone"],'
                '"solvent_filter_status":"partially_applied_with_fallback",'
                '"requested_solvent_filters":{"global":["Cyclohexane","isopropylamine"],'
                '"PE":["Cyclohexane"],"EVOH":["isopropylamine"]},'
                '"applied_solvent_filters":{"PE":["Cyclohexane"]},'
                '"solvent_filter_warnings":["No EVOH solvent overlap between upstream shortlist and the shared optimization catalog; falling back to the full EVOH candidate set."]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_opt",
        ),
    ]

    issues = _get_deterministic_optimization_issues(
        messages,
        "Cyclohexane was applied on the PE side, but the EVOH shortlist had no overlap with the optimization catalog so the solve fell back to the full EVOH candidate set.",
    )

    assert issues == []


def test_deterministic_verifier_fires_on_pareto_nested_candidate_summary_warnings():
    """Pareto payloads that only expose warnings under candidate_summary must still trip the verifier."""
    from strap.verifier import _get_deterministic_optimization_issues

    messages = [
        HumanMessage(content="Run a Pareto sweep on the shortlisted solvents."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_opt",
            "name": "task",
            "args": {"subagent_type": "optimization-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"optimization-engineer","schema_version":"1.2",'
                '"analysis_type":"pareto_front","x_metric":"total_cost","y_metric":"emissions",'
                '"n_points_feasible":3,"points":[],'
                '"candidate_summary":{"status":"fallback_to_full_catalog",'
                '"requested_filters":{"PE":["Cyclohexane"]},'
                '"applied_filters":{},'
                '"warnings":["No PE solvent overlap; falling back to full PE set."]}}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_opt",
        ),
    ]

    issues = _get_deterministic_optimization_issues(
        messages,
        "The Pareto sweep used the requested shortlist and produced three feasible points.",
    )

    assert any("fell back to the broader candidate set" in issue for issue in issues)


def test_verifier_helpers_hash_and_caveat():
    """Unit coverage for the stagnation hash and the caveat-append escape hatch."""
    from strap.verifier import _hash_issues, _response_with_caveat

    # Hash is stable and whitespace-normalized
    h1 = _hash_issues(["Issue A", "Issue B"])
    h2 = _hash_issues(["Issue B", "Issue A"])  # order-independent
    h3 = _hash_issues(["Issue A  ", "Issue   B"])  # whitespace-normalized
    h4 = _hash_issues(["Issue A", "Issue C"])
    assert h1 == h2 == h3
    assert h1 != h4

    # Empty issues returns empty string
    assert _hash_issues([]) == ""
    assert _hash_issues(None) == ""

    # Caveat is appended to AI content and additional_kwargs carries the issue list
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage

    original = AIMessage(content="Here is my answer.", additional_kwargs={"strap_origin": "test"})
    response = SimpleNamespace(result=[original])
    revised = _response_with_caveat(response, original, ["Unresolved issue 1", "Unresolved issue 2"])
    revised_msg = revised.result[0]
    assert "Unresolved issue 1" in revised_msg.content
    assert "Unresolved issue 2" in revised_msg.content
    assert revised_msg.additional_kwargs.get("strap_verifier_unresolved_issues") == [
        "Unresolved issue 1",
        "Unresolved issue 2",
    ]
    # Original origin tag preserved
    assert revised_msg.additional_kwargs.get("strap_origin") == "test"
    # Original content preserved (caveat is appended, not replaced)
    assert revised_msg.content.startswith("Here is my answer.")


def test_deterministic_verifier_fires_on_simulation_failures():
    """Dropped candidate pairs from BioSTEAM failures require explicit disclosure."""
    from strap.verifier import _get_deterministic_optimization_issues

    messages = [
        HumanMessage(content="Run the optimization."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_opt",
            "name": "task",
            "args": {"subagent_type": "optimization-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"optimization-engineer","schema_version":"1.0",'
                '"analysis_type":"point_optimum",'
                '"profit":1000,"emissions":100,"total_cost":500,'
                '"optimal_washes":["PE-Heptane"],'
                '"solvent_filter_status":"not_requested",'
                '"requested_solvent_filters":{},'
                '"applied_solvent_filters":{},'
                '"solvent_filter_warnings":[],'
                '"simulation_failures":[{"polymer":"EVOH","solvent":"Pyridazine"}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_opt",
        ),
    ]

    issues = _get_deterministic_optimization_issues(
        messages,
        "The optimizer found the optimum at Heptane for PE with total cost $500.",
    )

    assert any("BioSTEAM simulation failed" in issue for issue in issues)


def test_deterministic_verifier_blocks_upstream_sequence_prose_after_route_optimization():
    from strap.verifier import _get_deterministic_optimization_issues

    messages = [
        HumanMessage(content="Optimize the shortlisted routes."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_sep",
            "name": "task",
            "args": {"subagent_type": "separation-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"separation-engineer","schema_version":"1.0","polymers":["LDPE","EVOH"],'
                '"best_sequence":["LDPE","EVOH"],'
                '"steps":[{"step":1,"polymer":"LDPE","solvent":"Cyclohexane","temperature_c":76.0}],'
                '"solvent_mapping":{"LDPE":"Cyclohexane"},'
                '"top_k_sequences":[{"rank":1,"sequence":["LDPE","EVOH"],"solvent_mapping":{"LDPE":"Cyclohexane","EVOH":"Methanol"}}]}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_sep",
        ),
        AIMessage(content="", tool_calls=[{
            "id": "tc_opt",
            "name": "task",
            "args": {"subagent_type": "optimization-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"optimization-engineer","schema_version":"1.3","analysis_type":"pareto_front",'
                '"x_metric":"total_cost","y_metric":"emissions","n_points_feasible":1,'
                '"points":[{"point_id":1,"route_id":"route_1","total_cost":1000,"emissions":100}],'
                '"route_reports":[{"route_id":"route_1","status":"solved","polymer_solvent_map":{"PE":"Cyclohexane","EVOH":"Dimethyl sulfoxide"}}],'
                '"candidate_summary":{"status":"applied","warnings":[]}}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_opt",
        ),
    ]

    issues = _get_deterministic_optimization_issues(
        messages,
        "Recommended separation sequence: LDPE then EVOH. Step 1 uses Cyclohexane and Step 2 uses Methanol.",
    )

    assert any("route reports and Pareto points" in issue for issue in issues)
    assert any("does not appear in the validated optimization route reports" in issue for issue in issues)


def test_deterministic_verifier_requires_explicit_infeasible_optimization_language():
    from strap.verifier import _get_deterministic_optimization_issues

    messages = [
        HumanMessage(content="Optimize the shortlisted routes."),
        AIMessage(content="", tool_calls=[{
            "id": "tc_opt",
            "name": "task",
            "args": {"subagent_type": "optimization-engineer"},
        }]),
        ToolMessage(
            content=(
                "<STRUCTURED_RESULT>"
                '{"agent":"optimization-engineer","schema_version":"1.3","analysis_type":"infeasible",'
                '"constraint_mode":"hard","fallback_policy":"fail_closed",'
                '"failure_reason":"no_candidate_pair_overlap",'
                '"message":"No enforced route could be solved.",'
                '"requested_candidate_pairs":[],"applied_candidate_pairs":[],'
                '"suggested_relaxation":"Switch to ranked_soft."}'
                "</STRUCTURED_RESULT>"
            ),
            tool_call_id="tc_opt",
        ),
    ]

    issues = _get_deterministic_optimization_issues(
        messages,
        "Recommended separation sequence: LDPE then EVOH. Step 1 uses Cyclohexane.",
    )

    assert any("report the optimization as infeasible" in issue for issue in issues)
    assert any("explicitly state that the optimization result was infeasible" in issue for issue in issues)
