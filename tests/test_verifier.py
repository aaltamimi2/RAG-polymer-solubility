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
