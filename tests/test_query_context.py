from __future__ import annotations

from strap.query_context import extract_query_context


def test_extract_query_context_captures_process_entities_and_route_spans():
    query = (
        "Find an optimal separation sequence for an HDPE/EVOH mixed waste stream using "
        "selective dissolution at atmospheric pressure. Propose up to 1 additional wash "
        "step for phthalate removal. Then run a techno-economic analysis on solvent recovery."
    )

    context = extract_query_context(query)

    assert context.polymers == ("HDPE", "EVOH")
    assert context.contaminant_families == ("Phthalates",)
    assert {
        "separation.route",
        "route.atmospheric_pressure",
        "route.wash_step",
        "route.solvent_recovery",
    }.issubset(set(context.route_labels))
    for span in (*context.polymer_spans, *context.contaminant_family_spans, *context.route_spans):
        assert query[span.start:span.end] == span.text


def test_extract_query_context_resolves_contaminant_abbreviations():
    query = "For EVOH contaminated with DBP, compare leaching versus STRAP contaminant removal."

    context = extract_query_context(query)

    assert context.polymers == ("EVOH",)
    assert context.contaminants == ("di-n-butyl phthalate (DBP)",)
    assert "user.contaminants" in context.available_inputs


def test_extract_query_context_tracks_solvent_entities_for_route_inputs():
    query = "Compare safety of toluene and THF for PS dissolution."

    context = extract_query_context(query)

    assert context.polymers == ("PS",)
    assert context.solvents == ("toluene", "thf")
    assert "user.solvents_or_route" in context.available_inputs
    assert "user.target_polymer" in context.available_inputs


def test_extract_query_context_does_not_treat_generic_polymer_word_as_entity():
    query = "Do a literature search for multilayer polymer recycling methods, then create a chart summarizing the papers."

    context = extract_query_context(query)

    assert context.polymers == ()
    assert "user.polymers" not in context.available_inputs
    assert "user.research_question" in context.available_inputs
    assert "user.visualization_request" in context.available_inputs


def test_extract_query_context_ignores_ambiguous_tea_as_solvent_alias():
    query = "Then run a TEA on solvent recovery for the best option."

    context = extract_query_context(query)

    assert context.solvents == ()


def test_extract_query_context_prefers_specific_thermal_request_label():
    query = "Provide a thermal prediction for PET glass transition behavior."

    context = extract_query_context(query)

    assert "thermal.prediction" in context.request_labels


def test_extract_query_context_does_not_leak_route_goals_from_research_topic_text():
    query = (
        "Do a literature search and patent search for mixed-plastic solvent recovery process design. "
        "Answer the question with RAG, then create a chart visualization of the retrieved findings."
    )

    context = extract_query_context(query)

    assert "separation.route" not in context.route_labels
    assert {"literature.search", "patent.search", "literature.answer", "visualization.plot"}.issubset(
        set(context.request_labels)
    )
