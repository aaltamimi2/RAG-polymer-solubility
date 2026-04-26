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


def test_extract_query_context_captures_optimization_request_labels():
    query = (
        "Optimize waste management for an 8000 t/y multilayer feed of 40% PE, 40% PET, "
        "1% Nylon-6, and 19% EVOH. Maximize profit and report emissions."
    )

    context = extract_query_context(query)

    assert context.polymers == ("PE", "PET", "NYLON6", "EVOH")
    assert "optimization.pathway" in context.request_labels
    assert "user.optimization_request" in context.available_inputs


def test_extract_query_context_captures_plural_separation_route_phrasing():
    query = (
        "For an LDPE/EVOH/PET film, use the top separation routes as candidates, "
        "run route-constrained optimization, and generate a Pareto front plot."
    )

    context = extract_query_context(query)

    assert "separation.route" in context.route_labels
    assert "user.solvents_or_route" in context.available_inputs


def test_extract_query_context_captures_feed_composition_and_capacity():
    query = (
        "Optimize waste management for a mixed plastic feedstock of 8000 tonnes/year composed of "
        "5% LDPE, 5% EVOH, and 90% PET."
    )

    context = extract_query_context(query)

    assert context.feed_capacity_tpy == 8000.0
    assert context.feed_composition == {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9}
    assert "user.feed_composition" in context.available_inputs
    assert "user.feed_capacity" in context.available_inputs


def test_extract_query_context_captures_slash_delimited_composition_slices():
    query = (
        "For mixed LDPE/EVOH/PET feedstocks at 8000 tonnes/year, run Pareto slices "
        "for five fixed feed compositions: 20/60/20, 34/33/33, 60/20/20, "
        "20/20/60, and 5/5/90."
    )

    context = extract_query_context(query)

    assert context.feed_capacity_tpy == 8000.0
    assert context.feed_composition_slices == (
        {"LDPE": 0.2, "EVOH": 0.6, "PET": 0.2},
        {"LDPE": 0.34, "EVOH": 0.33, "PET": 0.33},
        {"LDPE": 0.6, "EVOH": 0.2, "PET": 0.2},
        {"LDPE": 0.2, "EVOH": 0.2, "PET": 0.6},
        {"LDPE": 0.05, "EVOH": 0.05, "PET": 0.9},
    )
    assert "user.feed_composition" in context.available_inputs
