from __future__ import annotations

from types import SimpleNamespace

from strap.services import advanced_separation_service as service


def _step(**overrides):
    data = {
        "target_polymer": "PS",
        "remaining_polymers": ["PE"],
        "is_viable": True,
        "step_number": 1,
        "solvent": "toluene",
        "temperature": 120.0,
        "selectivity": 22.5,
        "target_solubility": 80.0,
        "max_other_solubility": 57.5,
        "safety_score": 7.2,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _result(*, all_sequences: bool = False):
    steps = [_step(), _step(target_polymer="PE", remaining_polymers=[], step_number=2)]
    sequence = SimpleNamespace(
        steps=steps,
        status=SimpleNamespace(value="viable"),
        min_selectivity=22.5,
        avg_selectivity=22.5,
        unique_solvents={"toluene"},
    )
    result = SimpleNamespace(
        best_sequence=sequence,
        algorithm="dp",
        computation_time_ms=12.3,
        nodes_explored=7,
    )
    if all_sequences:
        result.all_sequences = [sequence]
    return result


def test_parse_polymer_list_normalizes_case():
    assert service.parse_polymer_list(" ps, pe , EVOH ") == ["PS", "PE", "EVOH"]


def test_parse_solvent_list_handles_empty_input():
    assert service.parse_solvent_list("") is None
    assert service.parse_solvent_list(" toluene, hexane ") == ["toluene", "hexane"]


def test_run_async_executes_coroutine():
    async def _value():
        return 42

    assert service.run_async(_value()) == 42


def test_format_top_k_results_includes_rank_detail():
    rendered = service.format_top_k_results(_result(all_sequences=True), "PS, PE")
    assert "# Top Separation Sequences" in rendered
    assert "Rank 1 Detail" in rendered
    assert "PS -> PE" in rendered


def test_build_solvent_ranking_report_includes_notes():
    scores = [
        SimpleNamespace(
            solvent="toluene",
            overall_score=0.91,
            selectivity_score=0.82,
            bp_score=0.73,
            logp_score=0.64,
            cp_score=0.55,
            energy_score=0.46,
            notes=["good balance", "widely available"],
        )
    ]

    rendered = service.build_solvent_ranking_report(
        scores,
        target_polymer="PS",
        other_polymers=["PE"],
        temperature=120.0,
    )

    assert "# Solvent Ranking" in rendered
    assert "| 1 | toluene |" in rendered
    assert "Notes for toluene" in rendered


def test_build_compatibility_matrix_report_renders_table():
    rendered = service.build_compatibility_matrix_report(
        {"PS": {"toluene": 88.0}, "PE": {"hexane": 65.0}},
        polymers=["PS", "PE"],
        temperature=100.0,
    )

    assert "# Polymer-Solvent Compatibility Matrix" in rendered
    assert "| Polymer |" in rendered
    assert "| PS |" in rendered
    assert "88.0%" in rendered


def test_build_challenging_pairs_report_handles_empty_and_populated():
    empty = service.build_challenging_pairs_report(
        [],
        polymers=["PS", "PE"],
        temperature=100.0,
        selectivity_threshold=10.0,
    )
    assert "No challenging pairs found" in empty

    populated = service.build_challenging_pairs_report(
        [("PS", "PMMA", 4.2)],
        polymers=["PS", "PMMA"],
        temperature=100.0,
        selectivity_threshold=10.0,
    )
    assert "# Challenging Polymer Pairs" in populated
    assert "(CRITICAL)" in populated


def test_score_separation_sequences_prefers_highest_min_selectivity(monkeypatch):
    scores = {
        ("PS", ("PE",)): [{"solvent": "toluene", "selectivity": 22.5, "target_sol": 80, "max_other": 57.5}],
        ("PE", ("PS",)): [{"solvent": "hexane", "selectivity": 8.0, "target_sol": 70, "max_other": 62.0}],
    }

    def _fake_find_top(target, remaining, *, temperature, k=3):
        return scores[(target, tuple(remaining))]

    monkeypatch.setattr(service, "_find_top_sequence_solvents", _fake_find_top)

    ranked = service.score_separation_sequences(["PS", "PE"], temperature=120.0)

    assert ranked[0]["sequence"] == ["PS", "PE"]
    assert ranked[0]["min_selectivity"] == 22.5
    assert ranked[1]["sequence"] == ["PE", "PS"]


def test_build_separation_tree_report_includes_plots():
    rendered = service.build_separation_tree_report(
        polymer_list=["PS", "PE"],
        sequence_scores=[{"sequence": ["PS", "PE"], "min_selectivity": 22.5, "steps": []}],
        temperature=120.0,
        rank1_plot="/tmp/rank1.png",
        topk_plot="/tmp/topk.png",
        plot_url_builder=lambda path: f"url:{path}",
    )

    assert "url:/tmp/rank1.png" in rendered
    assert "url:/tmp/topk.png" in rendered
    assert "**Best sequence:** PS -> PE" in rendered


def test_build_selectivity_heatmap_report_tracks_requested_solvents():
    rendered = service.build_selectivity_heatmap_report(
        filepath="/tmp/heatmap.png",
        polymer_list=["PS", "PE"],
        solvent_list=["toluene", "hexane"],
        temperature=90.0,
        matrix={"PS": {"toluene": 88.0}, "PE": {"hexane": 55.0}},
    )

    assert "Plot saved to" in rendered
    assert "**Requested solvents:** toluene, hexane" in rendered
    assert "**Matrix solvents:** 2" in rendered


def test_build_process_flow_report_summarizes_steps():
    result = _result()
    rendered = service.build_process_flow_report(
        filepath="/tmp/pfd.png",
        polymer_list=["PS", "PE"],
        result=result,
    )

    assert "# Process Flow Diagram" in rendered
    assert "Feed:** PS, PE" in rendered
    assert "Solvents Used:** toluene" in rendered


def test_plot_helpers_delegate_to_save_plot(monkeypatch):
    saved = []

    def _fake_save_plot(fig, filename, backend=None, **kwargs):
        saved.append((filename, backend, kwargs))
        return f"/tmp/{filename}.png"

    monkeypatch.setattr(service, "save_plot", _fake_save_plot)

    sequence_data = {
        "sequence": ["PS", "PE"],
        "min_selectivity": 22.5,
        "steps": [
            {
                "target": "PS",
                "remaining": ["PE"],
                "solvents": [
                    {
                        "solvent": "toluene",
                        "selectivity": 22.5,
                        "temperature": 120.0,
                        "optimal_temp": 120.0,
                        "optimal_selectivity": 22.5,
                    }
                ],
            }
        ],
    }

    path_one = service.plot_separation_sequence(["PS", "PE"], sequence_data, 120.0, total_sequences=1)
    path_two = service.plot_topk_comparison(["PS", "PE"], [sequence_data], 120.0)

    assert path_one.endswith("separation_sequence_rank1.png")
    assert path_two.endswith("separation_topk_comparison.png")
    assert [item[0] for item in saved] == ["separation_sequence_rank1", "separation_topk_comparison"]
