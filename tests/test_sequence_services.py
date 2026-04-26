import json


def test_build_greedy_planning_payload_includes_top_k_sequences():
    from strap.services.sequence_planning_payload_service import build_greedy_planning_payload

    payload = build_greedy_planning_payload(
        polymer_list=["A", "B"],
        temperature=60.0,
        sequence=["A", "B"],
        steps=[
            {"step": 1, "target": "A", "solvent": "S1", "selectivity": 20.0},
        ],
    )

    assert payload["tool_name"] == "plan_sequential_separation"
    assert payload["algorithm_used"] == "greedy"
    assert payload["top_k_sequences"][0]["solvent_mapping"] == {"A": "S1"}


def test_build_sequential_planning_payload_aggregates_candidates_for_final_residue():
    from strap.services.sequence_planning_payload_service import build_sequential_planning_payload

    payload = build_sequential_planning_payload(
        polymer_list=["LDPE", "EVOH", "PET"],
        temperature=120.0,
        excluded_set=set(),
        sequence_scores=[
            {
                "sequence": ("LDPE", "EVOH", "PET"),
                "min_selectivity": 20.0,
                "steps": [
                    {
                        "step": 1,
                        "target": "LDPE",
                        "solvents": [
                            {"solvent": "Cyclohexane", "selectivity": 40.0, "optimal_temp": 81.0}
                        ],
                    },
                    {
                        "step": 2,
                        "target": "EVOH",
                        "solvents": [
                            {"solvent": "Methanol", "selectivity": 20.0, "optimal_temp": 65.0}
                        ],
                    },
                ],
            },
            {
                "sequence": ("PET", "LDPE", "EVOH"),
                "min_selectivity": 10.0,
                "steps": [
                    {
                        "step": 1,
                        "target": "PET",
                        "solvents": [
                            {"solvent": "N,N-Dimethylformamide", "selectivity": 18.0, "optimal_temp": 153.0},
                            {"solvent": "Dimethyl sulfoxide", "selectivity": 16.0, "optimal_temp": 160.0},
                        ],
                    }
                ],
            },
        ],
    )

    candidates = payload["polymer_solvent_candidates"]
    assert candidates["LDPE"][0]["solvent"] == "Cyclohexane"
    assert candidates["EVOH"][0]["temperature_c"] == 65.0
    assert [entry["solvent"] for entry in candidates["PET"]] == [
        "N,N-Dimethylformamide",
        "Dimethyl sulfoxide",
    ]
    assert payload["top_k_sequences"][1]["steps"][0]["temperature_c"] == 153.0


def test_sequential_planning_payload_canonicalizes_solvent_aliases():
    from strap.services.sequence_planning_payload_service import build_sequential_planning_payload

    payload = build_sequential_planning_payload(
        polymer_list=["LDPE", "EVOH", "PET"],
        temperature=120.0,
        excluded_set=set(),
        sequence_scores=[
            {
                "sequence": ("LDPE", "EVOH", "PET"),
                "min_selectivity": 20.0,
                "steps": [
                    {
                        "step": 1,
                        "target": "LDPE",
                        "solvents": [
                            {"solvent": "THP", "selectivity": 40.0, "optimal_temp": 87.0},
                            {"solvent": "GVL", "selectivity": 32.0, "optimal_temp": 25.0},
                        ],
                    },
                    {
                        "step": 2,
                        "target": "EVOH",
                        "solvents": [
                            {"solvent": "Glycol", "selectivity": 20.0, "optimal_temp": 160.0},
                            {"solvent": "Propyleneglycol", "selectivity": 18.0, "optimal_temp": 160.0},
                        ],
                    },
                ],
            },
            {
                "sequence": ("PET", "LDPE", "EVOH"),
                "min_selectivity": 10.0,
                "steps": [
                    {
                        "step": 1,
                        "target": "PET",
                        "solvents": [
                            {"solvent": "CH\u2082Cl\u2082", "selectivity": 18.0, "optimal_temp": 39.0},
                        ],
                    }
                ],
            },
        ],
    )

    candidates = payload["polymer_solvent_candidates"]
    assert candidates["LDPE"][0]["solvent"] == "Tetrahydropyran"
    assert candidates["LDPE"][1]["solvent"] == "gamma-Valerolactone"
    assert candidates["EVOH"][0]["solvent"] == "Ethylene glycol"
    assert candidates["EVOH"][1]["solvent"] == "Propylene glycol"
    assert candidates["PET"][0]["solvent"] == "Dichloromethane"
    assert payload["steps"][0]["solvent"] == "Tetrahydropyran"
    assert payload["top_k_sequences"][1]["solvent_mapping"]["PET"] == "Dichloromethane"
    assert payload["top_k_sequences"][1]["steps"][0]["source_solvent"] == "CH\u2082Cl\u2082"


def test_sequence_analysis_display_canonicalizes_solvent_aliases():
    from strap.services.sequence_planning_exhaustive_display_service import build_sequence_analysis_output

    rendered = "\n".join(
        build_sequence_analysis_output(
            sequence=("LDPE", "PET"),
            seq_idx=1,
            seq_steps=[
                {
                    "step": 1,
                    "target": "LDPE",
                    "remaining": ["PET"],
                    "solvents": [
                        {
                            "solvent": "THP",
                            "selectivity": 25.0,
                            "target_sol": 30.0,
                            "max_other": 5.0,
                        },
                        {
                            "solvent": "CH\u2082Cl\u2082",
                            "selectivity": 20.0,
                            "target_sol": 28.0,
                            "max_other": 8.0,
                        },
                    ],
                }
            ],
        )
    )

    assert "Tetrahydropyran" in rendered
    assert "Dichloromethane" in rendered
    assert "**THP**" not in rendered
    assert "CH\u2082Cl\u2082" not in rendered


def test_build_multi_scheme_display_renders_scheme_table():
    from strap.services.sequence_planning_greedy_display_service import build_multi_scheme_display

    rendered = build_multi_scheme_display(
        polymer_list=["A", "B", "C"],
        temperature=80.0,
        n_variants=2,
        schemes=[
            {
                "tag": "SEL-v1",
                "name": "Max Selectivity",
                "seq": ["A", "B", "C"],
                "min_sel": 12.0,
                "avg_sel": 18.0,
                "n_solv": 2,
                "steps": [
                    {"step": 1, "target": "A", "solvent": "S1", "sel": 20.0, "temp": 75.0, "gsk": 7.0, "logp": 1.2},
                    {"step": 2, "target": "B", "solvent": "-", "sel": None, "temp": None, "gsk": None, "logp": None},
                ],
            }
        ],
    )

    assert "MULTI-SCHEME SEPARATION" in rendered
    assert "SEL-v1" in rendered
    assert "Max Selectivity" in rendered


def test_build_sequential_planning_display_mentions_alternatives():
    from strap.services.sequence_planning_exhaustive_display_service import build_sequential_planning_display

    rendered = build_sequential_planning_display(
        polymer_list=["A", "B"],
        temperature=60.0,
        top_k_solvents=2,
        excluded_set=set(),
        all_sequences=[("A", "B"), ("B", "A")],
        sequence_results=[
            {"output": ["Sequence body"]},
        ],
        sequence_scores=[
            {"sequence": ("A", "B"), "min_selectivity": 20.0},
            {"sequence": ("B", "A"), "min_selectivity": 10.0},
        ],
        rank1_plot_url="https://example.com/rank1.png",
        topk_plot_url="https://example.com/topk.png",
        visualization_errors=[],
    )

    assert "Top Recommended Separation Sequence" in rendered
    assert "Alternative sequences available" in rendered
    assert "https://example.com/topk.png" in rendered


def test_build_sequence_analysis_output_mentions_solvent_diversity():
    from strap.services.sequence_planning_exhaustive_display_service import build_sequence_analysis_output

    rendered = build_sequence_analysis_output(
        sequence=("A", "B"),
        seq_idx=1,
        seq_steps=[
            {
                "step": 1,
                "target": "A",
                "remaining": ["B"],
                "solvents": [{"solvent": "S1", "selectivity": 20.0, "target_sol": 80.0, "max_other": 60.0}],
            }
        ],
    )

    assert any("Solvent Diversity" in line for line in rendered)


def test_build_integrated_analysis_display_renders_summary_and_visualisation():
    from strap.services.sequence_analysis_service import build_integrated_analysis_display

    all_results = [
        {
            "sequence": ("PS", "PET"),
            "min_selectivity": 18.5,
            "steps": [
                {
                    "step": 1,
                    "target": "PS",
                    "remaining": ["PET"],
                    "best": {
                        "solvent": "THF",
                        "temperature": 55.0,
                        "selectivity": 18.5,
                        "target_sol": 80.0,
                        "max_other": 61.5,
                        "bp": 66.0,
                    },
                },
                {
                    "step": 2,
                    "target": "PET",
                    "remaining": [],
                    "best": {
                        "solvent": "N/A",
                        "temperature": 0.0,
                        "selectivity": float("inf"),
                        "note": "Isolated",
                    },
                },
            ],
        }
    ]

    rendered = build_integrated_analysis_display(
        polymer_list=["PS", "PET"],
        rank_by="selectivity",
        temperature_min=25.0,
        temperature_max=120.0,
        available_temps=[25.0, 30.0, 35.0],
        all_results=all_results,
        plot_url="https://example.com/plot.png",
        visualization_error=None,
        used_greedy=False,
    )

    assert "Integrated Multi-Polymer Separation Analysis" in rendered
    assert "Best Sequence:** PS -> PET" in rendered
    assert "![Separation Sequence](https://example.com/plot.png)" in rendered
    assert "THF" in rendered


def test_build_alternative_sequence_display_includes_best_comparison():
    from strap.services.sequence_analysis_service import build_alternative_sequence_display

    sequence_scores = [
        {
            "sequence": ("PS", "PET"),
            "min_selectivity": 20.0,
            "steps": [],
        },
        {
            "sequence": ("PET", "PS"),
            "min_selectivity": 12.5,
            "steps": [
                {
                    "step": 1,
                    "target": "PET",
                    "remaining": ["PS"],
                    "solvents": [{"solvent": "DMF", "selectivity": 12.5}],
                }
            ],
        },
    ]

    rendered = build_alternative_sequence_display(
        polymer_list=["PS", "PET"],
        target_sequence=sequence_scores[1],
        sequence_scores=sequence_scores,
        rank=2,
        temperature=80.0,
        plot_url=None,
        visualization_error="plot failed",
    )

    assert "Alternative Separation Sequence (Rank #2)" in rendered
    assert "Could not create visualisation: plot failed" in rendered
    assert "Comparison to Best Sequence" in rendered
    assert "PET -> PS" in rendered


def test_select_alternative_sequence_by_rank():
    from strap.services.sequence_analysis_service import select_alternative_sequence

    sequence_scores = [
        {"sequence": ("PS", "PET"), "min_selectivity": 20.0, "steps": []},
        {"sequence": ("PET", "PS"), "min_selectivity": 12.5, "steps": []},
    ]

    target_sequence, rank, error = select_alternative_sequence(
        sequence_scores=sequence_scores,
        sequence_rank=2,
        starting_polymer=None,
        polymer_list=["PS", "PET"],
        n_polymers=2,
        max_exhaustive=6,
    )

    assert error is None
    assert rank == 2
    assert target_sequence["sequence"] == ("PET", "PS")


def test_select_alternative_sequence_missing_selector_returns_error():
    from strap.services.sequence_analysis_service import select_alternative_sequence

    target_sequence, rank, error = select_alternative_sequence(
        sequence_scores=[],
        sequence_rank=None,
        starting_polymer=None,
        polymer_list=["PS", "PET"],
        n_polymers=2,
        max_exhaustive=6,
    )

    assert target_sequence is None
    assert rank is None
    assert error["error_code"] == "missing_sequence_selector"


def test_plot_integrated_separation_analysis_returns_saved_filepath(monkeypatch, tmp_path):
    from strap.services import sequence_analysis_plot_service

    expected = tmp_path / "integrated.png"
    monkeypatch.setattr(
        sequence_analysis_plot_service,
        "save_plot",
        lambda fig, name: str(expected),
    )

    filepath = sequence_analysis_plot_service.plot_integrated_separation_analysis(
        polymer_list=["PS", "PET"],
        best_result={
            "sequence": ("PS", "PET"),
            "min_selectivity": 18.5,
            "steps": [
                {
                    "step": 1,
                    "target": "PS",
                    "remaining": ["PET"],
                    "best": {
                        "solvent": "THF",
                        "temperature": 55.0,
                        "selectivity": 18.5,
                        "g_score": 7.0,
                        "bp": 66.0,
                    },
                },
                {
                    "step": 2,
                    "target": "PET",
                    "remaining": [],
                    "best": {"solvent": "N/A", "temperature": 0.0, "selectivity": float("inf")},
                },
            ],
        },
        rank_by="selectivity",
    )

    assert filepath == str(expected)


def test_plan_sequential_separation_two_polymer_success(monkeypatch):
    from strap.tools import sequence_planning_tools
    import strap.solubility as solubility

    def fake_get_solubility(polymer, solvent, temperature):
        values = {
            ("A", "S1"): 80.0,
            ("B", "S1"): 10.0,
            ("A", "S2"): 12.0,
            ("B", "S2"): 70.0,
        }
        return values.get((polymer, solvent), 0.0)

    monkeypatch.setattr(solubility, "get_solubility", fake_get_solubility)
    monkeypatch.setattr(solubility, "get_available_solvents_for_polymer", lambda _polymer: ["S1", "S2"])
    monkeypatch.setattr(sequence_planning_tools, "_get_solvent_table_name", lambda: None)

    parsed = json.loads(
        sequence_planning_tools.plan_sequential_separation(
            "A,B",
            temperature=60.0,
            top_k_solvents=2,
            create_decision_tree=False,
        )
    )

    assert parsed["data"]["tool_name"] == "plan_sequential_separation"
    assert parsed["data"]["success"] is True
    assert parsed["data"]["best_sequence"] in (["A", "B"], ["B", "A"])
    assert parsed["data"]["min_selectivity"] >= 58.0
