from __future__ import annotations

import pytest

from strap.planning.capability_registry import exported_tool_names, subagent_names
from strap.planning.query_bank import (
    DEFAULT_QUERY_BANK_PATH,
    load_query_bank,
    normalize_role_from_sheet,
    validated_query_bank_rows,
)


def _require_query_bank() -> None:
    if not DEFAULT_QUERY_BANK_PATH.exists():
        pytest.skip(f"query bank is not present at {DEFAULT_QUERY_BANK_PATH}")


def test_query_bank_loader_reads_validated_rows():
    _require_query_bank()

    rows = load_query_bank()
    validated = [row for row in rows if row.is_validated]
    p0 = [row for row in validated if row.is_p0]

    assert rows
    assert validated
    assert p0
    assert any(row.role == "safety-analyst" for row in validated)
    assert any(row.role == "statistics-ml" for row in validated)
    assert any(row.role == "optimization-engineer" for row in validated)


def test_query_bank_roles_normalize_to_known_subagents():
    _require_query_bank()

    known_roles = set(subagent_names())
    rows = load_query_bank()

    assert normalize_role_from_sheet("09 contaminant-removal") == "contaminant-removal-analyst"
    assert {row.role for row in rows} <= known_roles


def test_validated_p0_query_bank_rows_reference_registered_tools_when_named():
    _require_query_bank()

    exported = exported_tool_names()
    rows = validated_query_bank_rows(priority="P0")
    unmatched = [
        f"{row.sheet_name}:{row.row_number}:{row.expected_tools_or_handoffs}"
        for row in rows
        if row.expected_tools_or_handoffs and not row.expected_tool_names(exported)
    ]

    assert not unmatched


def test_query_bank_expected_tool_extraction_uses_exact_tool_names():
    _require_query_bank()

    exported = exported_tool_names()
    safety_row = next(
        row for row in validated_query_bank_rows(priority="P0")
        if row.query == "Show a safety card for THF at 60 C."
    )
    optimization_row = next(
        row for row in validated_query_bank_rows(priority="P0")
        if "Pareto sweep of total cost vs emissions" in row.query
    )

    assert safety_row.expected_tool_names(exported) == ["get_solvent_safety_card"]
    assert "run_waste_management_pareto" in optimization_row.expected_tool_names(exported)


def test_query_bank_expected_artifact_extraction_uses_registered_artifacts_and_aliases():
    _require_query_bank()

    safety_row = next(
        row for row in validated_query_bank_rows(priority="P0")
        if row.query == "Show a safety card for THF at 60 C."
    )
    pareto_row = next(
        row for row in validated_query_bank_rows(priority="P0")
        if "Pareto sweep of total cost vs emissions" in row.query
    )
    slices_row = next(
        row for row in validated_query_bank_rows(priority="P0")
        if "five fixed feed compositions" in row.query
    )
    single_card_row = next(
        row for row in validated_query_bank_rows(priority="P0")
        if row.query == "Peroxide formation for diethyl ether."
    )
    unsupported_hsp_row = next(
        row for row in validated_query_bank_rows(priority="P0")
        if "GVL for PET" in row.query
    )

    assert "solvent_safety_card" in safety_row.expected_artifact_types()
    assert "solvent_safety_card" in single_card_row.expected_artifact_types()
    assert "hsp_single_pair_summary" in unsupported_hsp_row.expected_artifact_types()
    assert "optimization_pareto_landscape" in pareto_row.expected_artifact_types()
    assert "optimization_pareto_slices_plot" in slices_row.expected_artifact_types()
