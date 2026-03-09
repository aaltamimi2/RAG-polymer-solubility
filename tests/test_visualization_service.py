"""Tests for visualization service helpers."""

import asyncio


def test_normalize_solvent_names_expands_aliases():
    from strap.services.visualization_service import normalize_solvent_names

    normalized = normalize_solvent_names(["xylene", "DMSO", "THF"])

    assert "1,2-dimethylbenzene" in normalized
    assert "1,4-dimethylbenzene" in normalized
    assert "dimethylsulfoxide" in normalized
    assert "thf" in normalized


def test_normalize_solvent_names_reconstructs_fragmented_names():
    from strap.services.visualization_service import normalize_solvent_names

    normalized = normalize_solvent_names(["2", "3-dihydropyran", "toluene"])

    assert normalized[0] == "2,3-dihydropyran"
    assert normalized[1] == "toluene"


def test_get_plot_url_formats_path():
    from strap.services.visualization_service import get_plot_url

    assert get_plot_url("/tmp/plot.png") == "Plot saved: `/tmp/plot.png`"


def test_execute_query_blocks_unsafe_sql():
    from strap.services.visualization_service import execute_query

    result = execute_query("DROP TABLE solvent_data")

    assert result["success"] is False
    assert "Unsafe" in result["error"]


def test_get_solvent_table_name_detects_property_table(conn):
    from strap.services.visualization_service import get_solvent_table_name

    table = get_solvent_table_name()

    assert table == "solvent_data"


def test_get_solvent_name_and_cosmobase_columns(conn):
    from strap.services.visualization_service import (
        get_cosmobase_column,
        get_solvent_name_column,
    )

    assert get_solvent_name_column("solvent_data") == "solvent_name"
    assert get_cosmobase_column("solvent_data") == "solvent_name_in_cosmobase"


def test_verify_inputs_accepts_known_table_and_columns(conn):
    from strap.services.visualization_service import verify_inputs

    ok, message = verify_inputs(
        "common_solvents_database",
        {
            "polymer": "polymer",
            "solvent": "solvent",
            "temperature": "temperature___c_",
        },
        {"polymer": ["LDPE"]},
    )

    assert ok is True
    assert message == "All inputs verified"


def test_lookup_solvent_properties_returns_expected_fields(conn):
    from strap.services.visualization_service import lookup_solvent_properties

    result = asyncio.run(lookup_solvent_properties(["Toluene", "THF"], "solvent_data"))

    assert "Toluene" in result
    assert "THF" in result
    assert result["Toluene"]["bp"] is not None
