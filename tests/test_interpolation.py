"""Tests for solubility interpolation tools."""

import json
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "solubility_coefficients.json"


def _parse_tool_result(raw: str) -> dict:
    parsed = json.loads(raw)
    assert "display" in parsed
    assert "data" in parsed
    return parsed


# ------------------------------------------------------------------
# Coefficient JSON structure tests
# ------------------------------------------------------------------

class TestCoefficientsJSON:
    @pytest.fixture(autouse=True, scope="class")
    def _load(self):
        with open(DATA_PATH) as f:
            self.__class__.data = json.load(f)

    def test_entry_count(self):
        assert self.data["n_entries"] == 362
        assert len(self.data["entries"]) == 362

    def test_categories_sum(self):
        cats = self.data["categories"]
        assert sum(cats.values()) == 362

    def test_known_insoluble_pairs(self):
        insol = [
            e for e in self.data["entries"] if e["category"] == "insoluble"
        ]
        pairs = {(e["polymer"], e["solvent"]) for e in insol}
        assert ("PC", "h2o") in pairs
        assert ("PET", "h2o") in pairs
        assert ("PS", "h2o") in pairs

    def test_fitted_r_squared_minimum(self):
        fitted = [
            e for e in self.data["entries"] if e["category"] == "fitted"
        ]
        assert len(fitted) > 300
        for e in fitted:
            assert e["r_squared"] >= 0.98, (
                f"{e['polymer']}/{e['solvent']} R²={e['r_squared']}"
            )

    def test_entry_fields(self):
        required = {
            "polymer", "solvent", "category", "A", "B", "C",
            "r_squared", "n_points", "t_min_c", "t_max_c",
        }
        for e in self.data["entries"]:
            assert required <= set(e.keys()), f"Missing keys in {e['polymer']}/{e['solvent']}"


# ------------------------------------------------------------------
# Tool function tests
# ------------------------------------------------------------------

class TestPredictSolubility:
    def test_returns_string(self):
        from strap.tools.interpolation import predict_solubility
        result = predict_solubility("HDPE", "toluene", 100.0)
        assert isinstance(result, str)
        parsed = _parse_tool_result(result)
        assert parsed["data"]["tool_name"] == "predict_solubility"
        assert parsed["data"]["success"] is True
        assert "%" in parsed["display"]
        assert "BP" in parsed["display"]

    def test_case_insensitive(self):
        from strap.tools.interpolation import predict_solubility
        r1 = predict_solubility("hdpe", "Toluene", 100.0)
        r2 = predict_solubility("HDPE", "toluene", 100.0)
        assert "%" in _parse_tool_result(r1)["display"]
        assert "%" in _parse_tool_result(r2)["display"]

    def test_alias_matching(self):
        from strap.tools.interpolation import predict_solubility
        result = predict_solubility("HDPE", "water", 100.0)
        # water → h2o
        assert isinstance(result, str)
        parsed = _parse_tool_result(result)
        assert parsed["data"]["tool_name"] == "predict_solubility"

    def test_unknown_polymer(self):
        from strap.tools.interpolation import predict_solubility
        result = predict_solubility("NONEXISTENT", "toluene", 100.0)
        parsed = _parse_tool_result(result)
        assert parsed["data"]["success"] is False
        assert "Unknown polymer" in parsed["display"]

    def test_unknown_solvent(self):
        from strap.tools.interpolation import predict_solubility
        result = predict_solubility("HDPE", "nonexistent_solvent", 100.0)
        parsed = _parse_tool_result(result)
        assert parsed["data"]["success"] is False
        assert "Unknown solvent" in parsed["display"]

    def test_insoluble_pair(self):
        from strap.tools.interpolation import predict_solubility
        result = predict_solubility("PC", "water", 100.0)
        parsed = _parse_tool_result(result)
        assert "Insoluble" in parsed["display"]

    def test_extrapolation_still_returns(self):
        from strap.tools.interpolation import predict_solubility
        # 0°C is below the fit range (25°C min) — should still return a result
        result = predict_solubility("HDPE", "toluene", 0.0)
        parsed = _parse_tool_result(result)
        assert "%" in parsed["display"]


class TestPredictSolubilityRange:
    def test_returns_table(self):
        from strap.tools.interpolation import predict_solubility_range
        result = predict_solubility_range("HDPE", "toluene", 25, 50, 5)
        assert isinstance(result, str)
        parsed = _parse_tool_result(result)
        assert parsed["data"]["tool_name"] == "predict_solubility_range"
        assert "T (C)" in parsed["display"]
        assert "25.0" in parsed["display"]

    def test_cap_at_200(self):
        from strap.tools.interpolation import predict_solubility_range
        # 0.1 step over 200 range → 2000 points, should be capped
        result = predict_solubility_range("HDPE", "toluene", 0, 200, 0.1)
        parsed = _parse_tool_result(result)
        lines = [l for l in parsed["display"].split("\n") if l.startswith("|") and "T (C)" not in l and "---" not in l]
        assert len(lines) <= 200


class TestListCoverage:
    def test_returns_string(self):
        from strap.tools.interpolation import list_interpolation_coverage
        result = list_interpolation_coverage()
        assert isinstance(result, str)
        parsed = _parse_tool_result(result)
        assert parsed["data"]["tool_name"] == "list_interpolation_coverage"
        assert "362" in parsed["display"]
        assert "HDPE" in parsed["display"]
