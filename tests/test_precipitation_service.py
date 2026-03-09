from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


class FakeAnalyzer:
    def __init__(self, *, differential=None, reverse=None, available_polymers=None, precipitation_point=None, curve_df=None, atmospheric=None, multi_atmospheric=None):
        self._differential = differential or []
        self._reverse = reverse or []
        self._available_polymers = available_polymers or ["LDPE", "EVOH"]
        self._precipitation_point = precipitation_point
        self._curve_df = curve_df if curve_df is not None else pd.DataFrame()
        self._atmospheric = atmospheric or []
        self._multi_atmospheric = multi_atmospheric or []
        self._calls = []

    def find_differential_precipitation_solvents(self, **kwargs):
        self._calls.append(kwargs)
        if len(self._calls) == 1:
            return self._differential
        return self._reverse

    def get_available_polymers(self):
        return self._available_polymers

    def analyze_precipitation(self, polymer, solvent, threshold):
        return self._precipitation_point

    def get_solubility_curve(self, polymer, solvent):
        return self._curve_df

    def check_atmospheric_feasibility(self, **kwargs):
        return self._atmospheric

    def check_multi_polymer_atmospheric_feasibility(self, **kwargs):
        return self._multi_atmospheric


def test_build_differential_precipitation_report_uses_reverse_order_when_needed():
    from strap.services.precipitation_service import build_differential_precipitation_report

    reverse = [SimpleNamespace(solvent="dmf")]
    analyzer = FakeAnalyzer(differential=[], reverse=reverse)

    result = build_differential_precipitation_report(
        analyzer,
        lambda rows: f"formatted:{rows[0].solvent}",
        polymer_to_precipitate="EVOH",
        polymer_to_retain="LDPE",
        min_temperature_gap=20.0,
        precipitation_threshold=1.0,
        top_k=5,
    )

    assert "REVERSE order works" in result
    assert "formatted:dmf" in result


def test_build_precipitation_temperature_report_formats_key_temperatures():
    from strap.services.precipitation_service import build_precipitation_temperature_report

    point = SimpleNamespace(
        max_solubility=42.0,
        max_solubility_temp=120.0,
        cloud_point=95.0,
        precipitation_temp=70.0,
        transition_width=25.0,
        data_points=6,
    )
    curve_df = pd.DataFrame(
        [
            {"temperature": 40.0, "solubility": 5.0},
            {"temperature": 70.0, "solubility": 12.0},
            {"temperature": 100.0, "solubility": 42.0},
        ]
    )
    analyzer = FakeAnalyzer(precipitation_point=point, curve_df=curve_df)

    result = build_precipitation_temperature_report(
        analyzer,
        polymer="LDPE",
        solvent="toluene",
        precipitation_threshold=1.0,
    )

    assert "Precipitation Analysis: LDPE in toluene" in result
    assert "Max Solubility" in result
    assert "120 deg C" in result


def test_build_atmospheric_feasibility_report_returns_formatted_results():
    from strap.services.precipitation_service import build_atmospheric_feasibility_report

    rows = [SimpleNamespace(solvent="THF")]
    analyzer = FakeAnalyzer(atmospheric=rows)

    result = build_atmospheric_feasibility_report(
        analyzer,
        lambda payload, include_infeasible=True: f"formatted:{payload[0].solvent}:{include_infeasible}",
        polymer1="PS",
        polymer2="PET",
        min_temperature_gap=20.0,
        precipitation_threshold=1.0,
        min_solubility=30.0,
    )

    assert result == "formatted:THF:True"


def test_build_multi_polymer_atmospheric_feasibility_report_validates_polymer_count():
    from strap.services.precipitation_service import build_multi_polymer_atmospheric_feasibility_report

    analyzer = FakeAnalyzer()

    result = build_multi_polymer_atmospheric_feasibility_report(
        analyzer,
        lambda payload, include_infeasible=True: "unused",
        polymers="LDPE",
        min_temperature_gap=20.0,
        precipitation_threshold=1.0,
        min_solubility=30.0,
    )

    assert "Need at least 2 polymers" in result
