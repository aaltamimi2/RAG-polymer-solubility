"""Tests for BioSTEAM sensitivity / uncertainty helper functions."""
import json
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helper-function unit tests (no subprocess calls)
# ---------------------------------------------------------------------------

class TestGetDefaultParameterRanges:
    def test_returns_valid_ranges_for_known_solvent(self):
        from strap.vendor.biosteam_runner import _get_default_parameter_ranges
        ranges = _get_default_parameter_ranges("Toluene")
        assert isinstance(ranges, dict)
        assert len(ranges) == 5
        for param, (lo, hi) in ranges.items():
            assert lo < hi, f"{param}: min ({lo}) >= max ({hi})"

    def test_returns_valid_ranges_for_unknown_solvent(self):
        from strap.vendor.biosteam_runner import _get_default_parameter_ranges
        ranges = _get_default_parameter_ranges("NonExistentSolvent")
        assert isinstance(ranges, dict)
        for param, (lo, hi) in ranges.items():
            assert lo < hi, f"{param}: min ({lo}) >= max ({hi})"

    def test_solvent_price_range_uses_base_price(self):
        from strap.vendor.biosteam_runner import (
            _get_default_parameter_ranges, _SOLVENT_DEFAULTS,
        )
        base_price = _SOLVENT_DEFAULTS["Toluene"][0]
        ranges = _get_default_parameter_ranges("Toluene")
        lo, hi = ranges["solvent_price"]
        assert lo == pytest.approx(base_price * 0.7, abs=0.01)
        assert hi == pytest.approx(base_price * 1.5, abs=0.01)

    def test_toluene_base_defaults_match_tea_lca_csv(self):
        from strap.vendor.biosteam_runner import _SOLVENT_DEFAULTS

        price, diss_temp, _ = _SOLVENT_DEFAULTS["Toluene"]
        assert price == pytest.approx(1.312, abs=1e-3)
        assert diss_temp == pytest.approx(109.6, abs=1e-3)

    def test_dissolution_temp_lower_bound_at_least_90(self):
        from strap.vendor.biosteam_runner import _get_default_parameter_ranges
        # Hexane has base dissolution temp 66 °C → 66-20 = 46, should clamp to 90
        ranges = _get_default_parameter_ranges("Hexane")
        lo, _ = ranges["dissolution_temperature_c"]
        assert lo >= 90.0

    def test_all_five_parameters_present(self):
        from strap.vendor.biosteam_runner import _get_default_parameter_ranges
        ranges = _get_default_parameter_ranges("Xylene")
        expected = {
            "solvent_price", "solvent_loss_pct",
            "dissolution_temperature_c", "precipitation_temperature_c",
            "feedstock_distance_km",
        }
        assert set(ranges.keys()) == expected


class TestBuildMonteCarloConfigs:
    def test_generates_correct_number(self):
        from strap.vendor.biosteam_runner import _build_monte_carlo_configs
        configs = _build_monte_carlo_configs("Toluene", n_samples=10)
        assert len(configs) == 10

    def test_caps_at_50(self):
        from strap.vendor.biosteam_runner import _build_monte_carlo_configs
        configs = _build_monte_carlo_configs("Toluene", n_samples=100)
        assert len(configs) == 50

    def test_values_within_ranges(self):
        from strap.vendor.biosteam_runner import (
            _build_monte_carlo_configs, _get_default_parameter_ranges,
        )
        ranges = _get_default_parameter_ranges("Toluene")
        configs = _build_monte_carlo_configs("Toluene", n_samples=20)
        for cfg in configs:
            for param, (lo, hi) in ranges.items():
                assert lo <= cfg[param] <= hi, (
                    f"{param}={cfg[param]} not in [{lo}, {hi}]"
                )

    def test_configs_have_required_keys(self):
        from strap.vendor.biosteam_runner import _build_monte_carlo_configs
        configs = _build_monte_carlo_configs("Toluene", n_samples=3)
        required = {"solvent", "target_plastic", "energy_case", "processing_capacity"}
        for cfg in configs:
            assert required.issubset(cfg.keys())

    def test_subset_parameters(self):
        from strap.vendor.biosteam_runner import (
            _build_monte_carlo_configs, _SOLVENT_DEFAULTS,
        )
        configs = _build_monte_carlo_configs(
            "Toluene", n_samples=5, parameters="solvent_price,feedstock_distance_km",
        )
        base_diss = _SOLVENT_DEFAULTS["Toluene"][1]
        for cfg in configs:
            # Non-swept params should stay at baseline
            assert cfg["dissolution_temperature_c"] == base_diss
            assert cfg["solvent_loss_pct"] == 0.01
            # Swept params should vary
            # (probabilistic, but 5 samples of uniform are almost certainly not all equal)


class TestBuildSweepConfigs:
    def test_auto_generates_five_values(self):
        from strap.vendor.biosteam_runner import _build_sweep_configs
        configs = _build_sweep_configs("Toluene")
        assert len(configs) == 5

    def test_explicit_values(self):
        from strap.vendor.biosteam_runner import _build_sweep_configs
        configs = _build_sweep_configs(
            "Toluene", parameter="solvent_price", values=[0.5, 1.0, 1.5],
        )
        assert len(configs) == 3
        vals = [c["_sweep_value"] for c in configs]
        assert vals == [0.5, 1.0, 1.5]

    def test_sweep_metadata_tags(self):
        from strap.vendor.biosteam_runner import _build_sweep_configs
        configs = _build_sweep_configs("Toluene", parameter="feedstock_distance_km")
        for cfg in configs:
            assert cfg["_sweep_param"] == "feedstock_distance_km"
            assert "_sweep_value" in cfg

    def test_unknown_parameter_raises(self):
        from strap.vendor.biosteam_runner import _build_sweep_configs
        with pytest.raises(ValueError, match="Unknown sweep parameter"):
            _build_sweep_configs("Toluene", parameter="bogus_param")


class TestBuildTornadoConfigs:
    def test_returns_correct_counts(self):
        from strap.vendor.biosteam_runner import _build_tornado_configs
        baseline, oat = _build_tornado_configs("Toluene")
        # 5 default params → 10 OAT configs (min+max each)
        assert len(oat) == 10

    def test_baseline_is_valid_config(self):
        from strap.vendor.biosteam_runner import _build_tornado_configs
        baseline, _ = _build_tornado_configs("Toluene")
        assert baseline["solvent"] == "Toluene"
        assert "energy_case" in baseline

    def test_oat_metadata_tags(self):
        from strap.vendor.biosteam_runner import _build_tornado_configs
        _, oat = _build_tornado_configs("Toluene")
        for cfg in oat:
            assert "_tornado_param" in cfg
            assert cfg["_tornado_bound"] in ("min", "max")

    def test_oat_pairs_cover_all_params(self):
        from strap.vendor.biosteam_runner import _build_tornado_configs
        _, oat = _build_tornado_configs("Toluene", parameters="solvent_price,feedstock_distance_km")
        assert len(oat) == 4  # 2 params × 2 bounds
        params_seen = {cfg["_tornado_param"] for cfg in oat}
        assert params_seen == {"solvent_price", "feedstock_distance_km"}

    def test_baseline_uses_midpoints(self):
        from strap.vendor.biosteam_runner import (
            _build_tornado_configs, _get_default_parameter_ranges,
        )
        ranges = _get_default_parameter_ranges("Toluene")
        baseline, _ = _build_tornado_configs("Toluene")
        for param, (lo, hi) in ranges.items():
            expected_mid = (lo + hi) / 2.0
            assert baseline[param] == pytest.approx(expected_mid, rel=1e-6), (
                f"{param}: baseline {baseline[param]} != midpoint {expected_mid}"
            )

    def test_subset_parameters_string(self):
        from strap.vendor.biosteam_runner import _build_tornado_configs
        _, oat = _build_tornado_configs("Toluene", parameters="solvent_price")
        assert len(oat) == 2
        assert all(c["_tornado_param"] == "solvent_price" for c in oat)


# ---------------------------------------------------------------------------
# Tool-level integration tests (mocked subprocess)
# ---------------------------------------------------------------------------

def _make_success_result(**overrides):
    """Build a fake successful simulation result dict."""
    base = {
        "success": True,
        "solvent": "Toluene",
        "energy_case": "C1",
        "tea": {"msp_usd_per_kg": 0.48, "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000},
        "lca": {"gwp_kg_co2e_per_kg": 1.22, "htc_ctuh_per_kg": 3e-7, "htnc_ctuh_per_kg": 2e-7, "etox_ctue_per_kg": 15.0},
        "operations": {"total_energy_mj_per_kg": 8.5},
        "runtime_seconds": 11.2,
    }
    base.update(overrides)
    return base


class TestRunBiosteamUncertaintyTool:
    """Integration tests for the uncertainty tool (mocked batch runner)."""

    @patch("strap.tools.biosteam_tea_lca.run_batch_simulations")
    def test_basic_output_structure(self, mock_batch):
        from strap.tools.biosteam_tea_lca import run_biosteam_uncertainty
        # Return 5 successful results with slightly varying MSP
        mock_batch.return_value = [
            _make_success_result(tea={"msp_usd_per_kg": 0.45 + i * 0.02,
                                      "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000})
            for i in range(5)
        ]
        raw = run_biosteam_uncertainty("Toluene", n_samples=5)
        parsed = json.loads(raw)
        assert "display" in parsed
        assert "data" in parsed
        data = parsed["data"]
        assert data["success"] is True
        assert data["n_succeeded"] == 5
        assert "msp_percentiles" in data
        assert data["msp_percentiles"]["p50"] is not None

    @patch("strap.tools.biosteam_tea_lca.run_batch_simulations")
    def test_all_failures_reported(self, mock_batch):
        from strap.tools.biosteam_tea_lca import run_biosteam_uncertainty
        mock_batch.return_value = [
            {"success": False, "error": "timeout"} for _ in range(5)
        ]
        raw = run_biosteam_uncertainty("Toluene", n_samples=5)
        parsed = json.loads(raw)
        assert "failed" in parsed["display"].lower()


class TestRunBiosteamParameterSweepTool:

    @patch("strap.tools.biosteam_tea_lca.run_batch_simulations")
    def test_basic_sweep(self, mock_batch):
        from strap.tools.biosteam_tea_lca import run_biosteam_parameter_sweep
        mock_batch.return_value = [
            _make_success_result(tea={"msp_usd_per_kg": 0.40 + i * 0.05,
                                      "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000})
            for i in range(5)
        ]
        raw = run_biosteam_parameter_sweep("Toluene", parameter="solvent_price")
        parsed = json.loads(raw)
        data = parsed["data"]
        assert data["success"] is True
        assert data["parameter"] == "solvent_price"
        assert len(data["sweep_data"]) == 5

    @patch("strap.tools.biosteam_tea_lca.run_batch_simulations")
    def test_explicit_values(self, mock_batch):
        from strap.tools.biosteam_tea_lca import run_biosteam_parameter_sweep
        mock_batch.return_value = [_make_success_result() for _ in range(3)]
        raw = run_biosteam_parameter_sweep(
            "Toluene", parameter="feedstock_distance_km", values="0,100,200",
        )
        parsed = json.loads(raw)
        assert parsed["data"]["n_values"] == 3

    def test_invalid_values_return_structured_error(self):
        from strap.tools.biosteam_tea_lca import run_biosteam_parameter_sweep

        raw = run_biosteam_parameter_sweep(
            "Toluene", parameter="feedstock_distance_km", values="a,b,c",
        )

        parsed = json.loads(raw)
        assert parsed["data"]["success"] is False
        assert "parse values" in parsed["data"]["error"].lower()


class TestRunBiosteamTornadoTool:

    @patch("strap.tools.biosteam_tea_lca.run_batch_simulations")
    def test_basic_tornado(self, mock_batch):
        from strap.tools.biosteam_tea_lca import run_biosteam_tornado
        # 1 baseline + 10 OAT (5 params × 2)
        results = [_make_success_result()]  # baseline
        for i in range(10):
            results.append(_make_success_result(
                tea={"msp_usd_per_kg": 0.40 + i * 0.02,
                     "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000}
            ))
        mock_batch.return_value = results
        raw = run_biosteam_tornado("Toluene", metric="msp")
        parsed = json.loads(raw)
        data = parsed["data"]
        assert data["success"] is True
        assert data["metric"] == "msp"
        assert data["baseline_value"] is not None
        assert len(data["rankings"]) > 0

    @patch("strap.tools.biosteam_tea_lca.run_batch_simulations")
    def test_baseline_failure_reported(self, mock_batch):
        from strap.tools.biosteam_tea_lca import run_biosteam_tornado
        mock_batch.return_value = [{"success": False, "error": "crash"}]
        raw = run_biosteam_tornado("Toluene")
        parsed = json.loads(raw)
        assert "failed" in parsed["display"].lower()

    @patch("strap.tools.biosteam_tea_lca.run_batch_simulations")
    def test_rankings_sorted_descending(self, mock_batch):
        from strap.tools.biosteam_tea_lca import run_biosteam_tornado
        # Build results: baseline + 4 OAT for 2 params
        baseline = _make_success_result(tea={"msp_usd_per_kg": 0.50,
                                              "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000})
        oat = [
            _make_success_result(tea={"msp_usd_per_kg": 0.45, "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000}),  # param1 min
            _make_success_result(tea={"msp_usd_per_kg": 0.55, "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000}),  # param1 max
            _make_success_result(tea={"msp_usd_per_kg": 0.49, "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000}),  # param2 min
            _make_success_result(tea={"msp_usd_per_kg": 0.51, "tci_usd": 12_000_000, "aoc_usd_per_yr": 5_000_000}),  # param2 max
        ]
        mock_batch.return_value = [baseline] + oat
        raw = run_biosteam_tornado(
            "Toluene", parameters="solvent_price,feedstock_distance_km",
        )
        parsed = json.loads(raw)
        rankings = parsed["data"]["rankings"]
        if len(rankings) >= 2:
            assert rankings[0]["range"] >= rankings[1]["range"]


class TestBioSteamToolErrorContracts:
    def test_visualize_invalid_json_returns_envelope(self):
        from strap.tools.biosteam_tea_lca import visualize_biosteam_results

        raw = visualize_biosteam_results("{bad json")
        parsed = json.loads(raw)

        assert parsed["data"]["success"] is False
        assert parsed["data"]["tool_name"] == "visualize_biosteam_results"

    def test_multi_polymer_invalid_json_returns_envelope(self):
        from strap.tools.biosteam_tea_lca import run_biosteam_multi_polymer

        raw = run_biosteam_multi_polymer("{bad json")
        parsed = json.loads(raw)

        assert parsed["data"]["success"] is False
        assert parsed["data"]["tool_name"] == "run_biosteam_multi_polymer"
