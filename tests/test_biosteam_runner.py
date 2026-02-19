"""Tests for BioSTEAM subprocess runner (mocked subprocess)."""
import json
import subprocess
import pytest
from unittest.mock import patch, MagicMock


class TestRunSingleSimulation:
    def test_timeout_returns_error(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=5)):
            result = run_single_simulation({"solvent": "Toluene", "energy_case": "C1"}, timeout=5)
        assert result["success"] is False
        assert (
            "timeout" in result.get("error", "").lower()
            or result.get("error_type") == "TimeoutExpired"
        )

    def test_timeout_includes_solvent_info(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=5)):
            result = run_single_simulation({"solvent": "Toluene", "energy_case": "C1"}, timeout=5)
        assert result.get("solvent") == "Toluene"
        assert result.get("energy_case") == "C1"

    def test_nonzero_returncode_returns_error(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=1,
            stdout="", stderr="ImportError: No module named biosteam"
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_single_simulation({"solvent": "Toluene", "energy_case": "C1"})
        assert result["success"] is False

    def test_nonzero_returncode_sets_error_type(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=1,
            stdout="", stderr="RuntimeError: something went wrong"
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_single_simulation({"solvent": "Toluene", "energy_case": "C1"})
        assert result.get("error_type") == "SubprocessError"
        assert result.get("returncode") == 1

    def test_valid_json_returned(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        payload = json.dumps({"success": True, "tea": {"npv": 1e6}})
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=0,
            stdout=payload, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_single_simulation({"solvent": "Toluene", "energy_case": "C1"})
        assert result["success"] is True

    def test_valid_json_preserves_payload(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        tea_data = {"npv": 1e6, "msp_usd_per_kg": 2.50}
        payload = json.dumps({"success": True, "tea": tea_data})
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=0,
            stdout=payload, stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_single_simulation({"solvent": "Toluene", "energy_case": "C1"})
        assert result.get("tea") == tea_data

    def test_invalid_json_returns_error(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=0,
            stdout="not-json", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_single_simulation({"solvent": "Toluene"})
        assert result["success"] is False

    def test_invalid_json_sets_error_type(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=0,
            stdout="{bad json", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_single_simulation({"solvent": "Xylene", "energy_case": "C2"})
        assert result.get("error_type") == "JSONDecodeError"

    def test_empty_stdout_returns_error(self):
        from strap.vendor.biosteam_runner import run_single_simulation
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=0,
            stdout="", stderr=""
        )
        with patch("subprocess.run", return_value=mock_result):
            result = run_single_simulation({"solvent": "Toluene", "energy_case": "C1"})
        assert result["success"] is False
        assert result.get("error_type") == "EmptyOutput"

    def test_lca_cfs_injected_automatically(self):
        """If lca_cfs is not in config, it should be auto-populated from the registry."""
        from strap.vendor.biosteam_runner import run_single_simulation
        payload = json.dumps({"success": True})
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=0,
            stdout=payload, stderr=""
        )
        captured_call = {}
        def capture_subprocess(args, **kwargs):
            captured_call["json_arg"] = args[2]
            return mock_result

        with patch("subprocess.run", side_effect=capture_subprocess):
            run_single_simulation({"solvent": "Toluene", "energy_case": "C1"})

        config_sent = json.loads(captured_call["json_arg"])
        assert "lca_cfs" in config_sent

    def test_existing_lca_cfs_not_overwritten(self):
        """If lca_cfs is already in config, it should NOT be overwritten."""
        from strap.vendor.biosteam_runner import run_single_simulation
        payload = json.dumps({"success": True})
        mock_result = subprocess.CompletedProcess(
            args=["python", "worker.py"], returncode=0,
            stdout=payload, stderr=""
        )
        custom_cfs = {"solvent_gwp": 999.0}
        captured_call = {}
        def capture_subprocess(args, **kwargs):
            captured_call["json_arg"] = args[2]
            return mock_result

        with patch("subprocess.run", side_effect=capture_subprocess):
            run_single_simulation(
                {"solvent": "Toluene", "energy_case": "C1", "lca_cfs": custom_cfs}
            )

        config_sent = json.loads(captured_call["json_arg"])
        assert config_sent["lca_cfs"] == custom_cfs


class TestGetSupportedSolvents:
    def test_returns_dict_with_pe_solvents(self):
        from strap.vendor.biosteam_runner import get_supported_solvents
        result = get_supported_solvents()
        assert isinstance(result, dict)
        assert "pe_solvents" in result
        assert "Toluene" in result["pe_solvents"]

    def test_returns_energy_cases(self):
        from strap.vendor.biosteam_runner import get_supported_solvents
        result = get_supported_solvents()
        assert "energy_cases" in result
        assert "C1" in result["energy_cases"]

    def test_chlorinated_blocklist_present(self):
        from strap.vendor.biosteam_runner import get_supported_solvents
        result = get_supported_solvents()
        assert "chlorinated_blocklist" in result
        assert isinstance(result["chlorinated_blocklist"], list)


class TestBuildBatchConfigs:
    def test_generates_one_config_per_solvent_energy_pair(self):
        from strap.vendor.biosteam_runner import build_batch_configs
        configs = build_batch_configs(
            solvents=["Toluene", "Xylene"],
            target_plastic="PE",
            energy_cases=["C1", "C2"],
        )
        assert len(configs) == 4  # 2 solvents x 2 energy cases

    def test_config_has_required_keys(self):
        from strap.vendor.biosteam_runner import build_batch_configs
        configs = build_batch_configs(solvents=["Toluene"])
        cfg = configs[0]
        assert "solvent" in cfg
        assert "energy_case" in cfg
        assert "target_plastic" in cfg

    def test_solvent_defaults_applied(self):
        """Known solvents should have price and dissolution temp defaults filled in."""
        from strap.vendor.biosteam_runner import build_batch_configs
        configs = build_batch_configs(solvents=["Toluene"])
        cfg = configs[0]
        assert "solvent_price" in cfg
        assert "dissolution_temperature_c" in cfg

    def test_empty_solvents_returns_empty_list(self):
        from strap.vendor.biosteam_runner import build_batch_configs
        configs = build_batch_configs(solvents=[])
        assert configs == []


class TestRankResults:
    def test_ranks_by_msp_ascending(self):
        from strap.vendor.biosteam_runner import rank_results
        results = [
            {"success": True, "tea": {"msp_usd_per_kg": 3.0}, "lca": {}},
            {"success": True, "tea": {"msp_usd_per_kg": 1.0}, "lca": {}},
            {"success": True, "tea": {"msp_usd_per_kg": 2.0}, "lca": {}},
        ]
        ranked = rank_results(results, metric="msp")
        msps = [r["tea"]["msp_usd_per_kg"] for r in ranked]
        assert msps == [1.0, 2.0, 3.0]

    def test_failed_results_sorted_to_end(self):
        from strap.vendor.biosteam_runner import rank_results
        results = [
            {"success": False, "error": "something"},
            {"success": True, "tea": {"msp_usd_per_kg": 1.0}, "lca": {}},
        ]
        ranked = rank_results(results, metric="msp")
        assert ranked[0]["success"] is True
        assert ranked[-1]["success"] is False

    def test_unknown_metric_falls_back_to_msp(self):
        from strap.vendor.biosteam_runner import rank_results
        results = [
            {"success": True, "tea": {"msp_usd_per_kg": 1.5}, "lca": {}},
        ]
        # Should not raise
        ranked = rank_results(results, metric="nonexistent_metric")
        assert len(ranked) == 1
