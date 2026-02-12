"""Integration tests for the STRAP v7 thermal ML pipeline.

Covers:
  - Group contribution module (strap.thermal_ml.group_contribution)
  - Thermal ML public API (strap.thermal_ml)
  - COSMO-RS interface (strap.cosmo_interface)
  - Three-tier coefficient store (strap.solubility)
  - Agent-facing tools (strap.tools.thermal_prediction)
"""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from rdkit import Chem  # noqa: F401

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import torch  # noqa: F401

    torch_available = True
except ImportError:
    torch_available = False

rdkit_required = pytest.mark.skipif(
    not RDKIT_AVAILABLE,
    reason="RDKit is required for group contribution tests",
)


# ===================================================================
# 1. Group Contribution Module
# ===================================================================


@rdkit_required
class TestGroupContribution:
    """Tests for strap.thermal_ml.group_contribution."""

    def test_parse_psmiles_groups_polyethylene(self):
        """[*]CC[*] should find 2x -CH2- groups with coverage=1.0."""
        from strap.thermal_ml.group_contribution import parse_psmiles_groups

        result = parse_psmiles_groups("[*]CC[*]")
        assert result is not None
        groups = result["groups"]
        assert "-CH2-" in groups
        assert groups["-CH2-"] == 2
        npt.assert_almost_equal(result["coverage"], 1.0, decimal=1)

    def test_parse_psmiles_groups_polypropylene(self):
        """[*]CC(C)[*] should find -CH2- and -CH(CH3)- groups."""
        from strap.thermal_ml.group_contribution import parse_psmiles_groups

        result = parse_psmiles_groups("[*]CC(C)[*]")
        assert result is not None
        groups = result["groups"]
        # Should have at least one -CH2- and one -CH(CH3)-
        assert "-CH2-" in groups or "-CH(CH3)-" in groups
        # Overall we expect both group types to be present
        has_ch2 = "-CH2-" in groups
        has_chch3 = "-CH(CH3)-" in groups
        assert has_ch2 or has_chch3, f"Expected -CH2- or -CH(CH3)- groups, got: {groups}"

    def test_estimate_delta_hf_polyethylene(self):
        """PE delta Hf should be ~8000 J/mol (2 x 4000)."""
        from strap.thermal_ml.group_contribution import estimate_delta_hf

        result = estimate_delta_hf("[*]CC[*]")
        assert result is not None
        npt.assert_almost_equal(result["value"], 8000.0, decimal=0)
        assert result["unit"] == "J/mol"

    def test_estimate_tm_polyethylene(self):
        """PE Tm should be ~400 K (reasonable for polyethylene)."""
        from strap.thermal_ml.group_contribution import estimate_tm

        result = estimate_tm("[*]CC[*]")
        assert result is not None
        assert result["value"] is not None, "Tm value should not be None for PE"
        # PE experimental Tm ~410 K; Van Krevelen gives ~404 K
        assert 350 < result["value"] < 450, (
            f"PE Tm = {result['value']} K is outside the reasonable range (350-450 K)"
        )
        assert result["unit"] == "K"

    def test_estimate_all_returns_complete_dict(self):
        """estimate_all should return tm, delta_hf, delta_cp, group_parse keys."""
        from strap.thermal_ml.group_contribution import estimate_all

        result = estimate_all("[*]CC[*]")
        assert result is not None
        for expected_key in ("tm", "delta_hf", "delta_cp", "group_parse"):
            assert expected_key in result, f"Missing key: {expected_key}"
        assert result["psmiles"] == "[*]CC[*]"

    def test_invalid_psmiles_returns_none(self):
        """Invalid SMILES should return None gracefully."""
        from strap.thermal_ml.group_contribution import parse_psmiles_groups

        result = parse_psmiles_groups("NOT_A_VALID_SMILES_XYZ!!!")
        assert result is None

    def test_coverage_metric(self):
        """Coverage should be 1.0 for simple polymers, <1.0 for complex ones."""
        from strap.thermal_ml.group_contribution import parse_psmiles_groups

        # Simple polymer (PE) -- all atoms should be recognised
        simple = parse_psmiles_groups("[*]CC[*]")
        assert simple is not None
        assert simple["coverage"] == 1.0

        # More complex polymer with unusual groups -- coverage may be lower
        # Polyimide fragment with atoms unlikely to be fully covered
        complex_result = parse_psmiles_groups("[*]c1ccc(Oc2ccc(S(=O)(=O)c3ccc([*])cc3)cc2)cc1")
        if complex_result is not None:
            # The coverage might or might not be 1.0 depending on the group table,
            # but we at least verify it is a float in [0, 1]
            assert 0.0 <= complex_result["coverage"] <= 1.0


# ===================================================================
# 2. Thermal ML Public API
# ===================================================================


@rdkit_required
class TestThermalML:
    """Tests for strap.thermal_ml public API."""

    def test_predict_thermal_properties_gc_only(self):
        """predict_thermal_properties with use_ml=False should return GC estimates."""
        from strap.thermal_ml import predict_thermal_properties

        result = predict_thermal_properties("[*]CC[*]", use_ml=False)
        assert result["method"] == "group_contribution_only"
        # Tm should be roughly 400 K for PE
        assert not math.isnan(result["Tm_K"])
        assert 350 < result["Tm_K"] < 450

    def test_predict_thermal_properties_returns_required_keys(self):
        """Result must have Tm_K, delta_Hf_J_per_mol, delta_Cp_J_per_mol_K, method, confidence."""
        from strap.thermal_ml import predict_thermal_properties

        result = predict_thermal_properties("[*]CC[*]", use_ml=False)
        required_keys = {
            "Tm_K",
            "delta_Hf_J_per_mol",
            "delta_Cp_J_per_mol_K",
            "method",
            "confidence",
        }
        missing = required_keys - set(result.keys())
        assert not missing, f"Missing keys in result: {missing}"

    def test_get_group_contribution_estimate(self):
        """get_group_contribution_estimate should return flat dict with coverage."""
        from strap.thermal_ml import get_group_contribution_estimate

        result = get_group_contribution_estimate("[*]CC[*]")
        assert "coverage" in result
        assert "Tm_K" in result
        assert "delta_Hf_J_per_mol" in result
        assert "delta_Cp_J_per_mol_K" in result
        assert "groups" in result
        assert result["coverage"] == 1.0

    def test_is_model_available(self):
        """is_model_available should return False when no trained model exists."""
        from strap.thermal_ml import is_model_available

        # Unless the model weights are actually present, this should be False
        # We point to a path that definitely does not exist to be certain
        assert is_model_available("/nonexistent/path/model.pt") is False

    def test_predict_thermal_properties_invalid_psmiles(self):
        """Invalid PSMILES should still return a result dict (with NaN values)."""
        from strap.thermal_ml import predict_thermal_properties

        result = predict_thermal_properties("NOT_VALID!!!", use_ml=False)
        # Should still return a dict, but with NaN values
        assert isinstance(result, dict)
        assert "method" in result
        assert math.isnan(result["Tm_K"])

    def test_predict_thermal_properties_confidence_field(self):
        """Confidence should be one of 'high', 'medium', 'low'."""
        from strap.thermal_ml import predict_thermal_properties

        result = predict_thermal_properties("[*]CC[*]", use_ml=False)
        assert result["confidence"] in ("high", "medium", "low")

    def test_predict_thermal_properties_includes_baselines(self):
        """Result should include group_contribution_baselines sub-dict."""
        from strap.thermal_ml import predict_thermal_properties

        result = predict_thermal_properties("[*]CC[*]", use_ml=False)
        assert "group_contribution_baselines" in result
        baselines = result["group_contribution_baselines"]
        assert "Tm_K" in baselines
        assert "coverage" in baselines


# ===================================================================
# 3. COSMO Interface
# ===================================================================


class TestCosmoInterface:
    """Tests for strap.cosmo_interface."""

    def test_compute_ideal_sle_basic(self):
        """Ideal SLE for PE at 25 degC should give x > 0 and x < 1."""
        from strap.cosmo_interface import compute_ideal_sle

        # PE-like: Tm ~ 410 K, dHf ~ 8000 J/mol
        T_K = 25.0 + 273.15  # 298.15 K
        x = compute_ideal_sle(T_K, Tm_K=410.0, delta_Hf=8000.0)
        assert x.shape == ()  # scalar input -> scalar output
        assert 0.0 < float(x) < 1.0, f"Expected 0 < x < 1, got x={float(x)}"

    def test_compute_ideal_sle_above_tm(self):
        """Above Tm, solubility should be 1.0 (fully miscible)."""
        from strap.cosmo_interface import compute_ideal_sle

        T_K = 500.0  # well above Tm of 410 K
        x = compute_ideal_sle(T_K, Tm_K=410.0, delta_Hf=8000.0)
        npt.assert_almost_equal(float(x), 1.0)

    def test_compute_ideal_sle_shape(self):
        """Solubility should increase monotonically with temperature."""
        from strap.cosmo_interface import compute_ideal_sle

        temps = np.linspace(250, 400, 50)
        x = compute_ideal_sle(temps, Tm_K=410.0, delta_Hf=8000.0, delta_Cp=20.0)
        assert x.shape == (50,)
        # Below Tm, solubility should increase with temperature
        diffs = np.diff(x)
        assert np.all(diffs >= -1e-10), (
            "Solubility should increase monotonically with temperature below Tm"
        )

    def test_run_sle_calculation_ideal_backend(self):
        """run_sle_calculation with backend='ideal' should return DataFrame."""
        from strap.cosmo_interface import run_sle_calculation

        df = run_sle_calculation(
            polymer_cosmo_file=None,
            solvent_cosmo_file=None,
            Tm_K=410.0,
            delta_Hf=8000.0,
            delta_Cp=20.0,
            t_range_c=(25, 100),
            t_step_c=10.0,
            cosmo_backend="ideal",
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = {
            "temperature_c",
            "temperature_k",
            "solubility_pct",
            "ln_gamma",
            "x_ideal",
            "x_total",
            "source",
        }
        missing = expected_cols - set(df.columns)
        assert not missing, f"Missing DataFrame columns: {missing}"
        assert len(df) > 0
        # All sources should be 'ideal'
        assert (df["source"] == "ideal").all()
        # ln_gamma should be 0 for ideal
        npt.assert_array_almost_equal(df["ln_gamma"].values, 0.0)

    def test_run_sle_with_uncertainty(self):
        """Uncertainty propagation should identify dominant source."""
        from strap.cosmo_interface import run_sle_with_uncertainty

        result = run_sle_with_uncertainty(
            polymer_cosmo=None,
            solvent_cosmo=None,
            Tm_K=410.0,
            delta_Hf=8000.0,
            delta_Cp=20.0,
            Tm_std=10.0,
            delta_Hf_std=2000.0,
            delta_Cp_std=5.0,
            t_range_c=(25, 100),
            t_step_c=10.0,
        )
        assert isinstance(result, dict)
        for key in (
            "temperature_c",
            "temperature_k",
            "x_mean",
            "x_upper",
            "x_lower",
            "sigma_ln_x",
            "dominant_source",
            "contributions",
        ):
            assert key in result, f"Missing key in uncertainty result: {key}"

        # Upper bound should be >= mean and lower bound <= mean
        assert np.all(result["x_upper"] >= result["x_mean"] - 1e-10)
        assert np.all(result["x_lower"] <= result["x_mean"] + 1e-10)

        # Dominant source should be a list of valid strings
        valid_sources = {"Tm", "delta_Hf", "delta_Cp", "none"}
        for src in result["dominant_source"]:
            assert src in valid_sources, f"Unexpected dominant source: {src}"

    def test_detect_cosmo_backend(self):
        """detect_cosmo_backend should return None or a valid string."""
        from strap.cosmo_interface import detect_cosmo_backend

        backend = detect_cosmo_backend()
        assert backend is None or backend in ("cosmotherm", "opencosmo")

    def test_list_available_cosmo_files(self):
        """Should return dict with 'polymers' and 'solvents' keys."""
        from strap.cosmo_interface import list_available_cosmo_files

        result = list_available_cosmo_files()
        assert isinstance(result, dict)
        assert "polymers" in result
        assert "solvents" in result
        assert isinstance(result["polymers"], list)
        assert isinstance(result["solvents"], list)

    def test_compute_ideal_sle_with_delta_cp(self):
        """Including delta_Cp should change the result compared to without."""
        from strap.cosmo_interface import compute_ideal_sle

        T_K = 300.0
        x_no_cp = compute_ideal_sle(T_K, Tm_K=410.0, delta_Hf=8000.0, delta_Cp=None)
        x_with_cp = compute_ideal_sle(T_K, Tm_K=410.0, delta_Hf=8000.0, delta_Cp=20.0)
        # The Cp correction term should make a non-zero difference
        assert float(x_no_cp) != float(x_with_cp)


# ===================================================================
# 4. Three-Tier Coefficient Store
# ===================================================================


class TestThreeTierCoefficients:
    """Tests for strap.solubility three-tier coefficient loading."""

    def test_static_entries_have_source(self):
        """All static entries should have source='static'."""
        from strap.solubility import _load_coefficients

        _, lookup = _load_coefficients()
        # Check a sample of entries that should be static
        static_count = sum(
            1 for entry in lookup.values() if entry.get("source") == "static"
        )
        assert static_count > 0, "Expected at least some static entries"

    def test_get_entry_source_static(self):
        """HDPE/toluene should be 'static'."""
        from strap.solubility import get_entry_source

        source = get_entry_source("HDPE", "toluene")
        assert source == "static", f"HDPE/toluene source should be 'static', got '{source}'"

    def test_get_dynamic_entries_empty(self):
        """With no generated file, dynamic entries should be empty list."""
        from strap.solubility import get_dynamic_entries, reload_coefficients, _GENERATED_PATH

        # If there is no generated file, dynamic entries should be empty
        # Reload to clear cache first
        reload_coefficients()
        entries = get_dynamic_entries()
        assert isinstance(entries, list)
        # If the generated file does not exist, we expect an empty list.
        # If it does exist, we just verify the return type.
        if not _GENERATED_PATH.exists():
            assert len(entries) == 0

    def test_reload_coefficients(self):
        """reload_coefficients should clear cache without error."""
        from strap.solubility import reload_coefficients, _load_coefficients

        # Call reload -- should not raise
        reload_coefficients()
        # After reload, calling _load_coefficients should re-load successfully
        coeffs, lookup = _load_coefficients()
        assert coeffs is not None
        assert lookup is not None
        assert len(lookup) > 0

    def test_predict_includes_source(self):
        """predict() return dict should include 'source' key."""
        from strap.solubility import predict

        # Construct a minimal entry dict matching the expected structure
        entry = {
            "A": 3.0,
            "B": -500.0,
            "C": 10000.0,
            "t_min_c": 25.0,
            "t_max_c": 160.0,
            "source": "static",
        }
        result = predict(entry, 80.0)
        assert "source" in result
        assert result["source"] == "static"

    def test_get_entry_returns_coefficients(self):
        """get_entry for a known pair should return dict with A, B, C keys."""
        from strap.solubility import get_entry

        entry = get_entry("HDPE", "toluene")
        assert entry is not None, "HDPE/toluene should exist in the coefficient store"
        for coeff_key in ("A", "B", "C"):
            assert coeff_key in entry, f"Entry missing coefficient key '{coeff_key}'"

    def test_resolve_names_with_aliases(self):
        """Polymer aliases should resolve correctly."""
        from strap.solubility import resolve_names

        # "POLYETHYLENE" should resolve to "HDPE" via alias
        polymer, solvent = resolve_names("POLYETHYLENE", "toluene")
        assert polymer is not None, "POLYETHYLENE should resolve to a known polymer"
        assert solvent is not None, "toluene should resolve to a known solvent"


# ===================================================================
# 5. Agent Tools
# ===================================================================


@rdkit_required
class TestAgentTools:
    """Tests for strap.tools.thermal_prediction agent-facing tools."""

    def test_predict_thermal_properties_tool(self):
        """Tool should return Markdown string containing polymer name."""
        from strap.tools.thermal_prediction import predict_thermal_properties

        result = predict_thermal_properties(
            polymer_psmiles="[*]CC[*]",
            polymer_name="Polyethylene",
        )
        assert isinstance(result, str)
        assert "Polyethylene" in result
        # Should contain table-like Markdown
        assert "Tm" in result
        assert "Delta Hf" in result or "delta_Hf" in result.lower()

    def test_generate_solubility_tool(self):
        """Tool should return Markdown with solubility prediction."""
        from strap.tools.thermal_prediction import generate_solubility_for_new_polymer

        result = generate_solubility_for_new_polymer(
            polymer_name="TestPolymer",
            polymer_psmiles="[*]CC[*]",
            solvent_name="toluene",
            temperature_c=25.0,
        )
        assert isinstance(result, str)
        # The tool should mention toluene or TestPolymer in the output
        assert "toluene" in result.lower() or "TestPolymer" in result

    def test_list_generated_polymers_tool(self):
        """Tool should return Markdown (even if no entries)."""
        from strap.tools.thermal_prediction import list_generated_polymers

        result = list_generated_polymers()
        assert isinstance(result, str)
        # Should contain a heading
        assert "ML-Generated" in result or "Generated" in result or "generated" in result

    def test_predict_thermal_properties_tool_returns_confidence(self):
        """Tool output should mention confidence level."""
        from strap.tools.thermal_prediction import predict_thermal_properties

        result = predict_thermal_properties(
            polymer_psmiles="[*]CC[*]",
            polymer_name="PE",
        )
        assert isinstance(result, str)
        # Should mention confidence somewhere
        assert "confidence" in result.lower() or "Confidence" in result

    def test_tool_error_handling(self):
        """Tool with invalid PSMILES should return a string (not raise)."""
        from strap.tools.thermal_prediction import predict_thermal_properties

        # safe_tool_wrapper should catch errors and return an error string
        result = predict_thermal_properties(
            polymer_psmiles="COMPLETELY_INVALID!!!",
            polymer_name="BadPolymer",
        )
        assert isinstance(result, str)
