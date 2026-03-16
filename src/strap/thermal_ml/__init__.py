"""Thermal property ML module for STRAP v7.

Physics-informed ML for polymer thermal properties (Tm, delta_Hf, delta_Cp).
Uses Van Krevelen group contribution baselines with ML residual learning.

Public API
----------
predict_thermal_properties(psmiles, use_ml=True) -> dict
    Main entry point: returns Tm, delta_Hf, delta_Cp with uncertainties.
get_group_contribution_estimate(psmiles) -> dict
    Direct Van Krevelen baseline estimates (no ML).
is_model_available() -> bool
    Check whether trained polyBERT weights exist on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from strap.paths import get_models_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = get_models_path("thermal", "polybert_thermal_best.pt")
_DEFAULT_SCALER_PATH = get_models_path("thermal", "thermal_scaler.pkl")
_DEFAULT_METADATA_PATH = get_models_path("thermal", "thermal_metadata.json")

# ---------------------------------------------------------------------------
# Group contribution imports (always available)
# ---------------------------------------------------------------------------

from strap.thermal_ml.group_contribution import (
    estimate_all,
    estimate_delta_cp,
    estimate_delta_hf,
    estimate_tm,
    parse_psmiles_groups,
)

# ---------------------------------------------------------------------------
# Lazy-loaded ML model singleton
# ---------------------------------------------------------------------------

_MODEL_INSTANCE: Optional[object] = None
_TOKENIZER_INSTANCE: Optional[object] = None
_SCALER_INSTANCE: Optional[object] = None
_MODEL_LOAD_ATTEMPTED: bool = False


def is_model_available(model_path: Optional[str] = None) -> bool:
    """Check whether trained polyBERT thermal weights exist on disk.

    Parameters
    ----------
    model_path : str, optional
        Override path to the model checkpoint.  Defaults to
        ``models/thermal/polybert_thermal_best.pt`` relative to the project root.

    Returns
    -------
    bool
        True if the weights file exists and torch can be imported.
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not path.is_file():
        return False
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load_model(
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
) -> tuple:
    """Lazy-load the polyBERT thermal model, tokenizer, and target scaler.

    Uses a singleton pattern so the model is only loaded once across the
    lifetime of the process.

    Returns
    -------
    tuple of (model, tokenizer, scaler) or (None, None, None) on failure.
    """
    global _MODEL_INSTANCE, _TOKENIZER_INSTANCE, _SCALER_INSTANCE, _MODEL_LOAD_ATTEMPTED

    if _MODEL_LOAD_ATTEMPTED:
        return _MODEL_INSTANCE, _TOKENIZER_INSTANCE, _SCALER_INSTANCE

    _MODEL_LOAD_ATTEMPTED = True

    mp = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    sp = Path(scaler_path) if scaler_path else _DEFAULT_SCALER_PATH

    if not mp.is_file():
        logger.info("Thermal model weights not found at %s; ML predictions disabled.", mp)
        return None, None, None

    try:
        import pickle
        import torch
        from transformers import AutoTokenizer
        from strap.thermal_ml.model import ThermalPropertyPredictor
    except ImportError as exc:
        logger.warning("Cannot load thermal ML model (missing dependency: %s).", exc)
        return None, None, None

    try:
        # Load tokenizer (polyBERT)
        tokenizer = AutoTokenizer.from_pretrained("kuelumbus/polyBERT")
        _TOKENIZER_INSTANCE = tokenizer

        # Reconstruct model and load weights
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ThermalPropertyPredictor()
        state_dict = torch.load(mp, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        _MODEL_INSTANCE = model

        # Load target scaler
        if sp.is_file():
            with open(sp, "rb") as f:
                _SCALER_INSTANCE = pickle.load(f)  # noqa: S301
            logger.info("Loaded target scaler from %s", sp)
        else:
            logger.warning("Target scaler not found at %s; predictions will be in scaled units.", sp)

        logger.info("Thermal ML model loaded from %s (device=%s).", mp, device)
    except Exception:
        logger.exception("Failed to load thermal ML model from %s.", mp)
        _MODEL_INSTANCE = None
        _TOKENIZER_INSTANCE = None
        _SCALER_INSTANCE = None

    return _MODEL_INSTANCE, _TOKENIZER_INSTANCE, _SCALER_INSTANCE


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_group_contribution_estimate(psmiles: str) -> dict:
    """Compute Van Krevelen group contribution estimates for a PSMILES string.

    Parameters
    ----------
    psmiles : str
        Polymer SMILES (PSMILES) representation.

    Returns
    -------
    dict
        Keys: ``Tm_K``, ``delta_Hf_J_per_mol``, ``delta_Cp_J_per_mol_K``,
        ``groups``, ``coverage``.
    """
    raw = estimate_all(psmiles)
    if raw is None:
        return {
            "Tm_K": float("nan"),
            "delta_Hf_J_per_mol": float("nan"),
            "delta_Cp_J_per_mol_K": float("nan"),
            "coverage": 0.0,
            "groups": {},
            "reliable": False,
        }
    # Flatten nested structure into expected keys
    tm_val = raw.get("tm", {}).get("value", float("nan"))
    hf_val = raw.get("delta_hf", {}).get("value", float("nan"))
    cp_val = raw.get("delta_cp", {}).get("value", float("nan"))
    coverage = raw.get("group_parse", {}).get("coverage", 0.0)
    groups = raw.get("group_parse", {}).get("groups", {})
    return {
        "Tm_K": tm_val if tm_val is not None else float("nan"),
        "delta_Hf_J_per_mol": hf_val if hf_val is not None else float("nan"),
        "delta_Cp_J_per_mol_K": cp_val if cp_val is not None else float("nan"),
        "coverage": coverage,
        "groups": groups,
        "reliable": raw.get("overall_reliable", False),
    }


# ---------------------------------------------------------------------------
# Confidence classification
# ---------------------------------------------------------------------------

# Thresholds will be refined after Phase 2 sensitivity analysis.
# Current values are conservative placeholders.
_UNCERTAINTY_THRESHOLDS = {
    "Tm": {"high": 10.0, "medium": 25.0},       # K
    "delta_Hf": {"high": 2000.0, "medium": 5000.0},  # J/mol
    "delta_Cp": {"high": 5.0, "medium": 15.0},   # J/(mol*K)
}

_COVERAGE_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.5,
}


def _classify_confidence(
    uncertainties: dict[str, float],
    coverage: float,
) -> str:
    """Classify overall prediction confidence as high / medium / low.

    Parameters
    ----------
    uncertainties : dict
        Mapping of property name to standard deviation, e.g.
        ``{"Tm": 8.2, "delta_Hf": 1500, "delta_Cp": 3.1}``.
    coverage : float
        Fraction of repeat-unit groups recognised by the group contribution
        scheme (0.0 to 1.0).

    Returns
    -------
    str
        One of ``"high"``, ``"medium"``, ``"low"``.
    """
    # Coverage gate
    if coverage < _COVERAGE_THRESHOLDS["medium"]:
        return "low"

    # Count how many properties have "high" or "medium" uncertainty
    high_count = 0
    medium_count = 0
    for prop, std in uncertainties.items():
        thresholds = _UNCERTAINTY_THRESHOLDS.get(prop)
        if thresholds is None:
            continue
        if std <= thresholds["high"]:
            high_count += 1
        elif std <= thresholds["medium"]:
            medium_count += 1

    total_props = len([p for p in uncertainties if p in _UNCERTAINTY_THRESHOLDS])
    if total_props == 0:
        # Fall back to coverage only
        if coverage >= _COVERAGE_THRESHOLDS["high"]:
            return "medium"
        return "low"

    # All properties high uncertainty + good coverage -> high
    if high_count == total_props and coverage >= _COVERAGE_THRESHOLDS["high"]:
        return "high"
    # Most properties at least medium
    if (high_count + medium_count) >= total_props and coverage >= _COVERAGE_THRESHOLDS["medium"]:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_thermal_properties(
    psmiles: str,
    use_ml: bool = True,
    model_path: Optional[str] = None,
    scaler_path: Optional[str] = None,
    n_mc_samples: int = 50,
) -> dict:
    """Predict Tm, delta_Hf, delta_Cp for a polymer from its PSMILES.

    The function always computes Van Krevelen group contribution baselines
    first.  When a trained polyBERT model is available and ``use_ml`` is True,
    it runs MC Dropout inference on the residual-learning heads and adds the
    residuals to the baselines.  If the ML model is unavailable the function
    falls back to pure group contribution estimates.

    Parameters
    ----------
    psmiles : str
        Polymer SMILES representation.
    use_ml : bool, optional
        Whether to attempt ML prediction (default True).
    model_path : str, optional
        Override path to model weights.
    scaler_path : str, optional
        Override path to target scaler pickle.
    n_mc_samples : int, optional
        Number of MC Dropout forward passes for uncertainty (default 50).

    Returns
    -------
    dict
        Keys:
        - ``Tm_K`` (float): melting temperature in Kelvin
        - ``Tm_std_K`` (float): standard deviation of Tm
        - ``delta_Hf_J_per_mol`` (float): enthalpy of fusion
        - ``delta_Hf_std`` (float): std of delta_Hf
        - ``delta_Cp_J_per_mol_K`` (float): heat capacity change at Tm
        - ``delta_Cp_std`` (float): std of delta_Cp
        - ``method`` (str): ``"ml+group_contribution"`` or ``"group_contribution_only"``
        - ``confidence`` (str): ``"high"`` | ``"medium"`` | ``"low"``
        - ``group_contribution_baselines`` (dict)
    """
    # ------------------------------------------------------------------
    # Step 1: Group contribution baselines (always computed)
    # ------------------------------------------------------------------
    gc_estimates = get_group_contribution_estimate(psmiles)
    coverage = gc_estimates.get("coverage", 0.0)

    baselines = {
        "Tm_K": gc_estimates.get("Tm_K", float("nan")),
        "delta_Hf_J_per_mol": gc_estimates.get("delta_Hf_J_per_mol", float("nan")),
        "delta_Cp_J_per_mol_K": gc_estimates.get("delta_Cp_J_per_mol_K", float("nan")),
        "coverage": coverage,
    }

    # ------------------------------------------------------------------
    # Step 2: ML prediction (residual learning on top of baselines)
    # ------------------------------------------------------------------
    if use_ml:
        model, tokenizer, scaler = _load_model(model_path, scaler_path)
    else:
        model, tokenizer, scaler = None, None, None

    if model is not None and tokenizer is not None:
        try:
            from strap.thermal_ml.uncertainty import mc_dropout_predict

            mc_results = mc_dropout_predict(
                model=model,
                psmiles_list=[psmiles],
                tokenizer=tokenizer,
                n_samples=n_mc_samples,
                group_baselines={
                    "Tm": baselines["Tm_K"],
                    "delta_Hf": baselines["delta_Hf_J_per_mol"],
                    "delta_Cp": baselines["delta_Cp_J_per_mol_K"],
                },
            )

            # Inverse-transform predictions if scaler is available
            property_keys = [("Tm", "Tm_K", "Tm_std_K"),
                             ("delta_Hf", "delta_Hf_J_per_mol", "delta_Hf_std"),
                             ("delta_Cp", "delta_Cp_J_per_mol_K", "delta_Cp_std")]

            predictions: dict = {}
            uncertainties: dict[str, float] = {}

            for mc_key, pred_key, std_key in property_keys:
                if mc_key not in mc_results:
                    continue
                mean_val = float(mc_results[mc_key]["mean"][0])
                std_val = float(mc_results[mc_key]["std"][0])

                # Inverse-transform if scaler available
                if scaler is not None and hasattr(scaler, "inverse_transform_property"):
                    mean_val, std_val = scaler.inverse_transform_property(
                        mc_key, mean_val, std_val,
                    )

                predictions[pred_key] = mean_val
                predictions[std_key] = std_val
                uncertainties[mc_key] = std_val

            confidence = _classify_confidence(uncertainties, coverage)

            return {
                "Tm_K": predictions.get("Tm_K", baselines["Tm_K"]),
                "Tm_std_K": predictions.get("Tm_std_K", float("nan")),
                "delta_Hf_J_per_mol": predictions.get("delta_Hf_J_per_mol", baselines["delta_Hf_J_per_mol"]),
                "delta_Hf_std": predictions.get("delta_Hf_std", float("nan")),
                "delta_Cp_J_per_mol_K": predictions.get("delta_Cp_J_per_mol_K", baselines["delta_Cp_J_per_mol_K"]),
                "delta_Cp_std": predictions.get("delta_Cp_std", float("nan")),
                "method": "ml+group_contribution",
                "confidence": confidence,
                "group_contribution_baselines": baselines,
            }

        except Exception:
            logger.exception("ML prediction failed for PSMILES=%s; falling back to group contribution.", psmiles)

    # ------------------------------------------------------------------
    # Step 3: Fallback — pure group contribution
    # ------------------------------------------------------------------
    # Assign conservative uncertainty estimates based on coverage alone.
    gc_std_Tm = 30.0 if coverage >= _COVERAGE_THRESHOLDS["medium"] else 60.0
    gc_std_Hf = 8000.0 if coverage >= _COVERAGE_THRESHOLDS["medium"] else 15000.0
    gc_std_Cp = 20.0 if coverage >= _COVERAGE_THRESHOLDS["medium"] else 40.0

    uncertainties_gc = {"Tm": gc_std_Tm, "delta_Hf": gc_std_Hf, "delta_Cp": gc_std_Cp}
    confidence = _classify_confidence(uncertainties_gc, coverage)

    return {
        "Tm_K": baselines["Tm_K"],
        "Tm_std_K": gc_std_Tm,
        "delta_Hf_J_per_mol": baselines["delta_Hf_J_per_mol"],
        "delta_Hf_std": gc_std_Hf,
        "delta_Cp_J_per_mol_K": baselines["delta_Cp_J_per_mol_K"],
        "delta_Cp_std": gc_std_Cp,
        "method": "group_contribution_only",
        "confidence": confidence,
        "group_contribution_baselines": baselines,
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Main API
    "predict_thermal_properties",
    "get_group_contribution_estimate",
    "is_model_available",
    # Group contribution re-exports
    "parse_psmiles_groups",
    "estimate_delta_hf",
    "estimate_delta_cp",
    "estimate_tm",
    "estimate_all",
]
