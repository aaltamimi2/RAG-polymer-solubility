"""MC Dropout uncertainty estimation for thermal property predictions.

This module provides Monte Carlo Dropout inference for quantifying prediction
uncertainty from the polyBERT-based thermal property model.  During inference,
dropout layers are kept active and multiple stochastic forward passes are
performed.  The resulting distribution of predictions yields calibrated
uncertainty estimates for each thermal property (Tm, delta_Hf, delta_Cp, Tg).

Usage
-----
>>> from strap.thermal_ml.uncertainty import mc_dropout_predict, classify_confidence
>>> results = mc_dropout_predict(model, ["[*]CC[*]"], tokenizer, n_samples=50)
>>> results["Tm"]["mean"]   # array of shape (n_polymers,)
>>> results["Tm"]["std"]    # array of shape (n_polymers,)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Confidence thresholds (placeholders — to be calibrated in Phase 2)
# ---------------------------------------------------------------------------

# Per-property standard-deviation thresholds used by classify_confidence.
# "high" means the std is *at most* this value; above "medium" -> low.
_STD_THRESHOLDS: dict[str, dict[str, float]] = {
    "Tm": {"high": 10.0, "medium": 25.0},           # K
    "delta_Hf": {"high": 2000.0, "medium": 5000.0},  # J/mol
    "delta_Cp": {"high": 5.0, "medium": 15.0},       # J/(mol*K)
    "Tg": {"high": 8.0, "medium": 20.0},             # K
}

_COVERAGE_THRESHOLDS: dict[str, float] = {
    "high": 0.8,
    "medium": 0.5,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _enable_dropout(model: nn.Module) -> None:
    """Set all Dropout layers in *model* to training mode (stochastic).

    All other layers (BatchNorm, etc.) remain in eval mode so that running
    statistics are used.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def _tokenize_batch(
    psmiles_list: list[str],
    tokenizer,
    max_length: int = 512,
    device: str = "cpu",
) -> dict:
    """Tokenize a list of PSMILES strings and move tensors to *device*.

    Parameters
    ----------
    psmiles_list : list[str]
        PSMILES strings to tokenize.
    tokenizer
        HuggingFace tokenizer (e.g. ``AutoTokenizer.from_pretrained("kuelumbus/polyBERT")``).
    max_length : int
        Maximum token sequence length (default 512).
    device : str
        Target device (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    dict
        Tokenizer output dict with tensors on *device*.
    """
    encoded = tokenizer(
        psmiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in encoded.items()}


# ---------------------------------------------------------------------------
# MC Dropout prediction
# ---------------------------------------------------------------------------

# Default property keys emitted by ThermalPropertyPredictor.forward()
_DEFAULT_PROPERTY_KEYS = ("Tm", "delta_Hf", "delta_Cp", "Tg")


def mc_dropout_predict(
    model: nn.Module,
    psmiles_list: list[str],
    tokenizer,
    n_samples: int = 50,
    device: str = "cpu",
    group_baselines: dict | None = None,
    max_length: int = 512,
    batch_size: int = 64,
    property_keys: tuple[str, ...] | None = None,
) -> dict:
    """Run MC Dropout inference and return per-property statistics.

    The model is set to eval mode (so that BatchNorm uses running stats),
    then only the Dropout layers are switched back to training mode.  For
    each of *n_samples* stochastic forward passes the model outputs are
    collected.  If *group_baselines* are provided the raw model outputs
    (residuals) are added to the baselines so that the returned statistics
    are in absolute units.

    Parameters
    ----------
    model : nn.Module
        A ``ThermalPropertyPredictor`` instance (already loaded to *device*).
    psmiles_list : list[str]
        PSMILES strings for which to predict properties.
    tokenizer
        HuggingFace tokenizer compatible with polyBERT.
    n_samples : int, optional
        Number of stochastic forward passes (default 50).
    device : str, optional
        ``"cpu"`` or ``"cuda"`` (default ``"cpu"``).
    group_baselines : dict or None, optional
        Mapping of property key to scalar baseline value.  When provided,
        each MC sample is ``baseline + model_residual``.
    max_length : int, optional
        Maximum token sequence length (default 512).
    batch_size : int, optional
        Maximum number of PSMILES to tokenize per micro-batch (default 64).
    property_keys : tuple[str, ...] or None, optional
        Expected output keys from the model.  If None, defaults to
        ``("Tm", "delta_Hf", "delta_Cp", "Tg")``.

    Returns
    -------
    dict
        Per-property results::

            {
                "Tm": {
                    "mean": np.ndarray of shape (n_polymers,),
                    "std": np.ndarray of shape (n_polymers,),
                    "samples": np.ndarray of shape (n_polymers, n_samples),
                },
                "delta_Hf": {...},
                "delta_Cp": {...},
                "Tg": {...},
            }
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for MC Dropout inference.  "
            "Install it with: pip install torch"
        )

    if property_keys is None:
        property_keys = _DEFAULT_PROPERTY_KEYS

    n_polymers = len(psmiles_list)
    if n_polymers == 0:
        return {
            key: {
                "mean": np.array([]),
                "std": np.array([]),
                "samples": np.empty((0, n_samples)),
            }
            for key in property_keys
        }

    # Prepare model: eval everything, then re-enable dropout
    model.eval()
    _enable_dropout(model)
    model.to(device)

    # Pre-allocate sample storage
    samples: dict[str, np.ndarray] = {
        key: np.zeros((n_polymers, n_samples), dtype=np.float64)
        for key in property_keys
    }

    # Run n_samples stochastic forward passes
    with torch.no_grad():
        for s_idx in range(n_samples):
            # Process in micro-batches to control memory
            all_outputs: dict[str, list[np.ndarray]] = {k: [] for k in property_keys}

            for batch_start in range(0, n_polymers, batch_size):
                batch_end = min(batch_start + batch_size, n_polymers)
                batch_psmiles = psmiles_list[batch_start:batch_end]

                inputs = _tokenize_batch(batch_psmiles, tokenizer, max_length, device)
                outputs = model(**inputs)

                # Model may return a dict or a NamedTuple / dataclass.
                # Normalise to dict.
                if isinstance(outputs, dict):
                    out_dict = outputs
                elif hasattr(outputs, "_asdict"):
                    out_dict = outputs._asdict()
                elif hasattr(outputs, "__dict__"):
                    out_dict = vars(outputs)
                else:
                    # Assume outputs is a single tensor (combined predictions)
                    # with columns ordered as property_keys.
                    out_tensor = outputs.cpu().numpy()
                    out_dict = {}
                    for col_idx, key in enumerate(property_keys):
                        if col_idx < out_tensor.shape[-1]:
                            out_dict[key] = out_tensor[..., col_idx]
                        else:
                            break

                for key in property_keys:
                    if key in out_dict:
                        val = out_dict[key]
                        if hasattr(val, "cpu"):
                            val = val.cpu().numpy()
                        val = np.asarray(val, dtype=np.float64).reshape(-1)
                        all_outputs[key].append(val)

            # Concatenate micro-batches and store
            for key in property_keys:
                if all_outputs[key]:
                    concatenated = np.concatenate(all_outputs[key])
                    # Apply group contribution baseline (residual learning)
                    if group_baselines is not None and key in group_baselines:
                        baseline_val = group_baselines[key]
                        if baseline_val is not None and not math.isnan(baseline_val):
                            concatenated = concatenated + baseline_val
                    samples[key][:, s_idx] = concatenated[: n_polymers]

    # Compute summary statistics
    results: dict = {}
    for key in property_keys:
        results[key] = {
            "mean": np.mean(samples[key], axis=1),
            "std": np.std(samples[key], axis=1, ddof=1),
            "samples": samples[key],
        }

    logger.info(
        "MC Dropout inference complete: %d polymers x %d samples.",
        n_polymers,
        n_samples,
    )
    return results


# ---------------------------------------------------------------------------
# Confidence classification
# ---------------------------------------------------------------------------

def classify_confidence(predictions: dict, coverage: float) -> str:
    """Classify prediction confidence as high / medium / low.

    The classification considers both the MC Dropout standard deviations
    (epistemic uncertainty) and the group contribution coverage (fraction of
    repeat-unit groups recognised by the Van Krevelen scheme).

    Parameters
    ----------
    predictions : dict
        Output from :func:`mc_dropout_predict`.  Each value must contain a
        ``"std"`` key with a numpy array.  If multiple polymers were
        predicted, the *maximum* std across the batch is used for
        classification (conservative).
    coverage : float
        Group contribution coverage (0.0 to 1.0).  A value of 1.0 means
        all structural groups in the repeat unit were recognised.

    Returns
    -------
    str
        One of ``"high"``, ``"medium"``, ``"low"``.

    Notes
    -----
    The threshold values are initial conservative placeholders.  They will
    be calibrated against experimental data during Phase 2 sensitivity
    analysis.
    """
    # Gate on coverage first
    if coverage < _COVERAGE_THRESHOLDS["medium"]:
        return "low"

    high_count = 0
    medium_count = 0
    total_evaluated = 0

    for prop, thresholds in _STD_THRESHOLDS.items():
        if prop not in predictions:
            continue
        std_arr = predictions[prop].get("std")
        if std_arr is None or len(std_arr) == 0:
            continue

        # Use maximum std across the batch (most conservative)
        max_std = float(np.max(std_arr))
        total_evaluated += 1

        if max_std <= thresholds["high"]:
            high_count += 1
        elif max_std <= thresholds["medium"]:
            medium_count += 1

    if total_evaluated == 0:
        # No evaluable properties; rely on coverage alone
        if coverage >= _COVERAGE_THRESHOLDS["high"]:
            return "medium"
        return "low"

    # All properties within "high" thresholds and good coverage
    if high_count == total_evaluated and coverage >= _COVERAGE_THRESHOLDS["high"]:
        return "high"

    # Most properties at least within "medium" thresholds
    if (high_count + medium_count) >= total_evaluated and coverage >= _COVERAGE_THRESHOLDS["medium"]:
        return "medium"

    return "low"


# ---------------------------------------------------------------------------
# Convenience: combined prediction + classification
# ---------------------------------------------------------------------------

def predict_with_confidence(
    model: nn.Module,
    psmiles_list: list[str],
    tokenizer,
    coverage_values: list[float] | None = None,
    n_samples: int = 50,
    device: str = "cpu",
    group_baselines: dict | None = None,
) -> tuple[dict, list[str]]:
    """Run MC Dropout and classify confidence for each polymer.

    This is a convenience wrapper combining :func:`mc_dropout_predict` and
    :func:`classify_confidence`.

    Parameters
    ----------
    model : nn.Module
        Loaded ThermalPropertyPredictor.
    psmiles_list : list[str]
        Polymer SMILES strings.
    tokenizer
        HuggingFace tokenizer.
    coverage_values : list[float] or None
        Per-polymer group contribution coverage.  If None, a default
        coverage of 0.0 is assumed (worst case).
    n_samples : int
        MC Dropout samples.
    device : str
        Compute device.
    group_baselines : dict or None
        Baselines for residual learning.

    Returns
    -------
    tuple[dict, list[str]]
        (mc_results, confidence_labels) where *confidence_labels* is a
        list of ``"high"``/``"medium"``/``"low"`` per polymer.
    """
    mc_results = mc_dropout_predict(
        model=model,
        psmiles_list=psmiles_list,
        tokenizer=tokenizer,
        n_samples=n_samples,
        device=device,
        group_baselines=group_baselines,
    )

    n_polymers = len(psmiles_list)
    if coverage_values is None:
        coverage_values = [0.0] * n_polymers

    confidence_labels: list[str] = []
    for i in range(n_polymers):
        # Build a single-polymer predictions dict for classify_confidence
        single = {}
        for key, result in mc_results.items():
            single[key] = {
                "std": result["std"][i: i + 1],
            }
        label = classify_confidence(single, coverage_values[i])
        confidence_labels.append(label)

    return mc_results, confidence_labels


__all__ = [
    "mc_dropout_predict",
    "classify_confidence",
    "predict_with_confidence",
]
