"""COSMO-RS SLE/LLE calculation interface for STRAP v7.

Pipeline position:
    PSMILES -> ML Model -> (Tm, dHf, dCp) -> **COSMO-RS SLE** -> Solubility Curve -> Fit A,B,C

This module takes thermal properties predicted by the ML model (Phase 1) plus
COSMO sigma-profile files and runs SLE calculations to produce temperature-
dependent solubility curves.

The core SLE equation:
    ln(x) = -(dHf/R)(1/T - 1/Tm)
           + (dCp/R)(Tm/T - 1 - ln(Tm/T))
           + ln(gamma)

Where gamma (activity coefficient) comes from COSMO-RS.  If no COSMO-RS
backend is available the module falls back to ideal SLE (gamma = 1), which
over-predicts solubility because the non-ideal penalty is absent.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from strap.paths import get_data_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_GAS: float = 8.314  # J/(mol*K)

_COSMO_ROOT = get_data_path("cosmo_files")
_POLYMER_DIR = _COSMO_ROOT / "polymers"
_SOLVENT_DIR = _COSMO_ROOT / "solvents"


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_cosmo_backend() -> str | None:
    """Check which COSMO-RS backend is available on this system.

    Detection order:
        1. COSMOtherm (commercial) -- via ``COSMOTHERM_PATH`` env-var or
           ``which cosmotherm``.
        2. openCOSMO-RS (open-source Python package ``opencosmo``).

    Returns
    -------
    str or None
        ``"cosmotherm"``, ``"opencosmo"``, or ``None`` if nothing found.
    """
    # 1. COSMOtherm
    cosmo_path = os.environ.get("COSMOTHERM_PATH")
    if cosmo_path and Path(cosmo_path).is_file():
        return "cosmotherm"
    if shutil.which("cosmotherm") is not None:
        return "cosmotherm"

    # 2. openCOSMO-RS
    try:
        import opencosmo  # noqa: F401
        return "opencosmo"
    except ImportError:
        pass

    return None


# ---------------------------------------------------------------------------
# COSMO file management
# ---------------------------------------------------------------------------

def _ensure_cosmo_dirs() -> None:
    """Create the default COSMO-file directories if they don't exist."""
    _POLYMER_DIR.mkdir(parents=True, exist_ok=True)
    _SOLVENT_DIR.mkdir(parents=True, exist_ok=True)


def list_available_cosmo_files(cosmo_dir: str | Path | None = None) -> dict[str, list[str]]:
    """Scan default (or custom) directories for ``.cosmo`` files.

    Parameters
    ----------
    cosmo_dir : str or Path, optional
        If given, scan this single directory instead of the defaults.

    Returns
    -------
    dict
        ``{"polymers": [...], "solvents": [...]}`` with stem names.
    """
    _ensure_cosmo_dirs()

    if cosmo_dir is not None:
        p = Path(cosmo_dir)
        files = sorted(f.stem for f in p.glob("*.cosmo"))
        return {"polymers": files, "solvents": files}

    polymers = sorted(f.stem for f in _POLYMER_DIR.glob("*.cosmo"))
    solvents = sorted(f.stem for f in _SOLVENT_DIR.glob("*.cosmo"))
    return {"polymers": polymers, "solvents": solvents}


# ---------------------------------------------------------------------------
# Ideal SLE
# ---------------------------------------------------------------------------

def compute_ideal_sle(
    T_K: float | np.ndarray,
    Tm_K: float,
    delta_Hf: float,
    delta_Cp: float | None = None,
) -> np.ndarray:
    """Compute mole-fraction solubility from the *ideal* SLE equation (gamma=1).

    Parameters
    ----------
    T_K : float or array-like
        Temperature(s) in Kelvin.
    Tm_K : float
        Melting temperature in Kelvin.
    delta_Hf : float
        Enthalpy of fusion in J/mol.
    delta_Cp : float or None
        Heat-capacity difference (liquid - solid) in J/(mol*K).  If ``None``
        the Cp correction term is omitted (equivalent to delta_Cp = 0).

    Returns
    -------
    np.ndarray
        Mole-fraction solubility at each temperature.  Values are clamped to
        [0, 1].  Temperatures >= Tm return 1.0 (fully molten / miscible).
    """
    T = np.asarray(T_K, dtype=np.float64)
    ln_x = -(delta_Hf / R_GAS) * (1.0 / T - 1.0 / Tm_K)

    if delta_Cp is not None and delta_Cp != 0.0:
        ln_x += (delta_Cp / R_GAS) * (Tm_K / T - 1.0 - np.log(Tm_K / T))

    x = np.exp(ln_x)
    # Clamp: T >= Tm means polymer melts -> solubility = 1
    x = np.where(T >= Tm_K, 1.0, x)
    # Clamp negatives (shouldn't happen with exp, but guard numerics)
    x = np.clip(x, 0.0, 1.0)
    return x


# ---------------------------------------------------------------------------
# COSMOtherm input generation
# ---------------------------------------------------------------------------

def generate_cosmotherm_input(
    polymer_cosmo: str | Path,
    solvent_cosmo: str | Path,
    Tm_K: float,
    delta_Hf: float,
    delta_Cp: float,
    temperatures_K: Sequence[float],
) -> str:
    """Generate a COSMOtherm ``.inp`` input file for an SLE calculation.

    Parameters
    ----------
    polymer_cosmo : path-like
        Path to the polymer ``.cosmo`` file.
    solvent_cosmo : path-like
        Path to the solvent ``.cosmo`` file.
    Tm_K : float
        Melting temperature in Kelvin.
    delta_Hf : float
        Enthalpy of fusion in J/mol.
    delta_Cp : float
        Heat-capacity change in J/(mol*K).
    temperatures_K : sequence of float
        Temperatures (K) at which to compute SLE.

    Returns
    -------
    str
        Full content of the COSMOtherm input file.
    """
    polymer_cosmo = Path(polymer_cosmo).resolve()
    solvent_cosmo = Path(solvent_cosmo).resolve()

    # COSMOtherm expects dHf in kJ/mol
    delta_Hf_kJ = delta_Hf / 1000.0

    temp_lines = "\n".join(f"  T={t:.2f}" for t in temperatures_K)

    inp = textwrap.dedent(f"""\
        ! COSMOtherm SLE input -- generated by STRAP v7
        ! Polymer: {polymer_cosmo.stem}
        ! Solvent: {solvent_cosmo.stem}

        ctd = BP_TZVPD_FINE_25.ctd  cdir = "/software/cosmo/COSMOthermX25/COSMOtherm/CTDATA-FILES"

        f = "{polymer_cosmo}" fdir="." comp = polymer
        f = "{solvent_cosmo}" fdir="." comp = solvent

        ! Thermal data for the polymer (melting species)
        therm
          Tm={Tm_K:.2f}
          dHf={delta_Hf_kJ:.4f}
          dCp={delta_Cp:.4f}
        end

        ! SLE calculation
        ftype=SLE

        ! Temperature list
        {temp_lines}
    """)
    return inp


# ---------------------------------------------------------------------------
# COSMOtherm output parsing
# ---------------------------------------------------------------------------

def parse_cosmotherm_output(output_path: str | Path) -> pd.DataFrame:
    """Parse a COSMOtherm ``.tab`` output file from an SLE job.

    The ``.tab`` file is whitespace-delimited.  We look for columns that
    contain temperature, activity coefficient (ln gamma), and solubility.

    Parameters
    ----------
    output_path : path-like
        Path to the ``.tab`` result file.

    Returns
    -------
    pd.DataFrame
        Columns: ``temperature_k``, ``ln_gamma``, ``x_solubility``.

    Raises
    ------
    FileNotFoundError
        If *output_path* does not exist.
    ValueError
        If the file cannot be parsed into the expected structure.
    """
    output_path = Path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"COSMOtherm output not found: {output_path}")

    # Read all lines, skip comment/header lines starting with '#' or empty
    lines: list[str] = []
    header: list[str] | None = None
    with open(output_path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # First non-comment line with text is the header
            if header is None:
                header = line.split()
                continue
            # Data lines
            lines.append(line)

    if header is None or not lines:
        raise ValueError(f"Could not parse COSMOtherm output: {output_path}")

    # Build DataFrame from whitespace-separated values
    data = [row.split() for row in lines]
    df = pd.DataFrame(data, columns=header[: len(data[0])])

    # Attempt to convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise column names -- COSMOtherm output names vary by version.
    # We search for common patterns.
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower()
        if "temp" in cl or cl == "t" or cl == "t/k":
            col_map[col] = "temperature_k"
        elif "ln" in cl and "gam" in cl:
            col_map[col] = "ln_gamma"
        elif "x(" in cl or "xsol" in cl or "solub" in cl:
            col_map[col] = "x_solubility"

    df.rename(columns=col_map, inplace=True)

    # Ensure required columns exist; fill ln_gamma with 0 if absent
    if "temperature_k" not in df.columns:
        # Fallback: assume first column is temperature
        df.rename(columns={df.columns[0]: "temperature_k"}, inplace=True)
    if "ln_gamma" not in df.columns:
        df["ln_gamma"] = 0.0
    if "x_solubility" not in df.columns:
        # Fallback: assume last column is solubility
        df.rename(columns={df.columns[-1]: "x_solubility"}, inplace=True)

    return df[["temperature_k", "ln_gamma", "x_solubility"]].copy()


# ---------------------------------------------------------------------------
# Full SLE calculation (main entry point)
# ---------------------------------------------------------------------------

def run_sle_calculation(
    polymer_cosmo_file: str | Path | None,
    solvent_cosmo_file: str | Path | None,
    Tm_K: float,
    delta_Hf: float,
    delta_Cp: float,
    t_range_c: tuple[float, float] = (25, 160),
    t_step_c: float = 5.0,
    cosmo_backend: str = "cosmotherm",
) -> pd.DataFrame:
    """Run a full SLE calculation and return a solubility-vs-temperature table.

    This is the main entry point.  It attempts to use a COSMO-RS backend for
    activity-coefficient-corrected SLE; if the requested backend is not
    available it falls back to ideal SLE with a logged warning.

    Parameters
    ----------
    polymer_cosmo_file : path-like or None
        Path to the polymer ``.cosmo`` file.  ``None`` forces ideal fallback.
    solvent_cosmo_file : path-like or None
        Path to the solvent ``.cosmo`` file.  ``None`` forces ideal fallback.
    Tm_K : float
        Melting temperature (K).
    delta_Hf : float
        Enthalpy of fusion (J/mol).
    delta_Cp : float
        Heat-capacity difference (J/(mol*K)).
    t_range_c : tuple[float, float]
        Temperature range in Celsius (inclusive endpoints).
    t_step_c : float
        Temperature step in Celsius.
    cosmo_backend : str
        ``"cosmotherm"``, ``"opencosmo"``, or ``"ideal"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``temperature_c``, ``temperature_k``, ``solubility_pct``,
        ``ln_gamma``, ``x_ideal``, ``x_total``, ``source``.
    """
    temps_c = np.arange(t_range_c[0], t_range_c[1] + t_step_c / 2, t_step_c)
    temps_k = temps_c + 273.15

    # Ideal contribution (always computed)
    x_ideal = compute_ideal_sle(temps_k, Tm_K, delta_Hf, delta_Cp)

    # Decide backend --------------------------------------------------------
    use_ideal_fallback = False
    source = cosmo_backend

    if cosmo_backend == "ideal":
        use_ideal_fallback = True
        source = "ideal"
    elif polymer_cosmo_file is None or solvent_cosmo_file is None:
        logger.warning(
            "COSMO files not provided -- falling back to ideal SLE. "
            "Solubility will be OVER-predicted (no activity-coefficient correction)."
        )
        use_ideal_fallback = True
        source = "ideal (no COSMO files)"
    else:
        available = detect_cosmo_backend()
        if cosmo_backend == "cosmotherm":
            if available != "cosmotherm":
                logger.warning(
                    "COSMOtherm not detected on this system -- falling back to ideal SLE. "
                    "Solubility values will be HIGHER than reality (gamma correction missing)."
                )
                use_ideal_fallback = True
                source = "ideal (COSMOtherm unavailable)"
            # else: run cosmotherm below
        elif cosmo_backend == "opencosmo":
            if available != "opencosmo" and available != "cosmotherm":
                logger.warning(
                    "openCOSMO-RS not installed -- falling back to ideal SLE."
                )
                use_ideal_fallback = True
                source = "ideal (openCOSMO-RS unavailable)"

    # --- Ideal fallback path -----------------------------------------------
    if use_ideal_fallback:
        ln_gamma = np.zeros_like(temps_k)
        x_total = x_ideal.copy()
    # --- COSMOtherm path ---------------------------------------------------
    elif cosmo_backend == "cosmotherm":
        ln_gamma, x_total = _run_cosmotherm(
            polymer_cosmo_file, solvent_cosmo_file,
            Tm_K, delta_Hf, delta_Cp, temps_k,
        )
        source = "cosmotherm"
    # --- openCOSMO-RS path -------------------------------------------------
    elif cosmo_backend == "opencosmo":
        ln_gamma, x_total = _run_opencosmo(
            polymer_cosmo_file, solvent_cosmo_file,
            Tm_K, delta_Hf, delta_Cp, temps_k,
        )
        source = "opencosmo"
    else:
        raise ValueError(f"Unknown COSMO backend: {cosmo_backend!r}")

    # Clamp
    x_total = np.clip(x_total, 0.0, 1.0)
    x_total = np.where(temps_k >= Tm_K, 1.0, x_total)

    solubility_pct = x_total * 100.0

    df = pd.DataFrame({
        "temperature_c": temps_c,
        "temperature_k": temps_k,
        "solubility_pct": solubility_pct,
        "ln_gamma": ln_gamma,
        "x_ideal": x_ideal,
        "x_total": x_total,
        "source": source,
    })
    return df


# ---------------------------------------------------------------------------
# Backend runners (private)
# ---------------------------------------------------------------------------

def _run_cosmotherm(
    polymer_cosmo: str | Path,
    solvent_cosmo: str | Path,
    Tm_K: float,
    delta_Hf: float,
    delta_Cp: float,
    temperatures_K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute COSMOtherm and return (ln_gamma, x_total) arrays."""
    import tempfile

    inp_content = generate_cosmotherm_input(
        polymer_cosmo, solvent_cosmo, Tm_K, delta_Hf, delta_Cp, temperatures_K,
    )

    with tempfile.TemporaryDirectory(prefix="strap_cosmo_") as tmpdir:
        inp_path = Path(tmpdir) / "sle_calc.inp"
        inp_path.write_text(inp_content)

        cosmo_bin = os.environ.get("COSMOTHERM_PATH", "cosmotherm")
        try:
            result = subprocess.run(
                [cosmo_bin, str(inp_path)],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=tmpdir,
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"COSMOtherm binary not found at {cosmo_bin!r}. "
                "Set COSMOTHERM_PATH or add cosmotherm to PATH."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("COSMOtherm calculation timed out (>600 s).")

        if result.returncode != 0:
            raise RuntimeError(
                f"COSMOtherm exited with code {result.returncode}.\n"
                f"STDOUT:\n{result.stdout[:2000]}\n"
                f"STDERR:\n{result.stderr[:2000]}"
            )

        # Look for .tab output
        tab_files = list(Path(tmpdir).glob("*.tab"))
        if not tab_files:
            raise RuntimeError(
                "COSMOtherm produced no .tab output file. "
                f"STDOUT:\n{result.stdout[:2000]}"
            )

        parsed = parse_cosmotherm_output(tab_files[0])

    # Align parsed data with our temperature grid
    ln_gamma = np.interp(
        temperatures_K, parsed["temperature_k"].values, parsed["ln_gamma"].values,
    )
    x_total = np.interp(
        temperatures_K, parsed["temperature_k"].values, parsed["x_solubility"].values,
    )
    return ln_gamma, x_total


def _run_opencosmo(
    polymer_cosmo: str | Path,
    solvent_cosmo: str | Path,
    Tm_K: float,
    delta_Hf: float,
    delta_Cp: float,
    temperatures_K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run openCOSMO-RS and return (ln_gamma, x_total) arrays."""
    try:
        from opencosmo import COSMORS  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "openCOSMO-RS is not installed.  Install via: pip install opencosmo-rs"
        )

    polymer_cosmo = str(Path(polymer_cosmo).resolve())
    solvent_cosmo = str(Path(solvent_cosmo).resolve())

    ln_gamma_arr = np.zeros(len(temperatures_K))
    x_total_arr = np.zeros(len(temperatures_K))

    for i, T in enumerate(temperatures_K):
        try:
            model = COSMORS([polymer_cosmo, solvent_cosmo], T=T)
            # openCOSMO-RS returns activity coefficients at infinite dilution
            gamma = model.activity_coefficients()
            # gamma[0] is the polymer activity coefficient
            ln_g = np.log(gamma[0]) if gamma[0] > 0 else 0.0
            ln_gamma_arr[i] = ln_g

            # Combine ideal + non-ideal
            ln_x_ideal = -(delta_Hf / R_GAS) * (1.0 / T - 1.0 / Tm_K)
            if delta_Cp != 0.0:
                ln_x_ideal += (delta_Cp / R_GAS) * (Tm_K / T - 1.0 - np.log(Tm_K / T))
            ln_x = ln_x_ideal + ln_g
            x_total_arr[i] = np.exp(ln_x)
        except Exception as exc:
            logger.warning("openCOSMO-RS failed at T=%.1f K: %s", T, exc)
            # Fall back to ideal for this temperature
            x_total_arr[i] = compute_ideal_sle(T, Tm_K, delta_Hf, delta_Cp)[0]
            ln_gamma_arr[i] = 0.0

    return ln_gamma_arr, x_total_arr


# ---------------------------------------------------------------------------
# Uncertainty propagation
# ---------------------------------------------------------------------------

def run_sle_with_uncertainty(
    polymer_cosmo: str | Path | None,
    solvent_cosmo: str | Path | None,
    Tm_K: float,
    delta_Hf: float,
    delta_Cp: float,
    Tm_std: float,
    delta_Hf_std: float,
    delta_Cp_std: float,
    n_samples: int = 100,  # kept for API compat; analytical method ignores it
    t_range_c: tuple[float, float] = (25, 160),
    t_step_c: float = 5.0,
) -> dict:
    """Analytical uncertainty propagation through the ideal SLE equation.

    Partial derivatives of ln(x) with respect to each thermal property are
    computed symbolically and combined via standard error propagation:

        sigma^2(ln_x) = (d_ln_x/d_Tm)^2 * sigma_Tm^2
                      + (d_ln_x/d_dHf)^2 * sigma_dHf^2
                      + (d_ln_x/d_dCp)^2 * sigma_dCp^2

    Parameters
    ----------
    polymer_cosmo, solvent_cosmo : path-like or None
        Not used for analytical propagation (ideal SLE only) but accepted for
        a consistent API.
    Tm_K, delta_Hf, delta_Cp : float
        Mean thermal property values.
    Tm_std, delta_Hf_std, delta_Cp_std : float
        Standard deviations (1-sigma) for each property.
    n_samples : int
        Unused; retained for forward-compatibility with Monte-Carlo mode.
    t_range_c : tuple
        Temperature range in Celsius.
    t_step_c : float
        Temperature step in Celsius.

    Returns
    -------
    dict
        Keys:

        - ``temperature_c`` -- array of temperatures (C)
        - ``temperature_k`` -- array of temperatures (K)
        - ``x_mean``        -- mean mole-fraction solubility
        - ``x_upper``       -- mean + 1 sigma
        - ``x_lower``       -- mean - 1 sigma
        - ``sigma_ln_x``    -- standard deviation in ln(x) space
        - ``dominant_source`` -- list of str, dominant uncertainty source per T
        - ``contributions``  -- dict of variance contribution arrays
    """
    temps_c = np.arange(t_range_c[0], t_range_c[1] + t_step_c / 2, t_step_c)
    T = temps_c + 273.15  # K

    # --- Mean curve (ideal SLE) ---
    x_mean = compute_ideal_sle(T, Tm_K, delta_Hf, delta_Cp)

    # --- Partial derivatives of ln(x) ---
    # ln(x) = -(dHf/R)(1/T - 1/Tm) + (dCp/R)(Tm/T - 1 - ln(Tm/T))
    #
    # d(ln_x)/d(Tm)  = -(dHf/R)*(1/Tm^2) + (dCp/R)*(1/T + 1/Tm)
    #   (note: d/dTm of -1/Tm = +1/Tm^2, so first term has - * 1/Tm^2
    #    actually: d/dTm [-(dHf/R)(1/T - 1/Tm)] = -(dHf/R)(-(-1/Tm^2)) = -(dHf/R)/Tm^2
    #    and d/dTm [(dCp/R)(Tm/T - 1 - ln(Tm/T))]
    #      = (dCp/R)(1/T - 1/Tm) ... let's be careful)

    # Recompute derivatives properly:
    # f = -(dHf/R)(1/T - 1/Tm)  +  (dCp/R)(Tm/T - 1 - ln(Tm/T))
    #
    # df/d(Tm) = -(dHf/R)*(-(-1/Tm^2))  +  (dCp/R)*(1/T - (1/Tm))
    #          = -(dHf/R)*(1/Tm^2)      +  (dCp/R)*(1/T - 1/Tm)
    #   Wait, let's redo: d/dTm of (1/T - 1/Tm) = +1/Tm^2
    #   So: d/dTm [-(dHf/R)(1/T - 1/Tm)] = -(dHf/R) * (1/Tm^2)  ... actually no:
    #     = -(dHf/R) * d/dTm(1/T - 1/Tm) = -(dHf/R)*(0 - (-1/Tm^2)) = -(dHf/R)/Tm^2
    #   Hmm that gives -(dHf/R)/Tm^2 which is negative. Let me be very explicit:
    #     d/dTm [1/T - 1/Tm] = 0 + 1/Tm^2
    #   So: -(dHf/R) * (1/Tm^2)
    #
    #   For the Cp term:
    #     d/dTm [Tm/T - 1 - ln(Tm/T)] = 1/T - 0 - (1/Tm)*(T/Tm)/1  ... no:
    #     ln(Tm/T) = ln(Tm) - ln(T), so d/dTm = 1/Tm
    #     d/dTm [Tm/T - 1 - ln(Tm) + ln(T)] = 1/T - 1/Tm
    #   So: (dCp/R)*(1/T - 1/Tm)

    dln_dTm = -(delta_Hf / R_GAS) * (1.0 / Tm_K**2)
    if delta_Cp != 0.0:
        dln_dTm = dln_dTm + (delta_Cp / R_GAS) * (1.0 / T - 1.0 / Tm_K)

    # df/d(dHf) = -(1/R)(1/T - 1/Tm)
    dln_dHf = -(1.0 / R_GAS) * (1.0 / T - 1.0 / Tm_K)

    # df/d(dCp) = (1/R)(Tm/T - 1 - ln(Tm/T))
    dln_dCp = (1.0 / R_GAS) * (Tm_K / T - 1.0 - np.log(Tm_K / T))

    # --- Variance contributions ---
    var_Tm = (dln_dTm ** 2) * (Tm_std ** 2)
    var_Hf = (dln_dHf ** 2) * (delta_Hf_std ** 2)
    var_Cp = (dln_dCp ** 2) * (delta_Cp_std ** 2)

    var_total = var_Tm + var_Hf + var_Cp
    sigma_ln_x = np.sqrt(var_total)

    # --- Bounds in mole-fraction space ---
    ln_x_mean = np.log(np.clip(x_mean, 1e-30, None))
    x_upper = np.clip(np.exp(ln_x_mean + sigma_ln_x), 0.0, 1.0)
    x_lower = np.clip(np.exp(ln_x_mean - sigma_ln_x), 0.0, 1.0)

    # At T >= Tm, uncertainty collapses (fully molten)
    above_tm = T >= Tm_K
    x_upper[above_tm] = 1.0
    x_lower[above_tm] = 1.0

    # --- Dominant source per temperature ---
    contributions = np.column_stack([var_Tm, var_Hf, var_Cp])
    labels = ["Tm", "delta_Hf", "delta_Cp"]
    dominant = []
    for row in contributions:
        total = row.sum()
        if total == 0:
            dominant.append("none")
        else:
            idx = int(np.argmax(row))
            dominant.append(labels[idx])

    return {
        "temperature_c": temps_c,
        "temperature_k": T,
        "x_mean": x_mean,
        "x_upper": x_upper,
        "x_lower": x_lower,
        "sigma_ln_x": sigma_ln_x,
        "dominant_source": dominant,
        "contributions": {
            "var_Tm": var_Tm,
            "var_delta_Hf": var_Hf,
            "var_delta_Cp": var_Cp,
        },
    }
