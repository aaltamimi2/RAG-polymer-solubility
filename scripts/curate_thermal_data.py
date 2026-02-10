#!/usr/bin/env python3
"""
Phase 1 Data Curation for STRAP v7 — Polymer Thermal Properties

Combines multiple open datasets into a unified, deduplicated, outlier-cleaned
CSV for downstream ML modelling of Tm, delta_Hf, and delta_Cp from PSMILES.

Datasets:
  - polyVERSE  (Ramprasad-Group/polyVERSE, Zenodo 10.5281/zenodo.13352644)
  - POINT2     (Jiaxin-Xu/POINT2)
  - PI1M       (RUIMINMA1996/PI1M)
  - Local ATHAS / thermal reference CSV

Outputs:
  data/thermal_properties/combined_thermal.csv
  data/thermal_properties/pretrain_psmiles.txt
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw_downloads"
THERMAL_DIR = DATA_DIR / "thermal_properties"
LOCAL_REF = THERMAL_DIR / "polymer_thermal_reference.csv"

OUTPUT_CSV = THERMAL_DIR / "combined_thermal.csv"
OUTPUT_PRETRAIN = THERMAL_DIR / "pretrain_psmiles.txt"

UNIFIED_COLS = [
    "psmiles",
    "polymer_name",
    "Tm_K",
    "delta_Hf_J_per_mol",
    "delta_Cp_J_per_mol_K",
    "Tg_K",
    "Td_K",
    "source",
    "split",
]

# Zenodo record for polyVERSE — resolve via API to get latest download URL.
POLYVERSE_ZENODO_RECORD = "13352644"
POLYVERSE_ZENODO_API = (
    f"https://zenodo.org/api/records/{POLYVERSE_ZENODO_RECORD}"
)

# GitHub raw-file URLs
POINT2_BASE = (
    "https://raw.githubusercontent.com/Jiaxin-Xu/POINT2/main"
)
PI1M_BASE = (
    "https://raw.githubusercontent.com/RUIMINMA1996/PI1M/main"
)

POLYVERSE_GH_BASE = (
    "https://raw.githubusercontent.com/Ramprasad-Group/polyVERSE/main"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RDKit helpers (graceful fallback)
# ---------------------------------------------------------------------------

_RDKIT_AVAILABLE: Optional[bool] = None


def _check_rdkit() -> bool:
    global _RDKIT_AVAILABLE
    if _RDKIT_AVAILABLE is None:
        try:
            from rdkit import Chem  # noqa: F401
            _RDKIT_AVAILABLE = True
        except ImportError:
            _RDKIT_AVAILABLE = False
            log.warning(
                "RDKit not available — PSMILES canonicalization will use "
                "fallback string normalization only."
            )
    return _RDKIT_AVAILABLE


def canonicalize_psmiles(smi: str) -> str:
    """Return a canonical PSMILES string.

    Uses RDKit if available; otherwise applies deterministic string cleaning.
    The [*] wildcard notation is preserved for polymerisation points.
    """
    if not isinstance(smi, str) or not smi.strip():
        return ""

    smi = smi.strip()

    # Normalise wildcard notation: convert various forms to [*]
    smi = re.sub(r"\[\d*\*\]", "[*]", smi)
    smi = re.sub(r"(?<!\[)\*(?!\])", "[*]", smi)

    if _check_rdkit():
        from rdkit import Chem

        # Temporarily replace [*] with a dummy atom for RDKit parsing
        temp = smi.replace("[*]", "[Xe]")
        mol = Chem.MolFromSmiles(temp, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                canon = Chem.MolToSmiles(mol, canonical=True)
                canon = canon.replace("[Xe]", "[*]")
                return canon
            except Exception:
                pass
        # RDKit failed on this molecule — fall through to string fallback

    # Fallback: lowercase, strip whitespace (simple deterministic form)
    return smi.strip()


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path, description: str = "") -> bool:
    """Download *url* to *dest* with caching. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Cached   %s -> %s", description or url, dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Download %s -> %s", description or url, dest.name)
    try:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return True
    except Exception as exc:
        log.error("Failed to download %s: %s", description or url, exc)
        if dest.exists():
            dest.unlink()
        return False


def _download_text(url: str, dest: Path, description: str = "") -> Optional[str]:
    """Download a text file, cache it, and return its contents."""
    if _download(url, dest, description):
        return dest.read_text(encoding="utf-8", errors="replace")
    return None


# ---------------------------------------------------------------------------
# Dataset parsers
# ---------------------------------------------------------------------------

def load_polyverse() -> pd.DataFrame:
    """Load polyVERSE thermal-property data.

    Strategy:
      1. Try the GitHub repo CSV first (lightweight).
      2. Fall back to Zenodo ZIP if the repo layout has changed.
    """
    frames: list[pd.DataFrame] = []

    # --- Try GitHub raw CSVs ------------------------------------------------
    # polyVERSE stores property CSVs under data/ in the repo.
    candidate_paths = [
        "data/polyVERSE.csv",
        "data/Tm.csv",
        "data/Tg.csv",
        "data/Td.csv",
        "polyVERSE.csv",
    ]

    for rel_path in candidate_paths:
        url = f"{POLYVERSE_GH_BASE}/{rel_path}"
        dest = RAW_DIR / "polyverse" / Path(rel_path).name
        txt = _download_text(url, dest, f"polyVERSE/{rel_path}")
        if txt is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(txt))
        except Exception as exc:
            log.warning("Could not parse %s: %s", rel_path, exc)
            continue

        # Identify SMILES column
        smi_col = None
        for c in df.columns:
            if "smiles" in c.lower() or "psmiles" in c.lower():
                smi_col = c
                break
        if smi_col is None:
            log.warning("No SMILES column found in %s — skipping", rel_path)
            continue

        mapped = pd.DataFrame()
        mapped["psmiles"] = df[smi_col].astype(str)

        # Map property columns (case-insensitive search)
        col_map = {
            "Tm_K": ["tm", "tm_k", "melting"],
            "Tg_K": ["tg", "tg_k", "glass"],
            "Td_K": ["td", "td_k", "decomposition"],
        }
        for unified, patterns in col_map.items():
            for c in df.columns:
                if any(p == c.lower() or p in c.lower() for p in patterns):
                    mapped[unified] = pd.to_numeric(df[c], errors="coerce")
                    break

        # Polymer name if available
        for c in df.columns:
            if "name" in c.lower() or "polymer" in c.lower() and "smiles" not in c.lower():
                mapped["polymer_name"] = df[c].astype(str)
                break

        mapped["source"] = "polyVERSE"
        frames.append(mapped)

    # --- Zenodo fallback (ZIP) -----------------------------------------------
    if not frames:
        log.info("Trying Zenodo ZIP download for polyVERSE ...")
        try:
            api_resp = requests.get(POLYVERSE_ZENODO_API, timeout=30)
            api_resp.raise_for_status()
            record = api_resp.json()
            zip_url = None
            for f in record.get("files", []):
                if f["key"].endswith(".zip") or f["key"].endswith(".csv"):
                    zip_url = f["links"]["self"]
                    break
            if zip_url:
                dest_zip = RAW_DIR / "polyverse" / "polyVERSE_zenodo.zip"
                if _download(zip_url, dest_zip, "polyVERSE Zenodo archive"):
                    with zipfile.ZipFile(dest_zip) as zf:
                        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                        for csv_name in csv_names:
                            with zf.open(csv_name) as cf:
                                df = pd.read_csv(cf)
                            smi_col = None
                            for c in df.columns:
                                if "smiles" in c.lower():
                                    smi_col = c
                                    break
                            if smi_col is None:
                                continue
                            mapped = pd.DataFrame()
                            mapped["psmiles"] = df[smi_col].astype(str)
                            for unified, patterns in col_map.items():
                                for c in df.columns:
                                    if any(p in c.lower() for p in patterns):
                                        mapped[unified] = pd.to_numeric(
                                            df[c], errors="coerce"
                                        )
                                        break
                            mapped["source"] = "polyVERSE"
                            frames.append(mapped)
        except Exception as exc:
            log.error("Zenodo fallback failed: %s", exc)

    if not frames:
        log.warning("polyVERSE: no data loaded")
        return pd.DataFrame(columns=UNIFIED_COLS)

    result = pd.concat(frames, ignore_index=True)
    log.info("polyVERSE: loaded %d rows", len(result))
    return result


def load_point2() -> pd.DataFrame:
    """Load POINT2 Tm train/test data."""
    frames: list[pd.DataFrame] = []

    # POINT2 typically stores data under dataset/ or data/ with train/test CSVs
    candidate_files = [
        ("dataset/Tm/train.csv", "train"),
        ("dataset/Tm/test.csv", "test"),
        ("data/Tm/train.csv", "train"),
        ("data/Tm/test.csv", "test"),
        ("Tm/train.csv", "train"),
        ("Tm/test.csv", "test"),
        ("dataset/Tm_train.csv", "train"),
        ("dataset/Tm_test.csv", "test"),
    ]

    for rel_path, split in candidate_files:
        url = f"{POINT2_BASE}/{rel_path}"
        dest = RAW_DIR / "point2" / Path(rel_path).name
        # Avoid name collisions between train/test
        if "train" in rel_path:
            dest = RAW_DIR / "point2" / f"Tm_train.csv"
        else:
            dest = RAW_DIR / "point2" / f"Tm_test.csv"

        txt = _download_text(url, dest, f"POINT2/{rel_path}")
        if txt is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(txt))
        except Exception:
            continue

        smi_col = None
        for c in df.columns:
            if "smiles" in c.lower() or "psmiles" in c.lower():
                smi_col = c
                break
        if smi_col is None:
            continue

        mapped = pd.DataFrame()
        mapped["psmiles"] = df[smi_col].astype(str)

        # Look for Tm column
        for c in df.columns:
            cl = c.lower()
            if cl in ("tm", "tm_k", "melting_temperature", "value", "target", "y"):
                mapped["Tm_K"] = pd.to_numeric(df[c], errors="coerce")
                break

        mapped["source"] = "POINT2"
        mapped["split"] = split
        frames.append(mapped)
        log.info("POINT2 %s: %d rows from %s", split, len(df), rel_path)

    if not frames:
        log.warning("POINT2: no data loaded")
        return pd.DataFrame(columns=UNIFIED_COLS)

    result = pd.concat(frames, ignore_index=True)
    log.info("POINT2: loaded %d rows total", len(result))
    return result


def load_pi1m() -> pd.DataFrame:
    """Load PI1M unlabelled PSMILES for self-supervised pretraining."""
    candidate_files = [
        "PI1M.csv",
        "data/PI1M.csv",
        "dataset/PI1M.csv",
        "PI1M_smiles.csv",
        "pi1m.csv",
    ]

    for rel_path in candidate_files:
        url = f"{PI1M_BASE}/{rel_path}"
        dest = RAW_DIR / "pi1m" / Path(rel_path).name
        txt = _download_text(url, dest, f"PI1M/{rel_path}")
        if txt is None:
            continue
        try:
            df = pd.read_csv(io.StringIO(txt), low_memory=False)
        except Exception:
            # Might be one-column file without header
            try:
                df = pd.read_csv(
                    io.StringIO(txt), header=None, names=["psmiles"]
                )
            except Exception:
                continue

        # Identify SMILES column
        smi_col = None
        if "psmiles" in [c.lower() for c in df.columns]:
            for c in df.columns:
                if c.lower() == "psmiles":
                    smi_col = c
                    break
        else:
            for c in df.columns:
                if "smiles" in c.lower():
                    smi_col = c
                    break
        if smi_col is None and len(df.columns) == 1:
            smi_col = df.columns[0]
        if smi_col is None:
            continue

        mapped = pd.DataFrame()
        mapped["psmiles"] = df[smi_col].astype(str)
        mapped["source"] = "PI1M"
        log.info("PI1M: loaded %d SMILES from %s", len(mapped), rel_path)
        return mapped

    # Try a .txt variant
    for fname in ["PI1M.txt", "data/PI1M.txt"]:
        url = f"{PI1M_BASE}/{fname}"
        dest = RAW_DIR / "pi1m" / Path(fname).name
        txt = _download_text(url, dest, f"PI1M/{fname}")
        if txt is None:
            continue
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if len(lines) > 100:
            mapped = pd.DataFrame({"psmiles": lines, "source": "PI1M"})
            log.info("PI1M: loaded %d SMILES from %s", len(mapped), fname)
            return mapped

    log.warning("PI1M: no data loaded")
    return pd.DataFrame(columns=UNIFIED_COLS)


def load_local_reference() -> pd.DataFrame:
    """Load the hand-curated local ATHAS / thermal reference CSV."""
    if not LOCAL_REF.exists():
        log.warning("Local reference not found at %s", LOCAL_REF)
        return pd.DataFrame(columns=UNIFIED_COLS)

    df = pd.read_csv(LOCAL_REF)
    log.info("Local reference: %d rows from %s", len(df), LOCAL_REF.name)

    # Well-known PSMILES for common polymers (used when CSV has names only)
    KNOWN_PSMILES = {
        "HDPE": "[*]CC[*]",
        "LDPE": "[*]CC[*]",
        "PE": "[*]CC[*]",
        "PP": "[*]CC(C)[*]",
        "PS": "[*]CC(c1ccccc1)[*]",
        "PVC": "[*]CC(Cl)[*]",
        "PET": "[*]OC(=O)c1ccc(C(=O)O[*])cc1",
        "NYLON6": "[*]CCCCCC(=O)N[*]",
        "NYLON66": "[*]NCCCCCCNC(=O)CCCCC(=O)[*]",
        "PC": "[*]OC(=O)Oc1ccc(C(C)(C)c2ccc(O[*])cc2)cc1",
        "PES": "[*]Oc1ccc(S(=O)(=O)c2ccc(O[*])cc2)cc1",
        "EVOH": "[*]CC(O)[*]",
    }

    mapped = pd.DataFrame()

    # Assign PSMILES from known dictionary
    name_col = None
    for c in df.columns:
        if "polymer" in c.lower() and "smiles" not in c.lower():
            name_col = c
            break
    if name_col is None and "polymer" in df.columns:
        name_col = "polymer"

    if name_col:
        mapped["polymer_name"] = df[name_col].astype(str).str.strip()
        mapped["psmiles"] = mapped["polymer_name"].str.upper().map(KNOWN_PSMILES)
    else:
        mapped["polymer_name"] = ""
        mapped["psmiles"] = ""

    # Map properties
    col_mapping = {
        "Tm_K": ["tm0_k", "tm_k"],
        "delta_Hf_J_per_mol": ["delta_hf0_j_per_mol", "delta_hf_j_per_mol"],
        "delta_Cp_J_per_mol_K": [],  # needs conversion from J/(g·K)
        "Tg_K": ["tg_k"],
        "Td_K": ["td_k"],
    }

    lower_cols = {c.lower(): c for c in df.columns}
    for unified, patterns in col_mapping.items():
        for p in patterns:
            if p in lower_cols:
                mapped[unified] = pd.to_numeric(
                    df[lower_cols[p]], errors="coerce"
                )
                break

    # delta_Cp: convert from J/(g·K) to J/(mol·K) using repeat-unit MW
    if "delta_cp_j_per_g_k" in lower_cols and "repeat_unit_mw" in lower_cols:
        cp_g = pd.to_numeric(df[lower_cols["delta_cp_j_per_g_k"]], errors="coerce")
        mw = pd.to_numeric(df[lower_cols["repeat_unit_mw"]], errors="coerce")
        mapped["delta_Cp_J_per_mol_K"] = cp_g * mw

    mapped["source"] = "ATHAS_local"
    return mapped


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

PROPERTY_COLS = [
    "Tm_K",
    "delta_Hf_J_per_mol",
    "delta_Cp_J_per_mol_K",
    "Tg_K",
    "Td_K",
]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all unified columns exist (fill missing with NaN / empty)."""
    for col in UNIFIED_COLS:
        if col not in df.columns:
            if col in PROPERTY_COLS:
                df[col] = np.nan
            else:
                df[col] = ""
    return df[UNIFIED_COLS]


def canonicalize_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply PSMILES canonicalization to the whole dataframe."""
    log.info("Canonicalizing %d PSMILES ...", len(df))
    df["psmiles"] = df["psmiles"].apply(canonicalize_psmiles)
    # Drop rows with empty PSMILES
    before = len(df)
    df = df[df["psmiles"].str.len() > 0].copy()
    dropped = before - len(df)
    if dropped:
        log.info("Dropped %d rows with empty/invalid PSMILES", dropped)
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by canonical PSMILES, keeping the entry with the most
    non-null property values. If tied, prefer labeled sources over PI1M."""
    if df.empty:
        return df

    df = df.copy()
    df["_n_props"] = df[PROPERTY_COLS].notna().sum(axis=1)
    # Prefer labelled sources
    df["_is_labelled"] = (~df["source"].isin(["PI1M"])).astype(int)
    df = df.sort_values(
        ["psmiles", "_is_labelled", "_n_props"], ascending=[True, False, False]
    )
    df = df.drop_duplicates(subset="psmiles", keep="first")
    df = df.drop(columns=["_n_props", "_is_labelled"])
    log.info("After deduplication: %d rows", len(df))
    return df


def iqr_outlier_removal(
    df: pd.DataFrame, factor: float = 1.5
) -> pd.DataFrame:
    """Set property values outside [Q1 - factor*IQR, Q3 + factor*IQR] to NaN.

    This follows the OpenPoly methodology for cleaning polymer-property data.
    Only applies to rows that have labelled values (NaNs are untouched).
    """
    total_removed = 0
    for col in PROPERTY_COLS:
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = df[col].notna() & ((df[col] < lower) | (df[col] > upper))
        n_out = mask.sum()
        if n_out > 0:
            log.info(
                "IQR outlier removal [%s]: %d outliers (range %.1f–%.1f)",
                col,
                n_out,
                lower,
                upper,
            )
            df.loc[mask, col] = np.nan
            total_removed += n_out
    log.info("Total property values removed as outliers: %d", total_removed)
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, pretrain_count: int) -> None:
    """Print a human-readable summary of the curated dataset."""
    print("\n" + "=" * 64)
    print("  STRAP v7 — Curated Thermal Dataset Summary")
    print("=" * 64)
    print(f"\nTotal labelled polymers : {len(df)}")
    print(f"Pretrain PSMILES (PI1M): {pretrain_count}")

    print("\n--- Counts per property ---")
    for col in PROPERTY_COLS:
        n = df[col].notna().sum()
        print(f"  {col:30s}: {n:>6d}")

    print("\n--- Counts per source ---")
    for src, grp in df.groupby("source"):
        print(f"  {src:30s}: {len(grp):>6d}")

    print("\n--- Train / Test split sizes ---")
    split_counts = df["split"].value_counts(dropna=False)
    for split_val, cnt in split_counts.items():
        label = split_val if split_val else "(unassigned)"
        print(f"  {label:30s}: {cnt:>6d}")

    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate polymer thermal property datasets for STRAP v7."
    )
    parser.add_argument(
        "--iqr-factor",
        type=float,
        default=1.5,
        help="IQR multiplier for outlier removal (default: 1.5)",
    )
    parser.add_argument(
        "--skip-pi1m",
        action="store_true",
        help="Skip PI1M download (large file).",
    )
    args = parser.parse_args()

    # Ensure output directories exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    THERMAL_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all datasets
    # ------------------------------------------------------------------
    frames: list[pd.DataFrame] = []

    log.info("=== Loading polyVERSE ===")
    df_pv = load_polyverse()
    if not df_pv.empty:
        frames.append(df_pv)

    log.info("=== Loading POINT2 ===")
    df_pt = load_point2()
    if not df_pt.empty:
        frames.append(df_pt)

    log.info("=== Loading local ATHAS reference ===")
    df_local = load_local_reference()
    if not df_local.empty:
        frames.append(df_local)

    # PI1M — keep separate for pretrain output
    pi1m_smiles: list[str] = []
    if not args.skip_pi1m:
        log.info("=== Loading PI1M ===")
        df_pi1m = load_pi1m()
        if not df_pi1m.empty:
            pi1m_smiles = df_pi1m["psmiles"].dropna().tolist()
            # Also add to frames so dedup removes any overlap with labelled data
            frames.append(df_pi1m)

    if not frames:
        log.error("No datasets loaded — nothing to output.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Combine into unified schema
    # ------------------------------------------------------------------
    combined = pd.concat(
        [_ensure_columns(f) for f in frames], ignore_index=True
    )
    log.info("Combined raw rows: %d", len(combined))

    # ------------------------------------------------------------------
    # 3. Canonicalize PSMILES
    # ------------------------------------------------------------------
    combined = canonicalize_all(combined)

    # ------------------------------------------------------------------
    # 4. Deduplicate
    # ------------------------------------------------------------------
    combined = deduplicate(combined)

    # ------------------------------------------------------------------
    # 5. Separate labelled vs unlabelled (PI1M pretrain set)
    # ------------------------------------------------------------------
    has_label = combined[PROPERTY_COLS].notna().any(axis=1)
    labelled = combined[has_label].copy()
    unlabelled = combined[~has_label].copy()

    # Pretrain list = PI1M SMILES (deduplicated, canonicalized)
    pretrain_smiles = sorted(
        set(
            unlabelled.loc[
                unlabelled["source"] == "PI1M", "psmiles"
            ].tolist()
        )
    )
    # Also include the raw PI1M list entries that may have been deduped out
    # (because they matched a labelled entry). The labelled entry stays in
    # the main CSV; the SMILES is *also* fine for pretraining.
    pretrain_set = set(pretrain_smiles)
    for smi in pi1m_smiles:
        c = canonicalize_psmiles(smi)
        if c and c not in pretrain_set:
            pretrain_smiles.append(c)
            pretrain_set.add(c)
    pretrain_smiles.sort()

    # ------------------------------------------------------------------
    # 6. IQR outlier removal on labelled data
    # ------------------------------------------------------------------
    labelled = iqr_outlier_removal(labelled, factor=args.iqr_factor)

    # ------------------------------------------------------------------
    # 7. Output
    # ------------------------------------------------------------------
    labelled.to_csv(OUTPUT_CSV, index=False)
    log.info("Wrote labelled dataset -> %s (%d rows)", OUTPUT_CSV, len(labelled))

    with open(OUTPUT_PRETRAIN, "w") as f:
        for smi in pretrain_smiles:
            f.write(smi + "\n")
    log.info(
        "Wrote pretrain PSMILES -> %s (%d entries)",
        OUTPUT_PRETRAIN,
        len(pretrain_smiles),
    )

    # ------------------------------------------------------------------
    # 8. Summary
    # ------------------------------------------------------------------
    print_summary(labelled, len(pretrain_smiles))


if __name__ == "__main__":
    main()
