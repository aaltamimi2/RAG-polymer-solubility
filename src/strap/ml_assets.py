from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from strap.paths import get_data_dir, get_models_dir

DATA_DIR = Path(os.environ.get("DATA_DIR", get_data_dir())).resolve()
ML_POLYMER_CATALOG_PATH = Path(
    os.environ.get("ML_POLYMER_CATALOG_PATH", DATA_DIR / "ml_polymer_catalog.json")
).resolve()
ML_HSP_LOOKUP_PATH = Path(
    os.environ.get("ML_HSP_LOOKUP_PATH", DATA_DIR / "ml_hsp_lookup.json")
).resolve()
ML_MODEL_DIR = Path(os.environ.get("ML_MODEL_DIR", get_models_dir())).resolve()
ML_MODEL_FILES = (
    "corrected_Random_Forest_20251231_212903_model.pkl",
    "corrected_Random_Forest_20251231_212903_scaler.pkl",
    "corrected_Random_Forest_20251231_212903_metadata.json",
)

# Common solvent aliases mapped to canonical names from the HSP dataset.
COMMON_SOLVENT_ALIASES = {
    "acetone": "Propan-2-one",
    "ethanol": "Ethanol",
    "methanol": "Methanol",
    "isopropanol": "Propan-2-ol",
    "ipa": "Propan-2-ol",
    "thf": "Tetrahydro-furan",
    "tetrahydrofuran": "Tetrahydro-furan",
    "dmf": "N,N-Dimethyl- formamide",
    "dimethylformamide": "N,N-Dimethyl- formamide",
    "n,n-dimethylformamide": "N,N-Dimethyl- formamide",
    "dmso": "Methylsulfi yl-methane",
    "dimethyl sulfoxide": "Methylsulfi yl-methane",
    "dimethylsulfoxide": "Methylsulfi yl-methane",
    "dma": "N,N-Dimethyl-acetamide",
    "nmp": "1-Methyl-pyrrolidin-2- one",
    "n-methyl-2-pyrrolidone": "1-Methyl-pyrrolidin-2- one",
    "mek": "Butan-2-one",
    "mibk": "4-Methyl-pentan-2-one",
    "dcm": "Dichloro-methane",
    "dichloromethane": "Dichloro-methane",
    "methylene chloride": "Dichloro-methane",
    "chloroform": "Trichloro-methane",
    "trichloromethane": "Trichloro-methane",
    "etoh": "Ethanol",
    "meoh": "Methanol",
    "acn": "Acetonitrile",
    "acetonitrile": "Acetonitrile",
    "dce": "1,2-Dichloro-ethane",
    "ea": "Acetic acid ethyl ester",
    "ethyl acetate": "Acetic acid ethyl ester",
    "ether": "Diethyl ether",
    "diethyl ether": "Diethyl ether",
    "hexane": "n-Hexane",
    "n-hexane": "n-Hexane",
    "heptane": "Heptane",
    "octane": "Octane",
    "decane": "Decane",
    "benzene": "Benzene",
    "toluene": "Toluene",
    "xylene": "o-Xylene",
    "water": "Water",
    "dioxane": "1,4-Dioxane",
    "pyridine": "Pyridine",
    "aniline": "Aniline",
    "nitromethane": "Nitromethane",
    "nitroethane": "Nitroethane",
    "cyclohexane": "Cyclohexane",
    "cyclohexanone": "Cyclohexanone",
    "ccl4": "Tetrachloro-methane",
    "carbon tetrachloride": "Tetrachloro-methane",
    "carbon disulfide": "Carbon disulfide",
    "cs2": "Carbon disulfide",
    "butanol": "Butan-1-ol",
    "propanol": "Propan-1-ol",
    "pentane": "Pentane",
    "butyl acetate": "Acetic acid butyl ester",
    "methyl acetate": "Acetic acid methyl ester",
    "propyl acetate": "Acetic acid propyl ester",
    "formic acid": "Formic acid",
}


def _normalize_key(name: str) -> str:
    return name.strip().lower()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ML asset not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def load_ml_polymer_catalog_data() -> dict[str, Any]:
    return _read_json(ML_POLYMER_CATALOG_PATH)


def load_ml_polymer_catalog() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    data = load_ml_polymer_catalog_data()
    return data["types"], data["grouped"]


@lru_cache(maxsize=1)
def load_ml_hsp_lookup() -> dict[str, Any]:
    data = _read_json(ML_HSP_LOOKUP_PATH)
    if "polymers" not in data or "solvents" not in data:
        raise ValueError(f"Invalid ML lookup payload: {ML_HSP_LOOKUP_PATH}")
    return data


def resolve_polymer_entry(polymer_name: str) -> dict[str, Any] | None:
    lookup = load_ml_hsp_lookup()
    entries = lookup["polymers"]
    key = _normalize_key(polymer_name)
    exact = entries.get(key)
    if exact:
        return exact

    needle = polymer_name.strip().upper()
    for canonical_name in lookup.get("polymer_names", []):
        if needle and needle in canonical_name.upper():
            return entries[_normalize_key(canonical_name)]
    return None


def suggest_polymer_names(polymer_name: str, limit: int = 10) -> list[str]:
    lookup = load_ml_hsp_lookup()
    names = lookup.get("polymer_names", [])
    tokens = ("PE", "POLY", "PET", "PP", "PVC", "PS")
    needle = polymer_name.strip().upper()
    suggestions = [name for name in names if needle and needle in name.upper()]
    if suggestions:
        return suggestions[:limit]
    return [name for name in names if any(token in name.upper() for token in tokens)][:limit]


def resolve_solvent_entry(solvent_name: str) -> dict[str, Any] | None:
    lookup = load_ml_hsp_lookup()
    entries = lookup["solvents"]
    key = _normalize_key(solvent_name)
    exact = entries.get(key)
    if exact:
        return exact

    alias = COMMON_SOLVENT_ALIASES.get(key)
    if alias:
        mapped = entries.get(_normalize_key(alias))
        if mapped:
            return mapped

    needle = solvent_name.strip().upper()
    for canonical_name in lookup.get("solvent_names", []):
        if needle and needle in canonical_name.upper():
            return entries[_normalize_key(canonical_name)]
    return None


def missing_ml_assets() -> list[str]:
    missing: list[str] = []
    for path in (ML_POLYMER_CATALOG_PATH, ML_HSP_LOOKUP_PATH):
        if not path.exists():
            missing.append(path.name)
    for model_name in ML_MODEL_FILES:
        if not (ML_MODEL_DIR / model_name).exists():
            missing.append(model_name)
    return missing
