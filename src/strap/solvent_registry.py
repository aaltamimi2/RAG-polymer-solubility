"""Unified solvent name registry — single source of truth for aliases.

Every subsystem that needs to resolve solvent names imports from here
instead of maintaining its own alias dictionary.

No imports from any tool, model, or database module — stdlib only.
"""

from __future__ import annotations

from typing import Optional


# ==================================================================
# Core registry: one entry per canonical solvent (keyed by interp-key)
# ==================================================================
#
# Fields:
#   interp_key  — lowercase key used in the solubility coefficient JSON
#   property_db — name in the solvent_data SQL table (title-case)
#   gsk_db      — name in the gsk_dataset SQL table (may differ or be None)
#   biosteam    — exact name for BioSTEAM / thermosteam registry (None if N/A)
#   bp_db_key   — lowercase key in the solvent_data table for BP/LogP cache
#   cas         — CAS registry number (for GreenSolventDB 10K lookup)
#   aliases     — every user-facing spelling that should resolve to this entry
#                  (the interp_key itself is added automatically)

SOLVENT_REGISTRY: dict[str, dict] = {
    # ── Water ──
    "h2o": {
        "interp_key": "h2o",
        "property_db": "Water",
        "gsk_db": "Water",
        "biosteam": None,
        "bp_db_key": "water",
        "cas": "7732-18-5",
        "aliases": ["water"],
    },
    # ── Ketones ──
    "propanone": {
        "interp_key": "propanone",
        "property_db": "Acetone",
        "gsk_db": "Acetone",
        "biosteam": "Acetone",
        "bp_db_key": "acetone",
        "cas": "67-64-1",
        "aliases": ["acetone", "2-propanone"],
    },
    "butanone": {
        "interp_key": "butanone",
        "property_db": "Methyl ethyl ketone",
        "gsk_db": "2-Butanone",
        "biosteam": "2-Butanone",
        "bp_db_key": "methyl ethyl ketone (mek)",
        "cas": "78-93-3",
        "aliases": ["methyl ethyl ketone", "mek"],
    },
    "acetylacetone": {
        "interp_key": "acetylacetone",
        "property_db": "2,4-Pentanedione",
        "gsk_db": None,
        "biosteam": "Acetylacetone",
        "bp_db_key": "2,4-pentanedione",
        "cas": "123-54-6",
        "aliases": ["2,4-pentanedione", "acac"],
    },
    # ── Polar aprotic ──
    "dimethylformamide": {
        "interp_key": "dimethylformamide",
        "property_db": "N,N-Dimethylformamide",
        "gsk_db": "DMF",
        "biosteam": "N,N-Dimethylformamide",
        "bp_db_key": "dimethyl formamide (dmf)",
        "cas": "68-12-2",
        "aliases": ["dmf", "n,n-dimethylformamide", "dimethyl formamide"],
    },
    "dimethylsulfoxide": {
        "interp_key": "dimethylsulfoxide",
        "property_db": "Dimethyl sulfoxide",
        "gsk_db": "Dimethyl sulfoxide",
        "biosteam": "Dimethyl sulfoxide",
        "bp_db_key": "dimethyl sulfoxide (dmso)",
        "cas": "67-68-5",
        "aliases": ["dmso", "dimethyl sulfoxide"],
    },
    # ── Halogenated ──
    "ch2cl2": {
        "interp_key": "ch2cl2",
        "property_db": "Dichloromethane",
        "gsk_db": "Dichloromethane",
        "biosteam": "Dichloromethane",
        "bp_db_key": "methylene dichloride (dichloromethane)",
        "cas": "75-09-2",
        "aliases": ["dichloromethane", "dcm", "methylene chloride"],
    },
    "chcl3": {
        "interp_key": "chcl3",
        "property_db": "Chloroform",
        "gsk_db": "Chloroform",
        "biosteam": None,
        "bp_db_key": "chloroform",
        "cas": "67-66-3",
        "aliases": ["chloroform", "trichloromethane"],
    },
    # ── Esters ──
    "ethylacetate": {
        "interp_key": "ethylacetate",
        "property_db": "Ethyl acetate",
        "gsk_db": "Ethyl acetate",
        "biosteam": "Ethyl acetate",
        "bp_db_key": "ethyl acetate",
        "cas": "141-78-6",
        "aliases": ["ethyl acetate", "etac"],
    },
    "methylacetate": {
        "interp_key": "methylacetate",
        "property_db": "Methyl acetate",
        "gsk_db": "Methyl acetate",
        "biosteam": "Methyl acetate",
        "bp_db_key": "methyl acetate",
        "cas": "79-20-9",
        "aliases": ["methyl acetate"],
    },
    # ── Ethers ──
    "thf": {
        "interp_key": "thf",
        "property_db": "Tetrahydrofuran (THF)",
        "gsk_db": "THF",
        "biosteam": "Tetrahydrofuran",
        "bp_db_key": "tetrahydrofuran (thf)",
        "cas": "109-99-9",
        "aliases": ["tetrahydrofuran"],
    },
    "thp": {
        "interp_key": "thp",
        "property_db": "Tetrahydropyran",
        "gsk_db": None,
        "biosteam": "Tetrahydropyran",
        "bp_db_key": "tetrahydropyran",
        "cas": "142-68-7",
        "aliases": ["tetrahydropyran"],
    },
    "diphenylether": {
        "interp_key": "diphenylether",
        "property_db": "Diphenyl ether",
        "gsk_db": "Diphenyl ether",
        "biosteam": "Diphenyl ether",
        "bp_db_key": "diphenyl ether",
        "cas": "101-84-8",
        "aliases": ["diphenyl ether"],
    },
    "2,3-dihydropyran": {
        "interp_key": "2,3-dihydropyran",
        "property_db": None,
        "gsk_db": None,
        "biosteam": "2,3-Dihydropyran",
        "bp_db_key": "dihydropyran",
        "cas": "110-87-2",
        "aliases": ["dihydropyran"],
    },
    # ── Glycols ──
    "glycol": {
        "interp_key": "glycol",
        "property_db": "Ethylene glycol",
        "gsk_db": "Ethylene glycol",
        "biosteam": "Ethylene Glycol",
        "bp_db_key": "ethylene glycol",
        "cas": "107-21-1",
        "aliases": ["ethylene glycol", "meg", "monoethylene glycol", "1,2-ethanediol"],
    },
    "propyleneglycol": {
        "interp_key": "propyleneglycol",
        "property_db": "Propylene glycol",
        "gsk_db": "1,2-Propanediol",
        "biosteam": "Propylene Glycol",
        "bp_db_key": "propylene glycol",
        "cas": "57-55-6",
        "aliases": ["propylene glycol", "1,2-propanediol"],
    },
    # ── Aromatics ──
    "toluene": {
        "interp_key": "toluene",
        "property_db": "Toluene",
        "gsk_db": "Toluene",
        "biosteam": "Toluene",
        "bp_db_key": None,
        "cas": "108-88-3",
        "aliases": ["phme", "toluol", "methylbenzene"],
    },
    "benzene": {
        "interp_key": "benzene",
        "property_db": "Benzene",
        "gsk_db": "Benzene",
        "biosteam": "Benzene",
        "bp_db_key": None,
        "cas": "71-43-2",
        "aliases": ["phh"],
    },
    "1,2-dimethylbenzene": {
        "interp_key": "1,2-dimethylbenzene",
        "property_db": "o-Xylene",
        "gsk_db": "o-Xylene",
        "biosteam": "o-Xylene",
        "bp_db_key": "o-xylene",
        "cas": "95-47-6",
        "aliases": ["o-xylene", "ortho-xylene"],
    },
    "1,4-dimethylbenzene": {
        "interp_key": "1,4-dimethylbenzene",
        "property_db": "p-Xylene",
        "gsk_db": "p-Xylene",
        "biosteam": "p-Xylene",
        "bp_db_key": "xylene",
        "cas": "106-42-3",
        "aliases": ["p-xylene", "para-xylene"],
    },
    # ── Alkanes ──
    "n-heptane": {
        "interp_key": "n-heptane",
        "property_db": "Heptane",
        "gsk_db": "n-Heptane",
        "biosteam": "Heptane",
        "bp_db_key": "heptane",
        "cas": "142-82-5",
        "aliases": ["heptane"],
    },
    "hexane": {
        "interp_key": "hexane",
        "property_db": "Hexane",
        "gsk_db": "n-Hexane",
        "biosteam": "Hexane",
        "bp_db_key": None,
        "cas": "110-54-3",
        "aliases": ["n-hexane"],
    },
    "dodecane": {
        "interp_key": "dodecane",
        "property_db": "Dodecane",
        "gsk_db": "Dodecane",
        "biosteam": "Dodecane",
        "bp_db_key": None,
        "cas": "112-40-3",
        "aliases": [],
    },
    "cyclohexane": {
        "interp_key": "cyclohexane",
        "property_db": "Cyclohexane",
        "gsk_db": "Cyclohexane",
        "biosteam": "Cyclohexane",
        "bp_db_key": None,
        "cas": "110-82-7",
        "aliases": [],
    },
    # ── Alcohols ──
    "methanol": {
        "interp_key": "methanol",
        "property_db": "Methanol",
        "gsk_db": "Methanol",
        "biosteam": "Methanol",
        "bp_db_key": None,
        "cas": "67-56-1",
        "aliases": ["meoh"],
    },
    "ethanol": {
        "interp_key": "ethanol",
        "property_db": "Ethanol",
        "gsk_db": "Ethanol",
        "biosteam": "Ethanol",
        "bp_db_key": None,
        "cas": "64-17-5",
        "aliases": ["etoh"],
    },
    "propanol": {
        "interp_key": "propanol",
        "property_db": "1-Propanol",
        "gsk_db": "1-Propanol",
        "biosteam": "1-Propanol",
        "bp_db_key": "1-propanol",
        "cas": "71-23-8",
        "aliases": ["1-propanol", "n-propanol"],
    },
    "2-propanol": {
        "interp_key": "2-propanol",
        "property_db": "2-Propanol",
        "gsk_db": "2-Propanol",
        "biosteam": "Isopropanol",
        "bp_db_key": "isopropanol",
        "cas": "67-63-0",
        "aliases": ["isopropanol", "ipa", "isopropyl alcohol"],
    },
    "tert-butanol": {
        "interp_key": "tert-butanol",
        "property_db": "tert-Butanol",
        "gsk_db": "t-Butanol",
        "biosteam": "tert-Butanol",
        "bp_db_key": "t-butyl alcohol",
        "cas": "75-65-0",
        "aliases": ["tert-butyl alcohol", "t-butyl alcohol"],
    },
    "cyclohexanol": {
        "interp_key": "cyclohexanol",
        "property_db": "Cyclohexanol",
        "gsk_db": "Cyclohexanol",
        "biosteam": "Cyclohexanol",
        "bp_db_key": None,
        "cas": "108-93-0",
        "aliases": [],
    },
    # ── Amines ──
    "triethylamine": {
        "interp_key": "triethylamine",
        "property_db": "Triethylamine",
        "gsk_db": "Triethylamine",
        "biosteam": "Triethylamine",
        "bp_db_key": None,
        "cas": "121-44-8",
        "aliases": ["tea"],
    },
    "isopropylamine": {
        "interp_key": "isopropylamine",
        "property_db": "Isopropylamine",
        "gsk_db": None,
        "biosteam": None,
        "bp_db_key": "isopropyl amine (2-propan amine)",
        "cas": "75-31-0",
        "aliases": ["isopropyl amine"],
    },
    # ── BioSTEAM-only solvents (no solubility/property/GSK data) ──
    "pyridazine": {
        "canonical": "Pyridazine",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "Pyridazine",
        "bp_db_key": None,
        "cas": "289-80-5",
        "aliases": ["pyridazine"],
    },
    "butanediol": {
        "canonical": "1,4-Butanediol",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "butane-1,4-diol",
        "bp_db_key": None,
        "cas": "110-63-4",
        "aliases": ["butanediol", "1,4-butanediol", "bdo", "butane-1,4-diol"],
    },
    "diethanolamine": {
        "canonical": "Diethanolamine",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "Diethanolamine",
        "bp_db_key": None,
        "cas": "111-42-2",
        "aliases": ["diethanolamine", "dea"],
    },
    "diethylene_glycol": {
        "canonical": "Diethylene glycol",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "Diethylene glycol",
        "bp_db_key": None,
        "cas": "111-46-6",
        "aliases": ["diethylene glycol", "deg"],
    },
    "gbl": {
        "canonical": "gamma-Butyrolactone",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "gamma-butyrolactone",
        "bp_db_key": None,
        "cas": "96-48-0",
        "aliases": ["gamma-butyrolactone", "gbl", "butyrolactone"],
    },
    "methylcyclohexane": {
        "canonical": "Methylcyclohexane",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "Methylcyclohexane",
        "bp_db_key": None,
        "cas": "108-87-2",
        "aliases": ["methylcyclohexane", "mch"],
    },
    "sec_butyl_acetate": {
        "canonical": "sec-Butyl Acetate",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "sec-Butyl Acetate",
        "bp_db_key": None,
        "cas": "105-46-4",
        "aliases": ["sec-butyl acetate", "sba"],
    },
    "isobutyl_acetate": {
        "canonical": "Isobutyl Acetate",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "Isobutyl Acetate",
        "bp_db_key": None,
        "cas": "110-19-0",
        "aliases": ["isobutyl acetate"],
    },
    "dodecanol": {
        "canonical": "1-Dodecanol",
        "interp_key": None,
        "property_db": None,
        "gsk_db": None,
        "biosteam": "Dodecanol",
        "bp_db_key": None,
        "cas": "112-53-8",
        "aliases": ["dodecanol", "1-dodecanol", "lauryl alcohol"],
    },
    # ── Lactones ──
    "gvl": {
        "interp_key": "gvl",
        "property_db": "gamma-Valerolactone",
        "gsk_db": None,
        "biosteam": None,
        "bp_db_key": "gvl",
        "cas": "108-29-2",
        "aliases": ["gamma-valerolactone", "γ-valerolactone"],
    },
}


# ==================================================================
# Shared abbreviation map (for SQL LIKE fuzzy matching)
# ==================================================================
# Used by: solvent_properties.py, advanced_separation.py, visualization.py

ABBREVIATION_MAP: dict[str, str] = {
    "dmf": "dimethylformamide",
    "thf": "tetrahydrofuran",
    "dme": "dimethoxyethane",
    "meoh": "methanol",
    "etoh": "ethanol",
    "ipa": "isopropanol",
    "nmp": "n-methyl-2-pyrrolidone",
    "dmso": "dimethyl sulfoxide",
    "dcm": "dichloromethane",
    "dce": "dichloroethane",
    "mecn": "acetonitrile",
    "etac": "ethyl acetate",
    "acac": "acetylacetone",
    "tfa": "trifluoroacetic acid",
    "tfe": "trifluoroethanol",
    "hfip": "hexafluoroisopropanol",
    "chcl3": "chloroform",
    "ccl4": "carbon tetrachloride",
    "phme": "toluene",
    "phh": "benzene",
    "mtbe": "methyl tert-butyl ether",
    "tbme": "tert-butyl methyl ether",
    "dipa": "diisopropylamine",
    "tea": "triethylamine",
    "dbu": "1,8-diazabicyclo[5.4.0]undec-7-ene",
    "pyr": "pyridine",
    "acn": "acetonitrile",
    "mibk": "methyl isobutyl ketone",
}


# ==================================================================
# Search aliases (for SQL LIKE term expansion in get_solvent_properties)
# ==================================================================
# Used by: solvent_properties.py::get_solvent_properties()

SEARCH_ALIASES: dict[str, list[str]] = {
    "decalin": ["decahydronaphthalene", "cis-decahydronaphthalene", "trans-decahydronaphthalene"],
    "d-limonene": ["limonene", "dipentene"],
    "limonene": ["limonene", "dipentene"],
    "dmf": ["dimethyl formamide", "dimethylformamide"],
    "thf": ["tetrahydrofuran"],
    "dcm": ["dichloromethane", "methylene chloride"],
    "nmp": ["n-methyl-2-pyrrolidone", "methylpyrrolidone"],
    "dmso": ["dimethyl sulfoxide"],
    "meg": ["ethylene glycol", "monoethylene glycol"],
    "mtbe": ["methyl tert-butyl ether"],
    "ipa": ["isopropanol", "isopropyl alcohol", "2-propanol"],
}


# ==================================================================
# Derived indexes (built once at import time)
# ==================================================================

# Flat map: any alias (lowercase) → interp_key
_ALIAS_TO_INTERP: dict[str, str] = {}
for _key, _info in SOLVENT_REGISTRY.items():
    _ALIAS_TO_INTERP[_key] = _key
    for _alias in _info["aliases"]:
        _ALIAS_TO_INTERP[_alias.lower()] = _key


# ==================================================================
# Resolver functions
# ==================================================================

def resolve_to_interp_key(name: str) -> Optional[str]:
    """Map any alias to the interp-key (lowercase, for solubility coefficients).

    Returns None if the name is not recognized.
    """
    return _ALIAS_TO_INTERP.get(name.strip().lower())


def resolve_to_property_db(name: str) -> Optional[str]:
    """Map any alias to the property-DB name (for solvent_data SQL queries)."""
    key = _ALIAS_TO_INTERP.get(name.strip().lower())
    if key is not None:
        return SOLVENT_REGISTRY[key]["property_db"]
    return None


def resolve_to_gsk_db(name: str) -> Optional[str]:
    """Map any alias to the GSK-DB name (for gsk_dataset SQL queries)."""
    key = _ALIAS_TO_INTERP.get(name.strip().lower())
    if key is not None:
        return SOLVENT_REGISTRY[key]["gsk_db"]
    return None


def resolve_to_biosteam(name: str) -> Optional[str]:
    """Map any alias to the BioSTEAM canonical name (title-case).

    Returns None if the solvent has no BioSTEAM mapping.
    """
    key = _ALIAS_TO_INTERP.get(name.strip().lower())
    if key is not None:
        return SOLVENT_REGISTRY[key]["biosteam"]
    return None


def resolve_to_bp_db_key(name: str) -> Optional[str]:
    """Map any alias to the lowercase key used in solvent_data BP/LogP cache.

    Returns None if the solvent has no BP-DB mapping.
    """
    key = _ALIAS_TO_INTERP.get(name.strip().lower())
    if key is not None:
        return SOLVENT_REGISTRY[key]["bp_db_key"]
    return None


def resolve_for_databases(name: str, target: str = "property") -> Optional[str]:
    """Drop-in replacement for normalize_solvent_name() in _helpers.py.

    Preserves original behavior: if the name is in the registry, returns
    the mapped value (which may be None for some entries). If the name is
    NOT in the registry, returns the original name unchanged.
    """
    norm = name.strip().lower()
    # Only resolve against interp-keys (not aliases) to match the original
    # SOLVENT_NAME_MAP behavior, which was keyed by interp-keys only.
    if norm in SOLVENT_REGISTRY:
        entry = SOLVENT_REGISTRY[norm]
        if target == "property":
            return entry["property_db"]
        if target == "gsk":
            return entry["gsk_db"]
    return name


def get_search_terms(name: str) -> list[str]:
    """Return SQL-friendly search terms for a solvent alias.

    Drop-in replacement for SOLVENT_ALIASES.get(name, [name]) in
    solvent_properties.py::get_solvent_properties().
    """
    name_lower = name.strip().lower()
    terms = SEARCH_ALIASES.get(name_lower, [name_lower])
    if name_lower not in terms:
        terms = list(terms) + [name_lower]
    return terms
