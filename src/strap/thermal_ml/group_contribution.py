"""Van Krevelen group contribution estimates for polymer thermal properties.

Implements the additive group contribution method from Van Krevelen's
"Properties of Polymers" (4th ed.) for estimating:
  - Enthalpy of fusion (DHf) in J/mol
  - Heat capacity difference at Tm (DCp) in J/(mol*K)
  - Melting temperature (Tm) in K via DHf/DSf

The ML model in STRAP v7 learns RESIDUALS against these baselines, which is
a much easier learning problem with limited data (~200-300 entries).

Usage::

    from strap.thermal_ml.group_contribution import estimate_all
    result = estimate_all("[*]CC[*]")  # polyethylene

"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Group contribution tables  (Van Krevelen, Properties of Polymers, 4th ed.)
# ---------------------------------------------------------------------------

# Each entry: (SMARTS, display_name, DHf [J/mol], DCp [J/(mol*K)], DSf [J/(mol*K)])
# None means no reliable value available for that property.
# Order matters: more specific patterns must come before less specific ones
# so that, e.g., -CONH- is matched before -CO- and -NH- separately.

_GROUP_TABLE: list[tuple[str, str, float | None, float | None, float | None]] = [
    # --- Biphenyl (must precede single phenylene) ---
    (
        "[c]1[c][c][c]([c][c]1)-[c]2[c][c][c][c][c]2",
        "-C6H4-C6H4- (biphenyl)",
        12_000.0,
        None,
        None,
    ),
    # --- Amide (must precede -CO- and -NH-) ---
    ("C(=O)[NH]", "-CONH- (amide)", 12_000.0, 20.0, 22.0),
    # --- Ester (must precede -CO- and -O-) ---
    ("C(=O)[OX2]", "-COO- (ester)", 10_000.0, 18.0, 18.0),
    # --- Sulfonyl ---
    ("[#16](=O)(=O)", "-SO2-", 7_000.0, None, None),
    # --- Hexafluoroisopropylidene ---
    ("C(C(F)(F)F)(C(F)(F)F)", "-C(CF3)2-", 8_000.0, None, None),
    # --- Siloxane ---
    ("[Si]([CH3])([CH3])[OX2]", "-Si(CH3)2-O-", 2_000.0, None, None),
    # --- Para-phenylene ---
    # Aromatic ring with two substituents at 1,4 positions is hard to
    # distinguish from meta purely by SMARTS in a general way.  We match
    # any non-biphenyl phenylene ring and default to the para value; users
    # can override meta assignment via the groups dict if needed.
    ("[c]1[c][c][c][c][c]1", "-C6H4- (phenylene)", 6_000.0, 25.0, 5.0),
    # --- Ketone (C=O not in ester/amide context) ---
    ("[CX3](=O)([#6])[#6]", "-CO- (ketone)", 5_000.0, None, None),
    # --- Hydroxyl ---
    ("[OX2H]", "-OH", 8_500.0, 12.0, None),
    # --- Ether oxygen (not in ester/siloxane) ---
    ("[OX2]([#6])[#6]", "-O- (ether)", 4_500.0, 8.0, 5.0),
    # --- Thioether ---
    ("[#16X2]([#6])[#6]", "-S-", 3_500.0, None, None),
    # --- Secondary amine (not in amide) ---
    ("[NX3H]([#6])[#6]", "-NH-", 6_000.0, None, None),
    # --- Dichloromethylene ---
    ("[CX4](Cl)(Cl)", "-CCl2-", 4_500.0, None, None),
    # --- Chloromethine ---
    ("[CX4H](Cl)", "-CHCl-", 4_000.0, 14.0, None),
    # --- Difluoromethylene ---
    ("[CX4](F)(F)", "-CF2-", 5_000.0, 12.0, 10.0),
    # --- trans-vinylene ---
    ("[#6]/[CH]=[CH]/[#6]", "-CH=CH- (trans)", 9_000.0, None, None),
    # --- Gem-dimethyl (isopropylidene, must precede -CH(CH3)-) ---
    ("[CX4]([CH3])([CH3])", "-C(CH3)2-", 6_500.0, 25.0, None),
    # --- Methyl-substituted methine ---
    ("[CX4H]([CH3])", "-CH(CH3)-", 7_500.0, 18.0, 14.0),
    # --- Methylene (backbone -CH2-) ---
    ("[CH2]", "-CH2-", 4_000.0, 10.5, 9.9),
]

# Pre-compile all SMARTS into a parallel list for performance.
_COMPILED_SMARTS: list[Any] = []  # populated lazily


def _ensure_compiled() -> None:
    """Lazily compile SMARTS patterns on first use (requires RDKit)."""
    if _COMPILED_SMARTS:
        return
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for Van Krevelen group contribution estimation. "
            "Install it with:  conda install -c conda-forge rdkit"
        )
    for smarts, name, *_ in _GROUP_TABLE:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("Failed to compile SMARTS for group %s: %s", name, smarts)
        _COMPILED_SMARTS.append(pattern)


# ---------------------------------------------------------------------------
# PSMILES handling
# ---------------------------------------------------------------------------

_STAR_TOKEN_RE = re.compile(r"\[\*\]|\[\*:(\d+)\]|\*")


def _psmiles_to_mol(psmiles: str) -> Chem.Mol | None:  # type: ignore[name-defined]
    """Convert a PSMILES string to an RDKit Mol object.

    Wildcard attachment points ``[*]`` are replaced with dummy atoms ``[Xe]``
    so that RDKit can parse the string as a valid molecule.  The dummy atoms
    are excluded from heavy-atom counts and group matching later.
    """
    if not RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for Van Krevelen group contribution estimation. "
            "Install it with:  conda install -c conda-forge rdkit"
        )
    sanitized = _STAR_TOKEN_RE.sub("[Xe]", psmiles.strip())
    mol = Chem.MolFromSmiles(sanitized, sanitize=True)
    if mol is None:
        # Try without sanitization as a fallback
        mol = Chem.MolFromSmiles(sanitized, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
    return mol


def _heavy_atom_count_no_dummy(mol: Chem.Mol) -> int:  # type: ignore[name-defined]
    """Count heavy atoms excluding Xe dummy atoms (attachment points)."""
    return sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 54  # Xe
    )


# ---------------------------------------------------------------------------
# Group parsing
# ---------------------------------------------------------------------------


def parse_psmiles_groups(psmiles: str) -> dict[str, Any] | None:
    """Parse a PSMILES string and identify Van Krevelen structural groups.

    Parameters
    ----------
    psmiles : str
        Polymer SMILES with ``[*]`` attachment points, e.g. ``[*]CC[*]``.

    Returns
    -------
    dict or None
        ``{"groups": {name: count}, "coverage": float, "heavy_atoms": int,
        "matched_atoms": int}`` on success, or ``None`` if the PSMILES is
        invalid (with a logged warning).
    """
    _ensure_compiled()

    mol = _psmiles_to_mol(psmiles)
    if mol is None:
        logger.warning("Invalid PSMILES (RDKit could not parse): %s", psmiles)
        return None

    total_heavy = _heavy_atom_count_no_dummy(mol)
    if total_heavy == 0:
        logger.warning("PSMILES has no heavy atoms: %s", psmiles)
        return None

    groups: dict[str, int] = {}
    matched_atom_indices: set[int] = set()

    # Iterate through patterns in priority order.  For each pattern we count
    # non-overlapping matches.  We track which atom indices have been
    # "consumed" to avoid double-counting atoms across different groups,
    # but we allow the *same* pattern to match the same atom in different
    # matches (RDKit's GetSubstructMatches does this naturally).
    for idx, (smarts, name, *_rest) in enumerate(_GROUP_TABLE):
        pat = _COMPILED_SMARTS[idx]
        if pat is None:
            continue
        matches = mol.GetSubstructMatches(pat, uniquify=True)
        if not matches:
            continue

        count = 0
        for match_atoms in matches:
            # Filter out dummy-atom matches
            real_atoms = {a for a in match_atoms if mol.GetAtomWithIdx(a).GetAtomicNum() != 54}
            if not real_atoms:
                continue
            # Only count if at least one atom in this match is not yet consumed
            if real_atoms - matched_atom_indices:
                count += 1
                matched_atom_indices.update(real_atoms)

        if count > 0:
            groups[name] = groups.get(name, 0) + count

    coverage = len(matched_atom_indices) / total_heavy if total_heavy > 0 else 0.0

    return {
        "groups": groups,
        "coverage": round(coverage, 4),
        "heavy_atoms": total_heavy,
        "matched_atoms": len(matched_atom_indices),
    }


# ---------------------------------------------------------------------------
# Property estimation helpers
# ---------------------------------------------------------------------------

# Build fast lookup dicts from the group table
_DHF_TABLE: dict[str, float] = {}
_DCP_TABLE: dict[str, float] = {}
_DSF_TABLE: dict[str, float] = {}

for _smarts, _name, _dhf, _dcp, _dsf in _GROUP_TABLE:
    if _dhf is not None:
        _DHF_TABLE[_name] = _dhf
    if _dcp is not None:
        _DCP_TABLE[_name] = _dcp
    if _dsf is not None:
        _DSF_TABLE[_name] = _dsf


def _estimate_property(
    psmiles: str,
    table: dict[str, float],
    property_name: str,
    unit: str,
) -> dict[str, Any] | None:
    """Generic estimator for an additive group-contribution property.

    Returns
    -------
    dict or None
        ``{"value": float, "unit": str, "groups_found": dict,
        "coverage": float, "reliable": bool}`` on success, ``None`` with
        a logged warning on failure.
    """
    parsed = parse_psmiles_groups(psmiles)
    if parsed is None:
        return None

    groups = parsed["groups"]
    coverage = parsed["coverage"]

    total = 0.0
    contributing_groups: dict[str, dict[str, Any]] = {}
    missing_groups: list[str] = []

    for group_name, count in groups.items():
        if group_name in table:
            contribution = table[group_name] * count
            total += contribution
            contributing_groups[group_name] = {
                "count": count,
                "contribution_per_group": table[group_name],
                "total_contribution": contribution,
            }
        else:
            missing_groups.append(group_name)

    reliable = coverage >= 0.7 and len(missing_groups) == 0

    result: dict[str, Any] = {
        "value": round(total, 2),
        "unit": unit,
        "groups_found": contributing_groups,
        "coverage": coverage,
        "reliable": reliable,
    }
    if missing_groups:
        result["missing_groups"] = missing_groups
        logger.info(
            "%s: groups without %s contribution data: %s",
            psmiles,
            property_name,
            missing_groups,
        )

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_delta_hf(psmiles: str) -> dict[str, Any] | None:
    """Estimate enthalpy of fusion (DHf) for a polymer repeat unit.

    Parameters
    ----------
    psmiles : str
        Polymer SMILES string with ``[*]`` attachment points.

    Returns
    -------
    dict or None
        Keys: ``value`` (J/mol), ``unit``, ``groups_found``, ``coverage``,
        ``reliable``.  Returns ``None`` if the PSMILES is invalid.

    Examples
    --------
    >>> estimate_delta_hf("[*]CC[*]")  # polyethylene
    {'value': 8000.0, 'unit': 'J/mol', ...}
    """
    return _estimate_property(psmiles, _DHF_TABLE, "DHf", "J/mol")


def estimate_delta_cp(psmiles: str) -> dict[str, Any] | None:
    """Estimate heat capacity difference at Tm (DCp) for a polymer repeat unit.

    Parameters
    ----------
    psmiles : str
        Polymer SMILES string with ``[*]`` attachment points.

    Returns
    -------
    dict or None
        Keys: ``value`` (J/(mol*K)), ``unit``, ``groups_found``, ``coverage``,
        ``reliable``.  Returns ``None`` if the PSMILES is invalid.
    """
    return _estimate_property(psmiles, _DCP_TABLE, "DCp", "J/(mol*K)")


def estimate_tm(psmiles: str) -> dict[str, Any] | None:
    """Estimate melting temperature (Tm) via DHf / DSf.

    Parameters
    ----------
    psmiles : str
        Polymer SMILES string with ``[*]`` attachment points.

    Returns
    -------
    dict or None
        Keys: ``value`` (K), ``unit``, ``groups_found``, ``coverage``,
        ``reliable``, ``delta_hf_J_per_mol``, ``delta_sf_J_per_mol_K``.
        Returns ``None`` if the PSMILES is invalid or DSf is zero.
    """
    parsed = parse_psmiles_groups(psmiles)
    if parsed is None:
        return None

    groups = parsed["groups"]
    coverage = parsed["coverage"]

    # Compute DHf
    total_dhf = 0.0
    dhf_groups: dict[str, dict[str, Any]] = {}
    for group_name, count in groups.items():
        if group_name in _DHF_TABLE:
            contribution = _DHF_TABLE[group_name] * count
            total_dhf += contribution
            dhf_groups[group_name] = {
                "count": count,
                "contribution_per_group": _DHF_TABLE[group_name],
                "total_contribution": contribution,
            }

    # Compute DSf
    total_dsf = 0.0
    dsf_groups: dict[str, dict[str, Any]] = {}
    missing_dsf: list[str] = []
    for group_name, count in groups.items():
        if group_name in _DSF_TABLE:
            contribution = _DSF_TABLE[group_name] * count
            total_dsf += contribution
            dsf_groups[group_name] = {
                "count": count,
                "contribution_per_group": _DSF_TABLE[group_name],
                "total_contribution": contribution,
            }
        else:
            missing_dsf.append(group_name)

    if total_dsf == 0.0:
        logger.warning(
            "Cannot estimate Tm for %s: DSf is zero (no matching DSf groups).",
            psmiles,
        )
        return {
            "value": None,
            "unit": "K",
            "groups_found": dhf_groups,
            "coverage": coverage,
            "reliable": False,
            "delta_hf_J_per_mol": round(total_dhf, 2),
            "delta_sf_J_per_mol_K": 0.0,
            "error": "DSf is zero; cannot compute Tm = DHf/DSf",
        }

    tm = total_dhf / total_dsf

    reliable = coverage >= 0.7 and len(missing_dsf) == 0

    result: dict[str, Any] = {
        "value": round(tm, 2),
        "unit": "K",
        "groups_found": {**dhf_groups, **dsf_groups},
        "coverage": coverage,
        "reliable": reliable,
        "delta_hf_J_per_mol": round(total_dhf, 2),
        "delta_sf_J_per_mol_K": round(total_dsf, 4),
    }
    if missing_dsf:
        result["missing_dsf_groups"] = missing_dsf

    return result


def estimate_all(psmiles: str) -> dict[str, Any] | None:
    """Estimate all thermal properties for a polymer repeat unit.

    This is a convenience function that calls :func:`estimate_delta_hf`,
    :func:`estimate_delta_cp`, and :func:`estimate_tm` and bundles the
    results.

    Parameters
    ----------
    psmiles : str
        Polymer SMILES string with ``[*]`` attachment points.

    Returns
    -------
    dict or None
        Keys: ``psmiles``, ``delta_hf``, ``delta_cp``, ``tm``,
        ``group_parse``.  Returns ``None`` if the PSMILES is entirely
        unparseable.

    Examples
    --------
    >>> result = estimate_all("[*]CC[*]")
    >>> result["delta_hf"]["value"]
    8000.0
    >>> result["tm"]["value"]  # ~404 K for polyethylene
    404.04
    """
    parsed = parse_psmiles_groups(psmiles)
    if parsed is None:
        return None

    dhf = estimate_delta_hf(psmiles)
    dcp = estimate_delta_cp(psmiles)
    tm = estimate_tm(psmiles)

    overall_reliable = all(
        r is not None and r.get("reliable", False) for r in [dhf, dcp, tm]
    )

    return {
        "psmiles": psmiles,
        "delta_hf": dhf,
        "delta_cp": dcp,
        "tm": tm,
        "group_parse": parsed,
        "overall_reliable": overall_reliable,
    }
