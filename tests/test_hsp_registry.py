from __future__ import annotations

from strap.hsp_registry import (
    get_hsp_polymer_asset,
    resolve_hsp_polymer,
    resolve_hsp_polymer_category,
    resolve_hsp_solvent,
    resolve_hsp_solvent_category,
    resolve_hsp_solvent_polarity,
)


def _names(entries):
    return {entry.display_name for entry in entries}


def test_hsp_polymer_resolution_handles_trimmed_raw_names():
    result = resolve_hsp_polymer("PC")

    assert result.status == "resolved"
    assert result.selected is not None
    assert result.selected.display_name == "PC"
    assert result.selected.raw_hsp_name == "PC "
    assert get_hsp_polymer_asset(result.selected)["polymer"] == "PC "


def test_hsp_polymer_resolution_flags_qualified_and_ambiguous_entries():
    pvdf = resolve_hsp_polymer("PVDF")
    ldpe = resolve_hsp_polymer("LDPE")

    assert pvdf.status == "resolved"
    assert pvdf.selected is not None
    assert pvdf.selected.raw_hsp_name == "POLYVINYLIDINE FLUORIDE SOL."
    assert pvdf.selected.quality == "qualified"

    assert ldpe.status == "ambiguous"
    assert {"LDPE PERM<0.8", "LDPE PERM>80"} <= _names(ldpe.matches)


def test_hsp_polymer_categories_include_family_and_polarity_groups():
    polyolefins = resolve_hsp_polymer_category("polyolefins")
    nylons = resolve_hsp_polymer("nylon")
    nonpolar = resolve_hsp_polymer_category("nonpolar polymers")
    hydrogen_bonding = resolve_hsp_polymer_category("hydrogen bonding polymers")

    assert polyolefins.status == "category"
    assert _names(polyolefins.category_members) == {"PE", "HDPE", "PP"}

    assert nylons.status == "category"
    assert {"Nylon 66", "PA6", "PA66"} <= _names(nylons.category_members)

    assert nonpolar.status == "category"
    assert {"PE", "HDPE", "PP", "PS"} <= _names(nonpolar.category_members)

    assert hydrogen_bonding.status == "category"
    assert {"EVOH", "PA6", "PA66"} <= _names(hydrogen_bonding.category_members)


def test_hsp_solvent_resolution_uses_curated_aliases_and_blocks_bad_proxies():
    ethylene_glycol = resolve_hsp_solvent("ethylene glycol")
    propylene_glycol = resolve_hsp_solvent("propylene glycol")
    gvl = resolve_hsp_solvent("GVL")
    dioxane = resolve_hsp_solvent("1,4-dioxane")
    dma = resolve_hsp_solvent("N,N-dimethylacetamide")

    assert ethylene_glycol.status == "resolved"
    assert ethylene_glycol.selected is not None
    assert ethylene_glycol.selected.raw_hsp_name == "Ethane-1,2-diol"

    assert propylene_glycol.status == "resolved"
    assert propylene_glycol.selected is not None
    assert propylene_glycol.selected.raw_hsp_name == "Propane-1,2-diol"

    assert gvl.status == "unsupported"
    assert "not present" in (gvl.unsupported_reason or "")

    assert dioxane.status == "unsupported"
    assert "1,4-dioxane" in (dioxane.unsupported_reason or "")

    assert dma.status == "unsupported"
    assert "dimethylacetamide" in (dma.unsupported_reason or "")


def test_hsp_solvent_categories_include_family_and_polarity_groups():
    nonpolar = resolve_hsp_solvent_polarity("nonpolar")
    polar_aprotic = resolve_hsp_solvent_category("polar aprotic")
    glycols = resolve_hsp_solvent_category("glycols")

    assert nonpolar.status == "category"
    assert {"n-Hexane", "Dodecane"} <= _names(nonpolar.category_members)

    assert polar_aprotic.status == "category"
    assert {"DMF", "DMSO", "NMP"} <= _names(polar_aprotic.category_members)

    assert glycols.status == "category"
    assert {"Ethylene glycol", "Propylene glycol"} == _names(glycols.category_members)
