from strap.ml_assets import (
    load_ml_polymer_catalog,
    missing_ml_assets,
    resolve_polymer_entry,
    resolve_solvent_entry,
)


def test_ml_assets_are_present_and_loadable():
    assert missing_ml_assets() == []
    types, grouped = load_ml_polymer_catalog()
    assert any(item["type"] == "Polyolefins" for item in types)
    assert grouped["Polyolefins"]


def test_ml_asset_lookup_resolves_polymer_and_solvent_entries():
    polymer = resolve_polymer_entry("HDPE")
    solvent = resolve_solvent_entry("acetone")

    assert polymer is not None
    assert polymer["polymer"] == "HDPE"
    assert polymer["interaction_radius"] > 0

    assert solvent is not None
    assert solvent["solvent"] == "Propan-2-one"
    assert solvent["dispersion"] > 0
