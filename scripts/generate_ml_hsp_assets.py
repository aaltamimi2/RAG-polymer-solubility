#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def hsp_type_for_polymer(polymer_name: str) -> str:
    upper = polymer_name.upper()
    if any(token in upper for token in ("HDPE", "LDPE", "LLDPE", "PE", "PP", "POLYOLEFIN", "EVA", "EPDM")):
        return "Polyolefins"
    if any(token in upper for token in ("PET", "PBT", "PLA", "POLYESTER", "CELLIT", "CELLULOSE ACETATE")):
        return "Polyesters and Cellulosics"
    if any(token in upper for token in ("PA", "NYLON", "POLYAMIDE", "ARAMID")):
        return "Polyamides"
    if any(token in upper for token in ("PS", "HIPS", "ABS", "STYRENE", "SAN")):
        return "Styrenics"
    if any(token in upper for token in ("PVC", "PVDC", "VINYL", "PVA", "PVAC", "EVOH")):
        return "Vinyl and Barrier Polymers"
    if any(token in upper for token in ("PC", "PMMA", "POM", "PPO", "PBT", "PEEK", "PPS", "ENGINEERING")):
        return "Engineering Thermoplastics"
    if any(token in upper for token in ("PTFE", "PVDF", "ETFE", "FEP", "PFA", "FLUOR")):
        return "Fluoropolymers"
    if any(token in upper for token in ("PU", "TPU", "POLYURETHANE", "ELASTOMER", "RUBBER")):
        return "Elastomers and Polyurethanes"
    return "Other Polymers"


def build_assets(source_csv: Path, output_dir: Path) -> tuple[Path, Path]:
    header = pd.read_csv(source_csv, nrows=0)
    available_columns = set(header.columns)

    polymer_columns = [
        "Polymer",
        "Polymer_Dispersion",
        "Polymer_Polar",
        "Polymer_Hydrogen",
        "R0",
    ]
    solvent_columns = [
        "Solvent",
        "Solvent_Dispersion",
        "Solvent_Polar",
        "Solvent_Hydrogen",
    ]
    if "Molar Volume" in available_columns:
        solvent_columns.append("Molar Volume")

    usecols = [column for column in polymer_columns + solvent_columns if column in available_columns]
    df = pd.read_csv(source_csv, usecols=usecols)

    polymers = (
        df[polymer_columns]
        .dropna(subset=["Polymer"])
        .drop_duplicates(subset=["Polymer"])
        .sort_values("Polymer")
        .copy()
    )
    polymers["type"] = polymers["Polymer"].map(hsp_type_for_polymer)

    grouped: dict[str, list[dict[str, float | str]]] = {}
    polymer_lookup: dict[str, dict[str, float | str]] = {}
    polymer_names: list[str] = []
    for _, row in polymers.iterrows():
        item = {
            "polymer": str(row["Polymer"]),
            "dispersion": float(row["Polymer_Dispersion"]),
            "polar": float(row["Polymer_Polar"]),
            "hydrogen_bonding": float(row["Polymer_Hydrogen"]),
            "interaction_radius": float(row["R0"]),
        }
        grouped.setdefault(str(row["type"]), []).append(item)
        polymer_lookup[str(row["Polymer"]).lower()] = item
        polymer_names.append(str(row["Polymer"]))

    polymer_types = [
        {"type": polymer_type, "count": len(items)}
        for polymer_type, items in sorted(grouped.items(), key=lambda item: item[0])
    ]

    solvent_frame = df[solvent_columns].dropna(subset=["Solvent"]).drop_duplicates(subset=["Solvent"]).sort_values("Solvent")
    solvent_lookup: dict[str, dict[str, float | str]] = {}
    solvent_names: list[str] = []
    for _, row in solvent_frame.iterrows():
        item = {
            "solvent": str(row["Solvent"]),
            "dispersion": float(row["Solvent_Dispersion"]),
            "polar": float(row["Solvent_Polar"]),
            "hydrogen_bonding": float(row["Solvent_Hydrogen"]),
            "molar_volume": float(row["Molar Volume"]) if "Molar Volume" in solvent_frame.columns else 100.0,
        }
        solvent_lookup[str(row["Solvent"]).lower()] = item
        solvent_names.append(str(row["Solvent"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = output_dir / "ml_polymer_catalog.json"
    lookup_path = output_dir / "ml_hsp_lookup.json"

    with catalog_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"types": polymer_types, "grouped": grouped},
            fh,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    with lookup_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "polymer_names": polymer_names,
                "polymers": polymer_lookup,
                "solvent_names": solvent_names,
                "solvents": solvent_lookup,
            },
            fh,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    return catalog_path, lookup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate compact ML HSP assets from the full CSV.")
    parser.add_argument(
        "--source",
        default="HSP-ML-integration/RED_values_complete_CORRECTED.csv",
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for the generated JSON assets.",
    )
    args = parser.parse_args()

    source_csv = Path(args.source).resolve()
    output_dir = Path(args.output_dir).resolve()
    catalog_path, lookup_path = build_assets(source_csv, output_dir)
    print(f"Wrote {catalog_path}")
    print(f"Wrote {lookup_path}")


if __name__ == "__main__":
    main()
