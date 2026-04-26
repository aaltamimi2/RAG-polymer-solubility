"""Curated Hansen-solubility registry and resolver helpers.

The raw HSP assets contain many qualified, conditioned, and trade-name entries.
This module adds a conservative registry layer so user-facing HSP tools resolve
common names and categories without relying on broad substring matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

from strap.ml_assets import load_ml_hsp_lookup

HspResolutionStatus = Literal["resolved", "ambiguous", "unsupported", "category"]


def normalize_hsp_key(value: str) -> str:
    """Normalize a user query or asset name for exact alias matching."""
    text = str(value or "").strip().casefold()
    text = text.replace("\u00b0", " ")
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[.,;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass(frozen=True)
class HspPolymerEntry:
    id: str
    display_name: str
    raw_hsp_name: str
    aliases: tuple[str, ...]
    polymer_family: str
    polymer_tags: tuple[str, ...] = ()
    quality: str = "canonical"
    qualifier: str | None = None
    default_include: bool = True
    warnings: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "polymer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "display_name": self.display_name,
            "raw_hsp_name": self.raw_hsp_name,
            "aliases": list(self.aliases),
            "polymer_family": self.polymer_family,
            "polymer_tags": list(self.polymer_tags),
            "quality": self.quality,
            "qualifier": self.qualifier,
            "default_include": self.default_include,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HspSolventEntry:
    id: str
    display_name: str
    raw_hsp_name: str
    aliases: tuple[str, ...]
    chemical_family: str
    polarity_class: str
    solvent_tags: tuple[str, ...] = ()
    default_include: bool = True
    warnings: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "solvent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "display_name": self.display_name,
            "raw_hsp_name": self.raw_hsp_name,
            "aliases": list(self.aliases),
            "chemical_family": self.chemical_family,
            "polarity_class": self.polarity_class,
            "solvent_tags": list(self.solvent_tags),
            "default_include": self.default_include,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HspResolverResult:
    status: HspResolutionStatus
    query: str
    kind: str
    selected: HspPolymerEntry | HspSolventEntry | None = None
    matches: tuple[HspPolymerEntry | HspSolventEntry, ...] = ()
    category_id: str | None = None
    category_label: str | None = None
    category_members: tuple[HspPolymerEntry | HspSolventEntry, ...] = ()
    excluded_members: tuple[HspPolymerEntry | HspSolventEntry, ...] = ()
    warnings: tuple[str, ...] = ()
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "query": self.query,
            "kind": self.kind,
            "selected": self.selected.to_dict() if self.selected else None,
            "matches": [entry.to_dict() for entry in self.matches],
            "category_id": self.category_id,
            "category_label": self.category_label,
            "category_members": [entry.to_dict() for entry in self.category_members],
            "excluded_members": [entry.to_dict() for entry in self.excluded_members],
            "warnings": list(self.warnings),
            "unsupported_reason": self.unsupported_reason,
        }


POLYMER_ENTRIES: tuple[HspPolymerEntry, ...] = (
    HspPolymerEntry(
        "pe",
        "PE",
        "PE",
        ("pe", "polyethylene"),
        "polyolefins",
        ("low_polarity", "semicrystalline"),
        warnings=("HSP does not model crystallinity barriers in polyethylene.",),
    ),
    HspPolymerEntry(
        "hdpe",
        "HDPE",
        "HDPE",
        ("hdpe", "high density polyethylene", "high-density polyethylene"),
        "polyolefins",
        ("low_polarity", "semicrystalline"),
        warnings=("HSP does not model crystallinity barriers in HDPE.",),
    ),
    HspPolymerEntry(
        "ldpe_perm_low",
        "LDPE PERM<0.8",
        "LDPE PERM<0.8",
        ("ldpe", "low density polyethylene", "low-density polyethylene"),
        "polyolefins",
        ("low_polarity", "permeation_record"),
        quality="swelling_or_permeation",
        qualifier="LDPE permeability-qualified HSP record; not a clean generic LDPE entry.",
        default_include=False,
        warnings=("LDPE HSP asset is a permeability-qualified record, not a clean generic LDPE entry.",),
    ),
    HspPolymerEntry(
        "ldpe_perm_high",
        "LDPE PERM>80",
        "LDPE PERM>80",
        ("ldpe", "low density polyethylene", "low-density polyethylene"),
        "polyolefins",
        ("low_polarity", "permeation_record"),
        quality="swelling_or_permeation",
        qualifier="LDPE permeability-qualified HSP record; not a clean generic LDPE entry.",
        default_include=False,
        warnings=("LDPE HSP asset is a permeability-qualified record, not a clean generic LDPE entry.",),
    ),
    HspPolymerEntry(
        "pp",
        "PP",
        "PP",
        ("pp", "polypropylene"),
        "polyolefins",
        ("low_polarity", "semicrystalline"),
        warnings=("HSP does not model crystallinity barriers in PP.",),
    ),
    HspPolymerEntry("ps", "PS", "PS", ("ps", "polystyrene"), "styrenics", ("aromatic", "amorphous")),
    HspPolymerEntry(
        "polystyrene_lg",
        "Polystyrene LG",
        "POLYSTYRENE LG",
        ("polystyrene lg",),
        "styrenics",
        ("aromatic",),
        quality="representative",
        qualifier="Specific polystyrene HSP record.",
    ),
    HspPolymerEntry("san", "SAN", "SAN", ("san", "styrene acrylonitrile"), "styrenics", ("aromatic", "polar")),
    HspPolymerEntry(
        "abs",
        "ABS",
        "ABS CR",
        ("abs", "acrylonitrile butadiene styrene"),
        "styrenics",
        ("aromatic", "rubbery_phase"),
        quality="qualified",
        qualifier="Raw HSP entry is ABS CR.",
    ),
    HspPolymerEntry("pvc", "PVC", "PVC", ("pvc", "polyvinyl chloride", "poly vinyl chloride"), "vinyl_barrier", ("chlorinated", "polar")),
    HspPolymerEntry(
        "evoh",
        "EVOH",
        "EVOH SOL",
        ("evoh", "ethylene vinyl alcohol", "ethylene-vinyl alcohol"),
        "vinyl_barrier",
        ("hydrogen_bonding", "barrier_polymer"),
        quality="qualified",
        qualifier="Raw HSP entry is EVOH SOL.",
    ),
    HspPolymerEntry("pvac", "PVAc", "PVAC", ("pvac", "polyvinyl acetate", "poly vinyl acetate"), "vinyl_barrier", ("ester_containing", "polar")),
    HspPolymerEntry("pvoh", "PVOH", "POLYVINYLALCOHOL", ("pvoh", "pva", "polyvinyl alcohol"), "vinyl_barrier", ("hydrogen_bonding", "polar")),
    HspPolymerEntry("petg", "PETG", "PETG", ("petg",), "polyesters", ("ester_containing", "polar")),
    HspPolymerEntry(
        "pet_mylar",
        "PET (Mylar)",
        "MYLAR PET",
        ("pet", "polyethylene terephthalate", "mylar pet", "mylar"),
        "polyesters",
        ("ester_containing", "aromatic"),
        quality="proxy",
        qualifier="PET resolves to a Mylar PET HSP proxy by default.",
    ),
    HspPolymerEntry(
        "petp",
        "PETP",
        "PETP CR",
        ("petp",),
        "polyesters",
        ("ester_containing", "aromatic"),
        quality="qualified",
        qualifier="Raw HSP entry is PETP CR.",
    ),
    HspPolymerEntry("r_pet", "R PET", "R PET", ("r pet",), "polyesters", ("ester_containing",), quality="qualified", default_include=False),
    HspPolymerEntry("nylon66", "Nylon 66", "NYLON 66", ("nylon 66", "nylon-66"), "polyamides", ("amide_containing", "hydrogen_bonding")),
    HspPolymerEntry(
        "pa6",
        "PA6",
        "PA6 CR",
        ("pa6", "nylon 6", "nylon-6"),
        "polyamides",
        ("amide_containing", "hydrogen_bonding"),
        quality="qualified",
        qualifier="Raw HSP entry is PA6 CR.",
    ),
    HspPolymerEntry(
        "pa66",
        "PA66",
        "PA66 SOL",
        ("pa66", "pa 66"),
        "polyamides",
        ("amide_containing", "hydrogen_bonding"),
        quality="qualified",
        qualifier="Raw HSP entry is PA66 SOL.",
    ),
    HspPolymerEntry("pa11", "PA11", "PA11 CR", ("pa11", "pa 11"), "polyamides", ("amide_containing",), quality="qualified", default_include=False),
    HspPolymerEntry("r_pa66", "R PA66", "R PA66", ("r pa66",), "polyamides", ("amide_containing",), quality="qualified", default_include=False),
    HspPolymerEntry("pc", "PC", "PC ", ("pc", "polycarbonate"), "engineering", ("carbonate_containing", "aromatic")),
    HspPolymerEntry("pmma", "PMMA", "PMMA", ("pmma", "polymethyl methacrylate"), "acrylics", ("ester_containing", "polar")),
    HspPolymerEntry("pes", "PES", "PES SOL", ("pes", "polyethersulfone"), "engineering", ("sulfone_containing", "aromatic"), quality="qualified", qualifier="Raw HSP entry is PES SOL."),
    HspPolymerEntry("pom", "POM", "POMH+POMC CR", ("pom", "acetal", "polyoxymethylene"), "engineering", ("ether_containing",), quality="qualified", qualifier="Raw HSP entry combines POMH/POMC CR."),
    HspPolymerEntry("psu", "PSU", "PSU CR", ("psu", "polysulfone"), "engineering", ("sulfone_containing", "aromatic"), quality="qualified", qualifier="Raw HSP entry is PSU CR."),
    HspPolymerEntry("pvdf", "PVDF", "POLYVINYLIDINE FLUORIDE SOL.", ("pvdf", "polyvinylidene fluoride", "polyvinylidine fluoride"), "fluoropolymers", ("fluorinated", "polar"), quality="qualified", qualifier="Raw HSP entry is POLYVINYLIDINE FLUORIDE SOL."),
    HspPolymerEntry("fep", "FEP", "FEP", ("fep",), "fluoropolymers", ("fluorinated",)),
    HspPolymerEntry("pctfe", "PCTFE", "PCTFE", ("pctfe",), "fluoropolymers", ("fluorinated", "chlorinated")),
    HspPolymerEntry("ptfe", "PTFE", "PTFE L80 CR", ("ptfe",), "fluoropolymers", ("fluorinated",), quality="qualified", qualifier="Raw HSP entry is PTFE L80 CR."),
    HspPolymerEntry("pmma_10", "PMMA (10%)", "PMMA (10%)", ("pmma 10", "pmma 10%"), "acrylics", ("conditioned",), quality="conditioned", default_include=False),
    HspPolymerEntry("pmma_30", "PMMA (30%)", "PMMA (30%)", ("pmma 30", "pmma 30%"), "acrylics", ("conditioned",), quality="conditioned", default_include=False),
    HspPolymerEntry("pmma_cr", "PMMA CR", "PMMA CR", ("pmma cr",), "acrylics", ("conditioned",), quality="qualified", default_include=False),
)


SOLVENT_ENTRIES: tuple[HspSolventEntry, ...] = (
    HspSolventEntry("water", "Water", "Water", ("water",), "water", "polar_protic", ("h_bond_donor", "h_bond_acceptor")),
    HspSolventEntry("methanol", "Methanol", "Methanol", ("methanol", "meoh"), "alcohol", "polar_protic", ("h_bond_donor", "h_bond_acceptor")),
    HspSolventEntry("ethanol", "Ethanol", "Ethanol", ("ethanol", "etoh"), "alcohol", "polar_protic", ("h_bond_donor", "h_bond_acceptor")),
    HspSolventEntry("propanol", "Propan-1-ol", "Propan-1-ol", ("propanol", "1-propanol", "propan-1-ol"), "alcohol", "polar_protic", ("h_bond_donor", "h_bond_acceptor")),
    HspSolventEntry("isopropanol", "Isopropanol", "Propan-2-ol", ("isopropanol", "ipa", "2-propanol", "propan-2-ol"), "alcohol", "polar_protic", ("h_bond_donor", "h_bond_acceptor")),
    HspSolventEntry("butanol", "Butan-1-ol", "Butan-1-ol", ("butanol", "1-butanol", "butan-1-ol"), "alcohol", "polar_protic", ("h_bond_donor", "h_bond_acceptor")),
    HspSolventEntry("ethylene_glycol", "Ethylene glycol", "Ethane-1,2-diol", ("ethylene glycol", "glycol", "ethane 1 2 diol", "ethane-1,2-diol"), "glycol", "polar_protic", ("glycol", "h_bond_donor", "h_bond_acceptor", "strap_common")),
    HspSolventEntry("propylene_glycol", "Propylene glycol", "Propane-1,2-diol", ("propylene glycol", "propane 1 2 diol", "propane-1,2-diol"), "glycol", "polar_protic", ("glycol", "h_bond_donor", "h_bond_acceptor", "strap_common")),
    HspSolventEntry("formic_acid", "Formic acid", "Formic acid", ("formic acid",), "acid", "ionic_or_acidic", ("acid", "h_bond_donor", "h_bond_acceptor")),
    HspSolventEntry("hexane", "n-Hexane", "n-Hexane", ("hexane", "n-hexane"), "hydrocarbon", "nonpolar", ("alkane", "hydrocarbon")),
    HspSolventEntry("heptane", "Heptane", "Heptane", ("heptane",), "hydrocarbon", "nonpolar", ("alkane", "hydrocarbon")),
    HspSolventEntry("cyclohexane", "Cyclohexane", "Cyclohexane", ("cyclohexane",), "hydrocarbon", "nonpolar", ("alkane", "hydrocarbon")),
    HspSolventEntry("dodecane", "Dodecane", "Dodecane", ("dodecane",), "hydrocarbon", "nonpolar", ("alkane", "hydrocarbon", "strap_common")),
    HspSolventEntry("benzene", "Benzene", "Benzene", ("benzene",), "aromatic", "weakly_polar", ("aromatic", "hazardous")),
    HspSolventEntry("toluene", "Toluene", "Toluene", ("toluene",), "aromatic", "weakly_polar", ("aromatic", "strap_common")),
    HspSolventEntry("o_xylene", "o-Xylene", "o-Xylene", ("xylene", "o-xylene", "ortho xylene"), "aromatic", "weakly_polar", ("aromatic", "strap_common")),
    HspSolventEntry("p_xylene", "p-Xylene", "p-Xylene", ("p-xylene", "para xylene"), "aromatic", "weakly_polar", ("aromatic",)),
    HspSolventEntry("dcm", "Dichloromethane", "Dichloro-methane", ("dcm", "dichloromethane", "methylene chloride"), "chlorinated", "weakly_polar", ("chlorinated", "h_bond_acceptor")),
    HspSolventEntry("chloroform", "Chloroform", "Trichloro-methane", ("chloroform", "trichloromethane"), "chlorinated", "weakly_polar", ("chlorinated", "h_bond_donor")),
    HspSolventEntry("dce", "1,2-Dichloroethane", "1,2-Dichloro-ethane", ("dce", "1,2-dichloroethane", "1 2 dichloroethane"), "chlorinated", "weakly_polar", ("chlorinated",)),
    HspSolventEntry("carbon_tetrachloride", "Carbon tetrachloride", "Tetrachloro-methane", ("carbon tetrachloride", "ccl4", "tetrachloromethane"), "chlorinated", "nonpolar", ("chlorinated",)),
    HspSolventEntry("thf", "THF", "Tetrahydro-furan", ("thf", "tetrahydrofuran", "tetrahydro furan"), "ether", "weakly_polar", ("ether", "h_bond_acceptor", "strap_common")),
    HspSolventEntry("acetone", "Acetone", "Propan-2-one", ("acetone", "propanone", "propan-2-one"), "ketone", "polar_aprotic", ("ketone", "h_bond_acceptor")),
    HspSolventEntry("mek", "MEK", "Butan-2-one", ("mek", "butanone", "2-butanone", "butan-2-one"), "ketone", "polar_aprotic", ("ketone", "h_bond_acceptor")),
    HspSolventEntry("mibk", "MIBK", "4-Methyl-pentan-2-one", ("mibk", "4-methyl-2-pentanone", "4-methyl-pentan-2-one"), "ketone", "polar_aprotic", ("ketone", "h_bond_acceptor")),
    HspSolventEntry("cyclohexanone", "Cyclohexanone", "Cyclohexanone", ("cyclohexanone",), "ketone", "polar_aprotic", ("ketone", "h_bond_acceptor")),
    HspSolventEntry("ethyl_acetate", "Ethyl acetate", "Acetic acid ethyl ester", ("ethyl acetate", "acetic acid ethyl ester", "ea"), "ester", "weakly_polar", ("ester", "h_bond_acceptor")),
    HspSolventEntry("methyl_acetate", "Methyl acetate", "Acetic acid methyl ester", ("methyl acetate", "acetic acid methyl ester"), "ester", "weakly_polar", ("ester", "h_bond_acceptor")),
    HspSolventEntry("propyl_acetate", "Propyl acetate", "Acetic acid propyl ester", ("propyl acetate", "acetic acid propyl ester"), "ester", "weakly_polar", ("ester", "h_bond_acceptor")),
    HspSolventEntry("butyl_acetate", "Butyl acetate", "Acetic acid butyl ester", ("butyl acetate", "acetic acid butyl ester"), "ester", "weakly_polar", ("ester", "h_bond_acceptor")),
    HspSolventEntry("acetonitrile", "Acetonitrile", "Acetonitrile", ("acetonitrile", "acn"), "nitrile", "polar_aprotic", ("nitrile", "h_bond_acceptor")),
    HspSolventEntry("dmf", "DMF", "N,N-Dimethyl- formamide", ("dmf", "dimethylformamide", "n,n-dimethylformamide", "n n dimethyl formamide"), "amide", "polar_aprotic", ("amide", "h_bond_acceptor", "strap_common")),
    HspSolventEntry("nmp", "NMP", "1-Methyl-pyrrolidin-2- one", ("nmp", "n-methyl-2-pyrrolidone", "1-methyl-pyrrolidin-2-one"), "amide", "polar_aprotic", ("amide", "h_bond_acceptor", "strap_common")),
    HspSolventEntry("dmso", "DMSO", "Methylsulfi yl-methane", ("dmso", "dimethyl sulfoxide", "dimethylsulfoxide"), "sulfoxide", "polar_aprotic", ("sulfoxide", "h_bond_acceptor", "strap_common"), warnings=("Raw HSP asset uses the name Methylsulfi yl-methane for DMSO.",)),
)


POLYMER_CATEGORY_ALIASES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "polyolefins": ("polyolefins", "Polyolefins", ("pe", "hdpe", "pp")),
    "polyolefin": ("polyolefins", "Polyolefins", ("pe", "hdpe", "pp")),
    "nylon": ("polyamides", "Nylons / Polyamides", ("nylon66", "pa6", "pa66")),
    "nylons": ("polyamides", "Nylons / Polyamides", ("nylon66", "pa6", "pa66")),
    "polyamide": ("polyamides", "Nylons / Polyamides", ("nylon66", "pa6", "pa66")),
    "polyamides": ("polyamides", "Nylons / Polyamides", ("nylon66", "pa6", "pa66")),
    "polyester": ("polyesters", "Polyesters", ("petg", "pet_mylar", "petp")),
    "polyesters": ("polyesters", "Polyesters", ("petg", "pet_mylar", "petp")),
    "styrenic": ("styrenics", "Styrenics", ("ps", "polystyrene_lg", "san", "abs")),
    "styrenics": ("styrenics", "Styrenics", ("ps", "polystyrene_lg", "san", "abs")),
    "vinyl": ("vinyl_barrier", "Vinyl / Barrier Polymers", ("pvc", "evoh", "pvac")),
    "vinyls": ("vinyl_barrier", "Vinyl / Barrier Polymers", ("pvc", "evoh", "pvac")),
    "barrier": ("vinyl_barrier", "Vinyl / Barrier Polymers", ("pvc", "evoh", "pvac")),
    "barrier polymers": ("vinyl_barrier", "Vinyl / Barrier Polymers", ("pvc", "evoh", "pvac")),
    "engineering": ("engineering", "Engineering Polymers", ("pc", "pmma", "pes", "pom", "psu")),
    "engineering polymers": ("engineering", "Engineering Polymers", ("pc", "pmma", "pes", "pom", "psu")),
    "fluoropolymer": ("fluoropolymers", "Fluoropolymers", ("pvdf", "fep", "pctfe", "ptfe")),
    "fluoropolymers": ("fluoropolymers", "Fluoropolymers", ("pvdf", "fep", "pctfe", "ptfe")),
    "acrylic": ("acrylics", "Acrylics", ("pmma",)),
    "acrylics": ("acrylics", "Acrylics", ("pmma",)),
    "nonpolar": ("nonpolar_polymers", "Nonpolar / Low-Polarity Polymers", ("pe", "hdpe", "pp", "ps", "polystyrene_lg", "fep", "ptfe")),
    "non polar": ("nonpolar_polymers", "Nonpolar / Low-Polarity Polymers", ("pe", "hdpe", "pp", "ps", "polystyrene_lg", "fep", "ptfe")),
    "nonpolar polymers": ("nonpolar_polymers", "Nonpolar / Low-Polarity Polymers", ("pe", "hdpe", "pp", "ps", "polystyrene_lg", "fep", "ptfe")),
    "low polarity": ("nonpolar_polymers", "Nonpolar / Low-Polarity Polymers", ("pe", "hdpe", "pp", "ps", "polystyrene_lg", "fep", "ptfe")),
    "low polarity polymers": ("nonpolar_polymers", "Nonpolar / Low-Polarity Polymers", ("pe", "hdpe", "pp", "ps", "polystyrene_lg", "fep", "ptfe")),
    "weakly polar": ("weakly_polar_polymers", "Weakly Polar Polymers", ("san", "abs", "pvc", "pvac", "petg", "pet_mylar", "petp", "pc", "pmma", "pvdf")),
    "weakly polar polymers": ("weakly_polar_polymers", "Weakly Polar Polymers", ("san", "abs", "pvc", "pvac", "petg", "pet_mylar", "petp", "pc", "pmma", "pvdf")),
    "polar": ("polar_polymers", "Polar Polymers", ("pvc", "evoh", "pvac", "pvoh", "petg", "pet_mylar", "petp", "nylon66", "pa6", "pa66", "pc", "pmma", "pes", "pom", "psu", "pvdf", "san")),
    "polar polymers": ("polar_polymers", "Polar Polymers", ("pvc", "evoh", "pvac", "pvoh", "petg", "pet_mylar", "petp", "nylon66", "pa6", "pa66", "pc", "pmma", "pes", "pom", "psu", "pvdf", "san")),
    "hydrogen bonding": ("hydrogen_bonding_polymers", "Hydrogen-Bonding Polymers", ("evoh", "pvoh", "nylon66", "pa6", "pa66")),
    "hydrogen bonding polymers": ("hydrogen_bonding_polymers", "Hydrogen-Bonding Polymers", ("evoh", "pvoh", "nylon66", "pa6", "pa66")),
    "h bonding": ("hydrogen_bonding_polymers", "Hydrogen-Bonding Polymers", ("evoh", "pvoh", "nylon66", "pa6", "pa66")),
    "semicrystalline": ("semicrystalline_polymers", "Semicrystalline Polymers", ("pe", "hdpe", "pp", "pet_mylar", "nylon66", "pa6", "pa66", "pvdf", "ptfe")),
    "semicrystalline polymers": ("semicrystalline_polymers", "Semicrystalline Polymers", ("pe", "hdpe", "pp", "pet_mylar", "nylon66", "pa6", "pa66", "pvdf", "ptfe")),
    "amorphous": ("amorphous_polymers", "Amorphous Polymers", ("ps", "polystyrene_lg", "san", "abs", "pvc", "pvac", "pc", "pmma", "pes", "pom", "psu")),
    "amorphous polymers": ("amorphous_polymers", "Amorphous Polymers", ("ps", "polystyrene_lg", "san", "abs", "pvc", "pvac", "pc", "pmma", "pes", "pom", "psu")),
    "aromatic polymers": ("aromatic_polymers", "Aromatic Polymers", ("ps", "polystyrene_lg", "san", "abs", "petg", "pet_mylar", "petp", "pc", "pes", "psu")),
}


SOLVENT_CATEGORY_ALIASES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "hydrocarbon": ("hydrocarbons", "Hydrocarbons", ("hexane", "heptane", "cyclohexane", "dodecane")),
    "hydrocarbons": ("hydrocarbons", "Hydrocarbons", ("hexane", "heptane", "cyclohexane", "dodecane")),
    "alkane": ("hydrocarbons", "Hydrocarbons", ("hexane", "heptane", "cyclohexane", "dodecane")),
    "alkanes": ("hydrocarbons", "Hydrocarbons", ("hexane", "heptane", "cyclohexane", "dodecane")),
    "aromatic": ("aromatics", "Aromatics", ("toluene", "o_xylene", "p_xylene", "benzene")),
    "aromatics": ("aromatics", "Aromatics", ("toluene", "o_xylene", "p_xylene", "benzene")),
    "chlorinated": ("chlorinated", "Chlorinated Solvents", ("dcm", "chloroform", "dce", "carbon_tetrachloride")),
    "halogenated": ("chlorinated", "Chlorinated Solvents", ("dcm", "chloroform", "dce", "carbon_tetrachloride")),
    "alcohol": ("alcohols", "Alcohols", ("methanol", "ethanol", "propanol", "isopropanol", "butanol")),
    "alcohols": ("alcohols", "Alcohols", ("methanol", "ethanol", "propanol", "isopropanol", "butanol")),
    "glycol": ("glycols", "Glycols", ("ethylene_glycol", "propylene_glycol")),
    "glycols": ("glycols", "Glycols", ("ethylene_glycol", "propylene_glycol")),
    "ketone": ("ketones", "Ketones", ("acetone", "mek", "mibk", "cyclohexanone")),
    "ketones": ("ketones", "Ketones", ("acetone", "mek", "mibk", "cyclohexanone")),
    "ester": ("esters", "Esters", ("ethyl_acetate", "methyl_acetate", "propyl_acetate", "butyl_acetate")),
    "esters": ("esters", "Esters", ("ethyl_acetate", "methyl_acetate", "propyl_acetate", "butyl_acetate")),
    "ether": ("ethers", "Ethers", ("thf",)),
    "ethers": ("ethers", "Ethers", ("thf",)),
    "amide": ("amides", "Amides", ("dmf", "nmp")),
    "amides": ("amides", "Amides", ("dmf", "nmp")),
    "sulfoxide": ("sulfoxides", "Sulfoxides", ("dmso",)),
    "sulfoxides": ("sulfoxides", "Sulfoxides", ("dmso",)),
    "acid": ("acids", "Acids", ("formic_acid",)),
    "acids": ("acids", "Acids", ("formic_acid",)),
    "strap common": ("strap_common", "STRAP-common HSP Solvents", ("toluene", "o_xylene", "dodecane", "thf", "dmf", "nmp", "dmso", "ethylene_glycol", "propylene_glycol")),
    "strap solvents": ("strap_common", "STRAP-common HSP Solvents", ("toluene", "o_xylene", "dodecane", "thf", "dmf", "nmp", "dmso", "ethylene_glycol", "propylene_glycol")),
}


SOLVENT_POLARITY_ALIASES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "nonpolar": ("nonpolar", "Nonpolar Solvents", ("hexane", "heptane", "cyclohexane", "dodecane")),
    "non polar": ("nonpolar", "Nonpolar Solvents", ("hexane", "heptane", "cyclohexane", "dodecane")),
    "apolar": ("nonpolar", "Nonpolar Solvents", ("hexane", "heptane", "cyclohexane", "dodecane")),
    "weakly polar": ("weakly_polar", "Weakly Polar Solvents", ("toluene", "o_xylene", "p_xylene", "dcm", "chloroform", "thf", "ethyl_acetate")),
    "moderately polar": ("weakly_polar", "Weakly Polar Solvents", ("toluene", "o_xylene", "p_xylene", "dcm", "chloroform", "thf", "ethyl_acetate")),
    "polar aprotic": ("polar_aprotic", "Polar Aprotic Solvents", ("acetone", "mek", "mibk", "cyclohexanone", "acetonitrile", "dmf", "dmso", "nmp")),
    "aprotic polar": ("polar_aprotic", "Polar Aprotic Solvents", ("acetone", "mek", "mibk", "cyclohexanone", "acetonitrile", "dmf", "dmso", "nmp")),
    "polar protic": ("polar_protic", "Polar Protic Solvents", ("water", "methanol", "ethanol", "isopropanol", "butanol", "ethylene_glycol", "propylene_glycol", "formic_acid")),
    "protic": ("polar_protic", "Polar Protic Solvents", ("water", "methanol", "ethanol", "isopropanol", "butanol", "ethylene_glycol", "propylene_glycol", "formic_acid")),
    "polar": ("polar", "Polar Solvents", ("acetone", "mek", "mibk", "cyclohexanone", "acetonitrile", "dmf", "dmso", "nmp", "water", "methanol", "ethanol", "isopropanol", "ethylene_glycol", "propylene_glycol", "formic_acid")),
    "polar solvents": ("polar", "Polar Solvents", ("acetone", "mek", "mibk", "cyclohexanone", "acetonitrile", "dmf", "dmso", "nmp", "water", "methanol", "ethanol", "isopropanol", "ethylene_glycol", "propylene_glycol", "formic_acid")),
    "hydrogen bonding": ("hydrogen_bonding", "Hydrogen-bonding Solvents", ("water", "methanol", "ethanol", "isopropanol", "butanol", "ethylene_glycol", "propylene_glycol", "formic_acid")),
    "h bonding": ("hydrogen_bonding", "Hydrogen-bonding Solvents", ("water", "methanol", "ethanol", "isopropanol", "butanol", "ethylene_glycol", "propylene_glycol", "formic_acid")),
    "h bond": ("hydrogen_bonding", "Hydrogen-bonding Solvents", ("water", "methanol", "ethanol", "isopropanol", "butanol", "ethylene_glycol", "propylene_glycol", "formic_acid")),
}


UNSUPPORTED_SOLVENT_ALIASES = {
    "gvl": "GVL/gamma-valerolactone is not present in the local HSP solvent asset.",
    "gamma valerolactone": "GVL/gamma-valerolactone is not present in the local HSP solvent asset.",
    "gamma-valerolactone": "GVL/gamma-valerolactone is not present in the local HSP solvent asset.",
    "gbl": "GBL/gamma-butyrolactone is not present in the local HSP solvent asset.",
    "gamma butyrolactone": "GBL/gamma-butyrolactone is not present in the local HSP solvent asset.",
    "gamma-butyrolactone": "GBL/gamma-butyrolactone is not present in the local HSP solvent asset.",
    "dma": "DMA/N,N-dimethylacetamide is not present in the local HSP solvent asset.",
    "dimethylacetamide": "DMA/N,N-dimethylacetamide is not present in the local HSP solvent asset.",
    "n,n dimethylacetamide": "DMA/N,N-dimethylacetamide is not present in the local HSP solvent asset.",
    "n n dimethylacetamide": "DMA/N,N-dimethylacetamide is not present in the local HSP solvent asset.",
    "n n dimethyl acetamide": "DMA/N,N-dimethylacetamide is not present in the local HSP solvent asset.",
    "n,n dimethyl acetamide": "DMA/N,N-dimethylacetamide is not present in the local HSP solvent asset.",
    "diethyl ether": "Diethyl ether is not present in the local HSP solvent asset.",
    "1 4 dioxane": "1,4-dioxane is not present in the local HSP solvent asset.",
    "1,4 dioxane": "1,4-dioxane is not present in the local HSP solvent asset.",
    "1,4-dioxane": "1,4-dioxane is not present in the local HSP solvent asset.",
    "dioxane": "Dioxane is ambiguous in the local HSP asset; do not substitute another isomer for 1,4-dioxane.",
}


def _raw_polymer_norm_map() -> dict[str, dict[str, Any]]:
    lookup = load_ml_hsp_lookup()
    return {
        normalize_hsp_key(entry["polymer"]): entry
        for entry in lookup["polymers"].values()
    }


def _raw_solvent_norm_map() -> dict[str, dict[str, Any]]:
    lookup = load_ml_hsp_lookup()
    return {
        normalize_hsp_key(entry["solvent"]): entry
        for entry in lookup["solvents"].values()
    }


def get_hsp_polymer_asset(entry: HspPolymerEntry) -> dict[str, Any]:
    raw = _raw_polymer_norm_map().get(normalize_hsp_key(entry.raw_hsp_name))
    if raw is None:
        raise KeyError(f"HSP polymer asset not found: {entry.raw_hsp_name}")
    return raw


def get_hsp_solvent_asset(entry: HspSolventEntry) -> dict[str, Any]:
    raw = _raw_solvent_norm_map().get(normalize_hsp_key(entry.raw_hsp_name))
    if raw is None:
        raise KeyError(f"HSP solvent asset not found: {entry.raw_hsp_name}")
    return raw


def _entry_alias_keys(entry: HspPolymerEntry | HspSolventEntry) -> set[str]:
    values = {entry.id, entry.display_name, entry.raw_hsp_name, *entry.aliases}
    return {normalize_hsp_key(value) for value in values if value}


def _polymer_by_id() -> dict[str, HspPolymerEntry]:
    return {entry.id: entry for entry in POLYMER_ENTRIES}


def _solvent_by_id() -> dict[str, HspSolventEntry]:
    return {entry.id: entry for entry in SOLVENT_ENTRIES}


def _matches_for_query(
    query: str,
    entries: tuple[HspPolymerEntry, ...] | tuple[HspSolventEntry, ...],
) -> tuple[HspPolymerEntry | HspSolventEntry, ...]:
    key = normalize_hsp_key(query)
    return tuple(entry for entry in entries if key in _entry_alias_keys(entry))


def _category_members(
    ids: tuple[str, ...],
    entries_by_id: dict[str, HspPolymerEntry] | dict[str, HspSolventEntry],
    *,
    include_excluded: bool = False,
) -> tuple[tuple[HspPolymerEntry | HspSolventEntry, ...], tuple[HspPolymerEntry | HspSolventEntry, ...]]:
    included: list[HspPolymerEntry | HspSolventEntry] = []
    excluded: list[HspPolymerEntry | HspSolventEntry] = []
    for entry_id in ids:
        entry = entries_by_id[entry_id]
        if entry.default_include or include_excluded:
            included.append(entry)
        else:
            excluded.append(entry)
    return tuple(included), tuple(excluded)


def resolve_hsp_polymer_category(category: str, *, include_excluded: bool = False) -> HspResolverResult:
    query_key = normalize_hsp_key(category)
    category_info = POLYMER_CATEGORY_ALIASES.get(query_key)
    if category_info is None:
        return HspResolverResult(
            status="unsupported",
            query=category,
            kind="polymer",
            unsupported_reason=f"Unsupported HSP polymer category: {category}",
        )
    category_id, label, member_ids = category_info
    members, excluded = _category_members(member_ids, _polymer_by_id(), include_excluded=include_excluded)
    return HspResolverResult(
        status="category",
        query=category,
        kind="polymer",
        category_id=category_id,
        category_label=label,
        category_members=members,
        excluded_members=excluded,
    )


def resolve_hsp_solvent_category(category: str, *, include_excluded: bool = False) -> HspResolverResult:
    query_key = normalize_hsp_key(category)
    category_info = SOLVENT_CATEGORY_ALIASES.get(query_key) or SOLVENT_POLARITY_ALIASES.get(query_key)
    if category_info is None:
        return HspResolverResult(
            status="unsupported",
            query=category,
            kind="solvent",
            unsupported_reason=f"Unsupported HSP solvent category: {category}",
        )
    category_id, label, member_ids = category_info
    members, excluded = _category_members(member_ids, _solvent_by_id(), include_excluded=include_excluded)
    return HspResolverResult(
        status="category",
        query=category,
        kind="solvent",
        category_id=category_id,
        category_label=label,
        category_members=members,
        excluded_members=excluded,
    )


def resolve_hsp_solvent_polarity(polarity: str, *, include_excluded: bool = False) -> HspResolverResult:
    query_key = normalize_hsp_key(polarity)
    category_info = SOLVENT_POLARITY_ALIASES.get(query_key)
    if category_info is None:
        return HspResolverResult(
            status="unsupported",
            query=polarity,
            kind="solvent",
            unsupported_reason=f"Unsupported HSP solvent polarity category: {polarity}",
        )
    category_id, label, member_ids = category_info
    members, excluded = _category_members(member_ids, _solvent_by_id(), include_excluded=include_excluded)
    return HspResolverResult(
        status="category",
        query=polarity,
        kind="solvent",
        category_id=category_id,
        category_label=label,
        category_members=members,
        excluded_members=excluded,
    )


def resolve_hsp_polymer(
    polymer_name: str,
    *,
    curated_only: bool = True,
    include_excluded: bool = False,
) -> HspResolverResult:
    key = normalize_hsp_key(polymer_name)
    if key in POLYMER_CATEGORY_ALIASES:
        return resolve_hsp_polymer_category(polymer_name, include_excluded=include_excluded)

    matches = _matches_for_query(polymer_name, POLYMER_ENTRIES)
    if len(matches) == 1:
        entry = matches[0]
        return HspResolverResult(
            status="resolved",
            query=polymer_name,
            kind="polymer",
            selected=entry,
            matches=matches,
            warnings=entry.warnings,
        )
    if len(matches) > 1:
        return HspResolverResult(
            status="ambiguous",
            query=polymer_name,
            kind="polymer",
            matches=matches,
            warnings=("Multiple curated HSP entries match this polymer query; choose a specific entry.",),
        )

    if not curated_only:
        raw = _raw_polymer_norm_map().get(key)
        if raw is not None:
            entry = HspPolymerEntry(
                id=key.replace(" ", "_"),
                display_name=str(raw["polymer"]).strip(),
                raw_hsp_name=str(raw["polymer"]),
                aliases=(str(raw["polymer"]),),
                polymer_family="uncurated",
                quality="ambiguous",
                qualifier="Raw HSP asset entry outside the curated registry.",
                default_include=False,
            )
            return HspResolverResult(status="resolved", query=polymer_name, kind="polymer", selected=entry, matches=(entry,))

    suggestions = [
        entry
        for entry in POLYMER_ENTRIES
        if key and key in normalize_hsp_key(" ".join((entry.id, entry.display_name, entry.raw_hsp_name, *entry.aliases)))
    ][:8]
    return HspResolverResult(
        status="unsupported",
        query=polymer_name,
        kind="polymer",
        matches=tuple(suggestions),
        unsupported_reason=f"HSP polymer not supported by curated registry: {polymer_name}",
    )


def resolve_hsp_solvent(
    solvent_name: str,
    *,
    curated_only: bool = True,
    include_excluded: bool = False,
) -> HspResolverResult:
    key = normalize_hsp_key(solvent_name)
    if key in UNSUPPORTED_SOLVENT_ALIASES:
        return HspResolverResult(
            status="unsupported",
            query=solvent_name,
            kind="solvent",
            unsupported_reason=UNSUPPORTED_SOLVENT_ALIASES[key],
        )
    if key in SOLVENT_CATEGORY_ALIASES or key in SOLVENT_POLARITY_ALIASES:
        return resolve_hsp_solvent_category(solvent_name, include_excluded=include_excluded)

    matches = _matches_for_query(solvent_name, SOLVENT_ENTRIES)
    if len(matches) == 1:
        entry = matches[0]
        return HspResolverResult(
            status="resolved",
            query=solvent_name,
            kind="solvent",
            selected=entry,
            matches=matches,
            warnings=entry.warnings,
        )
    if len(matches) > 1:
        return HspResolverResult(
            status="ambiguous",
            query=solvent_name,
            kind="solvent",
            matches=matches,
            warnings=("Multiple curated HSP entries match this solvent query; choose a specific entry.",),
        )

    if not curated_only:
        raw = _raw_solvent_norm_map().get(key)
        if raw is not None:
            entry = HspSolventEntry(
                id=key.replace(" ", "_"),
                display_name=str(raw["solvent"]).strip(),
                raw_hsp_name=str(raw["solvent"]),
                aliases=(str(raw["solvent"]),),
                chemical_family="uncurated",
                polarity_class="ambiguous",
                default_include=False,
            )
            return HspResolverResult(status="resolved", query=solvent_name, kind="solvent", selected=entry, matches=(entry,))

    suggestions = [
        entry
        for entry in SOLVENT_ENTRIES
        if key and key in normalize_hsp_key(" ".join((entry.id, entry.display_name, entry.raw_hsp_name, *entry.aliases)))
    ][:8]
    return HspResolverResult(
        status="unsupported",
        query=solvent_name,
        kind="solvent",
        matches=tuple(suggestions),
        unsupported_reason=f"HSP solvent not supported by curated registry: {solvent_name}",
    )


def list_hsp_polymer_entries(
    *,
    category: str | None = None,
    include_excluded: bool = False,
) -> tuple[HspPolymerEntry, ...]:
    if category:
        result = resolve_hsp_polymer_category(category, include_excluded=include_excluded)
        if result.status != "category":
            return ()
        return tuple(entry for entry in result.category_members if isinstance(entry, HspPolymerEntry))
    return tuple(entry for entry in POLYMER_ENTRIES if entry.default_include or include_excluded)


def list_hsp_solvent_entries(
    *,
    category: str | None = None,
    polarity_class: str | None = None,
    include_excluded: bool = False,
) -> tuple[HspSolventEntry, ...]:
    entries: list[HspSolventEntry] = []
    if category:
        result = resolve_hsp_solvent_category(category, include_excluded=include_excluded)
        if result.status == "category":
            entries.extend(entry for entry in result.category_members if isinstance(entry, HspSolventEntry))
    if polarity_class:
        result = resolve_hsp_solvent_polarity(polarity_class, include_excluded=include_excluded)
        if result.status == "category":
            entries.extend(entry for entry in result.category_members if isinstance(entry, HspSolventEntry))
    if not category and not polarity_class:
        entries.extend(entry for entry in SOLVENT_ENTRIES if entry.default_include or include_excluded)

    deduped: dict[str, HspSolventEntry] = {}
    for entry in entries:
        deduped.setdefault(entry.id, entry)
    return tuple(deduped.values())
