"""Data models for STRAP polymer separation analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class CompatibilityLevel(Enum):
    """Compatibility classification for polymer-solvent pairs."""
    EXCELLENT = "excellent"  # >80% solubility
    GOOD = "good"           # 50-80% solubility
    MODERATE = "moderate"   # 20-50% solubility
    POOR = "poor"           # 5-20% solubility
    INSOLUBLE = "insoluble" # <5% solubility
    UNKNOWN = "unknown"     # No data

    @classmethod
    def from_solubility(cls, solubility: float) -> "CompatibilityLevel":
        if solubility >= 80:
            return cls.EXCELLENT
        elif solubility >= 50:
            return cls.GOOD
        elif solubility >= 20:
            return cls.MODERATE
        elif solubility >= 5:
            return cls.POOR
        else:
            return cls.INSOLUBLE


@dataclass
class SelectivityMetrics:
    """Comprehensive selectivity metrics."""
    target_polymer: str
    other_polymers: List[str]
    solvent: str
    temperature: float
    selectivity: float
    target_solubility: float
    max_other_solubility: float
    avg_other_solubility: float
    selectivity_ratio: float
    is_viable: bool
    confidence: float  # Based on data quality

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target_polymer,
            "others": self.other_polymers,
            "solvent": self.solvent,
            "temperature": self.temperature,
            "selectivity": self.selectivity,
            "target_solubility": self.target_solubility,
            "max_other_solubility": self.max_other_solubility,
            "selectivity_ratio": self.selectivity_ratio,
            "is_viable": self.is_viable,
            "confidence": self.confidence,
        }


@dataclass
class SolventScore:
    """Ranking score for a solvent based on selectivity and physical properties."""
    solvent: str
    overall_score: float
    selectivity_score: float
    bp_score: float       # Boiling point (lower BP → higher score → easier recovery)
    logp_score: float     # LogP (normalized)
    cp_score: float       # Heat capacity (lower Cp → higher score → less energy)
    energy_score: float   # Vaporization energy (lower → higher score → easier recovery)
    notes: List[str] = field(default_factory=list)

    def __lt__(self, other: "SolventScore") -> bool:
        return self.overall_score < other.overall_score
