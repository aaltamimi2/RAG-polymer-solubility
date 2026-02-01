"""
Data Analysis Utilities for Polymer Separation

Provides utilities for:
- Selectivity calculations
- Solvent ranking and comparison
- Polymer compatibility analysis
- Data validation and quality checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from enum import Enum
import numpy as np


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
    """Ranking score for a solvent."""
    solvent: str
    overall_score: float
    selectivity_score: float
    availability_score: float
    safety_score: float
    cost_score: float
    environmental_score: float
    notes: List[str] = field(default_factory=list)

    def __lt__(self, other: "SolventScore") -> bool:
        return self.overall_score < other.overall_score


class SelectivityCalculator:
    """
    Calculate selectivity metrics for polymer separations.

    Selectivity = target_solubility - max(other_solubilities)
    Higher selectivity means easier separation.
    """

    def __init__(
        self,
        db_connection: Any,
        table_name: str = "common_solvents_database",
        polymer_column: str = "polymer",
        solvent_column: str = "solvent",
        solubility_column: str = "solubility____",
        temperature_column: str = "temperature___c_",
    ):
        self.conn = db_connection
        self.table_name = table_name
        self.polymer_col = polymer_column
        self.solvent_col = solvent_column
        self.solubility_col = solubility_column
        self.temp_col = temperature_column

    def calculate(
        self,
        target: str,
        others: List[str],
        solvent: str,
        temperature: float,
        temp_tolerance: float = 10.0,
    ) -> SelectivityMetrics:
        """
        Calculate selectivity metrics for separating target from others.

        Args:
            target: Target polymer to dissolve
            others: Other polymers that should remain
            solvent: Solvent to use
            temperature: Target temperature
            temp_tolerance: Temperature window for data matching

        Returns:
            SelectivityMetrics with detailed analysis
        """
        all_polymers = [target] + others
        polymer_filter = "', '".join(all_polymers)

        query = f"""
        SELECT {self.polymer_col}, AVG({self.solubility_col}) as avg_sol,
               COUNT(*) as n_points
        FROM {self.table_name}
        WHERE {self.solvent_col} = '{solvent}'
        AND {self.polymer_col} IN ('{polymer_filter}')
        AND {self.temp_col} BETWEEN {temperature - temp_tolerance} AND {temperature + temp_tolerance}
        GROUP BY {self.polymer_col}
        """

        try:
            result = self.conn.execute(query).fetchall()
        except Exception as e:
            return SelectivityMetrics(
                target_polymer=target,
                other_polymers=others,
                solvent=solvent,
                temperature=temperature,
                selectivity=-999.0,
                target_solubility=0.0,
                max_other_solubility=0.0,
                avg_other_solubility=0.0,
                selectivity_ratio=0.0,
                is_viable=False,
                confidence=0.0,
            )

        # Parse results
        solubilities = {row[0].upper(): (row[1], row[2]) for row in result}

        target_data = solubilities.get(target.upper())
        if target_data is None:
            target_sol = 0.0
            target_n = 0
        else:
            target_sol, target_n = target_data

        other_sols = []
        other_ns = []
        for p in others:
            data = solubilities.get(p.upper())
            if data:
                other_sols.append(data[0])
                other_ns.append(data[1])

        max_other = max(other_sols) if other_sols else 0.0
        avg_other = np.mean(other_sols) if other_sols else 0.0

        selectivity = target_sol - max_other
        ratio = target_sol / max_other if max_other > 0 else float('inf')

        # Calculate confidence based on data points
        total_n = target_n + sum(other_ns)
        confidence = min(1.0, total_n / (len(all_polymers) * 3))  # Expect ~3 points each

        return SelectivityMetrics(
            target_polymer=target,
            other_polymers=others,
            solvent=solvent,
            temperature=temperature,
            selectivity=selectivity,
            target_solubility=target_sol,
            max_other_solubility=max_other,
            avg_other_solubility=avg_other,
            selectivity_ratio=ratio,
            is_viable=selectivity >= 5.0,
            confidence=confidence,
        )

    def calculate_all_solvents(
        self,
        target: str,
        others: List[str],
        temperature: float,
        min_selectivity: float = 0.0,
    ) -> List[SelectivityMetrics]:
        """
        Calculate selectivity for all available solvents.

        Returns list sorted by selectivity (descending).
        """
        # Get all solvents with data for target
        query = f"""
        SELECT DISTINCT {self.solvent_col}
        FROM {self.table_name}
        WHERE UPPER({self.polymer_col}) = UPPER('{target}')
        """

        try:
            solvents = [row[0] for row in self.conn.execute(query).fetchall()]
        except Exception:
            return []

        results = []
        for solvent in solvents:
            metrics = self.calculate(target, others, solvent, temperature)
            if metrics.selectivity >= min_selectivity:
                results.append(metrics)

        # Sort by selectivity
        results.sort(key=lambda m: m.selectivity, reverse=True)
        return results


class SolventRanker:
    """
    Rank solvents based on multiple criteria.

    Considers:
    - Selectivity for separation
    - Availability/cost (from solvent database)
    - Safety profile
    - Environmental impact
    """

    # Default scores (0-1, higher is better)
    # These would ideally come from a solvent properties database
    DEFAULT_PROPERTIES = {
        # Solvent: (availability, safety, cost, environmental)
        "water": (1.0, 1.0, 1.0, 1.0),
        "ethanol": (0.9, 0.9, 0.8, 0.8),
        "methanol": (0.8, 0.6, 0.8, 0.6),
        "acetone": (0.9, 0.7, 0.8, 0.7),
        "toluene": (0.8, 0.4, 0.7, 0.4),
        "xylene": (0.7, 0.4, 0.6, 0.3),
        "dmf": (0.7, 0.3, 0.5, 0.3),
        "dmso": (0.8, 0.6, 0.6, 0.5),
        "thf": (0.7, 0.4, 0.5, 0.4),
        "dichloromethane": (0.8, 0.3, 0.6, 0.2),
        "chloroform": (0.7, 0.2, 0.5, 0.2),
        "hexane": (0.8, 0.5, 0.7, 0.4),
        "heptane": (0.7, 0.5, 0.6, 0.4),
        "cyclohexane": (0.7, 0.5, 0.6, 0.5),
        "diethyl ether": (0.6, 0.3, 0.5, 0.4),
        "ethyl acetate": (0.8, 0.7, 0.7, 0.6),
    }

    def __init__(
        self,
        selectivity_calculator: SelectivityCalculator,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.selectivity_calc = selectivity_calculator
        self.weights = weights or {
            "selectivity": 0.4,
            "safety": 0.25,
            "environmental": 0.2,
            "cost": 0.1,
            "availability": 0.05,
        }

    def rank_solvents(
        self,
        target: str,
        others: List[str],
        temperature: float,
        top_k: int = 10,
    ) -> List[SolventScore]:
        """
        Rank solvents for a separation task.

        Returns top_k solvents with detailed scores.
        """
        # Get selectivity for all solvents
        metrics = self.selectivity_calc.calculate_all_solvents(
            target, others, temperature, min_selectivity=-50
        )

        # Calculate scores
        scores = []
        max_selectivity = max(m.selectivity for m in metrics) if metrics else 1.0

        for m in metrics:
            solvent_lower = m.solvent.lower()
            props = self.DEFAULT_PROPERTIES.get(solvent_lower, (0.5, 0.5, 0.5, 0.5))

            # Normalize selectivity to 0-1
            sel_score = max(0, (m.selectivity + 50) / (max_selectivity + 50))

            overall = (
                self.weights["selectivity"] * sel_score +
                self.weights["availability"] * props[0] +
                self.weights["safety"] * props[1] +
                self.weights["cost"] * props[2] +
                self.weights["environmental"] * props[3]
            )

            notes = []
            if m.selectivity < 5:
                notes.append("Low selectivity - may require multiple passes")
            if props[1] < 0.5:
                notes.append("Safety concerns - check handling requirements")
            if props[3] < 0.4:
                notes.append("Environmental impact - consider alternatives")

            scores.append(SolventScore(
                solvent=m.solvent,
                overall_score=overall,
                selectivity_score=sel_score,
                availability_score=props[0],
                safety_score=props[1],
                cost_score=props[2],
                environmental_score=props[3],
                notes=notes,
            ))

        # Sort by overall score
        scores.sort(reverse=True)
        return scores[:top_k]

    def compare_solvents(
        self,
        solvents: List[str],
        target: str,
        others: List[str],
        temperature: float,
    ) -> Dict[str, SolventScore]:
        """Compare specific solvents head-to-head."""
        result = {}
        for solvent in solvents:
            metrics = self.selectivity_calc.calculate(
                target, others, solvent, temperature
            )
            props = self.DEFAULT_PROPERTIES.get(solvent.lower(), (0.5, 0.5, 0.5, 0.5))

            sel_score = max(0, min(1, (metrics.selectivity + 50) / 100))

            overall = (
                self.weights["selectivity"] * sel_score +
                self.weights["availability"] * props[0] +
                self.weights["safety"] * props[1] +
                self.weights["cost"] * props[2] +
                self.weights["environmental"] * props[3]
            )

            result[solvent] = SolventScore(
                solvent=solvent,
                overall_score=overall,
                selectivity_score=sel_score,
                availability_score=props[0],
                safety_score=props[1],
                cost_score=props[2],
                environmental_score=props[3],
            )

        return result


class PolymerCompatibilityMatrix:
    """
    Build and analyze polymer-solvent compatibility matrices.

    Useful for understanding separation feasibility and
    identifying challenging polymer combinations.
    """

    def __init__(
        self,
        db_connection: Any,
        table_name: str = "common_solvents_database",
        polymer_column: str = "polymer",
        solvent_column: str = "solvent",
        solubility_column: str = "solubility____",
        temperature_column: str = "temperature___c_",
    ):
        self.conn = db_connection
        self.table_name = table_name
        self.polymer_col = polymer_column
        self.solvent_col = solvent_column
        self.solubility_col = solubility_column
        self.temp_col = temperature_column

    def build_matrix(
        self,
        polymers: List[str],
        solvents: Optional[List[str]] = None,
        temperature: float = 100.0,
        temp_tolerance: float = 20.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Build polymer-solvent compatibility matrix.

        Returns:
            Dict of {polymer: {solvent: solubility}}
        """
        polymer_filter = "', '".join(polymers)

        if solvents:
            solvent_filter = f"AND {self.solvent_col} IN ('" + "', '".join(solvents) + "')"
        else:
            solvent_filter = ""

        query = f"""
        SELECT {self.polymer_col}, {self.solvent_col}, AVG({self.solubility_col}) as avg_sol
        FROM {self.table_name}
        WHERE {self.polymer_col} IN ('{polymer_filter}')
        AND {self.temp_col} BETWEEN {temperature - temp_tolerance} AND {temperature + temp_tolerance}
        {solvent_filter}
        GROUP BY {self.polymer_col}, {self.solvent_col}
        """

        try:
            result = self.conn.execute(query).fetchall()
        except Exception as e:
            return {}

        # Build matrix
        matrix: Dict[str, Dict[str, float]] = {p: {} for p in polymers}
        for row in result:
            polymer, solvent, sol = row
            if polymer in matrix:
                matrix[polymer][solvent] = sol

        return matrix

    def find_challenging_pairs(
        self,
        polymers: List[str],
        temperature: float = 100.0,
        threshold: float = 10.0,
    ) -> List[Tuple[str, str, float]]:
        """
        Find polymer pairs that are difficult to separate.

        Returns list of (polymer1, polymer2, best_selectivity) for
        pairs where best selectivity is below threshold.
        """
        matrix = self.build_matrix(polymers, temperature=temperature)

        if not matrix:
            return []

        # Get all solvents
        all_solvents: Set[str] = set()
        for sols in matrix.values():
            all_solvents.update(sols.keys())

        challenging = []

        for i, p1 in enumerate(polymers):
            for p2 in polymers[i+1:]:
                best_sel = -float('inf')

                for solvent in all_solvents:
                    sol1 = matrix.get(p1, {}).get(solvent, 0)
                    sol2 = matrix.get(p2, {}).get(solvent, 0)

                    # Selectivity for separating p1 from p2
                    sel = abs(sol1 - sol2)
                    best_sel = max(best_sel, sel)

                if best_sel < threshold:
                    challenging.append((p1, p2, best_sel))

        # Sort by selectivity (most challenging first)
        challenging.sort(key=lambda x: x[2])
        return challenging

    def find_universal_solvents(
        self,
        polymers: List[str],
        temperature: float = 100.0,
        min_solubility: float = 50.0,
    ) -> List[Tuple[str, int, float]]:
        """
        Find solvents that dissolve multiple polymers.

        Returns list of (solvent, n_polymers_dissolved, avg_solubility)
        """
        matrix = self.build_matrix(polymers, temperature=temperature)

        # Count how many polymers each solvent dissolves well
        solvent_stats: Dict[str, List[float]] = {}

        for polymer, solvents in matrix.items():
            for solvent, sol in solvents.items():
                if sol >= min_solubility:
                    if solvent not in solvent_stats:
                        solvent_stats[solvent] = []
                    solvent_stats[solvent].append(sol)

        result = [
            (solvent, len(sols), np.mean(sols))
            for solvent, sols in solvent_stats.items()
        ]

        # Sort by number of polymers dissolved
        result.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return result

    def get_compatibility_level(
        self,
        polymer: str,
        solvent: str,
        temperature: float = 100.0,
    ) -> CompatibilityLevel:
        """Get compatibility classification for a polymer-solvent pair."""
        query = f"""
        SELECT AVG({self.solubility_col}) as avg_sol
        FROM {self.table_name}
        WHERE UPPER({self.polymer_col}) = UPPER('{polymer}')
        AND UPPER({self.solvent_col}) = UPPER('{solvent}')
        AND {self.temp_col} BETWEEN {temperature - 20} AND {temperature + 20}
        """

        try:
            result = self.conn.execute(query).fetchone()
            if result and result[0] is not None:
                return CompatibilityLevel.from_solubility(result[0])
        except Exception:
            pass

        return CompatibilityLevel.UNKNOWN


# Convenience functions
def calculate_selectivity(
    target: str,
    others: List[str],
    solvent: str,
    temperature: float,
    db_connection: Any,
) -> SelectivityMetrics:
    """Quick selectivity calculation."""
    calc = SelectivityCalculator(db_connection)
    return calc.calculate(target, others, solvent, temperature)


def rank_solvents_for_separation(
    target: str,
    others: List[str],
    temperature: float,
    db_connection: Any,
    top_k: int = 10,
) -> List[SolventScore]:
    """Quick solvent ranking for a separation task."""
    calc = SelectivityCalculator(db_connection)
    ranker = SolventRanker(calc)
    return ranker.rank_solvents(target, others, temperature, top_k)
