"""Analysis utilities for polymer separation.

Provides:
- Selectivity calculations
- Solvent ranking and comparison
- Polymer compatibility analysis
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any, Set

import numpy as np

from .models import CompatibilityLevel, SelectivityMetrics, SolventScore


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
        """Calculate selectivity metrics for separating target from others.

        Uses the unified solubility API (interpolation model with SQL fallback).
        """
        from strap.solubility import get_solubility

        all_polymers = [target] + others

        target_sol = get_solubility(target, solvent, temperature)
        if target_sol is None:
            target_sol = 0.0
            target_n = 0
        else:
            target_n = 3  # interpolation model provides high confidence

        other_sols = []
        other_ns = []
        for p in others:
            sol = get_solubility(p, solvent, temperature)
            if sol is not None:
                other_sols.append(sol)
                other_ns.append(3)

        max_other = max(other_sols) if other_sols else 0.0
        avg_other = float(np.mean(other_sols)) if other_sols else 0.0

        selectivity = target_sol - max_other
        ratio = target_sol / max_other if max_other > 0 else float("inf")

        total_n = target_n + sum(other_ns)
        confidence = min(1.0, total_n / (len(all_polymers) * 3))

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
        """Calculate selectivity for all available solvents.

        Returns list sorted by selectivity (descending).
        """
        from strap.solubility import get_available_solvents_for_polymer

        solvents = get_available_solvents_for_polymer(target)

        results = []
        for solvent in solvents:
            metrics = self.calculate(target, others, solvent, temperature)
            if metrics.selectivity >= min_selectivity:
                results.append(metrics)

        results.sort(key=lambda m: m.selectivity, reverse=True)
        return results


class SolventRanker:
    """Rank solvents based on selectivity and physical properties from the database.

    Physical properties (from solvent_data table):
    - Boiling point (bp): lower → easier recovery → higher score
    - LogP: normalized as-is
    - Heat capacity (Cp): lower → less energy to heat → higher score
    - Vaporization energy: lower → easier recovery → higher score
    """

    def __init__(
        self,
        selectivity_calculator: SelectivityCalculator,
        db_connection: Any = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.selectivity_calc = selectivity_calculator
        self.conn = db_connection or selectivity_calculator.conn
        self.weights = weights or {
            "selectivity": 0.50,
            "bp": 0.15,
            "logp": 0.10,
            "cp": 0.10,
            "energy": 0.15,
        }
        self._property_cache: Optional[Dict[str, Dict[str, float]]] = None

    def _load_properties(self) -> Dict[str, Dict[str, float]]:
        """Load and normalize physical properties from solvent_data table."""
        if self._property_cache is not None:
            return self._property_cache

        try:
            rows = self.conn.execute(
                "SELECT solvent_name, bp__oc_, logp, cp__j_g_k_, energy__j_g_ "
                "FROM solvent_data WHERE bp__oc_ IS NOT NULL"
            ).fetchall()
        except Exception:
            self._property_cache = {}
            return self._property_cache

        if not rows:
            self._property_cache = {}
            return self._property_cache

        # Collect raw values
        raw: Dict[str, Tuple[float, float, float, float]] = {}
        bps, logps, cps, energies = [], [], [], []
        for name, bp, logp, cp, energy in rows:
            if bp is None:
                continue
            logp = logp if logp is not None else 0.0
            cp = cp if cp is not None else 0.0
            energy = energy if energy is not None else 0.0
            raw[name.lower()] = (bp, logp, cp, energy)
            bps.append(bp)
            logps.append(logp)
            cps.append(cp)
            energies.append(energy)

        def _minmax(vals: list) -> Tuple[float, float]:
            lo, hi = min(vals), max(vals)
            return (lo, hi) if hi > lo else (lo, lo + 1.0)

        bp_range = _minmax(bps)
        logp_range = _minmax(logps)
        cp_range = _minmax(cps)
        energy_range = _minmax(energies)

        # Normalize: bp/cp/energy inverted (lower = better), logp inverted (lower = better for green chemistry)
        self._property_cache = {}
        for name, (bp, logp, cp, energy) in raw.items():
            self._property_cache[name] = {
                "bp": 1.0 - (bp - bp_range[0]) / (bp_range[1] - bp_range[0]),
                "logp": 1.0 - (logp - logp_range[0]) / (logp_range[1] - logp_range[0]),
                "cp": 1.0 - (cp - cp_range[0]) / (cp_range[1] - cp_range[0]),
                "energy": 1.0 - (energy - energy_range[0]) / (energy_range[1] - energy_range[0]),
            }

        return self._property_cache

    def _get_props(self, solvent: str) -> Dict[str, float]:
        """Get normalized properties for a solvent, defaulting to 0.5 if unknown."""
        props = self._load_properties()
        return props.get(solvent.lower(), {"bp": 0.5, "logp": 0.5, "cp": 0.5, "energy": 0.5})

    def rank_solvents(
        self,
        target: str,
        others: List[str],
        temperature: float,
        top_k: int = 10,
    ) -> List[SolventScore]:
        """Rank solvents for a separation task.

        Returns top_k solvents with detailed scores.
        """
        metrics = self.selectivity_calc.calculate_all_solvents(
            target, others, temperature, min_selectivity=-50
        )

        if not metrics:
            return []
        scores = []
        max_selectivity = max(m.selectivity for m in metrics)

        for m in metrics:
            props = self._get_props(m.solvent)
            sel_score = max(0, (m.selectivity + 50) / (max_selectivity + 50))

            overall = (
                self.weights["selectivity"] * sel_score
                + self.weights["bp"] * props["bp"]
                + self.weights["logp"] * props["logp"]
                + self.weights["cp"] * props["cp"]
                + self.weights["energy"] * props["energy"]
            )

            notes = []
            if m.selectivity < 5:
                notes.append("Low selectivity - may require multiple passes")
            if props["bp"] > 0.8:
                notes.append("Very low boiling point - may evaporate too easily")
            if props["energy"] < 0.2:
                notes.append("High vaporization energy - recovery may be costly")

            scores.append(
                SolventScore(
                    solvent=m.solvent,
                    overall_score=overall,
                    selectivity_score=sel_score,
                    bp_score=props["bp"],
                    logp_score=props["logp"],
                    cp_score=props["cp"],
                    energy_score=props["energy"],
                    notes=notes,
                )
            )

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
            props = self._get_props(solvent)
            sel_score = max(0, min(1, (metrics.selectivity + 50) / 100))

            overall = (
                self.weights["selectivity"] * sel_score
                + self.weights["bp"] * props["bp"]
                + self.weights["logp"] * props["logp"]
                + self.weights["cp"] * props["cp"]
                + self.weights["energy"] * props["energy"]
            )

            result[solvent] = SolventScore(
                solvent=solvent,
                overall_score=overall,
                selectivity_score=sel_score,
                bp_score=props["bp"],
                logp_score=props["logp"],
                cp_score=props["cp"],
                energy_score=props["energy"],
            )

        return result


class PolymerCompatibilityMatrix:
    """Build and analyze polymer-solvent compatibility matrices."""

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
        """Build polymer-solvent compatibility matrix.

        Uses the unified solubility API (interpolation model with SQL fallback).

        Returns:
            Dict of {polymer: {solvent: solubility}}
        """
        from strap.solubility import get_solubility, get_available_solvents

        if solvents is None:
            solvents = sorted(get_available_solvents())

        matrix: Dict[str, Dict[str, float]] = {p: {} for p in polymers}
        for polymer in polymers:
            for solvent in solvents:
                sol = get_solubility(polymer, solvent, temperature)
                if sol is not None:
                    matrix[polymer][solvent] = sol

        return matrix

    def find_challenging_pairs(
        self,
        polymers: List[str],
        temperature: float = 100.0,
        threshold: float = 10.0,
    ) -> List[Tuple[str, str, float]]:
        """Find polymer pairs that are difficult to separate."""
        matrix = self.build_matrix(polymers, temperature=temperature)

        if not matrix:
            return []

        all_solvents: Set[str] = set()
        for sols in matrix.values():
            all_solvents.update(sols.keys())

        challenging = []

        for i, p1 in enumerate(polymers):
            for p2 in polymers[i + 1 :]:
                best_sel = -float("inf")

                for solvent in all_solvents:
                    sol1 = matrix.get(p1, {}).get(solvent, 0)
                    sol2 = matrix.get(p2, {}).get(solvent, 0)
                    sel = abs(sol1 - sol2)
                    best_sel = max(best_sel, sel)

                if best_sel < threshold:
                    challenging.append((p1, p2, best_sel))

        challenging.sort(key=lambda x: x[2])
        return challenging

    def find_universal_solvents(
        self,
        polymers: List[str],
        temperature: float = 100.0,
        min_solubility: float = 50.0,
    ) -> List[Tuple[str, int, float]]:
        """Find solvents that dissolve multiple polymers."""
        matrix = self.build_matrix(polymers, temperature=temperature)

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

        result.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return result

    def get_compatibility_level(
        self,
        polymer: str,
        solvent: str,
        temperature: float = 100.0,
    ) -> CompatibilityLevel:
        """Get compatibility classification for a polymer-solvent pair."""
        from strap.solubility import get_solubility
        sol = get_solubility(polymer, solvent, temperature)
        if sol is not None:
            return CompatibilityLevel.from_solubility(sol)
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
