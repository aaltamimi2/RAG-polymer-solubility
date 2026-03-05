"""
Temperature Optimization

Provides utilities for finding optimal temperature ranges for separation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Any
import numpy as np


@dataclass
class TemperatureWindow:
    """Valid temperature window for a separation."""
    polymer: str
    solvent: str
    temp_min: float
    temp_max: float
    optimal_temp: float
    selectivity_at_optimal: float
    notes: str = ""

    @property
    def window_size(self) -> float:
        return self.temp_max - self.temp_min


@dataclass
class OptimizationResult:
    """Result from optimization."""
    optimal_temperature: float
    temperature_windows: List[TemperatureWindow]
    overall_selectivity: float
    energy_score: float  # Lower is better
    feasibility_score: float  # 0-1, higher is better
    recommendations: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Optimal Temperature: {self.optimal_temperature}°C\n"
            f"Overall Selectivity: {self.overall_selectivity:.1f}%\n"
            f"Energy Score: {self.energy_score:.2f}\n"
            f"Feasibility: {self.feasibility_score:.1%}"
        )


class TemperatureOptimizer:
    """
    Optimize temperature for polymer separation sequences.

    Finds temperature windows where selectivity is maximized
    while considering energy costs and process constraints.
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

    async def find_optimal_temperature(
        self,
        target_polymer: str,
        other_polymers: List[str],
        solvent: str,
        temp_range: Tuple[float, float] = (25.0, 180.0),
        step_size: float = 5.0,
    ) -> OptimizationResult:
        """
        Find optimal temperature for separating target from others with given solvent.

        Scans temperature range and finds point with maximum selectivity.
        """
        from strap.solubility import get_solubility

        temp_min, temp_max = temp_range
        temperatures = np.arange(temp_min, temp_max + step_size, step_size)
        # Clamp to interpolation model range (25–160 °C)
        temperatures = [float(t) for t in temperatures if 25 <= t <= 160]

        if not temperatures:
            return OptimizationResult(
                optimal_temperature=100.0,
                temperature_windows=[],
                overall_selectivity=0.0,
                energy_score=1.0,
                feasibility_score=0.0,
                recommendations=["No solubility data found for this combination"]
            )

        # Find selectivity at each temperature
        best_temp = temp_min
        best_selectivity = -float('inf')
        windows = []

        for temp in temperatures:
            target_sol = get_solubility(target_polymer, solvent, temp)
            if target_sol is None:
                continue

            max_other = 0.0
            for p in other_polymers:
                sol = get_solubility(p, solvent, temp)
                if sol is not None and sol > max_other:
                    max_other = sol

            selectivity = target_sol - max_other

            if selectivity > best_selectivity:
                best_selectivity = selectivity
                best_temp = temp

            # Record if this is a viable window
            if selectivity >= 5.0:
                windows.append(TemperatureWindow(
                    polymer=target_polymer,
                    solvent=solvent,
                    temp_min=temp - step_size/2,
                    temp_max=temp + step_size/2,
                    optimal_temp=temp,
                    selectivity_at_optimal=selectivity,
                ))

        # Merge adjacent windows
        merged_windows = self._merge_windows(windows)

        # Calculate scores
        energy_score = best_temp / 100.0  # Higher temp = more energy
        feasibility_score = min(1.0, best_selectivity / 50.0) if best_selectivity > 0 else 0.0

        # Generate recommendations
        recommendations = []
        if best_selectivity < 5.0:
            recommendations.append("Warning: Low selectivity - consider alternative solvent")
        if best_temp > 150:
            recommendations.append("High temperature required - check polymer degradation limits")
        if len(merged_windows) > 1:
            recommendations.append(f"Multiple viable temperature windows found: {len(merged_windows)}")

        return OptimizationResult(
            optimal_temperature=best_temp,
            temperature_windows=merged_windows,
            overall_selectivity=best_selectivity,
            energy_score=energy_score,
            feasibility_score=feasibility_score,
            recommendations=recommendations,
        )

    def _merge_windows(self, windows: List[TemperatureWindow]) -> List[TemperatureWindow]:
        """Merge adjacent temperature windows."""
        if not windows:
            return []

        sorted_windows = sorted(windows, key=lambda w: w.temp_min)
        merged = [sorted_windows[0]]

        for window in sorted_windows[1:]:
            last = merged[-1]
            if window.temp_min <= last.temp_max + 5.0:  # Adjacent or overlapping
                # Extend the window
                merged[-1] = TemperatureWindow(
                    polymer=last.polymer,
                    solvent=last.solvent,
                    temp_min=last.temp_min,
                    temp_max=max(last.temp_max, window.temp_max),
                    optimal_temp=(last.optimal_temp if last.selectivity_at_optimal >= window.selectivity_at_optimal
                                  else window.optimal_temp),
                    selectivity_at_optimal=max(last.selectivity_at_optimal, window.selectivity_at_optimal),
                )
            else:
                merged.append(window)

        return merged
