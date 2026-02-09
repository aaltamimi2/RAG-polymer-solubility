"""
Precipitation Analysis Tools for Differential/Selective Polymer Separation

This module provides tools for analyzing temperature-dependent solubility data
to find optimal conditions for differential precipitation of polymer mixtures.

Key Concepts:
- Precipitation Temperature: Temperature where solubility drops to ~0% (<1%)
  This is the point where polymer is fully precipitated and can be filtered.
- Cloud Point: Temperature where polymer BEGINS to precipitate (~10% solubility)
  This is where the solution becomes cloudy/turbid as precipitation starts.
- Differential Precipitation: Different polymers precipitate at different temperatures
- Temperature Window: Gap between precipitation temperatures allowing selective separation

Thresholds:
- Precipitation: <1% solubility (polymer is fully out of solution)
- Cloud Point: ~10% solubility (onset of precipitation)
- Dissolved: >50% solubility (polymer is in solution)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_PRECIPITATION_THRESHOLD = 1.0   # Solubility below this = precipitated (approaching 0%)
DEFAULT_CLOUD_POINT_THRESHOLD = 10.0    # Solubility where precipitation begins (cloud point)
DEFAULT_DISSOLUTION_THRESHOLD = 50.0    # Solubility above this = dissolved
DEFAULT_MIN_TEMP_GAP = 20.0             # Minimum useful temperature gap

# Solvent boiling points at 1 atm (°C) - comprehensive database
SOLVENT_BOILING_POINTS = {
    # Common alcohols
    'methanol': 65, 'ethanol': 78, 'propanol': 97, '2-propanol': 82, 'isopropanol': 82,
    'butanol': 117, '1-butanol': 117, '2-butanol': 100, 'tert-butanol': 82,
    'pentanol': 138, 'hexanol': 157, 'cyclohexanol': 161,
    # Glycols
    'glycol': 197, 'ethylene glycol': 197, 'propyleneglycol': 188, 'propylene glycol': 188,
    # Ketones
    'propanone': 56, 'acetone': 56, 'butanone': 80, 'mek': 80, '2-butanone': 80,
    'acetylacetone': 140, 'cyclohexanone': 156,
    # Ethers
    'thf': 66, 'tetrahydrofuran': 66, 'thp': 88, 'tetrahydropyran': 88,
    '2,3-dihydropyran': 86, 'diethylether': 35, 'diphenylether': 259, 'diphenyl ether': 259,
    'dioxane': 101, '1,4-dioxane': 101,
    # Esters
    'methylacetate': 57, 'methyl acetate': 57, 'ethylacetate': 77, 'ethyl acetate': 77,
    'butylacetate': 126, 'propylacetate': 102,
    # Aromatic solvents
    'benzene': 80, 'toluene': 111, '1,2-dimethylbenzene': 144, 'o-xylene': 144,
    '1,4-dimethylbenzene': 138, 'p-xylene': 138, '1,3-dimethylbenzene': 139, 'm-xylene': 139,
    'ethylbenzene': 136, 'chlorobenzene': 132, 'nitrobenzene': 211,
    # Chlorinated solvents
    'chcl3': 61, 'chloroform': 61, 'ch2cl2': 40, 'dcm': 40, 'dichloromethane': 40,
    'carbon tetrachloride': 77, 'ccl4': 77,
    '1,2-dichloroethane': 83, 'tetrachloroethylene': 121,
    # Amides
    'dimethylformamide': 153, 'dmf': 153, 'dimethylacetamide': 165, 'dmac': 165,
    'n-methylpyrrolidone': 202, 'nmp': 202, 'formamide': 210,
    # Amines
    'triethylamine': 89, 'isopropylamine': 32, 'pyridine': 115, 'aniline': 184,
    # Sulfoxides
    'dimethylsulfoxide': 189, 'dmso': 189,
    # Alkanes
    'hexane': 69, 'n-hexane': 69, 'cyclohexane': 81, 'n-heptane': 98, 'heptane': 98,
    'octane': 126, 'decane': 174, 'dodecane': 216, 'petroleum ether': 60,
    # Water
    'h2o': 100, 'water': 100,
    # Other
    'carbon disulfide': 46, 'cs2': 46, 'acetonitrile': 82, 'nitromethane': 101,
    'morpholine': 128, 'furfural': 162, 'gamma-butyrolactone': 204, 'gbl': 204,
}


@dataclass
class PrecipitationPoint:
    """Precipitation characteristics for a polymer-solvent pair."""
    polymer: str
    solvent: str
    max_solubility: float           # Maximum solubility achieved (at high temp)
    max_solubility_temp: float      # Temperature of max solubility
    precipitation_temp: float       # Temp where solubility < threshold
    cloud_point: float              # Temp where solubility crosses 50%
    dissolution_temp: float         # Temp where solubility first exceeds threshold
    transition_width: float         # Temperature range of transition
    data_points: int                # Number of data points available


@dataclass
class DifferentialPrecipitationResult:
    """Result of differential precipitation analysis for two polymers."""
    solvent: str
    polymer_first: str              # Polymer that precipitates first (higher temp)
    polymer_first_precip_temp: float
    polymer_first_max_sol: float
    polymer_second: str             # Polymer that precipitates second (lower temp)
    polymer_second_precip_temp: float
    polymer_second_max_sol: float
    temperature_gap: float          # Separation window
    operating_window: Tuple[float, float]  # Recommended temp range for separation
    selectivity_score: float        # Quality metric (0-1)
    notes: List[str] = field(default_factory=list)


@dataclass
class MultiPolymerPrecipitationSequence:
    """Precipitation sequence for multiple polymers in a solvent."""
    solvent: str
    sequence: List[Tuple[str, float]]  # [(polymer, precip_temp), ...] ordered high to low
    max_solubilities: Dict[str, float]
    recommended_steps: List[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)


@dataclass
class AtmosphericFeasibilityResult:
    """Result of atmospheric pressure feasibility analysis for differential precipitation."""
    solvent: str
    boiling_point: float                # Solvent BP at 1 atm
    polymer_first: str                  # Polymer that precipitates first
    polymer_first_precip_temp: float
    polymer_first_max_sol: float
    polymer_second: str                 # Polymer that precipitates second
    polymer_second_precip_temp: float
    polymer_second_max_sol: float
    temperature_gap: float              # Separation window
    dissolution_temp_needed: float      # Min temp needed to dissolve both
    is_feasible: bool                   # True if process works below BP
    feasibility_margin: float           # How much below BP we can operate (negative = need pressure)
    notes: List[str] = field(default_factory=list)


@dataclass
class MultiPolymerAtmosphericResult:
    """Result of atmospheric feasibility analysis for 2+ polymer systems."""
    solvent: str
    boiling_point: float                          # Solvent BP at 1 atm
    polymers: List[str]                           # All polymers in order of precipitation
    precipitation_sequence: List[Tuple[str, float]]  # [(polymer, precip_temp), ...] high to low
    max_solubilities: Dict[str, float]            # {polymer: max_sol}
    temperature_gaps: List[Tuple[str, str, float]]   # [(p1, p2, gap), ...] between consecutive
    min_gap: float                                # Smallest gap in sequence
    dissolution_temp_needed: float                # Temp needed to dissolve all
    is_feasible: bool                             # True if entire process below BP
    feasibility_margin: float                     # BP - dissolution_temp
    recommended_steps: List[Dict[str, Any]]       # Step-by-step cooling protocol
    warnings: List[str] = field(default_factory=list)


class PrecipitationAnalyzer:
    """
    Analyzes temperature-dependent solubility data for differential precipitation.

    This class provides methods to:
    1. Find precipitation temperatures for polymer-solvent pairs
    2. Compare precipitation behavior across polymers
    3. Find optimal solvents for selective precipitation
    4. Design multi-step cooling protocols
    """

    def __init__(self, conn, table_name: str = "common_solvents_database"):
        """
        Initialize with database connection.

        Args:
            conn: DuckDB connection or similar
            table_name: Name of the solubility data table
        """
        self.conn = conn
        self.table_name = table_name
        self._cache = {}  # Cache for repeated queries

    def get_solubility_curve(
        self,
        polymer: str,
        solvent: str
    ) -> pd.DataFrame:
        """
        Get full temperature-solubility curve for a polymer-solvent pair.

        Uses the unified solubility API (interpolation model with SQL fallback).

        Returns:
            DataFrame with columns: temperature, solubility
        """
        cache_key = f"{polymer}_{solvent}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from strap.solubility import get_solubility_curve as _get_curve
        curve = _get_curve(polymer, solvent, t_start_c=25.0, t_end_c=160.0, t_step_c=5.0)
        if curve:
            df = pd.DataFrame(curve)
            self._cache[cache_key] = df
            return df

        logger.warning(f"No data found for {polymer}/{solvent}")
        return pd.DataFrame(columns=['temperature', 'solubility'])

    def find_precipitation_temperature(
        self,
        polymer: str,
        solvent: str,
        threshold: float = DEFAULT_PRECIPITATION_THRESHOLD
    ) -> Optional[float]:
        """
        Find the highest temperature where solubility drops below threshold.

        This is the temperature at which the polymer would precipitate during cooling.

        Args:
            polymer: Polymer name
            solvent: Solvent name
            threshold: Solubility threshold (%) below which polymer is precipitated

        Returns:
            Precipitation temperature in °C, or None if polymer never dissolves
        """
        df = self.get_solubility_curve(polymer, solvent)
        if df.empty:
            return None

        # Find where solubility crosses below threshold (from high temp to low)
        below_threshold = df[df['solubility'] < threshold]
        if below_threshold.empty:
            return None  # Never precipitates in this range

        # Return highest temp where it's below threshold
        return below_threshold['temperature'].max()

    def find_cloud_point(
        self,
        polymer: str,
        solvent: str,
        threshold: float = DEFAULT_DISSOLUTION_THRESHOLD
    ) -> Optional[float]:
        """
        Find cloud point - temperature where solubility crosses 50% (or specified threshold).

        Args:
            polymer: Polymer name
            solvent: Solvent name
            threshold: Solubility threshold for cloud point

        Returns:
            Cloud point temperature in °C
        """
        df = self.get_solubility_curve(polymer, solvent)
        if df.empty:
            return None

        # Find where solubility crosses the threshold
        above = df[df['solubility'] >= threshold]
        below = df[df['solubility'] < threshold]

        if above.empty or below.empty:
            return None

        # Return approximate crossing point
        return below['temperature'].max()

    def analyze_precipitation(
        self,
        polymer: str,
        solvent: str,
        precip_threshold: float = 1.0
    ) -> Optional[PrecipitationPoint]:
        """
        Full precipitation analysis for a polymer-solvent pair.

        Returns detailed information about dissolution and precipitation behavior.
        """
        df = self.get_solubility_curve(polymer, solvent)
        if df.empty:
            return None

        max_sol = df['solubility'].max()
        max_sol_temp = df.loc[df['solubility'].idxmax(), 'temperature']

        precip_temp = self.find_precipitation_temperature(polymer, solvent, precip_threshold)
        cloud_point = self.find_cloud_point(polymer, solvent, 50.0)

        # Find dissolution temperature (lowest temp where above threshold)
        above_threshold = df[df['solubility'] >= precip_threshold]
        dissolution_temp = above_threshold['temperature'].min() if not above_threshold.empty else None

        # Calculate transition width
        if cloud_point and precip_temp:
            transition_width = abs(cloud_point - precip_temp)
        else:
            transition_width = 0.0

        return PrecipitationPoint(
            polymer=polymer,
            solvent=solvent,
            max_solubility=max_sol,
            max_solubility_temp=max_sol_temp,
            precipitation_temp=precip_temp or 0.0,
            cloud_point=cloud_point or 0.0,
            dissolution_temp=dissolution_temp or 0.0,
            transition_width=transition_width,
            data_points=len(df)
        )

    def find_differential_precipitation_solvents(
        self,
        polymer_to_precipitate: str,
        polymer_to_retain: str,
        min_temp_gap: float = DEFAULT_MIN_TEMP_GAP,
        precip_threshold: float = 1.0,
        min_solubility: float = 30.0,
        top_k: int = 10
    ) -> List[DifferentialPrecipitationResult]:
        """
        Find solvents where one polymer precipitates before another.

        This is the key method for differential precipitation analysis.

        Args:
            polymer_to_precipitate: Polymer that should precipitate first (at higher temp)
            polymer_to_retain: Polymer that should stay dissolved
            min_temp_gap: Minimum temperature separation required
            precip_threshold: Solubility threshold for "precipitated"
            min_solubility: Minimum max solubility required for both polymers
            top_k: Number of top results to return

        Returns:
            List of DifferentialPrecipitationResult sorted by temperature gap
        """
        # Use interpolation curves for all available solvents
        from strap.solubility import get_available_solvents as _get_solvents

        results = []
        for solvent in _get_solvents():
            df1 = self.get_solubility_curve(polymer_to_precipitate, solvent)
            df2 = self.get_solubility_curve(polymer_to_retain, solvent)

            if df1.empty or df2.empty:
                continue

            max_sol_1 = float(df1['solubility'].max())
            max_sol_2 = float(df2['solubility'].max())

            if max_sol_1 < min_solubility or max_sol_2 < min_solubility:
                continue

            below_1 = df1[df1['solubility'] < precip_threshold]
            below_2 = df2[df2['solubility'] < precip_threshold]

            if below_1.empty or below_2.empty:
                continue

            precip_temp_1 = float(below_1['temperature'].max())
            precip_temp_2 = float(below_2['temperature'].max())

            if precip_temp_1 <= precip_temp_2:
                continue

            temp_gap = precip_temp_1 - precip_temp_2
            if temp_gap < min_temp_gap:
                continue

            operating_window = (precip_temp_2 + 5, precip_temp_1 - 5)
            selectivity_score = min(1.0, temp_gap / 50.0) * min(1.0, max_sol_1 / 100.0)

            notes = []
            if max_sol_1 < 50:
                notes.append(f"{polymer_to_precipitate} has limited solubility ({max_sol_1:.1f}%)")
            if max_sol_2 < 50:
                notes.append(f"{polymer_to_retain} has limited solubility ({max_sol_2:.1f}%)")

            results.append(DifferentialPrecipitationResult(
                solvent=solvent,
                polymer_first=polymer_to_precipitate,
                polymer_first_precip_temp=precip_temp_1,
                polymer_first_max_sol=max_sol_1,
                polymer_second=polymer_to_retain,
                polymer_second_precip_temp=precip_temp_2,
                polymer_second_max_sol=max_sol_2,
                temperature_gap=temp_gap,
                operating_window=operating_window,
                selectivity_score=selectivity_score,
                notes=notes
            ))

        results.sort(key=lambda r: r.temperature_gap, reverse=True)
        return results[:top_k]

    def analyze_multi_polymer_precipitation(
        self,
        polymers: List[str],
        solvent: str,
        precip_threshold: float = 1.0
    ) -> Optional[MultiPolymerPrecipitationSequence]:
        """
        Analyze precipitation sequence for multiple polymers in a single solvent.

        Args:
            polymers: List of polymer names
            solvent: Solvent to analyze
            precip_threshold: Solubility threshold for precipitation

        Returns:
            MultiPolymerPrecipitationSequence with ordered precipitation sequence
        """
        points = []
        max_sols = {}

        for polymer in polymers:
            p = self.analyze_precipitation(polymer, solvent, precip_threshold)
            if p:
                points.append((polymer, p.precipitation_temp, p.max_solubility))
                max_sols[polymer] = p.max_solubility

        if not points:
            return None

        # Sort by precipitation temperature (highest first = precipitates first)
        points.sort(key=lambda x: x[1], reverse=True)
        sequence = [(p[0], p[1]) for p in points]

        # Generate recommended cooling steps
        steps = []
        warnings = []

        for i, (polymer, temp) in enumerate(sequence):
            step = {
                "step": i + 1,
                "cool_to": temp - 5,  # Go 5C below precipitation temp
                "collect": polymer,
                "precipitation_temp": temp
            }
            steps.append(step)

            # Check for overlapping windows
            if i > 0:
                prev_temp = sequence[i-1][1]
                gap = prev_temp - temp
                if gap < 15:
                    warnings.append(
                        f"Small gap ({gap:.0f}°C) between {sequence[i-1][0]} and {polymer} - "
                        f"may have co-precipitation"
                    )

        return MultiPolymerPrecipitationSequence(
            solvent=solvent,
            sequence=sequence,
            max_solubilities=max_sols,
            recommended_steps=steps,
            warnings=warnings
        )

    def check_atmospheric_feasibility(
        self,
        polymer1: str,
        polymer2: str,
        min_temp_gap: float = DEFAULT_MIN_TEMP_GAP,
        precip_threshold: float = DEFAULT_PRECIPITATION_THRESHOLD,
        min_solubility: float = 30.0,
        top_k: int = 10
    ) -> List[AtmosphericFeasibilityResult]:
        """
        Find solvents where differential precipitation is feasible at atmospheric pressure.

        This checks if the entire process (dissolution → cooling → sequential precipitation)
        can be performed below the solvent's boiling point at 1 atm.

        Args:
            polymer1: First polymer name
            polymer2: Second polymer name
            min_temp_gap: Minimum temperature gap required for separation
            precip_threshold: Solubility threshold for "precipitated"
            min_solubility: Minimum max solubility required
            top_k: Number of results to return

        Returns:
            List of AtmosphericFeasibilityResult sorted by feasibility margin
        """
        # First get all differential precipitation results (both orderings)
        results_1_first = self.find_differential_precipitation_solvents(
            polymer1, polymer2, min_temp_gap, precip_threshold, min_solubility, top_k=50
        )
        results_2_first = self.find_differential_precipitation_solvents(
            polymer2, polymer1, min_temp_gap, precip_threshold, min_solubility, top_k=50
        )

        all_results = results_1_first + results_2_first
        atmospheric_results = []

        for r in all_results:
            solvent_lower = r.solvent.lower()

            # Look up boiling point
            bp = SOLVENT_BOILING_POINTS.get(solvent_lower)
            if bp is None:
                # Try without spaces/hyphens
                solvent_clean = solvent_lower.replace(' ', '').replace('-', '')
                bp = SOLVENT_BOILING_POINTS.get(solvent_clean)

            if bp is None:
                logger.debug(f"No boiling point data for {r.solvent}")
                continue

            # Need to dissolve both polymers - estimate dissolution temp
            # Add margin for good dissolution (higher than precipitation temp)
            dissolution_temp = max(r.polymer_first_precip_temp, r.polymer_second_precip_temp) + 20

            # Check if dissolution is possible below BP
            is_feasible = dissolution_temp < bp
            feasibility_margin = bp - dissolution_temp

            notes = []
            if is_feasible:
                notes.append(f"✅ Can operate at 1 atm (BP={bp}°C > dissolution at ~{dissolution_temp:.0f}°C)")
                notes.append(f"Safety margin: {feasibility_margin:.0f}°C below boiling point")
            else:
                notes.append(f"❌ Requires pressure (dissolution ~{dissolution_temp:.0f}°C > BP={bp}°C)")
                notes.append(f"Would need {-feasibility_margin:.0f}°C above BP - requires autoclave")

            atmospheric_results.append(AtmosphericFeasibilityResult(
                solvent=r.solvent,
                boiling_point=bp,
                polymer_first=r.polymer_first,
                polymer_first_precip_temp=r.polymer_first_precip_temp,
                polymer_first_max_sol=r.polymer_first_max_sol,
                polymer_second=r.polymer_second,
                polymer_second_precip_temp=r.polymer_second_precip_temp,
                polymer_second_max_sol=r.polymer_second_max_sol,
                temperature_gap=r.temperature_gap,
                dissolution_temp_needed=dissolution_temp,
                is_feasible=is_feasible,
                feasibility_margin=feasibility_margin,
                notes=notes
            ))

        # Sort: feasible first (by margin descending), then infeasible (by how close they are)
        # Use not instead of - for boolean to avoid numpy issues
        atmospheric_results.sort(key=lambda x: (not x.is_feasible, -x.feasibility_margin))

        return atmospheric_results[:top_k]

    def check_multi_polymer_atmospheric_feasibility(
        self,
        polymers: List[str],
        min_temp_gap: float = DEFAULT_MIN_TEMP_GAP,
        precip_threshold: float = DEFAULT_PRECIPITATION_THRESHOLD,
        min_solubility: float = 30.0,
        top_k: int = 10
    ) -> List[MultiPolymerAtmosphericResult]:
        """
        Find solvents where multi-polymer differential precipitation is feasible at 1 atm.

        For N polymers, this finds solvents where:
        1. All polymers dissolve below the solvent boiling point
        2. Each polymer precipitates at a different temperature during cooling
        3. Temperature gaps between consecutive precipitations are >= min_temp_gap

        Args:
            polymers: List of polymer names (2 or more)
            min_temp_gap: Minimum temperature gap between consecutive precipitations
            precip_threshold: Solubility threshold for "precipitated" (default 1%)
            min_solubility: Minimum max solubility required for each polymer
            top_k: Number of results to return

        Returns:
            List of MultiPolymerAtmosphericResult sorted by feasibility
        """
        if len(polymers) < 2:
            logger.warning("Need at least 2 polymers for multi-polymer analysis")
            return []

        # Get all solvents that have data for ALL polymers
        solvents = self.get_available_solvents()
        results = []

        for solvent in solvents:
            # Get precipitation data for each polymer in this solvent
            precip_data = []
            max_sols = {}
            all_valid = True

            for polymer in polymers:
                point = self.analyze_precipitation(polymer, solvent, precip_threshold)
                if point is None or point.max_solubility < min_solubility:
                    all_valid = False
                    break
                if point.precipitation_temp <= 0:
                    all_valid = False
                    break
                precip_data.append((polymer, point.precipitation_temp, point.max_solubility))
                max_sols[polymer] = point.max_solubility

            if not all_valid or len(precip_data) != len(polymers):
                continue

            # Sort by precipitation temperature (highest first = precipitates first during cooling)
            precip_data.sort(key=lambda x: x[1], reverse=True)
            sequence = [(p[0], p[1]) for p in precip_data]

            # Calculate gaps between consecutive precipitations
            gaps = []
            min_gap = float('inf')
            for i in range(len(sequence) - 1):
                p1, t1 = sequence[i]
                p2, t2 = sequence[i + 1]
                gap = t1 - t2
                gaps.append((p1, p2, gap))
                min_gap = min(min_gap, gap)

            # Check if all gaps are sufficient
            if min_gap < min_temp_gap:
                continue

            # Check atmospheric feasibility
            solvent_lower = solvent.lower()
            bp = SOLVENT_BOILING_POINTS.get(solvent_lower)
            if bp is None:
                solvent_clean = solvent_lower.replace(' ', '').replace('-', '')
                bp = SOLVENT_BOILING_POINTS.get(solvent_clean)

            if bp is None:
                continue

            # Dissolution temp = highest precip temp + margin
            dissolution_temp = sequence[0][1] + 20
            is_feasible = dissolution_temp < bp
            feasibility_margin = bp - dissolution_temp

            # Build recommended cooling steps
            steps = []
            warnings = []

            for i, (polymer, temp) in enumerate(sequence):
                step = {
                    "step": i + 1,
                    "cool_to": temp - 5,
                    "collect": polymer,
                    "precipitation_temp": temp,
                    "max_solubility": max_sols[polymer]
                }
                steps.append(step)

                # Warn about small gaps
                if i > 0:
                    prev_gap = gaps[i - 1][2]
                    if prev_gap < 25:
                        warnings.append(
                            f"Small gap ({prev_gap:.0f}°C) between {gaps[i-1][0]} and {polymer} - "
                            f"careful temperature control needed"
                        )

            if not is_feasible:
                warnings.append(f"Requires pressurized equipment (dissolution needs ~{dissolution_temp:.0f}°C, BP={bp}°C)")

            results.append(MultiPolymerAtmosphericResult(
                solvent=solvent,
                boiling_point=bp,
                polymers=[p[0] for p in sequence],
                precipitation_sequence=sequence,
                max_solubilities=max_sols,
                temperature_gaps=gaps,
                min_gap=min_gap,
                dissolution_temp_needed=dissolution_temp,
                is_feasible=is_feasible,
                feasibility_margin=feasibility_margin,
                recommended_steps=steps,
                warnings=warnings
            ))

        # Sort: feasible first, then by min_gap (larger is better), then by margin
        # Use int() for boolean to avoid numpy negation issues
        results.sort(key=lambda x: (not x.is_feasible, -x.min_gap, -x.feasibility_margin))

        return results[:top_k]

    def get_available_polymers(self) -> List[str]:
        """Get list of all polymers from the interpolation dataset."""
        from strap.solubility import get_available_polymers as _get_polymers
        return sorted(_get_polymers())

    def get_available_solvents(self) -> List[str]:
        """Get list of all solvents from the interpolation dataset."""
        from strap.solubility import get_available_solvents as _get_solvents
        return sorted(_get_solvents())


def format_differential_precipitation_results(
    results: List[DifferentialPrecipitationResult],
    include_details: bool = True
) -> str:
    """Format results as markdown for agent output."""
    if not results:
        return "No solvents found matching the criteria."

    lines = [
        "# Differential Precipitation Analysis Results\n",
        f"Found {len(results)} solvent(s) for selective precipitation.\n"
    ]

    # Summary table
    lines.append("| Rank | Solvent | Temp Gap | First Precip | Second Precip | Score |")
    lines.append("|------|---------|----------|--------------|---------------|-------|")

    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.solvent} | {r.temperature_gap:.0f}°C | "
            f"{r.polymer_first} @ {r.polymer_first_precip_temp:.0f}°C | "
            f"{r.polymer_second} @ {r.polymer_second_precip_temp:.0f}°C | "
            f"{r.selectivity_score:.2f} |"
        )

    if include_details and results:
        lines.append("\n## Top Recommendation\n")
        best = results[0]
        lines.append(f"**Solvent:** {best.solvent}\n")
        lines.append(f"**Temperature Gap:** {best.temperature_gap:.0f}°C\n")
        lines.append(f"**Process:**")
        lines.append(f"1. Dissolve both polymers at high temperature (>{best.polymer_first_precip_temp + 20:.0f}°C)")
        lines.append(f"2. Cool to {best.operating_window[1]:.0f}°C - {best.polymer_first} precipitates")
        lines.append(f"3. Filter to collect {best.polymer_first}")
        lines.append(f"4. Cool to {best.operating_window[0]:.0f}°C - {best.polymer_second} precipitates")
        lines.append(f"5. Filter to collect {best.polymer_second}")

        if best.notes:
            lines.append("\n**Notes:**")
            for note in best.notes:
                lines.append(f"- {note}")

    return "\n".join(lines)


def format_multi_polymer_sequence(seq: MultiPolymerPrecipitationSequence) -> str:
    """Format multi-polymer sequence as markdown."""
    lines = [
        f"# Multi-Polymer Precipitation Sequence in {seq.solvent.upper()}\n",
        "## Precipitation Order (cooling from high temperature)\n"
    ]

    lines.append("| Order | Polymer | Precip Temp | Max Solubility |")
    lines.append("|-------|---------|-------------|----------------|")

    for i, (polymer, temp) in enumerate(seq.sequence, 1):
        max_sol = seq.max_solubilities.get(polymer, 0)
        lines.append(f"| {i} | {polymer} | {temp:.0f}°C | {max_sol:.1f}% |")

    lines.append("\n## Recommended Cooling Protocol\n")
    for step in seq.recommended_steps:
        lines.append(f"**Step {step['step']}:** Cool to {step['cool_to']:.0f}°C → Collect {step['collect']}")

    if seq.warnings:
        lines.append("\n## Warnings\n")
        for w in seq.warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def format_atmospheric_feasibility_results(
    results: List[AtmosphericFeasibilityResult],
    include_infeasible: bool = True
) -> str:
    """Format atmospheric feasibility results as markdown."""
    if not results:
        return "No solvents found matching the criteria for atmospheric operation analysis."

    feasible = [r for r in results if r.is_feasible]
    infeasible = [r for r in results if not r.is_feasible]

    lines = [
        "# Atmospheric Pressure Feasibility Analysis\n",
        "Analysis of differential precipitation feasibility at 1 atm (no pressurization required).\n"
    ]

    # Summary
    lines.append(f"**Total solvents analyzed:** {len(results)}")
    lines.append(f"**Feasible at 1 atm:** {len(feasible)}")
    lines.append(f"**Requires pressurization:** {len(infeasible)}\n")

    # Feasible solvents
    if feasible:
        lines.append("## ✅ Feasible at Atmospheric Pressure\n")
        lines.append("| Solvent | BP (°C) | Temp Gap | First Precip | Second Precip | Margin |")
        lines.append("|---------|---------|----------|--------------|---------------|--------|")

        for r in feasible:
            lines.append(
                f"| {r.solvent} | {r.boiling_point:.0f} | {r.temperature_gap:.0f}°C | "
                f"{r.polymer_first} @ {r.polymer_first_precip_temp:.0f}°C | "
                f"{r.polymer_second} @ {r.polymer_second_precip_temp:.0f}°C | "
                f"+{r.feasibility_margin:.0f}°C |"
            )

        # Detailed recommendation for best option
        best = feasible[0]
        lines.append(f"\n### Recommended: {best.solvent.upper()}\n")
        lines.append(f"- **Boiling Point:** {best.boiling_point}°C (at 1 atm)")
        lines.append(f"- **Temperature Gap:** {best.temperature_gap:.0f}°C")
        lines.append(f"- **Safety Margin:** {best.feasibility_margin:.0f}°C below boiling point")
        lines.append(f"\n**Process at 1 atm:**")
        lines.append(f"1. Heat to ~{best.dissolution_temp_needed:.0f}°C to dissolve both polymers")
        lines.append(f"2. Cool to ~{best.polymer_first_precip_temp - 5:.0f}°C → {best.polymer_first} precipitates")
        lines.append(f"3. Filter to collect {best.polymer_first}")
        lines.append(f"4. Cool to ~{best.polymer_second_precip_temp - 5:.0f}°C → {best.polymer_second} precipitates")
        lines.append(f"5. Filter to collect {best.polymer_second}")

    else:
        lines.append("## ⚠️ No Solvents Feasible at Atmospheric Pressure\n")
        lines.append("All analyzed solvents require operating above their boiling point.")
        lines.append("Pressurized equipment (autoclave) would be needed.\n")

    # Infeasible solvents (optional)
    if include_infeasible and infeasible:
        lines.append("\n## ❌ Requires Pressurization\n")
        lines.append("| Solvent | BP (°C) | Dissolution Needed | Over BP by |")
        lines.append("|---------|---------|-------------------|------------|")

        for r in infeasible[:5]:  # Limit to top 5
            lines.append(
                f"| {r.solvent} | {r.boiling_point:.0f} | "
                f"~{r.dissolution_temp_needed:.0f}°C | "
                f"{-r.feasibility_margin:.0f}°C |"
            )

    return "\n".join(lines)


def format_multi_polymer_atmospheric_results(
    results: List[MultiPolymerAtmosphericResult],
    include_infeasible: bool = True
) -> str:
    """Format multi-polymer atmospheric feasibility results as markdown."""
    if not results:
        return "No solvents found where all polymers can be separated with sufficient temperature gaps."

    feasible = [r for r in results if r.is_feasible]
    infeasible = [r for r in results if not r.is_feasible]

    # Get polymer count from first result
    n_polymers = len(results[0].polymers) if results else 0

    lines = [
        f"# Multi-Polymer Atmospheric Feasibility Analysis ({n_polymers} polymers)\n",
        "Analysis of sequential precipitation feasibility at 1 atm (no pressurization required).\n"
    ]

    # Summary
    lines.append(f"**Polymers:** {', '.join(results[0].polymers) if results else 'N/A'}")
    lines.append(f"**Total solvents found:** {len(results)}")
    lines.append(f"**Feasible at 1 atm:** {len(feasible)}")
    lines.append(f"**Requires pressurization:** {len(infeasible)}\n")

    # Feasible solvents
    if feasible:
        lines.append("## ✅ Feasible at Atmospheric Pressure\n")

        # Header based on number of polymers
        header = "| Solvent | BP (°C) | Min Gap |"
        separator = "|---------|---------|---------|"
        for i in range(n_polymers):
            header += f" P{i+1} Precip |"
            separator += "-----------|"
        header += " Margin |"
        separator += "--------|"

        lines.append(header)
        lines.append(separator)

        for r in feasible:
            row = f"| {r.solvent} | {r.boiling_point:.0f} | {r.min_gap:.0f}°C |"
            for polymer, temp in r.precipitation_sequence:
                row += f" {polymer}@{temp:.0f}°C |"
            row += f" +{r.feasibility_margin:.0f}°C |"
            lines.append(row)

        # Detailed recommendation for best option
        best = feasible[0]
        lines.append(f"\n### Recommended: {best.solvent.upper()}\n")
        lines.append(f"- **Boiling Point:** {best.boiling_point}°C (at 1 atm)")
        lines.append(f"- **Minimum Gap:** {best.min_gap:.0f}°C")
        lines.append(f"- **Safety Margin:** {best.feasibility_margin:.0f}°C below boiling point")

        lines.append(f"\n**Sequential Cooling Protocol at 1 atm:**")
        lines.append(f"1. Heat to ~{best.dissolution_temp_needed:.0f}°C to dissolve all {n_polymers} polymers")

        step_num = 2
        for step in best.recommended_steps:
            lines.append(
                f"{step_num}. Cool to ~{step['cool_to']:.0f}°C → "
                f"{step['collect']} precipitates (precip temp: {step['precipitation_temp']:.0f}°C)"
            )
            step_num += 1
            lines.append(f"{step_num}. Filter to collect {step['collect']}")
            step_num += 1

        # Temperature gaps between consecutive polymers
        lines.append(f"\n**Temperature Gaps:**")
        for p1, p2, gap in best.temperature_gaps:
            lines.append(f"- {p1} → {p2}: {gap:.0f}°C")

        if best.warnings:
            lines.append(f"\n**Warnings:**")
            for w in best.warnings:
                lines.append(f"- {w}")

    else:
        lines.append("## ⚠️ No Solvents Feasible at Atmospheric Pressure\n")
        lines.append("All solvents that can separate these polymers require operating above their boiling point.")
        lines.append("Pressurized equipment (autoclave) would be needed.\n")

        if infeasible:
            lines.append("**Best option with pressurization:**")
            best = infeasible[0]
            lines.append(f"- **Solvent:** {best.solvent}")
            lines.append(f"- **Requires:** ~{best.dissolution_temp_needed:.0f}°C (BP={best.boiling_point}°C)")
            lines.append(f"- **Pressure needed for:** {-best.feasibility_margin:.0f}°C above BP")

    # Infeasible solvents (optional)
    if include_infeasible and infeasible and feasible:  # Only show if we have feasible options too
        lines.append("\n## ❌ Other Solvents (Require Pressurization)\n")
        lines.append("| Solvent | BP (°C) | Dissolution Needed | Over BP by |")
        lines.append("|---------|---------|-------------------|------------|")

        for r in infeasible[:5]:
            lines.append(
                f"| {r.solvent} | {r.boiling_point:.0f} | "
                f"~{r.dissolution_temp_needed:.0f}°C | "
                f"{-r.feasibility_margin:.0f}°C |"
            )

    return "\n".join(lines)
