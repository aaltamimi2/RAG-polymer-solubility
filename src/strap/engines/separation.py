"""
Polymer Separation Algorithms

Provides multiple algorithms for optimizing polymer separation sequences:
- Greedy: O(n²) fast heuristic
- Dynamic Programming: O(n² * 2^n) optimal for small n
- Branch and Bound: Prunes search space for larger problems

Each algorithm finds the best order to separate polymers based on
selectivity (difference in solubility between target and remaining polymers).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Callable, Any
from enum import Enum
import heapq
from abc import ABC, abstractmethod


class SeparationStatus(Enum):
    """Status of a separation operation."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some steps had low selectivity
    FAILED = "failed"    # Could not find valid separation
    NO_DATA = "no_data"  # Missing solubility data


@dataclass
class SeparationStep:
    """Result of one separation step."""
    step_number: int
    target_polymer: str
    remaining_polymers: List[str]
    solvent: str
    selectivity: float
    target_solubility: float
    max_other_solubility: float
    temperature: float
    is_viable: bool = True
    notes: str = ""
    safety_score: Optional[float] = None  # GSK G-score (0-10, higher = safer)

    @property
    def selectivity_ratio(self) -> float:
        """Ratio of target to max other solubility."""
        if self.max_other_solubility == 0:
            return float('inf')
        return self.target_solubility / self.max_other_solubility


@dataclass
class SeparationSequence:
    """Complete separation sequence with metrics."""
    polymers: List[str]
    steps: List[SeparationStep]
    total_selectivity: float = 0.0
    min_selectivity: float = float('inf')
    avg_selectivity: float = 0.0
    unique_solvents: Set[str] = field(default_factory=set)
    status: SeparationStatus = SeparationStatus.SUCCESS

    def __post_init__(self):
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute aggregate metrics from steps."""
        if not self.steps:
            return

        selectivities = [s.selectivity for s in self.steps if s.remaining_polymers]
        if selectivities:
            self.total_selectivity = sum(selectivities)
            self.min_selectivity = min(selectivities)
            self.avg_selectivity = self.total_selectivity / len(selectivities)

        self.unique_solvents = {s.solvent for s in self.steps if s.solvent not in ("N/A", "none", "error")}

        # Determine status
        if self.min_selectivity < 0:
            self.status = SeparationStatus.FAILED
        elif self.min_selectivity < 5.0:
            self.status = SeparationStatus.PARTIAL
        else:
            self.status = SeparationStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "polymers": self.polymers,
            "sequence": [s.target_polymer for s in self.steps],
            "steps": [
                {
                    "step": s.step_number,
                    "polymer": s.target_polymer,
                    "solvent": s.solvent,
                    "selectivity": s.selectivity,
                    "target_solubility": s.target_solubility,
                    "max_other_solubility": s.max_other_solubility,
                }
                for s in self.steps
            ],
            "metrics": {
                "total_selectivity": self.total_selectivity,
                "min_selectivity": self.min_selectivity,
                "avg_selectivity": self.avg_selectivity,
                "unique_solvents": list(self.unique_solvents),
                "status": self.status.value,
            }
        }

    def __str__(self) -> str:
        seq_str = " -> ".join(s.target_polymer for s in self.steps)
        return f"Sequence: {seq_str} (min_sel={self.min_selectivity:.1f}%, status={self.status.value})"


@dataclass
class SeparationResult:
    """Result from a separation algorithm."""
    best_sequence: SeparationSequence
    all_sequences: List[SeparationSequence] = field(default_factory=list)
    algorithm: str = ""
    computation_time_ms: float = 0.0
    nodes_explored: int = 0

    def top_k(self, k: int = 5) -> List[SeparationSequence]:
        """Return top k sequences by minimum selectivity."""
        sorted_seqs = sorted(self.all_sequences, key=lambda s: s.min_selectivity, reverse=True)
        return sorted_seqs[:k]


class SeparatorBase(ABC):
    """Base class for separation algorithms."""

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
        self.temperature_col = temperature_column
        self._cache: Dict[str, Any] = {}

    @abstractmethod
    async def find_optimal_sequence(
        self,
        polymers: List[str],
        temperature: float = 120.0,
    ) -> SeparationResult:
        """Find the optimal separation sequence."""
        pass

    async def get_selectivity(
        self,
        target: str,
        others: List[str],
        temperature: float,
        used_solvents: Optional[Set[str]] = None,
    ) -> Tuple[str, float, float, float]:
        """
        Find the best solvent to separate target from others.

        Uses the unified solubility API (interpolation model with SQL fallback).

        Returns: (solvent, selectivity, target_solubility, max_other_solubility)
        """
        cache_key = f"{target}|{'|'.join(sorted(others))}|{temperature}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        from strap.solubility import get_selectivity as _get_selectivity
        ret = _get_selectivity(target, others, temperature, used_solvents)
        self._cache[cache_key] = ret
        return ret

    def clear_cache(self):
        """Clear the selectivity cache."""
        self._cache.clear()


class GreedySeparator(SeparatorBase):
    """
    Greedy algorithm for polymer separation.

    At each step, separates the polymer with highest selectivity
    (easiest to separate from remaining polymers).

    Complexity: O(n²) queries
    Optimal: No (heuristic)
    Best for: Quick approximation, large polymer sets
    """

    async def find_optimal_sequence(
        self,
        polymers: List[str],
        temperature: float = 120.0,
        enforce_solvent_diversity: bool = True,
    ) -> SeparationResult:
        """Find separation sequence using greedy algorithm."""
        import time
        start_time = time.time()

        remaining = list(polymers)
        steps: List[SeparationStep] = []
        used_solvents: Set[str] = set()
        step_num = 0
        nodes_explored = 0

        while len(remaining) > 1:
            step_num += 1
            best_polymer = None
            best_selectivity = -float('inf')
            best_solvent = None
            best_target_sol = 0.0
            best_other_max = 0.0

            # Evaluate each polymer as potential target
            for target in remaining:
                others = [p for p in remaining if p != target]
                nodes_explored += 1

                solvent, selectivity, target_sol, other_max = await self.get_selectivity(
                    target, others, temperature,
                    used_solvents if enforce_solvent_diversity else None
                )

                if selectivity > best_selectivity:
                    best_selectivity = selectivity
                    best_polymer = target
                    best_solvent = solvent
                    best_target_sol = target_sol
                    best_other_max = other_max

            if best_polymer:
                others = [p for p in remaining if p != best_polymer]
                steps.append(SeparationStep(
                    step_number=step_num,
                    target_polymer=best_polymer,
                    remaining_polymers=others,
                    solvent=best_solvent,
                    selectivity=best_selectivity,
                    target_solubility=best_target_sol,
                    max_other_solubility=best_other_max,
                    temperature=temperature,
                    is_viable=best_selectivity >= 5.0,
                ))
                used_solvents.add(best_solvent)
                remaining.remove(best_polymer)
            else:
                break

        # Last polymer is isolated
        if remaining:
            steps.append(SeparationStep(
                step_number=step_num + 1,
                target_polymer=remaining[0],
                remaining_polymers=[],
                solvent="N/A",
                selectivity=100.0,
                target_solubility=100.0,
                max_other_solubility=0.0,
                temperature=temperature,
                notes="Last polymer - no separation needed",
            ))

        elapsed_ms = (time.time() - start_time) * 1000
        sequence = SeparationSequence(polymers=polymers, steps=steps)

        return SeparationResult(
            best_sequence=sequence,
            all_sequences=[sequence],
            algorithm="greedy",
            computation_time_ms=elapsed_ms,
            nodes_explored=nodes_explored,
        )


class DPSeparator(SeparatorBase):
    """
    Dynamic Programming algorithm for polymer separation.

    Uses bitmask DP to find the optimal sequence that maximizes
    the minimum selectivity across all steps.

    Complexity: O(n² * 2^n) - exponential but optimal
    Optimal: Yes
    Best for: Small polymer sets (n <= 10)
    """

    async def find_optimal_sequence(
        self,
        polymers: List[str],
        temperature: float = 120.0,
        objective: str = "max_min",  # "max_min", "max_sum", or "max_min_safety"
        top_k: int = 1,
        min_selectivity: float = 5.0,
    ) -> SeparationResult:
        """Find optimal sequence using dynamic programming.

        Args:
            polymers: List of polymer names.
            temperature: Target temperature in C.
            objective: Optimization objective:
                - "max_min": maximize bottleneck selectivity (default)
                - "max_sum": maximize total selectivity
                - "max_min_safety": maximize bottleneck G-score subject to
                  selectivity >= min_selectivity
            top_k: Number of top sequences to return (default: 1).
            min_selectivity: Selectivity floor for safety mode (default: 5.0).
        """
        import time
        start_time = time.time()

        n = len(polymers)
        if n > 12:
            raise ValueError(f"Too many polymers ({n}) for DP. Max is 12.")

        # Route to safety DP if requested
        if objective == "max_min_safety":
            return await self._dp_safety(
                polymers, temperature, min_selectivity, top_k,
            )

        # Precompute all selectivities
        selectivity_cache: Dict[Tuple[int, int], Tuple[str, float, float, float]] = {}
        nodes_explored = 0

        for target_idx in range(n):
            for mask in range(1, 1 << n):
                if not (mask & (1 << target_idx)):
                    continue

                others_mask = mask ^ (1 << target_idx)
                if others_mask == 0:
                    continue

                target = polymers[target_idx]
                others = [polymers[i] for i in range(n) if others_mask & (1 << i)]

                result = await self.get_selectivity(target, others, temperature)
                selectivity_cache[(target_idx, mask)] = result
                nodes_explored += 1

        # DP: dp[mask] = (min_selectivity_so_far, last_polymer_idx, prev_mask)
        full_mask = (1 << n) - 1
        INF = float('inf')

        if objective == "max_min":
            # Maximize minimum selectivity
            dp: Dict[int, Tuple[float, int, int]] = {}

            # Initialize: start by removing each polymer
            for i in range(n):
                mask = full_mask ^ (1 << i)
                if mask == 0:
                    dp[0] = (INF, i, full_mask)
                else:
                    _, sel, _, _ = selectivity_cache.get((i, full_mask), ("", -INF, 0, 0))
                    dp[mask] = (sel, i, full_mask)

            # Process remaining masks
            for remaining in range(full_mask - 1, -1, -1):
                if remaining not in dp:
                    continue

                current_min, _, _ = dp[remaining]
                if remaining == 0:
                    continue

                popcount = bin(remaining).count("1")
                if popcount == 1:
                    # Last polymer: isolation step — propagate to dp[0]
                    idx = next(i for i in range(n) if remaining & (1 << i))
                    if 0 not in dp or current_min > dp[0][0]:
                        dp[0] = (current_min, idx, remaining)
                    continue

                for i in range(n):
                    if not (remaining & (1 << i)):
                        continue

                    new_remaining = remaining ^ (1 << i)
                    cache_key = (i, remaining)

                    if cache_key in selectivity_cache:
                        _, sel, _, _ = selectivity_cache[cache_key]
                        new_min = min(current_min, sel)

                        if new_remaining not in dp or new_min > dp[new_remaining][0]:
                            dp[new_remaining] = (new_min, i, remaining)

            # Reconstruct best sequence
            if 0 not in dp:
                # Fallback - construct any valid sequence
                steps = []
                remaining_polymers = list(polymers)
                for step_num in range(1, n + 1):
                    if not remaining_polymers:
                        break
                    polymer = remaining_polymers.pop(0)
                    steps.append(SeparationStep(
                        step_number=step_num,
                        target_polymer=polymer,
                        remaining_polymers=list(remaining_polymers),
                        solvent="unknown",
                        selectivity=0.0,
                        target_solubility=0.0,
                        max_other_solubility=0.0,
                        temperature=temperature,
                        is_viable=False,
                    ))
            else:
                steps = self._reconstruct_sequence(
                    polymers, dp, selectivity_cache, temperature
                )
        else:
            # Maximize total selectivity (simpler)
            steps = await self._dp_max_sum(polymers, temperature, selectivity_cache)

        elapsed_ms = (time.time() - start_time) * 1000
        sequence = SeparationSequence(polymers=polymers, steps=steps)

        # Extract top-K sequences if requested
        if top_k > 1:
            all_sequences = self._extract_top_k_sequences(
                polymers, selectivity_cache, temperature, k=top_k,
            )
            if not all_sequences:
                all_sequences = [sequence]
        else:
            all_sequences = [sequence]

        return SeparationResult(
            best_sequence=sequence,
            all_sequences=all_sequences,
            algorithm="dynamic_programming",
            computation_time_ms=elapsed_ms,
            nodes_explored=nodes_explored,
        )

    def _extract_top_k_sequences(
        self,
        polymers: List[str],
        selectivity_cache: Dict[Tuple[int, int], Tuple[str, float, float, float]],
        temperature: float,
        k: int = 10,
    ) -> List[SeparationSequence]:
        """Extract top-K separation sequences using forward beam DP.

        Uses the precomputed selectivity_cache to enumerate the K best
        complete separation sequences ranked by min_selectivity (descending).
        """
        n = len(polymers)
        full_mask = (1 << n) - 1
        INF = float('inf')

        # beam[mask] = [(min_sel_so_far, [removal_indices])]
        beam: Dict[int, List[Tuple[float, List[int]]]] = {
            full_mask: [(INF, [])]
        }

        for mask in range(full_mask, 0, -1):
            if mask not in beam:
                continue

            popcount = bin(mask).count("1")
            if popcount == 1:
                # Last polymer: isolation step — propagate to empty set
                idx = next(i for i in range(n) if mask & (1 << i))
                for min_sel, seq in beam[mask]:
                    if 0 not in beam:
                        beam[0] = []
                    beam[0].append((min_sel, seq + [idx]))
                    beam[0].sort(key=lambda x: x[0], reverse=True)
                    if len(beam[0]) > k:
                        beam[0] = beam[0][:k]
                continue

            for min_sel, seq in beam[mask]:
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue
                    child = mask ^ (1 << i)
                    cache_key = (i, mask)
                    if cache_key not in selectivity_cache:
                        continue
                    _, sel, _, _ = selectivity_cache[cache_key]
                    new_min = min(min_sel, sel) if seq else sel

                    if child not in beam:
                        beam[child] = []
                    beam[child].append((new_min, seq + [i]))
                    beam[child].sort(key=lambda x: x[0], reverse=True)
                    if len(beam[child]) > k:
                        beam[child] = beam[child][:k]

        # Convert top-K paths to SeparationSequence objects
        sequences: List[SeparationSequence] = []
        for min_sel, idx_seq in beam.get(0, []):
            steps = self._build_steps_from_indices(
                polymers, idx_seq, selectivity_cache, temperature,
            )
            sequences.append(SeparationSequence(polymers=polymers, steps=steps))

        return sequences

    def _build_steps_from_indices(
        self,
        polymers: List[str],
        idx_sequence: List[int],
        cache: Dict[Tuple[int, int], Tuple[str, float, float, float]],
        temperature: float,
    ) -> List[SeparationStep]:
        """Build SeparationStep list from an index removal sequence."""
        n = len(polymers)
        steps = []
        remaining_mask = (1 << n) - 1

        for step_num, polymer_idx in enumerate(idx_sequence, 1):
            polymer = polymers[polymer_idx]
            cache_key = (polymer_idx, remaining_mask)
            child_mask = remaining_mask ^ (1 << polymer_idx)
            remaining_list = [polymers[i] for i in range(n) if child_mask & (1 << i)]

            if cache_key in cache:
                solvent, sel, target_sol, other_max = cache[cache_key]
            else:
                solvent, sel, target_sol, other_max = "N/A", 0.0, 0.0, 0.0

            is_last = (child_mask == 0) or (bin(remaining_mask).count("1") == 1)
            steps.append(SeparationStep(
                step_number=step_num,
                target_polymer=polymer,
                remaining_polymers=remaining_list,
                solvent=solvent if not is_last else "N/A",
                selectivity=sel if not is_last else 100.0,
                target_solubility=target_sol if not is_last else 100.0,
                max_other_solubility=other_max if not is_last else 0.0,
                temperature=temperature,
                is_viable=sel >= 5.0 if not is_last else True,
                notes="Last polymer - no separation needed" if is_last else "",
            ))
            remaining_mask = child_mask

        return steps

    def _reconstruct_sequence(
        self,
        polymers: List[str],
        dp: Dict[int, Tuple[float, int, int]],
        cache: Dict[Tuple[int, int], Tuple[str, float, float, float]],
        temperature: float,
    ) -> List[SeparationStep]:
        """Reconstruct the separation sequence from DP results."""
        steps = []
        n = len(polymers)

        current = 0
        step_num = n
        sequence_order = []

        visited = set()
        while current in dp and current not in visited:
            visited.add(current)
            _, last_idx, prev = dp[current]
            sequence_order.append((last_idx, current, prev))
            if prev == (1 << n) - 1:
                break
            current = prev

        # Reverse to get forward order
        sequence_order.reverse()

        for step_num, (polymer_idx, remaining_mask, full_mask) in enumerate(sequence_order, 1):
            polymer = polymers[polymer_idx]
            cache_key = (polymer_idx, full_mask)

            if cache_key in cache:
                solvent, sel, target_sol, other_max = cache[cache_key]
            else:
                solvent, sel, target_sol, other_max = "unknown", 0.0, 0.0, 0.0

            remaining = [polymers[i] for i in range(n)
                        if remaining_mask & (1 << i)]

            steps.append(SeparationStep(
                step_number=step_num,
                target_polymer=polymer,
                remaining_polymers=remaining,
                solvent=solvent,
                selectivity=sel,
                target_solubility=target_sol,
                max_other_solubility=other_max,
                temperature=temperature,
                is_viable=sel >= 5.0,
            ))

        # Add final polymer if needed
        if len(steps) < n:
            separated = {s.target_polymer for s in steps}
            last_polymer = [p for p in polymers if p not in separated][0]
            steps.append(SeparationStep(
                step_number=len(steps) + 1,
                target_polymer=last_polymer,
                remaining_polymers=[],
                solvent="N/A",
                selectivity=100.0,
                target_solubility=100.0,
                max_other_solubility=0.0,
                temperature=temperature,
            ))

        return steps

    async def _dp_max_sum(
        self,
        polymers: List[str],
        temperature: float,
        cache: Dict[Tuple[int, int], Tuple[str, float, float, float]],
    ) -> List[SeparationStep]:
        """DP variant maximizing total selectivity."""
        # Simpler greedy for now as fallback
        greedy = GreedySeparator(
            self.conn, self.table_name, self.polymer_col,
            self.solvent_col, self.solubility_col, self.temperature_col
        )
        result = await greedy.find_optimal_sequence(polymers, temperature)
        return result.best_sequence.steps

    # ------------------------------------------------------------------
    # Safety-constrained DP: maximize min G-score subject to selectivity floor
    # ------------------------------------------------------------------

    @staticmethod
    def _load_gscore_map(db_connection: Any) -> Dict[str, float]:
        """Load GSK G-scores into a {lowercase_name: score} dict."""
        gscore_map: Dict[str, float] = {}
        try:
            rows = db_connection.execute(
                "SELECT solvent_common_name, g_score FROM gsk_dataset"
            ).fetchall()
            for name, score in rows:
                if score is not None:
                    gscore_map[name.strip().lower()] = float(score)
        except Exception:
            pass

        # Abbreviation mappings (interpolation model names → GSK names)
        _ABBREV = {
            "dmf": "dimethylformamide", "thf": "tetrahydrofuran",
            "dcm": "dichloromethane", "ch2cl2": "dichloromethane",
            "chcl3": "chloroform", "meoh": "methanol", "etoh": "ethanol",
            "1,2-dimethylbenzene": "o-xylene", "1,4-dimethylbenzene": "p-xylene",
            "n-heptane": "heptane", "n-hexane": "hexane",
            "glycol": "ethylene glycol", "h2o": "water",
            "propanone": "acetone", "butanone": "methyl ethyl ketone",
            "ethylacetate": "ethyl acetate", "dimethylsulfoxide": "dimethyl sulfoxide",
            "dimethylformamide": "n,n-dimethylformamide",
            "acetylacetone": "2,4-pentanedione",
        }
        for abbr, full in _ABBREV.items():
            if full in gscore_map and abbr not in gscore_map:
                gscore_map[abbr] = gscore_map[full]
        return gscore_map

    def _build_safety_cache(
        self,
        polymers: List[str],
        temperature: float,
        gscore_map: Dict[str, float],
        min_selectivity: float = 5.0,
    ) -> Tuple[Dict[Tuple[int, int], Tuple[str, float, float]], int]:
        """Build per-edge cache choosing the safest viable solvent.

        For each (target_idx, mask), calls get_all_solvents_selectivity to
        get ALL solvents, annotates with G-scores, filters by selectivity
        threshold, and picks the solvent with the highest G-score.

        Returns:
            safety_cache: {(target_idx, mask) -> (solvent, selectivity, gscore)}
            nodes_explored: count of cache entries built
        """
        from strap.solubility import get_all_solvents_selectivity

        n = len(polymers)
        safety_cache: Dict[Tuple[int, int], Tuple[str, float, float]] = {}
        nodes_explored = 0

        for target_idx in range(n):
            for mask in range(1, 1 << n):
                if not (mask & (1 << target_idx)):
                    continue
                others_mask = mask ^ (1 << target_idx)
                if others_mask == 0:
                    continue

                target = polymers[target_idx]
                others = [polymers[i] for i in range(n) if others_mask & (1 << i)]
                results = get_all_solvents_selectivity(target, others, temperature)
                nodes_explored += 1

                if not results:
                    safety_cache[(target_idx, mask)] = ("N/A", 0.0, 0.0)
                    continue

                # Annotate with G-scores
                for r in results:
                    r["gscore"] = gscore_map.get(r["solvent"].lower(), 0.0)

                # Filter: selectivity >= threshold AND has G-score
                viable = [r for r in results
                          if r["selectivity"] >= min_selectivity and r["gscore"] > 0]
                if not viable:
                    viable = [r for r in results if r["gscore"] > 0]
                if not viable:
                    viable = results[:1]

                best = max(viable, key=lambda r: r["gscore"])
                safety_cache[(target_idx, mask)] = (
                    best["solvent"], best["selectivity"], best["gscore"],
                )

        return safety_cache, nodes_explored

    async def _dp_safety(
        self,
        polymers: List[str],
        temperature: float,
        min_selectivity: float = 5.0,
        top_k: int = 1,
    ) -> SeparationResult:
        """Safety-constrained DP: maximize min G-score subject to selectivity floor.

        Args:
            polymers: List of polymer names.
            temperature: Target temperature in C.
            min_selectivity: Minimum selectivity threshold for viable solvents.
            top_k: Number of top sequences to return.

        Returns:
            SeparationResult with best sequence and optional top-K.
        """
        import time
        start_time = time.time()

        n = len(polymers)
        gscore_map = self._load_gscore_map(self.conn)
        safety_cache, nodes_explored = self._build_safety_cache(
            polymers, temperature, gscore_map, min_selectivity,
        )

        # DP: maximize min G-score along path
        full_mask = (1 << n) - 1
        INF = float("inf")
        dp: Dict[int, Tuple[float, int, int]] = {}

        # Initialize: remove one polymer from full set
        for i in range(n):
            rem = full_mask ^ (1 << i)
            _, sel, gs = safety_cache.get((i, full_mask), ("N/A", 0.0, 0.0))
            if rem == 0:
                dp[0] = (gs, i, full_mask)
            elif rem not in dp or gs > dp[rem][0]:
                dp[rem] = (gs, i, full_mask)

        # Fill DP table
        for mask in range(full_mask - 1, -1, -1):
            if mask not in dp:
                continue
            cur_min_gs = dp[mask][0]
            if mask == 0:
                continue
            popcount = bin(mask).count("1")
            if popcount == 1:
                idx = next(i for i in range(n) if mask & (1 << i))
                if 0 not in dp or cur_min_gs > dp[0][0]:
                    dp[0] = (cur_min_gs, idx, mask)
                continue
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                new_mask = mask ^ (1 << i)
                _, sel, gs = safety_cache.get((i, mask), ("N/A", 0.0, 0.0))
                new_min = min(cur_min_gs, gs)
                if new_mask not in dp or new_min > dp[new_mask][0]:
                    dp[new_mask] = (new_min, i, mask)

        # Reconstruct best path
        steps = self._reconstruct_safety_sequence(
            polymers, dp, safety_cache, temperature,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        sequence = SeparationSequence(polymers=polymers, steps=steps)

        # Top-K
        if top_k > 1:
            all_sequences = self._extract_top_k_safety_sequences(
                polymers, safety_cache, temperature, k=top_k,
            )
            if not all_sequences:
                all_sequences = [sequence]
        else:
            all_sequences = [sequence]

        return SeparationResult(
            best_sequence=sequence,
            all_sequences=all_sequences,
            algorithm="dynamic_programming_safety",
            computation_time_ms=elapsed_ms,
            nodes_explored=nodes_explored,
        )

    def _reconstruct_safety_sequence(
        self,
        polymers: List[str],
        dp: Dict[int, Tuple[float, int, int]],
        cache: Dict[Tuple[int, int], Tuple[str, float, float]],
        temperature: float,
    ) -> List[SeparationStep]:
        """Reconstruct separation sequence from safety DP results."""
        n = len(polymers)
        sequence_order = []
        current = 0

        while current in dp:
            _, last_idx, prev = dp[current]
            sequence_order.append((last_idx, current, prev))
            if prev == (1 << n) - 1:
                break
            current = prev

        sequence_order.reverse()
        steps = []

        for step_num, (polymer_idx, remaining_mask, from_mask) in enumerate(sequence_order, 1):
            polymer = polymers[polymer_idx]
            cache_key = (polymer_idx, from_mask)

            if cache_key in cache:
                solvent, sel, gs = cache[cache_key]
            else:
                solvent, sel, gs = "unknown", 0.0, 0.0

            remaining = [polymers[i] for i in range(n)
                        if remaining_mask & (1 << i)]

            is_last = not remaining
            steps.append(SeparationStep(
                step_number=step_num,
                target_polymer=polymer,
                remaining_polymers=remaining,
                solvent=solvent if not is_last else "N/A",
                selectivity=sel if not is_last else 100.0,
                target_solubility=0.0,
                max_other_solubility=0.0,
                temperature=temperature,
                is_viable=sel >= 5.0 if not is_last else True,
                safety_score=gs if not is_last else None,
                notes="Last polymer - no separation needed" if is_last else "",
            ))

        # Add final polymer if needed
        if len(steps) < n:
            separated = {s.target_polymer for s in steps}
            last_polymer = [p for p in polymers if p not in separated][0]
            steps.append(SeparationStep(
                step_number=len(steps) + 1,
                target_polymer=last_polymer,
                remaining_polymers=[],
                solvent="N/A",
                selectivity=100.0,
                target_solubility=100.0,
                max_other_solubility=0.0,
                temperature=temperature,
                safety_score=None,
                notes="Last polymer - no separation needed",
            ))

        return steps

    def _extract_top_k_safety_sequences(
        self,
        polymers: List[str],
        safety_cache: Dict[Tuple[int, int], Tuple[str, float, float]],
        temperature: float,
        k: int = 10,
    ) -> List[SeparationSequence]:
        """Extract top-K safety-optimized sequences using forward beam DP."""
        n = len(polymers)
        full_mask = (1 << n) - 1
        INF = float("inf")

        beam: Dict[int, List[Tuple[float, List[int]]]] = {
            full_mask: [(INF, [])]
        }

        for mask in range(full_mask, 0, -1):
            if mask not in beam:
                continue
            popcount = bin(mask).count("1")
            if popcount == 1:
                idx = next(i for i in range(n) if mask & (1 << i))
                for min_gs, seq in beam[mask]:
                    if 0 not in beam:
                        beam[0] = []
                    beam[0].append((min_gs, seq + [idx]))
                    beam[0].sort(key=lambda x: x[0], reverse=True)
                    if len(beam[0]) > k:
                        beam[0] = beam[0][:k]
                continue

            for min_gs, seq in beam[mask]:
                for i in range(n):
                    if not (mask & (1 << i)):
                        continue
                    child = mask ^ (1 << i)
                    cache_key = (i, mask)
                    if cache_key not in safety_cache:
                        continue
                    _, sel, gs = safety_cache[cache_key]
                    new_min = min(min_gs, gs) if seq else gs

                    if child not in beam:
                        beam[child] = []
                    beam[child].append((new_min, seq + [i]))
                    beam[child].sort(key=lambda x: x[0], reverse=True)
                    if len(beam[child]) > k:
                        beam[child] = beam[child][:k]

        sequences: List[SeparationSequence] = []
        for min_gs, idx_seq in beam.get(0, []):
            steps = self._build_safety_steps_from_indices(
                polymers, idx_seq, safety_cache, temperature,
            )
            sequences.append(SeparationSequence(polymers=polymers, steps=steps))

        return sequences

    def _build_safety_steps_from_indices(
        self,
        polymers: List[str],
        idx_sequence: List[int],
        cache: Dict[Tuple[int, int], Tuple[str, float, float]],
        temperature: float,
    ) -> List[SeparationStep]:
        """Build SeparationStep list from safety cache index sequence."""
        n = len(polymers)
        steps = []
        remaining_mask = (1 << n) - 1

        for step_num, polymer_idx in enumerate(idx_sequence, 1):
            polymer = polymers[polymer_idx]
            cache_key = (polymer_idx, remaining_mask)
            child_mask = remaining_mask ^ (1 << polymer_idx)
            remaining_list = [polymers[i] for i in range(n) if child_mask & (1 << i)]

            if cache_key in cache:
                solvent, sel, gs = cache[cache_key]
            else:
                solvent, sel, gs = "N/A", 0.0, 0.0

            is_last = (child_mask == 0) or (bin(remaining_mask).count("1") == 1)
            steps.append(SeparationStep(
                step_number=step_num,
                target_polymer=polymer,
                remaining_polymers=remaining_list,
                solvent=solvent if not is_last else "N/A",
                selectivity=sel if not is_last else 100.0,
                target_solubility=0.0,
                max_other_solubility=0.0,
                temperature=temperature,
                is_viable=sel >= 5.0 if not is_last else True,
                safety_score=gs if not is_last else None,
                notes="Last polymer - no separation needed" if is_last else "",
            ))
            remaining_mask = child_mask

        return steps


class BranchAndBoundSeparator(SeparatorBase):
    """
    Branch and Bound algorithm for polymer separation.

    Explores the search tree with pruning based on upper bounds.
    Falls back to best solution found within time limit.

    Complexity: O(n!) worst case, but pruning helps significantly
    Optimal: Yes (if completed)
    Best for: Medium-sized problems where DP is too slow
    """

    async def find_optimal_sequence(
        self,
        polymers: List[str],
        temperature: float = 120.0,
        time_limit_ms: float = 5000.0,
    ) -> SeparationResult:
        """Find optimal sequence using branch and bound."""
        import time
        start_time = time.time()

        n = len(polymers)
        nodes_explored = 0

        # Priority queue: (-min_selectivity, sequence_so_far, remaining_polymers, used_solvents)
        # Negative because heapq is min-heap
        initial_remaining = set(range(n))
        pq: List[Tuple[float, List[int], Set[int], Set[str]]] = [
            (0.0, [], initial_remaining, set())
        ]

        best_sequence: Optional[List[SeparationStep]] = None
        best_min_selectivity = -float('inf')
        all_complete_sequences: List[SeparationSequence] = []

        while pq:
            # Check time limit
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > time_limit_ms:
                break

            neg_min_sel, seq_so_far, remaining, used_solvents = heapq.heappop(pq)
            current_min_sel = -neg_min_sel if seq_so_far else float('inf')
            nodes_explored += 1

            # Pruning: skip if worse than best found
            if current_min_sel < best_min_selectivity:
                continue

            # Complete sequence
            if not remaining:
                steps = self._build_steps(polymers, seq_so_far, temperature, used_solvents)
                sequence = SeparationSequence(polymers=polymers, steps=steps)
                all_complete_sequences.append(sequence)

                if sequence.min_selectivity > best_min_selectivity:
                    best_min_selectivity = sequence.min_selectivity
                    best_sequence = steps
                continue

            # Branch: try removing each remaining polymer
            for polymer_idx in remaining:
                others_idx = remaining - {polymer_idx}
                target = polymers[polymer_idx]
                others = [polymers[i] for i in others_idx]

                solvent, selectivity, _, _ = await self.get_selectivity(
                    target, others, temperature, used_solvents
                )

                new_min_sel = min(current_min_sel, selectivity) if seq_so_far else selectivity

                # Pruning: don't explore if already worse than best
                if new_min_sel >= best_min_selectivity:
                    new_seq = seq_so_far + [polymer_idx]
                    new_used = used_solvents | {solvent}
                    heapq.heappush(pq, (-new_min_sel, new_seq, others_idx, new_used))

        elapsed_ms = (time.time() - start_time) * 1000

        if best_sequence is None:
            # Fallback to greedy
            greedy = GreedySeparator(
                self.conn, self.table_name, self.polymer_col,
                self.solvent_col, self.solubility_col, self.temperature_col
            )
            result = await greedy.find_optimal_sequence(polymers, temperature)
            best_sequence = result.best_sequence.steps

        sequence = SeparationSequence(polymers=polymers, steps=best_sequence)

        return SeparationResult(
            best_sequence=sequence,
            all_sequences=all_complete_sequences,
            algorithm="branch_and_bound",
            computation_time_ms=elapsed_ms,
            nodes_explored=nodes_explored,
        )

    def _build_steps(
        self,
        polymers: List[str],
        sequence: List[int],
        temperature: float,
        used_solvents: Set[str],
    ) -> List[SeparationStep]:
        """Build SeparationStep list from index sequence."""
        steps = []
        remaining_set = set(range(len(polymers)))

        for step_num, polymer_idx in enumerate(sequence, 1):
            remaining_set.discard(polymer_idx)
            target = polymers[polymer_idx]
            others = [polymers[i] for i in remaining_set]

            # Would need to look up solvent info - simplified for now
            steps.append(SeparationStep(
                step_number=step_num,
                target_polymer=target,
                remaining_polymers=others,
                solvent="computed",
                selectivity=0.0,  # Would need lookup
                target_solubility=0.0,
                max_other_solubility=0.0,
                temperature=temperature,
            ))

        # Add last polymer
        if remaining_set:
            last_idx = remaining_set.pop()
            steps.append(SeparationStep(
                step_number=len(steps) + 1,
                target_polymer=polymers[last_idx],
                remaining_polymers=[],
                solvent="N/A",
                selectivity=100.0,
                target_solubility=100.0,
                max_other_solubility=0.0,
                temperature=temperature,
            ))

        return steps


# Convenience function for quick access
async def find_best_separation(
    polymers: List[str],
    db_connection: Any,
    temperature: float = 120.0,
    algorithm: str = "auto",
    **kwargs,
) -> SeparationResult:
    """
    Find the best separation sequence for given polymers.

    Args:
        polymers: List of polymer names to separate
        db_connection: Database connection (DuckDB)
        temperature: Target temperature in °C
        algorithm: "greedy", "dp", "branch_and_bound", or "auto"
        **kwargs: Additional arguments for the algorithm

    Returns:
        SeparationResult with optimal sequence and metrics
    """
    n = len(polymers)

    if algorithm == "auto":
        if n <= 6:
            algorithm = "dp"
        elif n <= 10:
            algorithm = "branch_and_bound"
        else:
            algorithm = "greedy"

    top_k = kwargs.pop("top_k", 1)
    objective = kwargs.pop("objective", "max_min")
    min_selectivity = kwargs.pop("min_selectivity", 5.0)

    # Safety objective requires DP
    if objective == "max_min_safety":
        algorithm = "dp"

    if algorithm == "greedy":
        separator = GreedySeparator(db_connection, **kwargs)
    elif algorithm == "dp":
        separator = DPSeparator(db_connection, **kwargs)
    elif algorithm == "branch_and_bound":
        separator = BranchAndBoundSeparator(db_connection, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if algorithm == "dp":
        return await separator.find_optimal_sequence(
            polymers, temperature, objective=objective,
            top_k=top_k, min_selectivity=min_selectivity,
        )
    return await separator.find_optimal_sequence(polymers, temperature)
