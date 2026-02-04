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

        Returns: (solvent, selectivity, target_solubility, max_other_solubility)
        """
        cache_key = f"{target}|{'|'.join(sorted(others))}|{temperature}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if not others:
            return ("N/A", float('inf'), 100.0, 0.0)

        all_polymers = [target] + others
        polymer_filter = "', '".join(all_polymers)

        query = f"""
        WITH solubility_data AS (
            SELECT
                {self.solvent_col} as solvent,
                {self.polymer_col} as polymer,
                AVG({self.solubility_col}) as avg_sol
            FROM {self.table_name}
            WHERE {self.polymer_col} IN ('{polymer_filter}')
            AND {self.temperature_col} BETWEEN {temperature - 10} AND {temperature + 10}
            GROUP BY {self.solvent_col}, {self.polymer_col}
        ),
        target_sol AS (
            SELECT solvent, avg_sol as target_solubility
            FROM solubility_data
            WHERE UPPER(polymer) = UPPER('{target}')
        ),
        others_max AS (
            SELECT solvent, MAX(avg_sol) as max_other
            FROM solubility_data
            WHERE UPPER(polymer) IN ({','.join([f"UPPER('{p}')" for p in others])})
            GROUP BY solvent
        )
        SELECT
            t.solvent,
            t.target_solubility,
            COALESCE(o.max_other, 0) as max_other,
            (t.target_solubility - COALESCE(o.max_other, 0)) as selectivity
        FROM target_sol t
        LEFT JOIN others_max o ON LOWER(t.solvent) = LOWER(o.solvent)
        WHERE t.target_solubility > 0
        ORDER BY selectivity DESC
        """

        try:
            result = self.conn.execute(query).fetchall()
            if not result:
                return ("none", -999.0, 0.0, 0.0)

            # Filter out used solvents if provided
            if used_solvents:
                used_lower = {s.lower() for s in used_solvents}
                filtered = [r for r in result if r[0].lower() not in used_lower]
                if filtered:
                    result = filtered

            best = result[0]
            ret = (best[0], best[3], best[1], best[2])
            self._cache[cache_key] = ret
            return ret

        except Exception as e:
            return ("error", -999.0, 0.0, 0.0)

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
        objective: str = "max_min",  # "max_min" or "max_sum"
    ) -> SeparationResult:
        """Find optimal sequence using dynamic programming."""
        import time
        start_time = time.time()

        n = len(polymers)
        if n > 12:
            raise ValueError(f"Too many polymers ({n}) for DP. Max is 12.")

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

        return SeparationResult(
            best_sequence=sequence,
            all_sequences=[sequence],
            algorithm="dynamic_programming",
            computation_time_ms=elapsed_ms,
            nodes_explored=nodes_explored,
        )

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

        while current in dp:
            _, last_idx, prev = dp[current]
            sequence_order.append((last_idx, current, prev))
            if prev == (1 << n) - 1:
                break
            current = prev ^ (1 << last_idx)

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

    if algorithm == "greedy":
        separator = GreedySeparator(db_connection, **kwargs)
    elif algorithm == "dp":
        separator = DPSeparator(db_connection, **kwargs)
    elif algorithm == "branch_and_bound":
        separator = BranchAndBoundSeparator(db_connection, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return await separator.find_optimal_sequence(polymers, temperature)
