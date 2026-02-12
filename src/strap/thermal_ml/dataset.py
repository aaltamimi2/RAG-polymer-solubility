"""PyTorch Dataset for polymer thermal property multitask learning.

Loads a combined CSV of polymer thermal data (Tm, delta_Hf, delta_Cp, Tg),
tokenises PSMILES strings with the polyBERT tokenizer, and handles missing
labels via per-task boolean masks.

Designed for use with :class:`~strap.thermal_ml.model.ThermalPropertyPredictor`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


TASK_NAMES: list[str] = ["Tm", "delta_Hf", "delta_Cp", "Tg"]
RESIDUAL_TASKS: list[str] = ["Tm", "delta_Hf", "delta_Cp"]

# Column name mapping — the CSV may use different names
_COLUMN_ALIASES: dict[str, list[str]] = {
    "Tm": ["Tm", "T_m", "Tm_K", "melting_temperature"],
    "delta_Hf": ["delta_Hf", "dHf", "Delta_Hf", "enthalpy_fusion"],
    "delta_Cp": ["delta_Cp", "dCp", "Delta_Cp", "heat_capacity_change"],
    "Tg": ["Tg", "T_g", "Tg_K", "glass_transition_temperature"],
    "psmiles": ["psmiles", "PSMILES", "smiles", "SMILES", "polymer_smiles"],
}


def _find_column(df: pd.DataFrame, task: str) -> str | None:
    """Find the actual column name in the DataFrame for a given task."""
    for alias in _COLUMN_ALIASES.get(task, [task]):
        if alias in df.columns:
            return alias
    return None


class ThermalPropertyDataset(Dataset):
    """PyTorch Dataset for multitask polymer thermal property prediction.

    Reads a CSV file containing PSMILES strings and (possibly sparse) thermal
    property labels. Missing labels are handled via per-task boolean masks so
    that the loss function can skip them.

    Parameters
    ----------
    csv_path : str or Path
        Path to the combined thermal property CSV file. Must contain a
        PSMILES column and at least one of Tm, delta_Hf, delta_Cp, Tg.
    tokenizer : PreTrainedTokenizer
        polyBERT tokenizer (from ``AutoTokenizer.from_pretrained``).
    max_length : int
        Maximum token sequence length for the tokenizer.
    group_contribution_fn : callable, optional
        A function ``fn(psmiles: str) -> dict`` that returns group
        contribution baseline estimates. Keys should be a subset of
        ``{"Tm", "delta_Hf", "delta_Cp"}``. When provided, baselines
        are precomputed and stored per sample.
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: Any,
        max_length: int = 512,
        group_contribution_fn: Callable[[str], dict[str, float]] | None = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for ThermalPropertyDataset. "
                "Install with: pip install torch"
            )
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for ThermalPropertyDataset. "
                "Install with: pip install pandas"
            )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_contribution_fn = group_contribution_fn

        # Load CSV
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows from %s", len(df), csv_path)

        # Resolve column names
        psmiles_col = _find_column(df, "psmiles")
        if psmiles_col is None:
            raise ValueError(
                f"No PSMILES column found in CSV. "
                f"Expected one of: {_COLUMN_ALIASES['psmiles']}. "
                f"Found columns: {list(df.columns)}"
            )

        self.psmiles: list[str] = df[psmiles_col].astype(str).tolist()

        # Extract target columns
        self.targets: dict[str, list[float]] = {}
        self.masks: dict[str, list[bool]] = {}
        task_counts: dict[str, int] = {}

        for task in TASK_NAMES:
            col = _find_column(df, task)
            if col is not None:
                values = pd.to_numeric(df[col], errors="coerce")
                self.targets[task] = values.tolist()
                self.masks[task] = (~values.isna()).tolist()
                task_counts[task] = sum(self.masks[task])
            else:
                # Task column not present — all masked out
                self.targets[task] = [float("nan")] * len(df)
                self.masks[task] = [False] * len(df)
                task_counts[task] = 0

        logger.info(
            "Task label counts: %s",
            ", ".join(f"{t}={c}" for t, c in task_counts.items()),
        )

        # Precompute group contribution baselines
        self.baselines: list[dict[str, float]] = []
        if group_contribution_fn is not None:
            logger.info("Computing group contribution baselines ...")
            for smi in self.psmiles:
                try:
                    bl = group_contribution_fn(smi)
                except Exception:
                    bl = {}
                self.baselines.append(
                    {task: bl.get(task, 0.0) for task in RESIDUAL_TASKS}
                )
        else:
            # Default to zero baselines (model does direct prediction)
            self.baselines = [
                {task: 0.0 for task in RESIDUAL_TASKS} for _ in range(len(df))
            ]

        # Pre-tokenise all PSMILES for speed
        logger.info("Tokenising %d PSMILES strings ...", len(self.psmiles))
        self._encodings = tokenizer(
            self.psmiles,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        logger.info("Dataset ready.")

    def __len__(self) -> int:
        return len(self.psmiles)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample.

        Returns
        -------
        dict
            - ``input_ids`` : Tensor [L]
            - ``attention_mask`` : Tensor [L]
            - ``targets`` : dict[str, Tensor scalar]
            - ``mask`` : dict[str, Tensor bool scalar]
            - ``baselines`` : dict[str, Tensor scalar]
        """
        item: dict[str, Any] = {
            "input_ids": self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
        }

        # Targets — replace NaN with 0.0 (will be masked out in loss)
        targets: dict[str, torch.Tensor] = {}
        mask: dict[str, torch.Tensor] = {}
        for task in TASK_NAMES:
            val = self.targets[task][idx]
            has_label = self.masks[task][idx]
            targets[task] = torch.tensor(
                val if has_label else 0.0, dtype=torch.float32
            )
            mask[task] = torch.tensor(has_label, dtype=torch.bool)

        item["targets"] = targets
        item["mask"] = mask

        # Baselines
        baselines: dict[str, torch.Tensor] = {}
        for task in RESIDUAL_TASKS:
            baselines[task] = torch.tensor(
                self.baselines[idx][task], dtype=torch.float32
            )
        item["baselines"] = baselines

        return item

    def __repr__(self) -> str:
        valid_counts = {
            t: sum(self.masks[t]) for t in TASK_NAMES
        }
        return (
            f"ThermalPropertyDataset(n_samples={len(self)}, "
            f"labels={valid_counts})"
        )


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collation for ThermalPropertyDataset batches.

    Stacks input tensors and assembles target/mask/baseline dicts into
    batched tensors, properly handling the variable-presence labels.

    Parameters
    ----------
    batch : list[dict]
        List of samples from ``ThermalPropertyDataset.__getitem__``.

    Returns
    -------
    dict
        Collated batch with:
        - ``input_ids`` : Tensor [B, L]
        - ``attention_mask`` : Tensor [B, L]
        - ``targets`` : dict[str, Tensor [B, 1]]
        - ``mask`` : dict[str, Tensor [B]]
        - ``baselines`` : dict[str, Tensor [B, 1]]
    """
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])

    targets: dict[str, torch.Tensor] = {}
    mask: dict[str, torch.Tensor] = {}
    for task in TASK_NAMES:
        targets[task] = torch.stack(
            [b["targets"][task] for b in batch]
        ).unsqueeze(-1)  # [B, 1]
        mask[task] = torch.stack([b["mask"][task] for b in batch])  # [B]

    baselines: dict[str, torch.Tensor] = {}
    for task in RESIDUAL_TASKS:
        baselines[task] = torch.stack(
            [b["baselines"][task] for b in batch]
        ).unsqueeze(-1)  # [B, 1]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "targets": targets,
        "mask": mask,
        "baselines": baselines,
    }
