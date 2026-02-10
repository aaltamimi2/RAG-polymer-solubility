"""Training script for the polyBERT thermal property predictor.

Trains a multi-task model that predicts Tm, delta_Hf, delta_Cp (and
optionally Tg) from PSMILES strings.  The model uses polyBERT as a frozen
backbone during an initial warm-up phase, then unfreezes for end-to-end
fine-tuning with a lower learning rate.

Usage
-----
Run as a module::

    python -m strap.thermal_ml.train --data_path data/thermal_properties/combined_thermal.csv

Or directly::

    python src/strap/thermal_ml/train.py --epochs 80 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, random_split

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_DATA_PATH = _PROJECT_ROOT / "data" / "thermal_properties" / "combined_thermal.csv"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "models" / "thermal"

_PROPERTY_NAMES = ("Tm", "delta_Hf", "delta_Cp", "Tg")
_PROPERTY_UNITS = {
    "Tm": "K",
    "delta_Hf": "J/mol",
    "delta_Cp": "J/(mol*K)",
    "Tg": "K",
}


# ---------------------------------------------------------------------------
# Learning rate scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _build_lr_lambda(warmup_steps: int, total_steps: int):
    """Return a lambda for ``torch.optim.lr_scheduler.LambdaLR``.

    Linear warmup for *warmup_steps*, then cosine decay to 0 over the
    remaining steps.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1.0, float(warmup_steps))
        progress = float(current_step - warmup_steps) / max(
            1.0, float(total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


# ---------------------------------------------------------------------------
# Target scaler wrapper
# ---------------------------------------------------------------------------

class ThermalTargetScaler:
    """Per-property StandardScaler that can inverse-transform individual properties.

    Attributes
    ----------
    scalers : dict[str, StandardScaler]
        One scaler per property key.
    """

    def __init__(self, property_keys: tuple[str, ...] = _PROPERTY_NAMES):
        self.property_keys = property_keys
        self.scalers: dict[str, StandardScaler] = {
            key: StandardScaler() for key in property_keys
        }

    def fit(self, targets: np.ndarray) -> ThermalTargetScaler:
        """Fit scalers.  *targets* has shape ``(N, n_properties)``."""
        for i, key in enumerate(self.property_keys):
            col = targets[:, i].reshape(-1, 1)
            # Only fit on non-NaN values
            mask = ~np.isnan(col.ravel())
            if mask.any():
                self.scalers[key].fit(col[mask].reshape(-1, 1))
        return self

    def transform(self, targets: np.ndarray) -> np.ndarray:
        """Transform targets to standardised values."""
        out = np.copy(targets)
        for i, key in enumerate(self.property_keys):
            col = targets[:, i].reshape(-1, 1)
            mask = ~np.isnan(col.ravel())
            if mask.any():
                out[mask, i] = self.scalers[key].transform(col[mask].reshape(-1, 1)).ravel()
        return out

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform all properties at once."""
        out = np.copy(scaled)
        for i, key in enumerate(self.property_keys):
            col = scaled[:, i].reshape(-1, 1)
            mask = ~np.isnan(col.ravel())
            if mask.any():
                out[mask, i] = self.scalers[key].inverse_transform(col[mask].reshape(-1, 1)).ravel()
        return out

    def inverse_transform_property(
        self, prop_key: str, mean: float, std: float
    ) -> tuple[float, float]:
        """Inverse-transform a single property's mean and std.

        The standard deviation is scaled by the scaler's ``scale_`` only
        (translation does not affect spread).
        """
        scaler = self.scalers.get(prop_key)
        if scaler is None or not hasattr(scaler, "mean_"):
            return mean, std
        inv_mean = float(scaler.inverse_transform(np.array([[mean]])).ravel()[0])
        inv_std = float(std * scaler.scale_[0])
        return inv_mean, inv_std


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    scaler: Optional[ThermalTargetScaler] = None,
) -> dict:
    """Run one validation pass and return per-task RMSE and MAE.

    Returns
    -------
    dict
        ``{"loss": float, "metrics": {prop: {"rmse": float, "mae": float}}}``
    """
    model.eval()
    criterion = nn.MSELoss(reduction="none")

    all_preds: dict[str, list[np.ndarray]] = {k: [] for k in _PROPERTY_NAMES}
    all_targets: dict[str, list[np.ndarray]] = {k: [] for k in _PROPERTY_NAMES}
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)  # (B, n_props)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Normalise outputs to (B, n_props) tensor
            if isinstance(outputs, dict):
                pred_tensor = torch.stack(
                    [outputs[k] for k in _PROPERTY_NAMES if k in outputs], dim=-1
                )
            elif hasattr(outputs, "shape") and outputs.dim() == 2:
                pred_tensor = outputs
            else:
                pred_tensor = outputs

            # Mask NaN targets
            valid_mask = ~torch.isnan(targets)
            per_element_loss = criterion(pred_tensor, targets)
            per_element_loss = per_element_loss * valid_mask.float()
            batch_loss = per_element_loss.sum() / valid_mask.float().sum().clamp(min=1.0)
            total_loss += batch_loss.item()
            n_batches += 1

            pred_np = pred_tensor.cpu().numpy()
            target_np = targets.cpu().numpy()

            for i, key in enumerate(_PROPERTY_NAMES):
                if i >= pred_np.shape[1]:
                    break
                mask = ~np.isnan(target_np[:, i])
                if mask.any():
                    all_preds[key].append(pred_np[mask, i])
                    all_targets[key].append(target_np[mask, i])

    avg_loss = total_loss / max(n_batches, 1)

    # Compute per-task metrics in original units
    metrics: dict[str, dict[str, float]] = {}
    for key in _PROPERTY_NAMES:
        if not all_preds[key]:
            continue
        preds_cat = np.concatenate(all_preds[key])
        targets_cat = np.concatenate(all_targets[key])

        # Inverse-transform to original units if scaler provided
        if scaler is not None:
            preds_inv = np.full((len(preds_cat), len(_PROPERTY_NAMES)), np.nan)
            targets_inv = np.full((len(targets_cat), len(_PROPERTY_NAMES)), np.nan)
            idx = _PROPERTY_NAMES.index(key)
            preds_inv[:, idx] = preds_cat
            targets_inv[:, idx] = targets_cat
            preds_cat = scaler.inverse_transform(preds_inv)[:, idx]
            targets_cat = scaler.inverse_transform(targets_inv)[:, idx]

        errors = preds_cat - targets_cat
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        metrics[key] = {
            "rmse": rmse,
            "mae": mae,
            "unit": _PROPERTY_UNITS.get(key, ""),
            "n_samples": len(preds_cat),
        }

    return {"loss": avg_loss, "metrics": metrics}


# ---------------------------------------------------------------------------
# Backbone freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def _freeze_backbone(model: nn.Module) -> None:
    """Freeze the polyBERT backbone parameters."""
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen.")
    elif hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen.")
    else:
        logger.warning("Could not identify backbone to freeze.")


def _unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze the polyBERT backbone parameters."""
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen.")
    elif hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen.")
    else:
        logger.warning("Could not identify backbone to unfreeze.")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Execute the full training pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments (see :func:`parse_args`).
    """
    if not _TORCH_AVAILABLE:
        logger.error("PyTorch is required for training.  pip install torch")
        sys.exit(1)
    if not _SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for training.  pip install scikit-learn")
        sys.exit(1)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s", device)

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    from strap.thermal_ml.dataset import ThermalPropertyDataset

    logger.info("Loading dataset from %s ...", args.data_path)
    full_dataset = ThermalPropertyDataset(
        csv_path=args.data_path,
        tokenizer_name="kuelumbus/polyBERT",
    )
    logger.info("Dataset size: %d samples", len(full_dataset))

    # Train / val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator,
    )
    logger.info("Train: %d, Val: %d", train_size, val_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # ------------------------------------------------------------------
    # Target scaler
    # ------------------------------------------------------------------
    # Collect all targets from training set to fit scaler
    all_targets_list = []
    for idx in train_dataset.indices:
        sample = full_dataset[idx]
        all_targets_list.append(sample["targets"].numpy())
    all_targets_arr = np.stack(all_targets_list, axis=0)

    target_scaler = ThermalTargetScaler(property_keys=_PROPERTY_NAMES)
    target_scaler.fit(all_targets_arr)

    scaler_path = output_dir / "thermal_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(target_scaler, f)
    logger.info("Target scaler saved to %s", scaler_path)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from strap.thermal_ml.model import ThermalPropertyPredictor

    model = ThermalPropertyPredictor()
    model.to(device)

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_path = output_dir / "polybert_thermal_checkpoint.pt"

    if args.resume and checkpoint_path.is_file():
        logger.info("Resuming from checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        patience_counter = checkpoint.get("patience_counter", 0)
        logger.info(
            "Resumed at epoch %d (best_val_loss=%.6f, patience=%d)",
            start_epoch, best_val_loss, patience_counter,
        )

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_build_lr_lambda(args.warmup_steps, total_steps),
    )

    criterion = nn.MSELoss(reduction="none")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_model_path = output_dir / "polybert_thermal_best.pt"
    training_log: list[dict] = []
    early_stop_patience = 10

    # Phase 1: freeze backbone
    if start_epoch < args.freeze_epochs:
        _freeze_backbone(model)
        logger.info(
            "Phase 1: training heads only for epochs %d-%d.",
            start_epoch, args.freeze_epochs - 1,
        )

    t_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Phase transition: unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs:
            _unfreeze_backbone(model)
            # Rebuild optimizer to include backbone params with lower LR
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if "backbone" in name or "encoder" in name:
                    backbone_params.append(param)
                else:
                    head_params.append(param)

            optimizer = AdamW(
                [
                    {"params": backbone_params, "lr": args.lr * 0.1},
                    {"params": head_params, "lr": args.lr},
                ],
                weight_decay=0.01,
            )
            remaining_steps = (args.epochs - epoch) * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=_build_lr_lambda(0, remaining_steps),
            )
            logger.info(
                "Phase 2: end-to-end fine-tuning (backbone LR=%.2e, heads LR=%.2e).",
                args.lr * 0.1, args.lr,
            )

        # --- Train one epoch ---
        model.train()
        epoch_loss = 0.0
        task_losses: dict[str, float] = {k: 0.0 for k in _PROPERTY_NAMES}
        task_counts: dict[str, int] = {k: 0 for k in _PROPERTY_NAMES}
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Normalise to (B, n_props)
            if isinstance(outputs, dict):
                pred_tensor = torch.stack(
                    [outputs[k] for k in _PROPERTY_NAMES if k in outputs], dim=-1,
                )
            elif hasattr(outputs, "shape") and outputs.dim() == 2:
                pred_tensor = outputs
            else:
                pred_tensor = outputs

            # Masked loss (ignore NaN targets)
            valid_mask = ~torch.isnan(targets)
            per_element = criterion(pred_tensor, targets.nan_to_num(0.0))
            masked_loss = (per_element * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)

            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += masked_loss.item()
            n_batches += 1

            # Track per-task losses
            with torch.no_grad():
                for i, key in enumerate(_PROPERTY_NAMES):
                    if i >= per_element.shape[1]:
                        break
                    task_mask = valid_mask[:, i]
                    if task_mask.any():
                        task_losses[key] += per_element[:, i][task_mask].mean().item()
                        task_counts[key] += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # --- Validation ---
        val_result = _validate(model, val_loader, device, target_scaler)
        val_loss = val_result["loss"]

        # Log epoch
        epoch_log = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "task_train_losses": {
                k: task_losses[k] / max(task_counts[k], 1) for k in _PROPERTY_NAMES
            },
            "val_metrics": val_result["metrics"],
            "elapsed_s": time.time() - t_start,
        }
        training_log.append(epoch_log)

        # Format validation metrics for logging
        metrics_str_parts = []
        for prop, m in val_result["metrics"].items():
            metrics_str_parts.append(
                f"{prop}: RMSE={m['rmse']:.2f}{m['unit']}, MAE={m['mae']:.2f}{m['unit']}"
            )
        metrics_str = " | ".join(metrics_str_parts)

        logger.info(
            "Epoch %3d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e  %s",
            epoch + 1, args.epochs, avg_train_loss, val_loss,
            optimizer.param_groups[0]["lr"], metrics_str,
        )

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info("  -> New best model saved (val_loss=%.6f).", val_loss)
        else:
            patience_counter += 1

        # Save resumable checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter,
            },
            checkpoint_path,
        )

        # --- Early stopping ---
        if patience_counter >= early_stop_patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d exceeded).",
                epoch + 1, early_stop_patience,
            )
            break

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    total_time = time.time() - t_start
    logger.info("Training complete in %.1f seconds.", total_time)

    # Final validation with best model
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False))
    final_val = _validate(model, val_loader, device, target_scaler)

    metadata = {
        "training_config": {
            "data_path": str(args.data_path),
            "epochs_completed": len(training_log),
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "freeze_epochs": args.freeze_epochs,
            "val_split": args.val_split,
            "seed": args.seed,
        },
        "dataset_stats": {
            "total_samples": len(full_dataset),
            "train_samples": train_size,
            "val_samples": val_size,
        },
        "best_val_loss": best_val_loss,
        "final_val_metrics": final_val["metrics"],
        "training_time_s": total_time,
        "device": device,
        "training_log": training_log,
    }

    metadata_path = output_dir / "thermal_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Metadata saved to %s", metadata_path)

    logger.info("Artifacts saved:")
    logger.info("  Best model:   %s", best_model_path)
    logger.info("  Scaler:       %s", scaler_path)
    logger.info("  Metadata:     %s", metadata_path)
    logger.info("  Checkpoint:   %s", checkpoint_path)

    # Print final metrics summary
    logger.info("--- Final validation metrics (best model) ---")
    for prop, m in final_val["metrics"].items():
        logger.info(
            "  %s: RMSE = %.4f %s, MAE = %.4f %s (n=%d)",
            prop, m["rmse"], m["unit"], m["mae"], m["unit"], m["n_samples"],
        )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Parameters
    ----------
    argv : list[str] or None
        Arguments to parse.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train the polyBERT thermal property predictor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(_DEFAULT_DATA_PATH),
        help="Path to the combined thermal property CSV dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Directory to save model weights, scaler, and metadata.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate.")
    parser.add_argument(
        "--warmup_steps", type=int, default=100,
        help="Linear warmup steps before cosine decay.",
    )
    parser.add_argument(
        "--freeze_epochs", type=int, default=5,
        help="Number of initial epochs to freeze the polyBERT backbone.",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.15,
        help="Fraction of data reserved for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for ``python -m strap.thermal_ml.train``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    logger.info("Parsed arguments: %s", args)
    train(args)


if __name__ == "__main__":
    main()
