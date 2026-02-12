"""polyBERT multitask fine-tuning model for polymer thermal property prediction.

Fine-tunes polyBERT (kuelumbus/polyBERT, a DeBERTa-based transformer pretrained
on 80M polymer SMILES) with multitask regression heads to predict polymer thermal
properties from PSMILES strings.

Architecture
------------
PSMILES -> polyBERT Encoder (768-dim CLS token)
    -> Shared MLP (768 -> 256 -> 128)
    -> Head 1: Tm residual (vs group contribution baseline)
    -> Head 2: delta_Hf residual (vs group contribution baseline)
    -> Head 3: delta_Cp residual (vs group contribution baseline)
    -> Head 4: Tg (direct prediction, auxiliary task)

Heads 1-3 use residual learning against Van Krevelen group contribution
baselines. The uncertainty-weighted multitask loss (Kendall & Gal, 2018)
automatically balances losses across tasks with different data sizes and scales.

MC Dropout is supported for uncertainty quantification at inference time.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for ThermalPropertyPredictor. "
            "Install with: pip install torch"
        )


def _require_transformers() -> None:
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "HuggingFace Transformers is required for ThermalPropertyPredictor. "
            "Install with: pip install transformers"
        )


TASK_NAMES: list[str] = ["Tm", "delta_Hf", "delta_Cp", "Tg"]
RESIDUAL_TASKS: list[str] = ["Tm", "delta_Hf", "delta_Cp"]
DIRECT_TASKS: list[str] = ["Tg"]
DEFAULT_BACKBONE = "kuelumbus/polyBERT"


class ThermalPropertyPredictor(nn.Module):
    """Multitask regression model for polymer thermal properties.

    Fine-tunes polyBERT with shared MLP layers and per-task regression heads.
    Supports residual learning against group contribution baselines and
    uncertainty-weighted multitask loss.

    Parameters
    ----------
    backbone_name : str
        HuggingFace model name or path for the polyBERT encoder.
    dropout : float
        Dropout probability for shared MLP layers and MC Dropout.
    n_tasks : int
        Number of prediction tasks (default 4: Tm, delta_Hf, delta_Cp, Tg).
    hidden_dim : int
        Hidden dimension of the encoder (768 for polyBERT / DeBERTa-base).
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        dropout: float = 0.1,
        n_tasks: int = 4,
        hidden_dim: int = 768,
    ) -> None:
        _require_torch()
        super().__init__()

        self.backbone_name = backbone_name
        self.dropout_p = dropout
        self.n_tasks = n_tasks
        self.hidden_dim = hidden_dim

        # Encoder — loaded lazily by from_polybert / from_pretrained
        self.encoder: nn.Module | None = None

        # Shared MLP: 768 -> 256 -> 128
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Per-task regression heads
        self.head_Tm = nn.Linear(128, 1)
        self.head_delta_Hf = nn.Linear(128, 1)
        self.head_delta_Cp = nn.Linear(128, 1)
        self.head_Tg = nn.Linear(128, 1)

        self._heads = {
            "Tm": self.head_Tm,
            "delta_Hf": self.head_delta_Hf,
            "delta_Cp": self.head_delta_Cp,
            "Tg": self.head_Tg,
        }

        # Learnable log-variance parameters for uncertainty-weighted loss
        # (Kendall & Gal, 2018). We parameterise as log(sigma^2) for
        # numerical stability.
        self.log_vars = nn.ParameterDict(
            {task: nn.Parameter(torch.zeros(1)) for task in TASK_NAMES}
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_baselines: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        input_ids : Tensor [B, L]
            Tokenised PSMILES input ids.
        attention_mask : Tensor [B, L]
            Attention mask for the tokeniser output.
        group_baselines : dict, optional
            Dict mapping residual task names ("Tm", "delta_Hf", "delta_Cp")
            to tensors of shape [B, 1] with group contribution estimates.
            When provided the model adds its residual prediction to the
            baseline. When absent the model predicts directly.

        Returns
        -------
        dict[str, Tensor]
            Predictions keyed by task name, each of shape [B, 1].
        """
        if self.encoder is None:
            raise RuntimeError(
                "Encoder not loaded. Use ThermalPropertyPredictor.from_polybert() "
                "or from_pretrained() to initialise the model."
            )

        # Encode — use CLS token (first token) representation
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # DeBERTa returns last_hidden_state; take CLS token
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]  # [B, 768]

        # Shared MLP
        shared_features = self.shared_mlp(cls_embedding)  # [B, 128]

        # Task heads
        predictions: dict[str, torch.Tensor] = {}
        for task_name, head in self._heads.items():
            raw = head(shared_features)  # [B, 1]

            # Add group contribution baseline for residual tasks
            if task_name in RESIDUAL_TASKS and group_baselines is not None:
                baseline = group_baselines.get(task_name)
                if baseline is not None:
                    raw = raw + baseline

            predictions[task_name] = raw

        return predictions

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Uncertainty-weighted multitask loss (Kendall & Gal, 2018).

        L = sum_task [ 1/(2*sigma_task^2) * MSE_task + log(sigma_task) ]

        Parameters
        ----------
        predictions : dict[str, Tensor [B, 1]]
            Model predictions for each task.
        targets : dict[str, Tensor [B, 1]]
            Ground truth values for each task.
        mask : dict[str, Tensor [B]]
            Boolean mask indicating which samples have valid labels for
            each task.

        Returns
        -------
        total_loss : Tensor
            Scalar combined loss.
        per_task_losses : dict[str, Tensor]
            Unweighted MSE loss for each task (for logging).
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        per_task_losses: dict[str, torch.Tensor] = {}

        for task in TASK_NAMES:
            if task not in predictions or task not in targets:
                continue

            task_mask = mask[task]  # [B]
            if not task_mask.any():
                per_task_losses[task] = torch.tensor(0.0, device=total_loss.device)
                continue

            pred = predictions[task][task_mask]  # [N_valid, 1]
            tgt = targets[task][task_mask]  # [N_valid, 1]

            mse = F.mse_loss(pred, tgt)
            per_task_losses[task] = mse.detach()

            # Uncertainty weighting
            log_var = self.log_vars[task]  # log(sigma^2)
            precision = torch.exp(-log_var)  # 1/sigma^2
            weighted = 0.5 * precision * mse + 0.5 * log_var

            total_loss = total_loss + weighted

        return total_loss, per_task_losses

    # ------------------------------------------------------------------
    # Uncertainty estimation via MC Dropout
    # ------------------------------------------------------------------

    def predict_with_uncertainty(
        self,
        psmiles_list: list[str],
        tokenizer: Any,
        group_baselines: dict[str, list[float]] | None = None,
        n_samples: int = 50,
        device: str = "cpu",
    ) -> dict[str, dict[str, Any]]:
        """Predict with MC Dropout uncertainty estimation.

        Runs *n_samples* stochastic forward passes (dropout kept active)
        and returns mean and standard deviation for each property.

        Parameters
        ----------
        psmiles_list : list[str]
            List of PSMILES strings.
        tokenizer : PreTrainedTokenizer
            polyBERT tokenizer.
        group_baselines : dict, optional
            Dict mapping residual task names to lists of baseline values
            (one per PSMILES string).
        n_samples : int
            Number of MC Dropout forward passes.
        device : str
            Device to run inference on.

        Returns
        -------
        dict[str, dict[str, ndarray]]
            ``{property: {"mean": array, "std": array}}`` for each task.
        """
        import numpy as np

        self.to(device)

        # Tokenise
        encoded = tokenizer(
            psmiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Build baseline tensors
        baseline_tensors: dict[str, torch.Tensor] | None = None
        if group_baselines is not None:
            baseline_tensors = {}
            for key in RESIDUAL_TASKS:
                if key in group_baselines:
                    baseline_tensors[key] = torch.tensor(
                        group_baselines[key], dtype=torch.float32, device=device
                    ).unsqueeze(-1)

        # Enable dropout for MC sampling
        self.train()  # activates dropout
        # But freeze batch-norm etc. if any — encoder likely has LayerNorm which
        # behaves the same in train/eval, so this is fine.

        all_predictions: dict[str, list[torch.Tensor]] = {t: [] for t in TASK_NAMES}

        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.forward(input_ids, attention_mask, baseline_tensors)
                for task in TASK_NAMES:
                    all_predictions[task].append(preds[task].cpu())

        self.eval()  # restore eval mode

        results: dict[str, dict[str, Any]] = {}
        for task in TASK_NAMES:
            stacked = torch.cat(
                [p.unsqueeze(0) for p in all_predictions[task]], dim=0
            )  # [n_samples, B, 1]
            stacked = stacked.squeeze(-1)  # [n_samples, B]
            results[task] = {
                "mean": stacked.mean(dim=0).numpy(),
                "std": stacked.std(dim=0).numpy(),
            }

        return results

    # ------------------------------------------------------------------
    # Backbone freezing
    # ------------------------------------------------------------------

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze the polyBERT encoder parameters.

        Parameters
        ----------
        freeze : bool
            If True, sets ``requires_grad=False`` for all encoder parameters.
            If False, unfreezes them.
        """
        if self.encoder is None:
            logger.warning("Encoder not loaded; nothing to freeze.")
            return
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
        action = "Frozen" if freeze else "Unfrozen"
        logger.info("%s polyBERT encoder (%s parameters).", action, sum(1 for _ in self.encoder.parameters()))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_polybert(
        cls,
        backbone_name: str = DEFAULT_BACKBONE,
        dropout: float = 0.1,
    ) -> "ThermalPropertyPredictor":
        """Initialise from HuggingFace polyBERT weights.

        Downloads the pretrained polyBERT encoder from HuggingFace Hub
        and attaches freshly-initialised MLP heads.

        Parameters
        ----------
        backbone_name : str
            HuggingFace model identifier.
        dropout : float
            Dropout probability.

        Returns
        -------
        ThermalPropertyPredictor
            Model with pretrained encoder and fresh heads.
        """
        _require_transformers()
        model = cls(backbone_name=backbone_name, dropout=dropout)
        logger.info("Loading polyBERT encoder from '%s' ...", backbone_name)
        model.encoder = AutoModel.from_pretrained(backbone_name)
        logger.info("polyBERT encoder loaded successfully.")
        return model

    def save_pretrained(self, path: str | Path) -> None:
        """Save model weights and config to a directory.

        Parameters
        ----------
        path : str or Path
            Directory to save into (created if needed).
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save full state dict
        torch.save(self.state_dict(), path / "model.pt")

        # Save config
        config = {
            "backbone_name": self.backbone_name,
            "dropout": self.dropout_p,
            "n_tasks": self.n_tasks,
            "hidden_dim": self.hidden_dim,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save learned log-variance values for inspection
        log_var_values = {k: v.item() for k, v in self.log_vars.items()}
        with open(path / "log_variances.json", "w") as f:
            json.dump(log_var_values, f, indent=2)

        logger.info("Model saved to %s", path)

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "ThermalPropertyPredictor":
        """Load a previously saved model.

        Parameters
        ----------
        path : str or Path
            Directory containing ``model.pt`` and ``config.json``.

        Returns
        -------
        ThermalPropertyPredictor
            Model with loaded weights.
        """
        _require_torch()
        _require_transformers()

        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        model = cls(
            backbone_name=config["backbone_name"],
            dropout=config.get("dropout", 0.1),
            n_tasks=config.get("n_tasks", 4),
            hidden_dim=config.get("hidden_dim", 768),
        )

        # Load encoder from HuggingFace first to build the architecture,
        # then overwrite all weights from checkpoint
        model.encoder = AutoModel.from_pretrained(config["backbone_name"])
        state_dict = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        logger.info("Model loaded from %s", path)
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_learned_task_weights(self) -> dict[str, float]:
        """Return the effective task weights derived from learned log-variances.

        Returns
        -------
        dict[str, float]
            ``{task: 1/(2*sigma^2)}`` for each task — higher means the
            model is weighting that task more heavily.
        """
        weights = {}
        for task in TASK_NAMES:
            log_var = self.log_vars[task].item()
            sigma_sq = math.exp(log_var)
            weights[task] = 1.0 / (2.0 * sigma_sq)
        return weights

    def __repr__(self) -> str:
        encoder_status = "loaded" if self.encoder is not None else "not loaded"
        return (
            f"ThermalPropertyPredictor("
            f"backbone='{self.backbone_name}', "
            f"encoder={encoder_status}, "
            f"dropout={self.dropout_p}, "
            f"tasks={TASK_NAMES})"
        )
