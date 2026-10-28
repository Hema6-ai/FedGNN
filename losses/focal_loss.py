"""
Focal Loss for MAGNN-Plus.

Focal loss down-weights easy (well-classified) samples and focuses training
on hard examples, which is critical for class-imbalanced network traffic
datasets where benign traffic vastly outnumbers malicious traffic.

FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

where p_t is the probability of the true class, γ is the focusing parameter
(typically 2.0), and α_t is a per-class weight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # per-class weight tensor (C,) or None
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (N, C) raw logits
            labels: (N,) integer class labels
        Returns:
            scalar focal loss
        """
        C = logits.size(1)

        # Label smoothing: soft targets
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, self.label_smoothing / (C - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = F.log_softmax(logits, dim=-1)          # (N, C)
        probs = log_probs.exp()                             # (N, C)

        # p_t for the true class
        p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)  # (N,)
        focal_weight = (1.0 - p_t) ** self.gamma               # (N,)

        # Soft cross-entropy with smooth labels
        ce = -(smooth_labels * log_probs).sum(dim=-1)      # (N,)
        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[labels]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
