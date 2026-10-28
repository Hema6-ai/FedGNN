"""
Multi-Task Loss (Eq. 9 in MAGNN paper).

Combines classification and regression objectives with the integrated contrastive loss:
  L_total = L_cls + α * L_reg + β * L_integrated

  L_cls = cross-entropy on binary malicious/benign node prediction
  L_reg = MSE between predicted malicious score and binary label (0/1)
          (the multi-task regression head learns a soft malicious probability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,    # regression weight
        beta: float = 0.5,     # contrastive weight
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,          # (N, C) classification logits
        scores: torch.Tensor,          # (N,) regression scores (sigmoid output)
        labels: torch.Tensor,          # (N,) integer labels (0 or 1)
        contrastive_loss: torch.Tensor,  # scalar from IntegratedContrastiveLoss
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, cls_loss, reg_loss
        """
        l_cls = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        targets = labels.float()
        l_reg = F.mse_loss(scores, targets)

        total = l_cls + self.alpha * l_reg + self.beta * contrastive_loss
        return total, l_cls, l_reg
