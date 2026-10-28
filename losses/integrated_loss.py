"""
Integrated Contrastive Loss (Eq. 8 in MAGNN paper).

Combines the three contrastive signals into a single weighted objective:
  L_integrated = λ1 * L_TN + λ2 * L_EL + λ3 * L_MHHC
"""

import torch
import torch.nn as nn
from .temporal_node_loss import TemporalNodeLoss
from .edge_level_loss import EdgeLevelLoss
from .mhhc_loss import MHHCLoss


class IntegratedContrastiveLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        lambda1: float = 1.0,   # temporal-node weight
        lambda2: float = 1.0,   # edge-level weight
        lambda3: float = 1.0,   # MHHC weight
        num_heads: int = 4,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.tn_loss = TemporalNodeLoss(temperature=temperature)
        self.el_loss = EdgeLevelLoss(temperature=temperature)
        self.mhhc_loss = MHHCLoss(num_heads=num_heads, temperature=temperature)

    def forward(
        self,
        # Temporal-node inputs
        h_t1: torch.Tensor,
        h_t2: torch.Tensor,
        # Edge-level inputs
        h_orig: torch.Tensor,
        edge_index_orig: torch.Tensor,
        h_aug: torch.Tensor,
        edge_index_aug: torch.Tensor,
        # MHHC inputs
        head_v1: torch.Tensor,
        head_v2: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        l_tn = self.tn_loss(h_t1, h_t2)
        l_el = self.el_loss(h_orig, edge_index_orig, h_aug, edge_index_aug)
        l_mhhc = self.mhhc_loss(head_v1, head_v2, batch)

        return self.lambda1 * l_tn + self.lambda2 * l_el + self.lambda3 * l_mhhc
