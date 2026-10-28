"""
MAGNN-Plus: Improved variant designed to beat the paper's baseline results.

Enhancements over base MAGNN (with design rationale):

1. Graph Transformer layers instead of GCNConv
   - TransformerConv computes attention weights per edge and directly uses edge features.
   - More expressive than isotropic GCN which averages all neighbors equally.

2. Learnable positional encodings (Degree PE + RWSE)
   - Breaks the WL-isomorphism limitations of standard MPNNs.
   - Gives the model structural awareness beyond local neighborhoods.

3. Cross-dataset contrastive regularizer (disabled by default; enabled during
   multi-dataset training via the --cross_dataset flag in train.py)
   - Forces shared node representations to be invariant to dataset-specific artifacts.
   - Implemented as InfoNCE between same-label nodes from different datasets.

4. Focal Loss (γ=2) inside the classification objective
   - Handles the severe class imbalance in network traffic (benign >> malicious).
   - Combined with label smoothing (ε=0.05) to prevent over-confident predictions.

5. Transformer-based temporal encoder
   - Replaces simple embedding comparison with a sequence model over T snapshots.
   - Learns *how* representations evolve over time (trend detection).

6. Cosine annealing LR scheduler with warm restarts (T_0=10, T_mult=2)
   - Applied externally in train.py; MAGNN-Plus signals this preference via
     the `scheduler_type` attribute.

7. Label smoothing ε=0.05 on classification via FocalLoss
   - Reduces over-confidence on easy majority-class samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

from .graph_transformer import MultiScaleGraphTransformer
from .positional_encoding import LearnablePositionalEncoding
from .temporal_encoder import TemporalTransformerEncoder
from losses.focal_loss import FocalLoss


class MAGNNPlus(nn.Module):
    """
    MAGNN-Plus: drop-in replacement for MAGNN with improved components.
    All forward outputs are dict-compatible with the same keys as MAGNN
    so the training loop is shared.
    """

    scheduler_type: str = "cosine_warm_restart"   # read by train.py

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_heads: int = 4,
        num_layers: int = 3,
        edge_dim: int = 4,
        dropout: float = 0.1,
        # Positional encoding
        pe_dim: int = 64,
        walk_length: int = 16,
        max_degree: int = 512,
        # Temporal encoder
        num_time_steps: int = 5,
        temp_nhead: int = 4,
        temp_layers: int = 2,
        # Focal loss
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.05,
        # Cross-dataset contrast
        cross_dataset: bool = False,
        cross_contrast_weight: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.cross_dataset = cross_dataset
        self.cross_contrast_weight = cross_contrast_weight

        # Input projection (node feature dim → hidden_dim)
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        # Positional encoding (added to projected features before GNN)
        self.pe = LearnablePositionalEncoding(
            out_dim=hidden_dim,
            max_degree=max_degree,
            walk_length=walk_length,
            pe_dim=pe_dim,
        )

        # Graph Transformer backbone
        self.gnn = MultiScaleGraphTransformer(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            edge_dim=edge_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Per-head linear projections to split hidden_dim into num_heads views
        head_dim = hidden_dim // num_heads
        self.head_projs = nn.ModuleList([
            nn.Linear(hidden_dim, head_dim) for _ in range(num_heads)
        ])
        self.head_dim = head_dim

        # Temporal Transformer encoder
        self.temporal_enc = TemporalTransformerEncoder(
            d_model=hidden_dim,
            nhead=temp_nhead,
            num_layers=temp_layers,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            max_time_steps=num_time_steps + 2,
        )

        # Classification head with Focal Loss (loss applied externally)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Regression head (malicious score)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Focal loss (used in loss computation when the model is queried for its criterion)
        self.focal_loss = FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)

        # Cross-dataset contrastive projector
        if cross_dataset:
            self.cross_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
            )

    def get_per_head_embeddings(self, h: torch.Tensor) -> torch.Tensor:
        """Project h to (num_heads, N, head_dim) for MHHC loss."""
        per_head = torch.stack([proj(h) for proj in self.head_projs], dim=0)  # (H, N, d)
        return per_head

    def cross_dataset_contrastive(
        self,
        h_a: torch.Tensor,   # (N_a, D) embeddings from dataset A
        labels_a: torch.Tensor,
        h_b: torch.Tensor,   # (N_b, D) embeddings from dataset B
        labels_b: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Cross-dataset contrastive: nodes with the same label across two datasets
        should have similar representations (label-semantic invariance).
        """
        z_a = F.normalize(self.cross_proj(h_a), dim=-1)
        z_b = F.normalize(self.cross_proj(h_b), dim=-1)

        sim = torch.mm(z_a, z_b.T) / temperature   # (N_a, N_b)

        # Positive mask: same label
        pos_mask = (labels_a.unsqueeze(1) == labels_b.unsqueeze(0)).float()

        # InfoNCE: for each anchor in A, positives are same-label nodes in B
        exp_sim = sim.exp()
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        total_sum = exp_sim.sum(dim=1)
        loss = -torch.log(pos_sum / (total_sum + 1e-8) + 1e-8)
        return loss.mean()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        batch: torch.Tensor = None,
        snapshots: list[torch.Tensor] = None,   # pre-computed snapshot embeddings for temporal enc
    ) -> dict:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E)
            edge_attr: (E, edge_dim) edge features
            batch: (N,) node-to-graph assignment
            snapshots: list of T (N, hidden_dim) embeddings from consecutive time windows;
                       if provided, the temporal encoder runs and its output replaces h_att.
        Returns:
            dict with same keys as MAGNN.forward() plus:
              temporal_emb  (N, D) — temporally-fused embedding (if snapshots provided)
              contrast_pair (N, D) — second temporal view for contrastive loss
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        h = F.relu(self.in_proj(x))    # (N, hidden_dim)
        h0 = h                          # kept for the initial-residual skip

        # Positional encoding (added before GNN, not replacing node features)
        pe = self.pe(edge_index, x.size(0))
        h = h + pe                      # (N, hidden_dim)

        # Graph Transformer backbone
        h_gnn, layer_embs = self.gnn(h, edge_index, edge_attr)   # (N, hidden_dim)

        # Per-head embeddings for MHHC loss
        h_per_head = self.get_per_head_embeddings(h_gnn)          # (H, N, head_dim)

        # Temporal encoder (optional — used during multi-snapshot training)
        temporal_emb = None
        contrast_pair = h_gnn    # default: use same snapshot as "second view"
        if snapshots is not None and len(snapshots) >= 2:
            temporal_emb, contrast_pair = self.temporal_enc(snapshots)
            n_t = temporal_emb.size(0)
            if n_t < h_gnn.size(0):
                # Snapshots cover fewer nodes than the full graph. Truncating
                # h_final to n_t silently dropped supervision for every later
                # node — with time-sorted flow graphs those are label-skewed,
                # collapsing training to the majority class. Keep GNN
                # embeddings for nodes beyond temporal coverage instead.
                h_final = torch.cat([temporal_emb, h_gnn[n_t:]], dim=0)
                contrast_pair = torch.cat([contrast_pair, h_gnn[n_t:]], dim=0)
            else:
                h_final = temporal_emb
        else:
            h_final = h_gnn

        # Initial-residual skip (GCNII/APPNP-style): task heads see the raw
        # projected features alongside the graph embedding, so graph smoothing
        # can only add information relative to a feature-only model. On flow
        # graphs the GNN otherwise underfits far below the tabular ceiling.
        h_final = h_final + h0

        logits = self.cls_head(h_final)           # (N, num_classes)
        scores = self.reg_head(h_final).squeeze(-1)  # (N,)

        out = {
            "logits": logits,
            "scores": scores,
            "node_emb": h_final,
            "head_emb": h_per_head,
            "layer_embs": layer_embs,
            "gcn_emb": h_gnn,
            "temporal_emb": temporal_emb,
            "contrast_pair": contrast_pair,
        }
        return out

    def classification_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns Focal Loss (used in MultiTaskLoss as the cls component for MAGNN-Plus)."""
        return self.focal_loss(logits, labels)
