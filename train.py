"""
Unified training script for MAGNN and MAGNN-Plus.

Usage examples:
  # Train base MAGNN on CTU-13:
  python train.py --config configs/ctu13.yaml --model magnn

  # Train MAGNN-Plus on CICIDS-2017:
  python train.py --config configs/cicids.yaml --model magnn_plus

  # Quick dev run (subsample 5000 flows, 10 epochs):
  python train.py --config configs/ctu13.yaml --model magnn_plus --sample 5000 --epochs 10

  # Evaluate only (load checkpoint):
  python train.py --config configs/ctu13.yaml --model magnn_plus --eval_only --ckpt results/best.pt
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

from utils.seed import set_seed
from utils.logger import get_logger
from utils.augmentation import edge_modification, random_walk_restart
from utils.sampling import batch_dirichlet_sample
from utils.metrics import compute_metrics, format_metrics_table
from data import get_dataset
from models import MAGNN, MAGNNPlus
from losses import IntegratedContrastiveLoss, MultiTaskLoss


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MAGNN / MAGNN-Plus training")
    p.add_argument("--config", required=True, help="Path to dataset YAML config")
    p.add_argument("--model", choices=["magnn", "magnn_plus"], default="magnn")
    p.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--sample", type=int, default=None, help="Subsample N flows (dev mode)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="cuda / cpu / auto")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--ckpt", default=None, help="Checkpoint path for eval_only")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--no_cache", action="store_true", help="Disable dataset caching")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_args(cfg: dict, args: argparse.Namespace) -> dict:
    """Command-line args override config values."""
    tr = cfg.setdefault("training", {})
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.lr is not None:
        tr["lr"] = args.lr
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size
    if args.sample is not None:
        cfg.setdefault("dataset", {})["sample_size"] = args.sample
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, cfg: dict, in_dim: int) -> nn.Module:
    mc = cfg.get("model", {})
    pc = cfg.get("magnn_plus", {})
    hidden_dim = mc.get("hidden_dim", 128)
    num_heads = mc.get("num_heads", 4)
    num_layers = mc.get("num_layers", 3)
    dropout = mc.get("dropout", 0.1)
    edge_dim = mc.get("edge_dim", 4)

    if model_name == "magnn":
        return MAGNN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    else:
        return MAGNNPlus(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=2,
            num_heads=num_heads,
            num_layers=num_layers,
            edge_dim=edge_dim,
            dropout=dropout,
            pe_dim=pc.get("pe_dim", 64),
            walk_length=pc.get("walk_length", 16),
            max_degree=pc.get("max_degree", 512),
            focal_gamma=pc.get("focal_gamma", 2.0),
            label_smoothing=pc.get("label_smoothing", 0.05),
            cross_dataset=pc.get("cross_dataset", False),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler factory
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, model_name: str, cfg: dict):
    pc = cfg.get("magnn_plus", {})
    if model_name == "magnn_plus":
        sched_type = pc.get("scheduler_type", "cosine_warm_restart")
        if sched_type == "cosine_warm_restart":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=pc.get("T_0", 10),
                T_mult=pc.get("T_mult", 2),
                eta_min=pc.get("eta_min", 1e-6),
            )
    # Base MAGNN: no LR scheduling (constant lr as in paper)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation helpers
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_device(data: Data, device: torch.device) -> Data:
    return Data(
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        edge_attr=data.edge_attr.to(device) if data.edge_attr is not None else None,
        y=data.y.to(device),
        num_nodes=data.num_nodes,
    )


def apply_dirichlet_sampling(data: Data, size: int, alpha: float, seed: int) -> Data:
    """Return a new Data with Dirichlet-sampled edges (CPU op, then move back)."""
    ei_np = data.edge_index.cpu().numpy()
    ei_sampled = batch_dirichlet_sample(ei_np, data.num_nodes, size=size, alpha=alpha, seed=seed)
    device = data.edge_index.device
    new_ei = torch.tensor(ei_sampled, dtype=torch.long, device=device)

    # Edge attr alignment: keep only edges present in sampled set
    # (Approximation: edge_attr is not re-indexed; use mean attr for training stability)
    ea = data.edge_attr
    if ea is not None and new_ei.shape[1] != ea.shape[0]:
        mean_ea = ea.mean(dim=0, keepdim=True).expand(new_ei.shape[1], -1)
        ea = mean_ea

    return Data(x=data.x, edge_index=new_ei, edge_attr=ea, y=data.y, num_nodes=data.num_nodes)


# ─────────────────────────────────────────────────────────────────────────────
# Single epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    data: Data,
    snapshots: list,
    optimizer: torch.optim.Optimizer,
    contrastive_loss_fn: IntegratedContrastiveLoss,
    task_loss_fn: MultiTaskLoss,
    cfg: dict,
    model_name: str,
    device: torch.device,
    epoch: int,
) -> dict:
    model.train()
    optimizer.zero_grad()

    tr = cfg["training"]
    aug_args = dict(
        drop_ratio=tr.get("augment_drop_ratio", 0.1),
        add_ratio=tr.get("augment_add_ratio", 0.1),
        seed=epoch,
    )
    rwr_args = dict(
        restart_prob=tr.get("rwr_restart_prob", 0.15),
        num_steps=tr.get("rwr_steps", 100),
        seed=epoch + 1000,
    )

    # Build two augmented views (on CPU, then move to device)
    data_cpu = Data(
        x=data.x.cpu(), edge_index=data.edge_index.cpu(),
        edge_attr=data.edge_attr.cpu() if data.edge_attr is not None else None,
        y=data.y.cpu(), num_nodes=data.num_nodes,
    )
    aug1 = graph_to_device(edge_modification(data_cpu, **aug_args), device)
    aug2 = graph_to_device(random_walk_restart(data_cpu, **rwr_args), device)

    # Apply Dirichlet sampling to the main graph
    sampled = apply_dirichlet_sampling(
        data,
        size=tr.get("dirichlet_size", 30),
        alpha=tr.get("dirichlet_alpha", 1.0),
        seed=epoch,
    )

    # Prepare snapshot embeddings for temporal encoder (MAGNN-Plus only)
    snap_embs = None
    if model_name == "magnn_plus" and len(snapshots) >= 2:
        model.eval()
        with torch.no_grad():
            snap_embs = []
            for snap in snapshots:
                s = graph_to_device(snap, device)
                out = model(s.x, s.edge_index, s.edge_attr)
                snap_embs.append(out["gcn_emb"])
        model.train()

    # Forward pass on main sampled graph
    out = model(sampled.x, sampled.edge_index, sampled.edge_attr, snapshots=snap_embs)
    out_aug1 = model(aug1.x, aug1.edge_index, aug1.edge_attr)
    out_aug2 = model(aug2.x, aug2.edge_index, aug2.edge_attr)

    # Align node counts across views (different augmentations may have the same N,
    # but edge-level views are sized by min(E, E') in the loss function)
    N = min(out["node_emb"].size(0), out_aug1["node_emb"].size(0), out_aug2["node_emb"].size(0))

    h_t1 = out["node_emb"][:N]
    h_t2 = out_aug1["node_emb"][:N]

    batch = torch.zeros(N, dtype=torch.long, device=device)

    # Per-head embeddings for MHHC (H, N, d)
    head_v1 = out["head_emb"][:, :N, :]
    head_v2 = out_aug2["head_emb"][:, :N, :]

    # Contrastive loss (Eq. 8)
    l_contrast = contrastive_loss_fn(
        h_t1=h_t1,
        h_t2=h_t2,
        h_orig=out["node_emb"][:N],
        edge_index_orig=sampled.edge_index,
        h_aug=out_aug1["node_emb"][:N],
        edge_index_aug=aug1.edge_index,
        head_v1=head_v1,
        head_v2=head_v2,
        batch=batch,
    )

    # Classification loss (Focal for MAGNN-Plus, CE for MAGNN)
    labels = sampled.y[:N]
    logits = out["logits"][:N]
    scores = out["scores"][:N]

    if model_name == "magnn_plus" and hasattr(model, "classification_loss"):
        # Replace standard CE in MultiTaskLoss with Focal loss
        l_cls = model.classification_loss(logits, labels)
        l_reg = torch.nn.functional.mse_loss(scores, labels.float())
        l_total = l_cls + task_loss_fn.alpha * l_reg + task_loss_fn.beta * l_contrast
        l_task = l_cls
    else:
        l_total, l_task, l_reg = task_loss_fn(logits, scores, labels, l_contrast)

    l_total.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return {
        "total": l_total.item(),
        "cls": l_task.item() if isinstance(l_task, torch.Tensor) else l_task,
        "reg": l_reg.item() if isinstance(l_reg, torch.Tensor) else l_reg,
        "contrastive": l_contrast.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, data: Data, device: torch.device) -> dict:
    model.eval()
    data = graph_to_device(data, device)
    out = model(data.x, data.edge_index, data.edge_attr)

    logits = out["logits"]
    scores = out["scores"]
    labels = data.y

    probs = torch.softmax(logits, dim=-1)[:, 1]   # P(malicious)
    preds = logits.argmax(dim=-1)

    return compute_metrics(
        y_true=labels.cpu().numpy(),
        y_pred_cls=preds.cpu().numpy(),
        y_pred_score=probs.cpu().numpy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str, metadata: dict) -> None:
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(model: nn.Module, path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    return ckpt.get("metadata", {})


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_args(cfg, args)

    set_seed(args.seed)
    device = get_device(args.device)

    os.makedirs(args.results_dir, exist_ok=True)
    logger = get_logger(
        name=f"{cfg['dataset']['name']}_{args.model}",
        log_dir=args.results_dir,
    )

    logger.info(f"Config: {args.config}  |  Model: {args.model}  |  Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds_cfg = cfg["dataset"]
    dataset = get_dataset(
        name=ds_cfg["name"],
        data_root=ds_cfg["data_root"],
        num_snapshots=ds_cfg.get("num_snapshots", 5),
        sample_size=ds_cfg.get("sample_size"),
        seed=args.seed,
        cache=not args.no_cache,
    )
    logger.info(f"Dataset loaded: {dataset}")

    train_graph, val_graph, test_graph = dataset.get_graphs()
    train_snapshots = dataset.get_snapshots("train")

    in_dim = train_graph.x.shape[1]
    logger.info(f"Node feature dim: {in_dim} | Train nodes: {train_graph.num_nodes} "
                f"| Val nodes: {val_graph.num_nodes} | Test nodes: {test_graph.num_nodes}")

    # Move training graph to device
    train_graph = graph_to_device(train_graph, device)
    val_graph = graph_to_device(val_graph, device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(args.model, cfg, in_dim).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.eval_only:
        if args.ckpt is None:
            logger.error("--ckpt required with --eval_only")
            sys.exit(1)
        load_checkpoint(model, args.ckpt)
        test_graph = graph_to_device(test_graph, device)
        metrics = evaluate(model, test_graph, device)
        logger.info(f"Test metrics:\n{json.dumps(metrics, indent=2)}")
        return

    # ── Optimizer & Loss ─────────────────────────────────────────────────────
    tr = cfg["training"]
    lc = cfg.get("loss", {})

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=tr.get("lr", 0.001),
        weight_decay=tr.get("weight_decay", 1e-5),
    )
    scheduler = build_scheduler(optimizer, args.model, cfg)

    contrastive_loss_fn = IntegratedContrastiveLoss(
        temperature=lc.get("temperature", 0.07),
        lambda1=lc.get("lambda1", 1.0),
        lambda2=lc.get("lambda2", 1.0),
        lambda3=lc.get("lambda3", 1.0),
        num_heads=cfg["model"].get("num_heads", 4),
    ).to(device)

    task_loss_fn = MultiTaskLoss(
        alpha=lc.get("alpha", 0.5),
        beta=lc.get("beta", 0.5),
        label_smoothing=0.0 if args.model == "magnn" else 0.0,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    epochs = tr.get("epochs", 100)
    best_f1 = -1.0
    best_ckpt = os.path.join(args.results_dir, f"best_{ds_cfg['name']}_{args.model}.pt")
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss_dict = train_epoch(
            model=model,
            data=train_graph,
            snapshots=train_snapshots,
            optimizer=optimizer,
            contrastive_loss_fn=contrastive_loss_fn,
            task_loss_fn=task_loss_fn,
            cfg=cfg,
            model_name=args.model,
            device=device,
            epoch=epoch,
        )
        if scheduler is not None:
            scheduler.step(epoch - 1)

        val_metrics = evaluate(model, val_graph, device)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss={loss_dict['total']:.4f} "
            f"(cls={loss_dict['cls']:.4f}, reg={loss_dict['reg']:.4f}, "
            f"ctr={loss_dict['contrastive']:.4f}) | "
            f"val_acc={val_metrics['accuracy']:.2f}% val_f1={val_metrics['f1']:.2f}% | "
            f"{elapsed:.1f}s"
        )

        history.append({"epoch": epoch, "loss": loss_dict, "val": val_metrics})

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(model, best_ckpt, {
                "epoch": epoch,
                "val_metrics": val_metrics,
                "model": args.model,
                "dataset": ds_cfg["name"],
            })
            logger.info(f"  ↑ New best val F1: {best_f1:.2f}% — checkpoint saved")

    # ── Final evaluation on test set ─────────────────────────────────────────
    logger.info("\n=== Loading best checkpoint for test evaluation ===")
    load_checkpoint(model, best_ckpt)
    test_graph = graph_to_device(test_graph, device)
    test_metrics = evaluate(model, test_graph, device)

    logger.info("\n=== TEST RESULTS ===")
    logger.info(json.dumps(test_metrics, indent=2))

    # Compare to paper targets
    targets = cfg.get("targets", {})
    if targets:
        logger.info("\n=== COMPARISON TO PAPER (Table 3) ===")
        logger.info(f"{'Metric':<12} {'Paper':>10} {'Ours':>10} {'Delta':>10}")
        for k in ["accuracy", "f1", "mse", "rmse", "mae", "mape"]:
            if k in targets and k in test_metrics:
                delta = test_metrics[k] - targets[k]
                sign = "+" if delta > 0 else ""
                # For error metrics (lower is better), negative delta = improvement
                logger.info(f"  {k:<12} {targets[k]:>10.4f} {test_metrics[k]:>10.4f} {sign}{delta:>9.4f}")

    # Save results JSON
    results_path = os.path.join(
        args.results_dir,
        f"results_{ds_cfg['name']}_{args.model}.json"
    )
    with open(results_path, "w") as f:
        json.dump({
            "config": cfg,
            "test_metrics": test_metrics,
            "best_val_f1": best_f1,
            "history": history,
        }, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
