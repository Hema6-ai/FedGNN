"""
FedMAGNN experiment driver.

Modes:
  local    — each silo trains alone, no communication (lower bound)
  fedavg   — federated averaging of the shared encoder
  fedprox  — fedavg + proximal term (mu > 0)

Flags compose the ablation matrix:
  --align            enable DP prototype alignment (cross-silo contrastive)
  --dp_sigma S       Gaussian noise multiplier on model updates (0 = off)
  --dp_clip C        L2 clip norm for client deltas
  --share full       share all params instead of encoder-only

Examples:
  python run_federated.py --mode local   --tag local_only
  python run_federated.py --mode fedavg  --tag fedavg
  python run_federated.py --mode fedavg  --align --tag fedavg_align
  python run_federated.py --mode fedprox --align --dp_sigma 1.0 --tag full_dp
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml

from utils.seed import set_seed
from utils.logger import get_logger
from fedmagnn import load_clients_data, FedClient, FedServer, GaussianDPMechanism


def parse_args():
    p = argparse.ArgumentParser(description="FedMAGNN federated training")
    p.add_argument("--config", default="configs/fed_default.yaml")
    p.add_argument("--mode", choices=["local", "fedavg", "fedprox"], default="fedavg")
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--local_epochs", type=int, default=None)
    p.add_argument("--sample", type=int, default=None, help="Subsample flows per split (dev)")
    p.add_argument("--datasets", nargs="+", default=None, help="Subset of silos (dev)")
    p.add_argument("--share", choices=["encoder", "full"], default=None)
    p.add_argument("--align", action="store_true")
    p.add_argument("--dp_sigma", type=float, default=None)
    p.add_argument("--dp_clip", type=float, default=None)
    p.add_argument("--fedprox_mu", type=float, default=None)
    p.add_argument("--zeroshot_silo", default=None,
                   help="Silo trained with NO malicious examples (zero-shot transfer experiment)")
    p.add_argument("--data_root", default=None)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default=None, help="Results filename tag")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fed = cfg["federated"]
    if args.rounds is not None:        fed["rounds"] = args.rounds
    if args.local_epochs is not None:  fed["local_epochs"] = args.local_epochs
    if args.share is not None:         fed["share_mode"] = args.share
    if args.dp_sigma is not None:      fed["dp_sigma"] = args.dp_sigma
    if args.dp_clip is not None:       fed["dp_clip"] = args.dp_clip
    if args.fedprox_mu is not None:    fed["fedprox_mu"] = args.fedprox_mu
    if args.align:                     fed["align"] = True

    mode = args.mode
    if mode != "fedprox":
        fed["fedprox_mu"] = 0.0

    set_seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    tag = args.tag or mode
    os.makedirs(args.results_dir, exist_ok=True)
    logger = get_logger(name=f"fedmagnn_{tag}", log_dir=args.results_dir)
    logger.info(f"Mode: {mode} | tag: {tag} | device: {device}")
    logger.info(f"Federated config: {json.dumps(fed)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    clients_data = load_clients_data(
        data_root=args.data_root,
        num_snapshots=cfg["dataset"].get("num_snapshots", 5),
        sample_size=args.sample,
        seed=args.seed,
        datasets=args.datasets,
        zeroshot_silo=args.zeroshot_silo,
    )
    for cd in clients_data:
        logger.info(f"  silo {cd.name}: train_nodes={cd.train_graph.num_nodes} "
                    f"val_nodes={cd.val_graph.num_nodes} test_nodes={cd.test_graph.num_nodes} "
                    f"malicious_ratio={float(cd.train_graph.y.float().mean()):.3f}")

    # ── Clients ──────────────────────────────────────────────────────────────
    clients = [
        FedClient(
            client_data=cd,
            cfg=cfg,
            device=device,
            share_mode=fed.get("share_mode", "encoder"),
            fedprox_mu=fed.get("fedprox_mu", 0.0),
            align_weight=fed.get("align_weight", 0.5) if fed.get("align") else 0.0,
            align_temperature=fed.get("align_temperature", 0.1),
            proto_clip=fed.get("proto_clip", 1.0),
            proto_noise_multiplier=fed.get("proto_sigma", fed.get("dp_sigma", 0.0)),
            seed=args.seed,
        )
        for cd in clients_data
    ]

    rounds = fed.get("rounds", 30)
    local_epochs = fed.get("local_epochs", 2)

    t0 = time.time()
    if mode == "local":
        # No communication: each silo trains rounds*local_epochs epochs alone.
        results = run_local_baseline(clients, rounds * local_epochs, logger)
    else:
        dp = None
        if fed.get("dp_sigma", 0.0) > 0:
            dp = GaussianDPMechanism(
                clip_norm=fed.get("dp_clip", 1.0),
                noise_multiplier=fed["dp_sigma"],
                delta=fed.get("dp_delta", 1e-5),
                seed=args.seed,
            )
        server = FedServer(
            clients=clients,
            rounds=rounds,
            local_epochs=local_epochs,
            dp=dp,
            use_alignment=fed.get("align", False),
            zeroshot_silo=args.zeroshot_silo,
            logger=logger,
        )
        results = server.run()

    results["wall_time_sec"] = time.time() - t0
    results["mode"] = mode
    results["federated_config"] = fed
    results["seed"] = args.seed

    out_path = os.path.join(args.results_dir, f"fed_results_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


def run_local_baseline(clients, total_epochs, logger):
    """Isolated per-silo training — the no-federation lower bound."""
    import copy
    test, best_states, best_f1s = {}, {}, {}
    history = []
    for client in clients:
        best_f1, best_sd = -1.0, None
        for ep in range(1, total_epochs + 1):
            stats = client.local_train(local_epochs=1)
            val = client.evaluate_split("val")
            if val["f1"] > best_f1:
                best_f1 = val["f1"]
                best_sd = copy.deepcopy({k: v.cpu() for k, v in client.model.state_dict().items()})
            if ep % 10 == 0 or ep == total_epochs:
                logger.info(f"  {client.name} epoch {ep}/{total_epochs} "
                            f"loss={stats['total']:.4f} val_f1={val['f1']:.2f}%")
        client.model.load_state_dict(best_sd)
        test[client.name] = client.evaluate_split("test")
        best_f1s[client.name] = best_f1
        logger.info(f"{client.name}: TEST acc={test[client.name]['accuracy']:.2f} "
                    f"f1={test[client.name]['f1']:.2f}")

    mean_f1 = float(np.mean([m["f1"] for m in test.values()]))
    mean_acc = float(np.mean([m["accuracy"] for m in test.values()]))
    logger.info(f"LOCAL BASELINE | mean test f1={mean_f1:.2f}% acc={mean_acc:.2f}%")
    return {
        "test_metrics": test,
        "mean_test_f1": mean_f1,
        "mean_test_acc": mean_acc,
        "best_val_f1": best_f1s,
        "history": history,
    }


if __name__ == "__main__":
    main()
