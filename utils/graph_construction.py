"""
Host-centric graph construction from network flow records.

Nodes  = unique IP addresses (hosts)
Edges  = directed network flows between hosts
Node features: aggregated statistics per IP (packet counts, bytes, durations, etc.)
Edge features: per-flow attributes (timestamp, size, protocol, flags)

The binary node label is set to 1 (malicious) if ANY flow involving that IP is labeled
malicious. This is the standard convention for host-centric threat detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional


# Standard columns we expect after dataset-specific preprocessing
_SRC_IP = "src_ip"
_DST_IP = "dst_ip"
_LABEL = "label"          # 0=benign, 1=malicious (already binarized by loader)
_TIMESTAMP = "timestamp"  # float seconds since epoch (or flow index if absent)
_DURATION = "duration"
_PKT_COUNT = "pkt_count"
_BYTE_COUNT = "byte_count"
_PROTO = "protocol"       # integer-encoded
_SRC_PORT = "src_port"
_DST_PORT = "dst_port"
_TCP_FLAGS = "tcp_flags"  # integer bitmask

_NODE_FEATURE_COLS = [
    "out_pkt_mean", "out_pkt_std",
    "out_byte_mean", "out_byte_std",
    "out_dur_mean",
    "in_pkt_mean", "in_pkt_std",
    "in_byte_mean", "in_byte_std",
    "in_dur_mean",
    "degree_out", "degree_in",
    "proto_tcp_ratio", "proto_udp_ratio",
    "port_entropy",
]


def _port_entropy(ports: pd.Series) -> float:
    counts = ports.value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-10)).sum())


def _build_ip_map(df: pd.DataFrame) -> Dict[str, int]:
    all_ips = pd.concat([df[_SRC_IP], df[_DST_IP]]).unique()
    return {ip: idx for idx, ip in enumerate(sorted(all_ips))}


def _aggregate_node_features(df: pd.DataFrame, ip_map: Dict[str, int]) -> np.ndarray:
    n = len(ip_map)
    feat = np.zeros((n, len(_NODE_FEATURE_COLS)), dtype=np.float32)

    # Map IPs to integer indices
    src_idx = df[_SRC_IP].map(ip_map).values
    dst_idx = df[_DST_IP].map(ip_map).values

    pkt = df[_PKT_COUNT].values.astype(np.float32) if _PKT_COUNT in df.columns else np.ones(len(df), np.float32)
    byt = df[_BYTE_COUNT].values.astype(np.float32) if _BYTE_COUNT in df.columns else np.full(len(df), 64.0, np.float32)
    dur = df[_DURATION].values.astype(np.float32) if _DURATION in df.columns else np.zeros(len(df), np.float32)
    proto = df[_PROTO].values.astype(np.float32) if _PROTO in df.columns else np.zeros(len(df), np.float32)
    dport = df[_DST_PORT].values.astype(np.float32) if _DST_PORT in df.columns else np.zeros(len(df), np.float32)

    # Vectorized accumulation using numpy.add.at
    out_pkt_sum = np.zeros(n, np.float32); out_pkt_sq = np.zeros(n, np.float32)
    out_byte_sum = np.zeros(n, np.float32); out_byte_sq = np.zeros(n, np.float32)
    out_dur_sum = np.zeros(n, np.float32); out_cnt = np.zeros(n, np.float32)
    in_pkt_sum = np.zeros(n, np.float32); in_pkt_sq = np.zeros(n, np.float32)
    in_byte_sum = np.zeros(n, np.float32); in_byte_sq = np.zeros(n, np.float32)
    in_dur_sum = np.zeros(n, np.float32); in_cnt = np.zeros(n, np.float32)
    tcp_cnt = np.zeros(n, np.float32); udp_cnt = np.zeros(n, np.float32); all_cnt = np.zeros(n, np.float32)
    port_entropy_arr = np.zeros(n, np.float32)

    np.add.at(out_pkt_sum, src_idx, pkt); np.add.at(out_pkt_sq, src_idx, pkt**2)
    np.add.at(out_byte_sum, src_idx, byt); np.add.at(out_byte_sq, src_idx, byt**2)
    np.add.at(out_dur_sum, src_idx, dur); np.add.at(out_cnt, src_idx, 1.0)
    np.add.at(in_pkt_sum, dst_idx, pkt); np.add.at(in_pkt_sq, dst_idx, pkt**2)
    np.add.at(in_byte_sum, dst_idx, byt); np.add.at(in_byte_sq, dst_idx, byt**2)
    np.add.at(in_dur_sum, dst_idx, dur); np.add.at(in_cnt, dst_idx, 1.0)
    np.add.at(tcp_cnt, src_idx, (proto == 6).astype(np.float32))
    np.add.at(udp_cnt, src_idx, (proto == 17).astype(np.float32))
    np.add.at(all_cnt, src_idx, 1.0); np.add.at(all_cnt, dst_idx, 1.0)

    # Port entropy per source IP — fully vectorized
    if _DST_PORT in df.columns and dport.max() > 0:
        d_int = dport.astype(np.int64)
        # Encode each (src, dport) pair as a single int64 key
        max_port = int(d_int.max()) + 1
        pair_key = src_idx.astype(np.int64) * max_port + d_int
        pair_unique, pair_counts = np.unique(pair_key, return_counts=True)
        pair_src = pair_unique // max_port
        # Total count per src
        src_total = np.zeros(n, np.float64)
        np.add.at(src_total, pair_src, pair_counts.astype(np.float64))
        # Probability per (src, port) pair
        probs = pair_counts / src_total[pair_src]
        ent_contrib = -(probs * np.log2(probs + 1e-10))
        np.add.at(port_entropy_arr, pair_src, ent_contrib.astype(np.float32))

    def safe_mean(s, c): return np.where(c > 0, s / np.maximum(c, 1), 0.0)
    def safe_std(sq, s, c): return np.where(c > 0, np.sqrt(np.maximum(sq / np.maximum(c, 1) - (s / np.maximum(c, 1))**2, 0.0)), 0.0)

    feat[:, 0] = safe_mean(out_pkt_sum, out_cnt)
    feat[:, 1] = safe_std(out_pkt_sq, out_pkt_sum, out_cnt)
    feat[:, 2] = safe_mean(out_byte_sum, out_cnt)
    feat[:, 3] = safe_std(out_byte_sq, out_byte_sum, out_cnt)
    feat[:, 4] = safe_mean(out_dur_sum, out_cnt)
    feat[:, 5] = safe_mean(in_pkt_sum, in_cnt)
    feat[:, 6] = safe_std(in_pkt_sq, in_pkt_sum, in_cnt)
    feat[:, 7] = safe_mean(in_byte_sum, in_cnt)
    feat[:, 8] = safe_std(in_byte_sq, in_byte_sum, in_cnt)
    feat[:, 9] = safe_mean(in_dur_sum, in_cnt)
    feat[:, 10] = out_cnt
    feat[:, 11] = in_cnt
    feat[:, 12] = np.where(all_cnt > 0, tcp_cnt / np.maximum(all_cnt, 1), 0.0)
    feat[:, 13] = np.where(all_cnt > 0, udp_cnt / np.maximum(all_cnt, 1), 0.0)
    feat[:, 14] = port_entropy_arr
    return feat


def _build_node_labels(df: pd.DataFrame, ip_map: Dict[str, int]) -> np.ndarray:
    """Label an IP malicious if ANY flow involving it is malicious.

    Vectorized: collect IPs of malicious flows once instead of scanning the
    whole DataFrame per IP (the per-IP scan is O(N*F) — ~1.5h at 134k nodes).
    """
    labels = np.zeros(len(ip_map), dtype=np.int64)
    mal = df[df[_LABEL] > 0]
    mal_ips = pd.concat([mal[_SRC_IP], mal[_DST_IP]]).unique()
    for ip in mal_ips:
        idx = ip_map.get(ip)
        if idx is not None:
            labels[idx] = 1
    return labels


def build_host_graph(df: pd.DataFrame) -> Data:
    """
    Convert a DataFrame of flow records into a PyG Data object.

    The DataFrame must have columns defined by the _SRC_IP, _DST_IP, _LABEL
    constants (plus optional feature columns). Dataset-specific loaders are
    responsible for renaming columns to this schema before calling this function.

    Returns:
        torch_geometric.data.Data with:
          .x           — node feature matrix  (N, F)
          .edge_index  — directed edges       (2, E)
          .edge_attr   — edge feature matrix  (E, 4): [timestamp, pkt_count, protocol, tcp_flags]
          .y           — node labels          (N,)
          .num_nodes   — N
    """
    ip_map = _build_ip_map(df)
    n = len(ip_map)

    # Node features
    node_feat = _aggregate_node_features(df, ip_map)

    # Node labels
    node_labels = _build_node_labels(df, ip_map)

    # Edges
    src_ids = df[_SRC_IP].map(ip_map).values
    dst_ids = df[_DST_IP].map(ip_map).values
    edge_index = np.stack([src_ids, dst_ids], axis=0).astype(np.int64)

    # Edge features: [normalized_timestamp, pkt_count, protocol, tcp_flags]
    ts = df[_TIMESTAMP].values.astype(np.float32) if _TIMESTAMP in df else np.arange(len(df), dtype=np.float32)
    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
    pkt = df[_PKT_COUNT].values.astype(np.float32) if _PKT_COUNT in df else np.zeros(len(df), np.float32)
    proto = df[_PROTO].values.astype(np.float32) if _PROTO in df else np.zeros(len(df), np.float32)
    flags = df[_TCP_FLAGS].values.astype(np.float32) if _TCP_FLAGS in df else np.zeros(len(df), np.float32)
    edge_attr = np.stack([ts_norm, pkt, proto, flags], axis=1).astype(np.float32)

    data = Data(
        x=torch.tensor(node_feat, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(node_labels, dtype=torch.long),
        num_nodes=n,
    )
    return data


def build_flow_graph(
    df: pd.DataFrame,
    feature_cols: list,
    knn_k: int = 5,
    chain_by_host: bool = False,
) -> Data:
    """
    Flow-level graph for datasets where host-centric graphs degenerate
    (ISCXVPN has no real IPs at all; DoHBrw has only ~15 unique hosts).

    Nodes  = individual flows; features = `feature_cols` (already normalized).
    Labels = per-flow labels (no aggregation).
    Edges  = k-NN in feature space (labels never used — no leakage),
             plus optional temporal chains linking consecutive flows of the
             same src/dst host when real IPs exist (chain_by_host=True).
    """
    from sklearn.neighbors import NearestNeighbors

    df = df.sort_values(_TIMESTAMP).reset_index(drop=True) if _TIMESTAMP in df.columns else df.reset_index(drop=True)
    n = len(df)
    x = df[feature_cols].values.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[_LABEL].values.astype(np.int64)

    srcs, dsts = [], []

    # k-NN feature-similarity edges (directed i -> neighbor)
    if n > knn_k + 1:
        nn = NearestNeighbors(n_neighbors=knn_k + 1, n_jobs=-1).fit(x)
        _, idx = nn.kneighbors(x)
        nbr = idx[:, 1:]                      # drop self
        srcs.append(np.repeat(np.arange(n), knn_k))
        dsts.append(nbr.reshape(-1))

    # Temporal same-host chains (real IPs required)
    if chain_by_host:
        for col in (_SRC_IP, _DST_IP):
            if col in df.columns:
                grp = df.groupby(col).indices
                for positions in grp.values():
                    if len(positions) > 1:
                        srcs.append(positions[:-1])
                        dsts.append(positions[1:])

    if srcs:
        edge_index = np.stack([np.concatenate(srcs), np.concatenate(dsts)]).astype(np.int64)
        edge_index = np.unique(edge_index, axis=1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    # Edge features from the source flow: [ts_norm, pkt_count, protocol, tcp_flags]
    ts = df[_TIMESTAMP].values.astype(np.float32) if _TIMESTAMP in df.columns else np.arange(n, dtype=np.float32)
    ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
    pkt = df[_PKT_COUNT].values.astype(np.float32) if _PKT_COUNT in df.columns else np.zeros(n, np.float32)
    proto = df[_PROTO].values.astype(np.float32) if _PROTO in df.columns else np.zeros(n, np.float32)
    flags = df[_TCP_FLAGS].values.astype(np.float32) if _TCP_FLAGS in df.columns else np.zeros(n, np.float32)
    e_src = edge_index[0]
    edge_attr = np.stack([ts_norm[e_src], pkt[e_src], proto[e_src], flags[e_src]], axis=1).astype(np.float32)

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        num_nodes=n,
    )


def temporal_graph_snapshots(
    df: pd.DataFrame,
    num_snapshots: int = 5,
    graph_mode: str = "host",
    flow_feature_cols: list = None,
    chain_by_host: bool = False,
) -> list[Data]:
    """
    Split flow DataFrame into equal time windows and build one graph per window.
    Used by the temporal-node contrastive loss.
    """
    if _TIMESTAMP not in df.columns:
        df = df.copy()
        df[_TIMESTAMP] = np.arange(len(df), dtype=np.float64)

    df = df.sort_values(_TIMESTAMP).reset_index(drop=True)
    splits = np.array_split(df, num_snapshots)
    if graph_mode == "flow":
        return [build_flow_graph(part, flow_feature_cols, chain_by_host=chain_by_host)
                for part in splits if len(part) > 1]
    return [build_host_graph(part) for part in splits if len(part) > 1]
