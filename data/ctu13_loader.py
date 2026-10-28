"""
CTU-13 Dataset Loader.

Actual data location:  CTU-13/CTU-13-Dataset/{1..13}/capture*.binetflow
Format: ARGUS biNetFlow CSV
Columns: StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,
         sTos,dTos,TotPkts,TotBytes,SrcBytes,Label

Label format examples:
  flow=Background-Established-cmpgw-CVUT   → benign (background)
  flow=Normal-P2P-File                     → benign (normal)
  flow=From-Botnet-V45-UDP-Attempt         → malicious (botnet)
  flow=To-Botnet-V45-UDP-Attempt           → malicious (botnet)

Relative path from magnn_improved/: ../CTU-13/CTU-13-Dataset/
"""

from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from pathlib import Path

from .base_dataset import BaseTrafficDataset
from .preprocessing import encode_protocol

MALICIOUS_KEYWORDS = ("botnet", "malicious", "attack")
BENIGN_KEYWORDS = ("background", "normal")


class CTU13Dataset(BaseTrafficDataset):
    name = "ctu13"
    graph_mode = "host"   # host-centric: has real SrcAddr/DstAddr

    def load_raw(self) -> pd.DataFrame:
        # Search for .binetflow files across all 13 scenario folders
        pattern = str(self.data_root / "**" / "*.binetflow")
        binetflow_files = sorted(glob.glob(pattern, recursive=True))

        if not binetflow_files:
            raise FileNotFoundError(
                f"No .binetflow files found under {self.data_root}.\n"
                f"Expected path: CTU-13/CTU-13-Dataset/{{1..13}}/capture*.binetflow"
            )

        print(f"[CTU13] Found {len(binetflow_files)} .binetflow files")
        dfs = []
        for f in binetflow_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                dfs.append(df)
                print(f"  Loaded {Path(f).parent.name}/... ({len(df):,} rows)")
            except Exception as e:
                print(f"  Warning: {f}: {e}")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"[CTU13] Total rows: {len(combined):,}")
        return combined

    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Rename binetflow columns to canonical schema
        df = df.rename(columns={
            "StartTime": "timestamp_str",
            "Dur":       "duration",
            "Proto":     "protocol",
            "SrcAddr":   "src_ip",
            "Sport":     "src_port",
            "DstAddr":   "dst_ip",
            "Dport":     "dst_port",
            "TotPkts":   "pkt_count",
            "TotBytes":  "byte_count",
            "Label":     "label_raw",
        })

        # Timestamp: convert datetime string → epoch seconds
        if "timestamp_str" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp_str"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce"
            ).astype(np.int64) / 1e9
            df["timestamp"] = df["timestamp"].fillna(0.0)
        else:
            df["timestamp"] = np.arange(len(df), dtype=np.float64)

        # Protocol
        if "protocol" in df.columns:
            df["protocol"] = encode_protocol(df["protocol"])
        else:
            df["protocol"] = 6.0

        # TCP flags: not directly in binetflow → use State field as proxy
        df["tcp_flags"] = 0.0   # binetflow has State column but not numeric flags

        # Fill missing columns with defaults
        for col, default in [("src_port", 0.0), ("dst_port", 0.0),
                              ("pkt_count", 1.0), ("byte_count", 64.0), ("duration", 0.0)]:
            if col not in df.columns:
                df[col] = default

        # Ensure numeric
        for col in ["src_port", "dst_port", "pkt_count", "byte_count", "duration"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Fill missing IPs
        df["src_ip"] = df["src_ip"].fillna("0.0.0.0").astype(str)
        df["dst_ip"] = df["dst_ip"].fillna("0.0.0.1").astype(str)

        # Binarize label
        # CTU-13 format: "flow=From-Botnet-..." or "flow=Background-..." etc.
        if "label_raw" in df.columns:
            raw = df["label_raw"].astype(str).str.lower()
            is_malicious = raw.str.contains("|".join(MALICIOUS_KEYWORDS), regex=True)
            df["label"] = is_malicious.astype(np.int32)
        else:
            df["label"] = 0

        # Drop rows with NaN IPs
        df = df.dropna(subset=["src_ip", "dst_ip"]).reset_index(drop=True)

        print(f"[CTU13] After canonicalize: {len(df):,} rows | "
              f"malicious={df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")

        return df[["src_ip", "dst_ip", "label", "timestamp",
                   "duration", "pkt_count", "byte_count",
                   "protocol", "src_port", "dst_port", "tcp_flags"]]
