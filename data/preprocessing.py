"""
Common preprocessing utilities shared across all dataset loaders.

Responsibilities:
  - RobustScaler normalization (handles outliers common in traffic features)
  - Protocol integer encoding
  - Train/val/test stratified split (70/10/20)
  - Label binarization (any non-benign class → 1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from typing import Tuple, List, Optional


def binarize_labels(labels: pd.Series, benign_values: list) -> pd.Series:
    """Map benign label values to 0, everything else to 1."""
    return labels.apply(lambda x: 0 if x in benign_values else 1).astype(np.int32)


def encode_protocol(proto_series: pd.Series) -> pd.Series:
    """
    Encode protocol strings or numbers to integers.
    Common: TCP=6, UDP=17, ICMP=1; unknown → 0.
    """
    proto_map = {
        "tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58,
        "igmp": 2, "gre": 47, "esp": 50, "arp": 2054,
        6: 6, 17: 17, 1: 1,    # pass-through for numeric inputs
    }
    return proto_series.apply(
        lambda x: proto_map.get(str(x).lower().strip(), 0)
    ).astype(np.float32)


def normalize_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler: Optional[RobustScaler] = None,
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Apply RobustScaler to `feature_cols` in-place.
    If scaler is None, fits a new one (train set).
    Returns updated DataFrame and the fitted scaler.
    """
    if scaler is None:
        scaler = RobustScaler()
        df = df.copy()
        df[feature_cols] = scaler.fit_transform(df[feature_cols].values.astype(np.float32))
    else:
        df = df.copy()
        df[feature_cols] = scaler.transform(df[feature_cols].values.astype(np.float32))
    return df, scaler


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/10/20 split.
    Returns (train_df, val_df, test_df).
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    # First split: train vs (val + test)
    df_train, df_temp = train_test_split(
        df, test_size=(val_ratio + test_ratio),
        stratify=df[label_col], random_state=seed
    )

    # Second split: val vs test
    val_frac = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(
        df_temp, test_size=(1.0 - val_frac),
        stratify=df_temp[label_col], random_state=seed
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


def fill_and_clip(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Replace inf/nan with column median and clip to finite range."""
    df = df.copy()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if not np.isnan(median) else 0.0)
    return df


def safe_cast_float(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce columns to float, silently replacing non-numeric values with 0."""
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df
