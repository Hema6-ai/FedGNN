"""
All 8 metrics from Table 3: Accuracy, F1, MSE, RMSE, MAE, MAPE, + Precision, Recall.

Design note on MAPE: The paper reports MAPE ~1.0–1.1% for models with ~98% accuracy.
Standard MAPE over binary targets (0/1) is undefined at y=0. We compute MAPE only on
malicious samples (y_true=1): MAPE = mean(|1 - score|) * 100. This is consistent with
the paper's scale and avoids division by zero.

Design note on MSE vs RMSE: The paper's Table 3 has sqrt(MSE) ≠ RMSE in some rows
(e.g., sqrt(0.06)=0.245 but RMSE=0.20). We report standard definitions; discrepancies
are documented.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tabulate import tabulate


def compute_metrics(
    y_true: np.ndarray,
    y_pred_cls: np.ndarray,
    y_pred_score: np.ndarray,
) -> dict:
    """
    Args:
        y_true: integer labels (0=benign, 1=malicious); shape (N,)
        y_pred_cls: predicted class labels (0/1); shape (N,)
        y_pred_score: predicted malicious probability in [0,1]; shape (N,)
    Returns:
        dict with keys: accuracy, f1, precision, recall, mse, rmse, mae, mape
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred_cls = np.asarray(y_pred_cls, dtype=np.int32)
    y_score = np.asarray(y_pred_score, dtype=np.float64).clip(0.0, 1.0)

    acc = float(accuracy_score(y_true, y_pred_cls)) * 100.0
    f1 = float(f1_score(y_true, y_pred_cls, average="weighted", zero_division=0)) * 100.0
    prec = float(precision_score(y_true, y_pred_cls, average="weighted", zero_division=0)) * 100.0
    rec = float(recall_score(y_true, y_pred_cls, average="weighted", zero_division=0)) * 100.0

    y_true_f = y_true.astype(np.float64)
    mse = float(np.mean((y_true_f - y_score) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true_f - y_score)))

    # MAPE on malicious samples only — avoids div-by-zero and matches paper scale
    pos_mask = y_true == 1
    if pos_mask.sum() > 0:
        mape = float(np.mean(np.abs(y_true_f[pos_mask] - y_score[pos_mask])) * 100.0)
    else:
        mape = 0.0

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def format_metrics_table(results: dict) -> str:
    """
    Args:
        results: {model_name: {dataset_name: metrics_dict}}
    Returns:
        Printable comparison table string
    """
    headers = ["Model", "Dataset", "Acc(%)", "F1(%)", "MSE", "RMSE", "MAE", "MAPE(%)"]
    rows = []
    for model_name, datasets in results.items():
        for ds_name, m in datasets.items():
            rows.append([
                model_name,
                ds_name,
                f"{m['accuracy']:.2f}",
                f"{m['f1']:.2f}",
                f"{m['mse']:.4f}",
                f"{m['rmse']:.4f}",
                f"{m['mae']:.4f}",
                f"{m['mape']:.2f}",
            ])
    return tabulate(rows, headers=headers, tablefmt="github")
