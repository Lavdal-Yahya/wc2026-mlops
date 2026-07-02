"""Evaluation metrics for the 3-class outcome problem.

Selection primary: **log loss** (probabilistic). Secondary: Brier (multiclass).
Reported alongside: accuracy, macro F1, per-class precision/recall/F1, confusion matrix,
and per-class reliability curves for calibration analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

# Lexicographic order — matches sklearn's default `classes_` and what log_loss expects.
CLASSES = np.array(["away_win", "draw", "home_win"])

# Display order used in plots/tables (more intuitive than lexicographic).
DISPLAY_ORDER = ["home_win", "draw", "away_win"]


def _onehot(y_true: np.ndarray) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CLASSES)}
    n = len(y_true)
    oh = np.zeros((n, len(CLASSES)), dtype=float)
    for i, label in enumerate(y_true):
        oh[i, idx[label]] = 1.0
    return oh


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean over samples of sum over classes of squared error — the standard multiclass Brier."""
    oh = _onehot(y_true)
    return float(np.mean(np.sum((proba - oh) ** 2, axis=1)))


@dataclass
class MetricResult:
    accuracy: float
    macro_f1: float
    log_loss: float
    brier: float
    per_class: dict          # {class_name: {precision, recall, f1, support}}
    confusion: list[list[int]]  # 3x3 list[list], row=true, col=pred

    def to_dict(self) -> dict:
        return asdict(self)


def compute_all_metrics(y_true: np.ndarray, proba: np.ndarray) -> MetricResult:
    """Single entry point: ``proba`` is (n, 3) aligned with ``CLASSES``."""
    y_pred = CLASSES[proba.argmax(axis=1)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    per_class = {
        c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(CLASSES)
    }
    return MetricResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        log_loss=float(log_loss(y_true, proba, labels=CLASSES)),
        brier=multiclass_brier(y_true, proba),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    )


def calibration_curve_per_class(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> dict[str, pd.DataFrame]:
    """One-vs-rest reliability table for each class.

    Returns a dict ``{class_name: DataFrame[bin_mean_proba, empirical_rate, count]}``.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: dict[str, pd.DataFrame] = {}
    for i, c in enumerate(CLASSES):
        prob_c = proba[:, i]
        truth_c = (y_true == c).astype(int)
        bin_id = np.clip(np.digitize(prob_c, bins, right=False) - 1, 0, n_bins - 1)
        df = (
            pd.DataFrame({"bin": bin_id, "proba": prob_c, "truth": truth_c})
            .groupby("bin")
            .agg(bin_mean_proba=("proba", "mean"),
                 empirical_rate=("truth", "mean"),
                 count=("truth", "size"))
            .reset_index(drop=True)
        )
        out[c] = df
    return out
