"""Time-aware splits.

Rules (see DECISIONS.md, D3):
- Train  : pre-2019 modern era (with a >=1995 floor recommended for feature quality).
- Val    : 2019-01-01 .. 2022-12-31 — covers Euro 2020/21 and the 2022 WC era.
- Test   : 2023-01-01 .. just before WC 2026 (held-out, untouched until the final report).
- CV     : expanding-window inside train+val for hyperparameter selection. Never shuffled.

Use :func:`chronological_split` for the single held-out cut and
:func:`expanding_window_splits` for CV folds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """Train/val/test index arrays, plus the date cut points used."""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    val_end: pd.Timestamp


def chronological_split(
    df: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    date_col: str = "date",
    min_date: str | pd.Timestamp | None = None,
) -> TimeSplit:
    """Single chronological cut.

    Rows with ``date <= train_end`` go to train; ``train_end < date <= val_end`` to val;
    ``date > val_end`` to test. Optional ``min_date`` drops earlier rows entirely
    (useful to skip pre-modern football where Elo/form features are sparse).
    """
    dates = pd.to_datetime(df[date_col])
    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    keep = dates >= pd.Timestamp(min_date) if min_date else pd.Series(True, index=df.index)

    train_idx = df.index[(dates <= train_end) & keep].to_numpy()
    val_idx = df.index[(dates > train_end) & (dates <= val_end) & keep].to_numpy()
    test_idx = df.index[(dates > val_end) & keep].to_numpy()
    return TimeSplit(train_idx, val_idx, test_idx, train_end, val_end)


def expanding_window_splits(
    df: pd.DataFrame,
    *,
    n_splits: int = 5,
    val_years: int = 2,
    initial_train_end: str | pd.Timestamp = "2005-12-31",
    date_col: str = "date",
    min_date: str | pd.Timestamp | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield ``n_splits`` expanding-window (train_idx, val_idx) folds.

    Fold k uses train = [start .. cut_k] and val = (cut_k .. cut_k + val_years].
    Each cut advances by ``val_years`` so val windows do not overlap.
    """
    dates = pd.to_datetime(df[date_col])
    keep_mask = dates >= pd.Timestamp(min_date) if min_date else pd.Series(True, index=df.index)
    initial_cut = pd.Timestamp(initial_train_end)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        train_end = initial_cut + pd.DateOffset(years=val_years * k)
        val_end = train_end + pd.DateOffset(years=val_years)
        train_idx = df.index[(dates <= train_end) & keep_mask].to_numpy()
        val_idx = df.index[(dates > train_end) & (dates <= val_end) & keep_mask].to_numpy()
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds
