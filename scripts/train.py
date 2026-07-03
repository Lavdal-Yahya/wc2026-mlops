"""Train stage — owned by Mohamed (implemented by Mouna).

Model candidates and time-aware cross-validation helpers for the training stage.
Selection metric is log loss (probabilistic), matching src/evaluation/metrics.py.
CV is a forward-chaining TimeSeriesSplit so no future match informs a past fold.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import CLASSES, compute_all_metrics


def build_candidates(models_cfg: dict, seed: int):
    """Yield ``(model_name, params, estimator)`` for every hyperparameter combo in params.yaml.

    Logistic regression is wrapped in a StandardScaler pipeline (it is scale-sensitive);
    the tree ensembles are used directly.
    """
    lr = models_cfg.get("logistic_regression", {})
    for c in lr.get("C", [1.0]):
        params = {"C": c, "max_iter": lr.get("max_iter", 1000)}
        est = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=c, max_iter=params["max_iter"])),
        ])
        yield "logistic_regression", params, est

    rf = models_cfg.get("random_forest", {})
    for n, depth, leaf in itertools.product(
        rf.get("n_estimators", [200]), rf.get("max_depth", [None]), rf.get("min_samples_leaf", [1])
    ):
        params = {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf}
        est = RandomForestClassifier(
            n_estimators=n, max_depth=depth, min_samples_leaf=leaf, random_state=seed, n_jobs=-1
        )
        yield "random_forest", params, est

    gb = models_cfg.get("gradient_boosting", {})
    for n, depth, rate in itertools.product(
        gb.get("n_estimators", [200]), gb.get("max_depth", [3]), gb.get("learning_rate", [0.1])
    ):
        params = {"n_estimators": n, "max_depth": depth, "learning_rate": rate}
        est = GradientBoostingClassifier(
            n_estimators=n, max_depth=depth, learning_rate=rate, random_state=seed
        )
        yield "gradient_boosting", params, est


def aligned_proba(est, X: np.ndarray) -> np.ndarray:
    """predict_proba reordered to the canonical CLASSES order (away_win, draw, home_win)."""
    proba = est.predict_proba(X)
    classes = list(est.classes_)
    return proba[:, [classes.index(c) for c in CLASSES]]


def cv_scores(est, X: np.ndarray, y: np.ndarray, n_splits: int) -> tuple[float, float]:
    """Forward-chaining CV; returns (mean log loss, mean accuracy)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses, accs = [], []
    for tr, va in tscv.split(X):
        est.fit(X[tr], y[tr])
        metrics = compute_all_metrics(y[va], aligned_proba(est, X[va]))
        losses.append(metrics.log_loss)
        accs.append(metrics.accuracy)
    return float(np.mean(losses)), float(np.mean(accs))
