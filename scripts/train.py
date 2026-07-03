"""Train stage — owned by Mohamed (implemented by Mouna).

Trains the candidate models declared in params.yaml with time-aware CV, logs every
configuration to MLflow, and registers ONLY the best model — lowest cross-validated
log loss — as ``wc-outcome-model`` in the MLflow Model Registry (PDF §4).

Inputs:
  data/processed/train.csv   (produced by scripts/preprocess.py)
  params.yaml                (model grids, CV, mlflow config)

Outputs:
  models/model.joblib        (best refit pipeline + feature list; DVC-trackable, Flask fallback)
  registered ``wc-outcome-model`` in the MLflow registry (when a server is configured)

Selection metric is log loss (probabilistic), matching src/evaluation/metrics.py.
CV is a forward-chaining TimeSeriesSplit so no future match informs a past fold.

Run from the project root:  python -m scripts.train
"""
from __future__ import annotations

import itertools
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.preprocess import META_COLS
from src.evaluation.metrics import CLASSES, compute_all_metrics
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeds import set_global_seed

load_dotenv()


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


def _init_mlflow(config):
    """Return (mlflow module, active?) — active only when a tracking server is reachable."""
    uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("mlflow", {}).get("tracking_uri")
    if not uri:
        print("MLFLOW_TRACKING_URI not set; training locally without MLflow logging/registry.")
        return None, False
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config["mlflow"]["experiment"])
        return mlflow, True
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: MLflow unavailable ({exc}); training locally.")
        return None, False


def _register_best(mlflow, config, est, name, params, cv_log_loss, X, feat_cols) -> None:
    """Log + register the winning model as the sole registered version (PDF §4)."""
    import mlflow.sklearn
    from mlflow.models import infer_signature

    reg_name = config["mlflow"]["registered_model"]
    mlflow.set_tag("best_model", name)
    mlflow.log_params({f"best_{k}": v for k, v in params.items()})
    mlflow.log_metric("best_cv_log_loss", cv_log_loss)
    signature = infer_signature(pd.DataFrame(X[:5], columns=feat_cols), aligned_proba(est, X[:5]))
    try:
        mlflow.sklearn.log_model(
            est, artifact_path="model", registered_model_name=reg_name,
            signature=signature, input_example=X[:2],
        )
        print(f"Registered best model as '{reg_name}'.")
    except Exception as exc:  # noqa: BLE001 — file-store backends can't register
        print(f"WARNING: registry unavailable ({exc}); logging model artifact without registration.")
        try:
            mlflow.sklearn.log_model(est, artifact_path="model", signature=signature)
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    config = load_config()
    seed = config.get("seed", 42)
    set_global_seed(seed)

    paths = config["paths"]
    target = config["target"]["column"]
    version = config["features"]["version"]
    n_splits = config["validation"]["cv"]["n_splits"]

    train_path = PROJECT_ROOT / paths["train_csv"]
    if not train_path.exists():
        raise SystemExit(
            f"{train_path} not found. Run `python -m scripts.preprocess` (or `dvc repro`) first."
        )

    df = pd.read_csv(train_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS + [target]]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy()
    print(f"[train:{version}] {len(df):,} rows  {len(feat_cols)} features  {n_splits}-fold TS-CV")

    mlflow, active = _init_mlflow(config)
    if active:
        mlflow.start_run(run_name=f"train-{version}")
        mlflow.set_tag("stage", "train")
        mlflow.set_tag("feature_version", version)

    best = None  # (log_loss, name, params, estimator)
    for name, params, est in build_candidates(config["models"], seed):
        log_loss, acc = cv_scores(est, X, y, n_splits)
        print(f"  {name:<20} {params}  -> cv_log_loss={log_loss:.4f}  cv_acc={acc:.3f}")
        if active:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tag("stage", "train")
                mlflow.set_tag("model", name)
                mlflow.log_params({**params, "model": name, "feature_version": version})
                mlflow.log_metric("cv_log_loss", log_loss)
                mlflow.log_metric("cv_accuracy", acc)
        if best is None or log_loss < best[0]:
            best = (log_loss, name, params, est)

    best_ll, best_name, best_params, best_est = best
    print(f"  BEST: {best_name} {best_params}  cv_log_loss={best_ll:.4f}")

    best_est.fit(X, y)  # refit on the full train set

    models_dir = PROJECT_ROOT / paths["model_output"]
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best_est,
        "features": feat_cols,
        "classes": [str(c) for c in CLASSES],
        "feature_version": version,
        "model_name": best_name,
        "params": best_params,
        "cv_log_loss": best_ll,
    }
    model_path = models_dir / "model.joblib"
    joblib.dump(bundle, model_path)
    print(f"  -> {model_path}")

    if active:
        _register_best(mlflow, config, best_est, best_name, best_params, best_ll, X, feat_cols)
        mlflow.end_run()


if __name__ == "__main__":
    main()
