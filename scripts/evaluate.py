"""Evaluate stage — owned by Souleyman.

Scores the model produced by scripts/train.py (the same estimator registered as
``wc-outcome-model``) on the held-out test set and produces the evaluation
artifacts required by the PDF: metrics JSON, per-class calibration (reliability)
curves, and a confusion matrix — all logged to MLflow (PDF §4).

Inputs:
  data/processed/test.csv    (produced by scripts/preprocess.py)
  models/model.joblib        (produced by scripts/train.py; falls back to the
                              MLflow registry when the local bundle is absent)

Outputs:
  reports/metrics.json             (log loss, Brier, accuracy, macro F1, per-class, confusion)
  reports/calibration_curve.png
  reports/confusion_matrix.png

Run from the project root:  python -m scripts.evaluate
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib.colors import LinearSegmentedColormap

from scripts.preprocess import META_COLS
from scripts.train import aligned_proba
from src.evaluation.metrics import (
    CLASSES,
    MetricResult,
    calibration_curve_per_class,
    compute_all_metrics,
)
from src.utils.config import PROJECT_ROOT, load_config

load_dotenv()

# Fixed per-class colors (validated categorical palette, slots 1-3 in order).
CLASS_COLORS = {"away_win": "#2a78d6", "draw": "#1baf7a", "home_win": "#eda100"}
# Single-hue sequential ramp for the confusion heatmap (light -> dark blue).
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_MUTED = "#1a1a19", "#5f5e57"


def load_model_bundle(config: dict) -> dict:
    """Local bundle first (DVC-tracked, deterministic); registry as fallback."""
    model_path = PROJECT_ROOT / config["paths"]["model_output"] / "model.joblib"
    if model_path.exists():
        bundle = joblib.load(model_path)
        print(f"Loaded model bundle from {model_path}")
        return bundle

    uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("mlflow", {}).get("tracking_uri")
    if not uri:
        raise SystemExit(
            f"{model_path} not found and MLFLOW_TRACKING_URI is not set. "
            "Run `python -m scripts.train` (or `dvc repro`) first."
        )
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(uri)
    reg_name = config["mlflow"]["registered_model"]
    est = mlflow.sklearn.load_model(f"models:/{reg_name}/latest")
    print(f"Loaded '{reg_name}' (latest) from the MLflow registry at {uri}")
    return {"pipeline": est, "features": None, "model_name": reg_name,
            "feature_version": config["features"]["version"]}


def plot_calibration(y_true: np.ndarray, proba: np.ndarray, out_path) -> None:
    """One-vs-rest reliability curves, one line per class, diagonal = perfect."""
    curves = calibration_curve_per_class(y_true, proba)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=150)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK_MUTED, zorder=1,
            label="perfect calibration")
    for cls in CLASSES:
        df = curves[cls]
        ax.plot(df["bin_mean_proba"], df["empirical_rate"], marker="o", ms=6,
                lw=2, color=CLASS_COLORS[cls], label=cls, zorder=2)
    ax.set_xlabel("Predicted probability", color=INK)
    ax.set_ylabel("Observed frequency", color=INK)
    ax.set_title("Reliability curves (one-vs-rest, 10 bins)", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, lw=0.5, alpha=0.35)
    ax.legend(frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(confusion: list[list[int]], out_path) -> None:
    """Row-normalized heatmap (row = true class), annotated with raw counts."""
    counts = np.asarray(confusion, dtype=float)
    rates = counts / counts.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)

    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=150)
    im = ax.imshow(rates, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted", color=INK)
    ax.set_ylabel("True", color=INK)
    ax.set_title("Confusion matrix (test set)", color=INK)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ink = "#ffffff" if rates[i, j] > 0.5 else INK
            ax.text(j, i, f"{int(counts[i, j]):,}\n{rates[i, j]:.0%}",
                    ha="center", va="center", color=ink, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of true class")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _log_to_mlflow(config: dict, bundle: dict, result: MetricResult, artifacts) -> None:
    """Best-effort MLflow logging — never blocks the stage on a down server."""
    uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("mlflow", {}).get("tracking_uri")
    if not uri:
        print("MLFLOW_TRACKING_URI not set; skipping MLflow logging.")
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config["mlflow"]["experiment"])
        version = bundle.get("feature_version", config["features"]["version"])
        with mlflow.start_run(run_name=f"evaluate-{version}"):
            mlflow.set_tag("stage", "evaluate")
            mlflow.set_tag("model", bundle.get("model_name", "unknown"))
            mlflow.set_tag("feature_version", version)
            mlflow.log_metrics({
                "test_log_loss": result.log_loss,
                "test_brier": result.brier,
                "test_accuracy": result.accuracy,
                "test_macro_f1": result.macro_f1,
                **{f"test_f1_{cls}": vals["f1"] for cls, vals in result.per_class.items()},
            })
            for art in artifacts:
                mlflow.log_artifact(str(art), artifact_path="evaluation")
        print(f"Logged evaluate run to MLflow at {uri}")
    except Exception as exc:  # noqa: BLE001 — never let tracking break evaluation
        print(f"WARNING: MLflow logging failed ({exc}). Local reports were still written.")


def main() -> None:
    config = load_config()
    paths = config["paths"]
    target = config["target"]["column"]

    test_path = PROJECT_ROOT / paths["test_csv"]
    if not test_path.exists():
        raise SystemExit(
            f"{test_path} not found. Run `python -m scripts.preprocess` (or `dvc repro`) first."
        )

    bundle = load_model_bundle(config)
    df = pd.read_csv(test_path, parse_dates=["date"])
    feat_cols = bundle.get("features") or [c for c in df.columns if c not in META_COLS + [target]]
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy()

    proba = aligned_proba(bundle["pipeline"], X)
    result = compute_all_metrics(y, proba)

    reports_dir = PROJECT_ROOT / paths["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    payload = {
        "model_name": bundle.get("model_name", "unknown"),
        "feature_version": bundle.get("feature_version", config["features"]["version"]),
        "n_test": len(df),
        "cv_log_loss": bundle.get("cv_log_loss"),
        "test": result.to_dict(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")

    calib_path = reports_dir / "calibration_curve.png"
    conf_path = reports_dir / "confusion_matrix.png"
    plot_calibration(y, proba, calib_path)
    plot_confusion(result.confusion, conf_path)

    print(
        f"[evaluate:{payload['feature_version']}] {payload['model_name']}  "
        f"n_test={len(df):,}  log_loss={result.log_loss:.4f}  brier={result.brier:.4f}  "
        f"acc={result.accuracy:.3f}  macro_f1={result.macro_f1:.3f}"
    )
    for out in (metrics_path, calib_path, conf_path):
        print(f"  -> {out}")

    _log_to_mlflow(config, bundle, result, [metrics_path, calib_path, conf_path])


if __name__ == "__main__":
    main()
