"""Flask prediction app — WC 2026 match outcome forecaster.

Loads the registered model from the MLflow Model Registry (falling back to
the local models/model.joblib bundle) and ratings_snapshot.joblib (produced
by the preprocess stage) to serve live match predictions.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Only attempt the MLflow registry when a tracking server was explicitly
# configured — the default below is just so `mlflow.set_tracking_uri` always
# has a value to point at for any other MLflow calls in this process.
_RAW_MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_URI = _RAW_MLFLOW_TRACKING_URI or "http://127.0.0.1:5000"
REGISTERED_MODEL = "wc-outcome-model"
MODEL_BUNDLE_PATH = Path(os.getenv("MODEL_BUNDLE_PATH", "models/model.joblib"))
SNAPSHOT_PATH = Path(os.getenv("SNAPSHOT_PATH", "data/processed/ratings_snapshot.joblib"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model = None
_bundle = None
_snapshot = None


def _load_bundle() -> dict:
    """Load the local model.joblib bundle — the only source of the training
    ``features`` list and ``classes``. The MLflow registry only stores the bare
    sklearn pipeline, so this is loaded regardless of where the pipeline itself
    comes from.
    """
    global _bundle
    if _bundle is None:
        if not MODEL_BUNDLE_PATH.exists():
            raise RuntimeError(
                f"Model bundle not found at {MODEL_BUNDLE_PATH}. "
                "Run `python -m scripts.train` first."
            )
        _bundle = joblib.load(MODEL_BUNDLE_PATH)
    return _bundle


def _load_model():
    """Return the serving pipeline.

    Tries the MLflow registry's latest ``wc-outcome-model`` version when
    MLFLOW_TRACKING_URI is configured; falls back to the local bundle's
    pipeline when the registry is unreachable or has no versions registered
    (no version is ever transitioned to a stage, so we ask for "latest").
    """
    global _model
    if _model is not None:
        return _model

    if _RAW_MLFLOW_TRACKING_URI:
        try:
            _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/latest")
            return _model
        except Exception as exc:  # noqa: BLE001 — registry may be down or empty
            print(f"WARNING: could not load '{REGISTERED_MODEL}' from MLflow registry "
                  f"({exc}); falling back to local model bundle.")

    _model = _load_bundle()["pipeline"]
    return _model


def _load_snapshot():
    global _snapshot
    if _snapshot is None:
        _snapshot = joblib.load(SNAPSHOT_PATH)
    return _snapshot


def _build_feature_vector(home: str, away: str, neutral: bool, snapshot: dict) -> np.ndarray:
    """Build the same feature vector that scripts/preprocess.py produces.

    Uses ratings_snapshot — a dict keyed by team name with Elo + rolling stats.
    Feature order must match training: see scripts/preprocess.py for the canonical list.
    """
    home_stats = snapshot.get(home, {})
    away_stats = snapshot.get(away, {})

    elo_diff = home_stats.get("elo", 1500) - away_stats.get("elo", 1500)
    home_elo = home_stats.get("elo", 1500)
    away_elo = away_stats.get("elo", 1500)
    home_form = home_stats.get("rolling_pts", 0.0)
    away_form = away_stats.get("rolling_pts", 0.0)
    home_gf = home_stats.get("rolling_gf", 0.0)
    away_gf = away_stats.get("rolling_gf", 0.0)
    home_ga = home_stats.get("rolling_ga", 0.0)
    away_ga = away_stats.get("rolling_ga", 0.0)

    return np.array([[
        home_elo, away_elo, elo_diff,
        home_form, away_form,
        home_gf, away_gf, home_ga, away_ga,
        int(neutral),
        1,  # is_world_cup (assume for this app)
        0,  # is_friendly
        0,  # is_qualification
        1,  # is_competitive
    ]])


@app.route("/")
def index():
    snapshot = _load_snapshot()
    teams = sorted(snapshot.keys())
    return render_template("index.html", teams=teams)


@app.route("/predict", methods=["POST"])
def predict():
    home = request.form["home_team"]
    away = request.form["away_team"]
    neutral = request.form.get("neutral") == "on"

    if home == away:
        return jsonify({"error": "Home and away teams must differ."}), 400

    snapshot = _load_snapshot()
    model = _load_model()
    X = _build_feature_vector(home, away, neutral, snapshot)
    proba = model.predict_proba(X)[0]

    return render_template(
        "index.html",
        teams=sorted(snapshot.keys()),
        result={
            "home": home,
            "away": away,
            "neutral": neutral,
            "p_home_win": round(float(proba[2]) * 100, 1),
            "p_draw": round(float(proba[1]) * 100, 1),
            "p_away_win": round(float(proba[0]) * 100, 1),
        },
    )


@app.route("/healthz")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
