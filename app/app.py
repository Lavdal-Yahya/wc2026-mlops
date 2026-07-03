"""Flask prediction app — WC 2026 match outcome forecaster.

Loads the latest registered model from MLflow Model Registry and
ratings_snapshot.joblib (produced by the preprocess stage) to serve
live match predictions.

NOTE: Real registry + snapshot wired in L6 after gate confirmation.
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

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
REGISTERED_MODEL = "wc-outcome-model"
SNAPSHOT_PATH = Path(os.getenv("SNAPSHOT_PATH", "data/processed/ratings_snapshot.joblib"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

_model = None
_snapshot = None


def _load_model():
    global _model
    if _model is None:
        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(REGISTERED_MODEL, stages=["Production"])[0]
        _model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/Production")
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
