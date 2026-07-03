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
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from src.features.elo import INITIAL_RATING, expected_score
from src.features.h2h import _pair_key

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


def _load_snapshot() -> dict:
    global _snapshot
    if _snapshot is None:
        _snapshot = joblib.load(SNAPSHOT_PATH)
    return _snapshot


def _rest_days(stats: dict) -> float:
    last = stats.get("last_match_date")
    if not last:
        return 0.0
    today = pd.Timestamp.now().normalize()
    return float((today - pd.Timestamp(last)).days)


def _feature_values(home: str, away: str, neutral: bool, snapshot: dict) -> dict[str, float]:
    """Compute every column scripts/preprocess.py knows how to produce, keyed by name.

    ``_build_feature_vector`` below only reads the subset the trained model
    actually needs (``bundle["features"]``), so v1- and v2-trained models are
    both served correctly from this single map. Missing team/pairing data
    (unknown team, first meeting) falls back to 0.0 — the same value training
    uses for cold-start rows (see scripts/preprocess.py's fillna(0.0)).
    """
    teams = snapshot.get("teams", {})
    h2h_state = snapshot.get("h2h", {})
    home_stats = teams.get(home, {})
    away_stats = teams.get(away, {})

    home_elo = home_stats.get("elo", INITIAL_RATING)
    away_elo = away_stats.get("elo", INITIAL_RATING)

    values = {
        "elo_home": home_elo,
        "elo_away": away_elo,
        "elo_diff": home_elo - away_elo,
        "elo_expected_home": expected_score(home_elo, away_elo),

        "form_home_winrate": home_stats.get("rolling_winrate", 0.0),
        "form_home_drawrate": home_stats.get("rolling_drawrate", 0.0),
        "form_home_ppg": home_stats.get("rolling_pts", 0.0),
        "form_home_gs": home_stats.get("rolling_gf", 0.0),
        "form_home_gc": home_stats.get("rolling_ga", 0.0),
        "rest_home_days": _rest_days(home_stats),

        "form_away_winrate": away_stats.get("rolling_winrate", 0.0),
        "form_away_drawrate": away_stats.get("rolling_drawrate", 0.0),
        "form_away_ppg": away_stats.get("rolling_pts", 0.0),
        "form_away_gs": away_stats.get("rolling_gf", 0.0),
        "form_away_gc": away_stats.get("rolling_ga", 0.0),
        "rest_away_days": _rest_days(away_stats),

        # This app only serves World Cup fixtures — exactly one tournament
        # one-hot may be 1 (see src/features/elo.py::categorize_tournament).
        "is_friendly": 0,
        "is_qualification": 0,
        "is_competitive": 0,
        "is_world_cup": 1,
        "neutral_int": int(neutral),
    }

    key = _pair_key(home, away)
    pair = h2h_state.get(key, {})
    nm, wins_first, draws, gd_first = pair.get("n", 0), pair.get("wins_first", 0), pair.get("draws", 0), pair.get("gd_first", 0)
    if nm > 0:
        if key[0] == home:
            home_wins, home_gd_sum = wins_first, gd_first
        else:
            home_wins, home_gd_sum = nm - wins_first - draws, -gd_first
        values["h2h_total"] = float(nm)
        values["h2h_home_winrate"] = home_wins / nm
        values["h2h_draw_rate"] = draws / nm
        values["h2h_goal_diff_mean"] = home_gd_sum / nm
    else:
        values["h2h_total"] = 0.0
        values["h2h_home_winrate"] = 0.0
        values["h2h_draw_rate"] = 0.0
        values["h2h_goal_diff_mean"] = 0.0

    return values


def _build_feature_vector(home: str, away: str, neutral: bool, snapshot: dict,
                           feature_columns: list[str]) -> np.ndarray:
    """Build the feature vector in the exact column order the model was trained on."""
    values = _feature_values(home, away, neutral, snapshot)
    missing = [c for c in feature_columns if c not in values]
    if missing:
        raise KeyError(f"Serving cannot compute feature columns: {missing}")
    return np.array([[float(values[c]) for c in feature_columns]])


@app.route("/")
def index():
    snapshot = _load_snapshot()
    teams = sorted(snapshot.get("teams", {}).keys())
    return render_template("index.html", teams=teams)


@app.route("/predict", methods=["POST"])
def predict():
    home = request.form["home_team"]
    away = request.form["away_team"]
    neutral = request.form.get("neutral") == "on"

    if home == away:
        return jsonify({"error": "Home and away teams must differ."}), 400

    snapshot = _load_snapshot()
    bundle = _load_bundle()
    model = _load_model()

    X = _build_feature_vector(home, away, neutral, snapshot, bundle["features"])
    proba = model.predict_proba(X)[0]
    classes = list(getattr(model, "classes_", bundle["classes"]))
    proba_by_class = dict(zip(classes, proba))

    return render_template(
        "index.html",
        teams=sorted(snapshot.get("teams", {}).keys()),
        result={
            "home": home,
            "away": away,
            "neutral": neutral,
            "p_home_win": round(float(proba_by_class.get("home_win", 0.0)) * 100, 1),
            "p_draw": round(float(proba_by_class.get("draw", 0.0)) * 100, 1),
            "p_away_win": round(float(proba_by_class.get("away_win", 0.0)) * 100, 1),
        },
    )


@app.route("/healthz")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
