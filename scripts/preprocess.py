"""Preprocess stage — owned by Souleyman (implemented by Mouna).

Builds the model-ready train/test tables and the serving-time ratings snapshot,
then logs the run to MLflow so the pipeline has history from day one (PDF §4).

Outputs:
  data/processed/train.csv               (train + val, chronological)
  data/processed/test.csv                (held-out: date > validation.val_end)
  data/processed/ratings_snapshot.joblib (per-team Elo + rolling form for the Flask app)

The feature set is controlled by ``params.yaml -> features.version``:
  v1 = Elo + match-context
  v2 = Elo + rolling form + head-to-head + match-context
This is the knob DVC uses to produce two different data versions (PDF §5): change
``features.version`` in params.yaml, re-run the stage, and commit the new .dvc state.

Run from the project root:  python -m scripts.preprocess
"""
from __future__ import annotations

import os
from collections import deque

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.data.load import load_raw
from src.data.splits import chronological_split
from src.features.build import CONTEXT_COLS, build_features
from src.features.elo import INITIAL_RATING, final_ratings
from src.features.h2h import _pair_key
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeds import set_global_seed

load_dotenv()

# Feature groups, matched exactly to the column names emitted by src/features/*.
ELO_COLS = ["elo_home", "elo_away", "elo_diff", "elo_expected_home"]
ROLLING_COLS = [
    "form_home_winrate", "form_home_drawrate", "form_home_ppg",
    "form_home_gs", "form_home_gc", "rest_home_days",
    "form_away_winrate", "form_away_drawrate", "form_away_ppg",
    "form_away_gs", "form_away_gc", "rest_away_days",
]
H2H_COLS = ["h2h_total", "h2h_home_winrate", "h2h_draw_rate", "h2h_goal_diff_mean"]
META_COLS = ["date", "home_team", "away_team", "tournament", "neutral"]


def select_feature_columns(version: str) -> list[str]:
    """Return the ordered feature columns for the requested feature version."""
    version = (version or "v1").lower()
    if version == "v1":
        return ELO_COLS + CONTEXT_COLS
    if version == "v2":
        return ELO_COLS + ROLLING_COLS + H2H_COLS + CONTEXT_COLS
    raise ValueError(f"Unknown features.version={version!r}; expected 'v1' or 'v2'.")


def build_ratings_snapshot(matches: pd.DataFrame, *, window: int) -> dict[str, dict]:
    """Post-history serving state: latest Elo, rolling form and head-to-head record.

    The Flask app (app/app.py) reads this snapshot to build a prediction feature
    vector for a fixture that has not been played yet, so we want the state
    *after* the most recent match — not the pre-match features stored per row.
    Mirrors the same rolling window (src/features/rolling.py) and head-to-head
    (src/features/h2h.py) bookkeeping used at training time so v2 columns can be
    reconstructed at serving time.

    Returns ``{"teams": {team: {...}}, "h2h": {(team_a, team_b): {...}}}`` where
    ``team_a <= team_b`` (see ``_pair_key``). Per-team keys: elo, rolling_gf,
    rolling_ga, rolling_pts, rolling_winrate, rolling_drawrate, matches,
    last_match_date (ISO date string, or None if the team never played).
    """
    elo = final_ratings(matches)  # team -> Elo after the last played match
    ordered = matches.sort_values("date", kind="stable")

    history: dict[str, deque] = {}
    last_date: dict[str, str] = {}
    h2h_state: dict[tuple[str, str], tuple[int, int, int, int]] = {}

    for h, a, hs, as_, dt in zip(
        ordered["home_team"], ordered["away_team"],
        ordered["home_score"], ordered["away_score"], ordered["date"],
    ):
        hs, as_ = int(hs), int(as_)
        pts_h = 3 if hs > as_ else 1 if hs == as_ else 0
        pts_a = 3 if as_ > hs else 1 if hs == as_ else 0
        history.setdefault(h, deque(maxlen=window)).append((hs, as_, pts_h))
        history.setdefault(a, deque(maxlen=window)).append((as_, hs, pts_a))
        date_str = pd.Timestamp(dt).date().isoformat()
        last_date[h] = date_str
        last_date[a] = date_str

        key = _pair_key(h, a)
        nm, wins_first, draws, gd_first = h2h_state.get(key, (0, 0, 0, 0))
        gd = hs - as_
        gd_in_key = gd if key[0] == h else -gd
        if gd > 0:
            won_first = 1 if key[0] == h else 0
        elif gd < 0:
            won_first = 1 if key[0] == a else 0
        else:
            won_first = 0
        is_draw = 1 if gd == 0 else 0
        h2h_state[key] = (nm + 1, wins_first + won_first, draws + is_draw, gd_first + gd_in_key)

    teams: dict[str, dict] = {}
    for team in set(elo) | set(history):
        dq = history.get(team)
        if dq:
            wins = sum(1 for gs, gc, _ in dq if gs > gc)
            draws = sum(1 for gs, gc, _ in dq if gs == gc)
            gf = sum(gs for gs, _, _ in dq) / len(dq)
            ga = sum(gc for _, gc, _ in dq) / len(dq)
            pts = sum(p for _, _, p in dq) / len(dq)
            winrate = wins / len(dq)
            drawrate = draws / len(dq)
        else:
            gf = ga = pts = winrate = drawrate = 0.0
        teams[team] = {
            "elo": float(elo.get(team, INITIAL_RATING)),
            "rolling_gf": float(gf),
            "rolling_ga": float(ga),
            "rolling_pts": float(pts),
            "rolling_winrate": float(winrate),
            "rolling_drawrate": float(drawrate),
            "matches": len(dq) if dq else 0,
            "last_match_date": last_date.get(team),
        }

    # Store the raw counters — app.py re-orients them to the requested home/away
    # side the same way src/features/h2h.py does for a specific match's frame.
    h2h = {
        key: {"n": nm, "wins_first": wins_first, "draws": draws, "gd_first": gd_first}
        for key, (nm, wins_first, draws, gd_first) in h2h_state.items()
    }

    return {"teams": teams, "h2h": h2h}


def _log_to_mlflow(config, version, train_df, test_df, feat_cols, target_col, artifacts) -> None:
    """Log params, dataset stats and produced artifacts to the MLflow server.

    Best-effort: if the tracking server is unreachable, local outputs are still
    written so the pipeline is never blocked by a down EC2 instance.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI") or config.get("mlflow", {}).get("tracking_uri")
    if not uri:
        print("MLFLOW_TRACKING_URI not set; skipping MLflow logging.")
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config["mlflow"]["experiment"])
        with mlflow.start_run(run_name=f"preprocess-{version}"):
            mlflow.set_tag("stage", "preprocess")
            mlflow.set_tag("feature_version", version)
            mlflow.log_params({
                "feature_version": version,
                "seed": config.get("seed", 42),
                "train_end": config["validation"]["train_end"],
                "val_end": config["validation"]["val_end"],
                "rolling_window": config["features"]["rolling"]["window_matches"],
                "n_features": len(feat_cols),
            })
            mlflow.log_metrics({"n_train": len(train_df), "n_test": len(test_df)})
            for split_name, frame in (("train", train_df), ("test", test_df)):
                for cls, frac in frame[target_col].value_counts(normalize=True).items():
                    mlflow.log_metric(f"{split_name}_frac_{cls}", float(frac))
            for art in artifacts:
                mlflow.log_artifact(str(art), artifact_path="processed")
        print(f"Logged preprocess run to MLflow at {uri}")
    except Exception as exc:  # noqa: BLE001 — never let tracking break preprocessing
        print(f"WARNING: MLflow logging failed ({exc}). Local outputs were still written.")


def main() -> None:
    config = load_config()
    set_global_seed(config.get("seed", 42))

    paths = config["paths"]
    version = config["features"]["version"]
    target_col = config["target"]["column"]
    validation = config["validation"]
    window = config["features"]["rolling"]["window_matches"]

    results_csv = PROJECT_ROOT / paths["results_csv"]
    if not results_csv.exists():
        raise SystemExit(
            f"Raw data not found at {results_csv}.\n"
            f"Download the Kaggle dataset ({config['dataset']['kaggle_slug']}) into "
            "data/raw/ or pull it via DVC/S3 before running preprocess."
        )

    # Full leakage-safe feature table (save=False to avoid the parquet/pyarrow dep;
    # this stage only needs the in-memory frame to slice train/test CSVs).
    features = build_features(save=False)

    feat_cols = select_feature_columns(version)
    missing = [c for c in feat_cols if c not in features.columns]
    if missing:
        raise KeyError(f"Feature columns missing from build output: {missing}")

    df = features[META_COLS + feat_cols].copy()
    df[target_col] = features["outcome"].astype(str)
    # Simple, documented imputation: cold-start rows (early history / first H2H
    # meeting) carry NaNs. Fill with 0.0 so downstream sklearn estimators are happy;
    # the train stage may revisit this if a smarter imputer is warranted.
    df[feat_cols] = df[feat_cols].fillna(0.0)

    split = chronological_split(
        features, train_end=validation["train_end"], val_end=validation["val_end"]
    )
    train_idx = np.concatenate([split.train_idx, split.val_idx])
    train_df = df.loc[train_idx].sort_values("date").reset_index(drop=True)
    test_df = df.loc[split.test_idx].sort_values("date").reset_index(drop=True)

    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = PROJECT_ROOT / paths["train_csv"]
    test_path = PROJECT_ROOT / paths["test_csv"]
    snap_path = PROJECT_ROOT / paths["ratings_snapshot"]

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    snapshot = build_ratings_snapshot(load_raw(config).matches, window=window)
    joblib.dump(snapshot, snap_path)

    print(
        f"[preprocess:{version}] "
        f"train={len(train_df):,} rows  test={len(test_df):,} rows  "
        f"features={len(feat_cols)}  teams={len(snapshot['teams']):,}"
    )
    print(f"  -> {train_path}")
    print(f"  -> {test_path}")
    print(f"  -> {snap_path}")

    _log_to_mlflow(config, version, train_df, test_df, feat_cols, target_col,
                   [train_path, test_path, snap_path])


if __name__ == "__main__":
    main()
