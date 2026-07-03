"""Preprocessing stage: raw results -> processed train/test + ratings snapshot.

Wraps src/features/build.py and logs dataset params/stats to MLflow.
Reads MLFLOW_TRACKING_URI from the environment (falls back to the local
./mlruns store), so it works before the EC2 tracking server is up.

Usage:
    python scripts/preprocess.py [--params params.yaml] [--feature-version v1|v2]
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import joblib
import mlflow
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import build  # noqa: E402


def ensure_raw_data(raw_path: str, raw_url: str) -> None:
    if Path(raw_path).exists():
        return
    print(f"Raw data missing, downloading from {raw_url}")
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(raw_url, raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--feature-version", choices=["v1", "v2"],
                        help="override preprocess.feature_version from params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)
    if args.feature_version:
        params["preprocess"]["feature_version"] = args.feature_version
    prep = params["preprocess"]
    data_cfg = params["data"]
    version = prep["feature_version"]

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    with mlflow.start_run(run_name=f"preprocess-{version}"):
        mlflow.set_tag("stage", "preprocess")

        ensure_raw_data(data_cfg["raw_path"], data_cfg["raw_url"])
        raw = build.load_raw(data_cfg["raw_path"])
        df, snapshot = build.build_features(raw, params)
        train, test = build.time_split(df, prep["train_start"], prep["test_cut"])

        out_dir = Path(data_cfg["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        keep = (["date", "home_team", "away_team", "tournament"]
                + build.feature_columns(version) + [build.TARGET])
        train[keep].to_csv(data_cfg["train_path"], index=False)
        test[keep].to_csv(data_cfg["test_path"], index=False)
        joblib.dump(snapshot, data_cfg["snapshot_path"])

        mlflow.log_params({
            "feature_version": version,
            "train_start": prep["train_start"],
            "test_cut": prep["test_cut"],
            "elo_initial_rating": prep["elo"]["initial_rating"],
            "elo_k_factor": prep["elo"]["k_factor"],
            "elo_home_advantage": prep["elo"]["home_advantage"],
            "form_window": prep["form"]["window"],
            "h2h_window": prep["h2h"]["window"],
        })

        counts = train[build.TARGET].value_counts(normalize=True)
        stats = {
            "n_raw_matches": len(raw),
            "n_train": len(train),
            "n_test": len(test),
            "n_teams": len(snapshot["teams"]),
            "n_features": len(build.feature_columns(version)),
            "train_share_home_win": round(counts.get(0, 0.0), 4),
            "train_share_draw": round(counts.get(1, 0.0), 4),
            "train_share_away_win": round(counts.get(2, 0.0), 4),
        }
        mlflow.log_metrics(stats)

        summary = {
            **stats,
            "feature_version": version,
            "features": build.feature_columns(version),
            "train_dates": [str(train["date"].min().date()),
                            str(train["date"].max().date())],
            "test_dates": [str(test["date"].min().date()),
                           str(test["date"].max().date())],
            "outputs": [data_cfg["train_path"], data_cfg["test_path"],
                        data_cfg["snapshot_path"]],
        }
        summary_path = out_dir / f"dataset_summary_{version}.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(summary_path))

        print(json.dumps(summary, indent=2))
        print(f"\nDone. Tracking URI: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()
