"""Feature pipeline orchestrator.

Glues Elo + rolling form + H2H + match-context features into a single match-level
table and writes it to ``data/processed/matches_features.parquet``.

By construction, every feature is computed from data strictly before each
match's date — see the leakage notes in each feature module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load import load_raw
from src.features.elo import categorize_tournament, compute_elo_features
from src.features.h2h import compute_h2h_features
from src.features.rolling import compute_rolling_features
from src.utils.config import PROJECT_ROOT, load_config

CONTEXT_COLS = ["is_friendly", "is_qualification", "is_competitive", "is_world_cup", "neutral_int"]


def _context_features(matches: pd.DataFrame) -> pd.DataFrame:
    cat = matches["tournament"].map(categorize_tournament)
    return pd.DataFrame(
        {
            "is_friendly": (cat == "friendly").astype(int),
            "is_qualification": (cat == "qualification").astype(int),
            "is_competitive": (cat == "competitive").astype(int),
            "is_world_cup": (cat == "world_cup").astype(int),
            "neutral_int": matches["neutral"].astype(int),
        },
        index=matches.index,
    )


def build_features(save: bool = True) -> pd.DataFrame:
    """Compute the full match-level feature table and optionally persist it."""
    config = load_config()
    bundle = load_raw(config)
    matches = bundle.matches.sort_values("date", kind="stable").reset_index(drop=True)

    elo = compute_elo_features(matches)
    roll = compute_rolling_features(matches, window=config["features"]["rolling"]["window_matches"])
    h2h = compute_h2h_features(matches)
    ctx = _context_features(matches)

    keep_cols = ["date", "home_team", "away_team", "tournament", "neutral",
                 "home_score", "away_score", "outcome"]
    out = pd.concat([matches[keep_cols], elo, roll, h2h, ctx], axis=1)

    if save:
        path = PROJECT_ROOT / config["paths"]["processed_data"]
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        print(f"Wrote {len(out):,} rows × {out.shape[1]} cols → {path}")
    return out


if __name__ == "__main__":
    build_features()
