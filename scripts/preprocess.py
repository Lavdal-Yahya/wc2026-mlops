"""Preprocess stage — owned by Souleyman (implemented by Mouna).

Builds the model-ready train/test tables and the serving-time ratings snapshot.

The feature set is controlled by ``params.yaml -> features.version``:
  v1 = Elo + match-context
  v2 = Elo + rolling form + head-to-head + match-context
This is the knob DVC uses to produce two different data versions (PDF §5): change
``features.version`` in params.yaml, re-run the stage, and commit the new .dvc state.
"""
from __future__ import annotations

from collections import deque

import pandas as pd

from src.features.build import CONTEXT_COLS
from src.features.elo import INITIAL_RATING, final_ratings

# Feature groups, matched exactly to the column names emitted by src/features/*.
ELO_COLS = ["elo_home", "elo_away", "elo_diff", "elo_expected_home"]
ROLLING_COLS = [
    "form_home_winrate", "form_home_drawrate", "form_home_ppg",
    "form_home_gs", "form_home_gc", "rest_home_days",
    "form_away_winrate", "form_away_drawrate", "form_away_ppg",
    "form_away_gs", "form_away_gc", "rest_away_days",
]
H2H_COLS = ["h2h_total", "h2h_home_winrate", "h2h_draw_rate", "h2h_goal_diff_mean"]


def select_feature_columns(version: str) -> list[str]:
    """Return the ordered feature columns for the requested feature version."""
    version = (version or "v1").lower()
    if version == "v1":
        return ELO_COLS + CONTEXT_COLS
    if version == "v2":
        return ELO_COLS + ROLLING_COLS + H2H_COLS + CONTEXT_COLS
    raise ValueError(f"Unknown features.version={version!r}; expected 'v1' or 'v2'.")


def build_ratings_snapshot(matches: pd.DataFrame, *, window: int) -> dict[str, dict]:
    """Post-history per-team serving state: latest Elo + last-`window` form.

    The Flask app (app/app.py) reads this snapshot to build a prediction feature
    vector for a fixture that has not been played yet, so we want the state
    *after* the most recent match — not the pre-match features stored per row.

    Keys per team match what app.py expects: elo, rolling_pts, rolling_gf, rolling_ga.
    """
    elo = final_ratings(matches)  # team -> Elo after the last played match
    ordered = matches.sort_values("date", kind="stable")

    history: dict[str, deque] = {}
    for h, a, hs, as_ in zip(
        ordered["home_team"], ordered["away_team"],
        ordered["home_score"], ordered["away_score"],
    ):
        history.setdefault(h, deque(maxlen=window)).append((int(hs), int(as_)))
        history.setdefault(a, deque(maxlen=window)).append((int(as_), int(hs)))

    snapshot: dict[str, dict] = {}
    for team in set(elo) | set(history):
        dq = history.get(team)
        if dq:
            gf = sum(gs for gs, _ in dq) / len(dq)
            ga = sum(gc for _, gc in dq) / len(dq)
            pts = sum(3 if gs > gc else 1 if gs == gc else 0 for gs, gc in dq) / len(dq)
        else:
            gf = ga = pts = 0.0
        snapshot[team] = {
            "elo": float(elo.get(team, INITIAL_RATING)),
            "rolling_gf": float(gf),
            "rolling_ga": float(ga),
            "rolling_pts": float(pts),
            "matches": len(dq) if dq else 0,
        }
    return snapshot
