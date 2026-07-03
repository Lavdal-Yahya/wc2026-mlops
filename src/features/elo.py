"""Leakage-safe Elo ratings.

Standard logistic Elo. Per-match output columns are the *pre-match* ratings;
the rating update only happens after the row is read, so each match's features
use information strictly available before kick-off.

K-factor varies by tournament importance — friendlies move ratings less than
World Cup games. Tournament categorization is intentionally simple (string
matching) to stay legible; refinements should be argued in DECISIONS.md.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

INITIAL_RATING = 1500.0

K_BY_CATEGORY = {
    "friendly": 20.0,
    "qualification": 30.0,
    "competitive": 40.0,
    "world_cup": 60.0,
}


def categorize_tournament(name: str) -> str:
    n = name.lower()
    if n == "friendly":
        return "friendly"
    if "qualification" in n:
        return "qualification"
    if "fifa world cup" in n:
        return "world_cup"
    return "competitive"


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_features(
    matches: pd.DataFrame,
    *,
    initial_rating: float = INITIAL_RATING,
    k_table: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run a sequential pass over (date-sorted) matches and emit pre-match Elo features.

    Returned DataFrame is indexed like ``matches`` and contains:
    ``elo_home, elo_away, elo_diff, elo_expected_home``.
    """
    k_table = k_table or K_BY_CATEGORY
    ratings: dict[str, float] = defaultdict(lambda: initial_rating)

    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date", kind="stable")

    n = len(matches)
    elo_home = np.empty(n, dtype=float)
    elo_away = np.empty(n, dtype=float)
    expected = np.empty(n, dtype=float)

    home_teams = matches["home_team"].to_numpy()
    away_teams = matches["away_team"].to_numpy()
    home_scores = matches["home_score"].to_numpy()
    away_scores = matches["away_score"].to_numpy()
    tournaments = matches["tournament"].to_numpy()

    for i in range(n):
        h, a = home_teams[i], away_teams[i]
        r_h, r_a = ratings[h], ratings[a]
        e_h = expected_score(r_h, r_a)

        elo_home[i] = r_h
        elo_away[i] = r_a
        expected[i] = e_h

        # Actual score from home perspective: 1 / 0.5 / 0.
        hs, as_ = home_scores[i], away_scores[i]
        if hs > as_:
            s_h = 1.0
        elif hs < as_:
            s_h = 0.0
        else:
            s_h = 0.5
        k = k_table[categorize_tournament(tournaments[i])]
        delta = k * (s_h - e_h)
        ratings[h] = r_h + delta
        ratings[a] = r_a - delta

    out = pd.DataFrame(
        {
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": elo_home - elo_away,
            "elo_expected_home": expected,
        },
        index=matches.index,
    )
    return out


def final_ratings(matches: pd.DataFrame, **kwargs) -> dict[str, float]:
    """Convenience: same sequential pass but returns the post-history rating dict.

    Useful as the simulation's starting state (Elo after the last played match).
    """
    _ = compute_elo_features(matches, **kwargs)
    ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)
    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date", kind="stable")
    k_table = kwargs.get("k_table") or K_BY_CATEGORY
    for h, a, hs, as_, t in zip(
        matches["home_team"], matches["away_team"],
        matches["home_score"], matches["away_score"], matches["tournament"],
    ):
        r_h, r_a = ratings[h], ratings[a]
        e_h = expected_score(r_h, r_a)
        s_h = 1.0 if hs > as_ else 0.0 if hs < as_ else 0.5
        k = k_table[categorize_tournament(t)]
        d = k * (s_h - e_h)
        ratings[h] = r_h + d
        ratings[a] = r_a - d
    return dict(ratings)
