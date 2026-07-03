"""Feature engineering for international match outcome prediction.

Builds pre-match features from the raw results.csv history:
  - v1: Elo ratings only
  - v2: Elo + rolling form (last N matches) + head-to-head (last N meetings)

All features are computed strictly from matches played *before* each match
(no leakage). Also produces a per-team "ratings snapshot" so the Flask app
can rebuild the same feature vector from two team names at serving time.
"""

from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd

# Feature contract shared with scripts/train.py (Mohamed) and the Flask app.
FEATURES_V1 = ["elo_home", "elo_away", "elo_diff", "neutral"]
FEATURES_V2 = FEATURES_V1 + [
    "form_home_pts", "form_home_gf", "form_home_ga",
    "form_away_pts", "form_away_gf", "form_away_ga",
    "h2h_home_wr", "h2h_home_gd", "h2h_played",
]
TARGET = "outcome"  # 0 = home win, 1 = draw, 2 = away win

# Neutral priors used when a team/pair has no history yet.
DEFAULTS = {
    "form_pts": 1.0,   # one point per match ~ draw-level form
    "form_gf": 1.25,   # historical average goals per team per match
    "form_ga": 1.25,
    "h2h_home_wr": 0.5,
    "h2h_home_gd": 0.0,
}


def load_raw(path: str) -> pd.DataFrame:
    """Load raw results, drop unplayed fixtures, sort chronologically."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(int)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def add_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Target from the home side's perspective: 0 win, 1 draw, 2 loss."""
    df = df.copy()
    df[TARGET] = 1
    df.loc[df["home_score"] > df["away_score"], TARGET] = 0
    df.loc[df["home_score"] < df["away_score"], TARGET] = 2
    return df


def compute_elo(df: pd.DataFrame, initial_rating: float, k_factor: float,
                home_advantage: float) -> tuple[pd.DataFrame, dict]:
    """Sequential Elo over the full history.

    Adds pre-match ratings (elo_home, elo_away, elo_diff) and returns the
    final per-team ratings for the serving snapshot.
    """
    ratings: dict[str, float] = defaultdict(lambda: initial_rating)
    elo_home, elo_away = [], []

    for home, away, hs, as_, neutral in zip(
        df["home_team"], df["away_team"],
        df["home_score"], df["away_score"], df["neutral"],
    ):
        r_home, r_away = ratings[home], ratings[away]
        elo_home.append(r_home)
        elo_away.append(r_away)

        adv = 0 if neutral else home_advantage
        expected_home = 1.0 / (1.0 + 10 ** ((r_away - r_home - adv) / 400.0))
        score_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        delta = k_factor * (score_home - expected_home)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    df = df.copy()
    df["elo_home"] = elo_home
    df["elo_away"] = elo_away
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df, dict(ratings)


def compute_form(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, dict]:
    """Rolling form per team over its last `window` matches (any venue).

    Pre-match averages of points (3/1/0), goals for and goals against.
    Returns the final rolling state for the serving snapshot.
    """
    history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
    cols = {name: [] for name in (
        "form_home_pts", "form_home_gf", "form_home_ga",
        "form_away_pts", "form_away_gf", "form_away_ga",
    )}

    def stats(team: str, side: str) -> None:
        past = history[team]
        if past:
            n = len(past)
            cols[f"form_{side}_pts"].append(sum(p for p, _, _ in past) / n)
            cols[f"form_{side}_gf"].append(sum(g for _, g, _ in past) / n)
            cols[f"form_{side}_ga"].append(sum(g for _, _, g in past) / n)
        else:
            cols[f"form_{side}_pts"].append(DEFAULTS["form_pts"])
            cols[f"form_{side}_gf"].append(DEFAULTS["form_gf"])
            cols[f"form_{side}_ga"].append(DEFAULTS["form_ga"])

    for home, away, hs, as_ in zip(df["home_team"], df["away_team"],
                                   df["home_score"], df["away_score"]):
        stats(home, "home")
        stats(away, "away")
        home_pts = 3 if hs > as_ else (1 if hs == as_ else 0)
        away_pts = 3 if as_ > hs else (1 if hs == as_ else 0)
        history[home].append((home_pts, hs, as_))
        history[away].append((away_pts, as_, hs))

    df = df.copy()
    for name, values in cols.items():
        df[name] = values
    state = {team: list(past) for team, past in history.items()}
    return df, state


def compute_h2h(df: pd.DataFrame, window: int) -> tuple[pd.DataFrame, dict]:
    """Head-to-head over the last `window` direct meetings of each pair.

    Stored per sorted pair from the perspective of the alphabetically first
    team, then flipped to the current home side when emitting features.
    Returns the final pair state for the serving snapshot.
    """
    history: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=window))
    wr, gd, played = [], [], []

    for home, away, hs, as_ in zip(df["home_team"], df["away_team"],
                                   df["home_score"], df["away_score"]):
        key = (home, away) if home < away else (away, home)
        sign = 1 if home == key[0] else -1
        past = history[key]
        if past:
            n = len(past)
            wins = sum(1 for d in past if d * sign > 0)
            wr.append(wins / n)
            gd.append(sign * sum(past) / n)
            played.append(n)
        else:
            wr.append(DEFAULTS["h2h_home_wr"])
            gd.append(DEFAULTS["h2h_home_gd"])
            played.append(0)
        history[key].append(sign * (hs - as_))  # goal diff, first-team view

    df = df.copy()
    df["h2h_home_wr"] = wr
    df["h2h_home_gd"] = gd
    df["h2h_played"] = played
    state = {key: list(past) for key, past in history.items()}
    return df, state


def build_features(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, dict]:
    """Full feature build over the raw history, per params['preprocess'].

    Returns the enriched dataframe and the serving snapshot (final Elo
    ratings, plus form/h2h state when feature_version == v2).
    """
    prep = params["preprocess"]
    version = prep["feature_version"]

    df = add_outcome(df)
    df, ratings = compute_elo(df, **prep["elo"])
    snapshot = {
        "feature_version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "params": prep,
        "ratings": ratings,
        "teams": sorted(ratings),
        "defaults": DEFAULTS,
    }

    if version == "v2":
        df, form_state = compute_form(df, prep["form"]["window"])
        df, h2h_state = compute_h2h(df, prep["h2h"]["window"])
        snapshot["form_state"] = form_state
        snapshot["h2h_state"] = h2h_state
    elif version != "v1":
        raise ValueError(f"Unknown feature_version: {version!r}")

    return df, snapshot


def time_split(df: pd.DataFrame, train_start: str,
               test_cut: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-aware split: train = [train_start, test_cut), test = [test_cut, )."""
    train = df[(df["date"] >= train_start) & (df["date"] < test_cut)]
    test = df[df["date"] >= test_cut]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def feature_columns(version: str) -> list[str]:
    return FEATURES_V1 if version == "v1" else FEATURES_V2
