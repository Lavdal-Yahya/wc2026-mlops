"""Raw data loading and normalization.

Loads the four-file Kaggle dataset (martj42/international-football-results-from-1872-to-2026)
and returns clean, typed pandas DataFrames. Splits played matches from unplayed
future fixtures (notably WC 2026 group games yet to be played).

Team-name normalization uses ``former_names.csv``: matches whose date falls inside
[start_date, end_date] of a renaming row are mapped to the *current* country name.
This is important for Elo/feature continuity (e.g. Dahomey -> Benin in 1975).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import PROJECT_ROOT, load_config


OUTCOME_HOME_WIN = "home_win"
OUTCOME_DRAW = "draw"
OUTCOME_AWAY_WIN = "away_win"


@dataclass
class FootballData:
    """Container for the three working tables."""
    matches: pd.DataFrame      # past matches with known scores + derived outcome label
    fixtures: pd.DataFrame     # scheduled-but-unplayed matches (e.g. WC 2026 future games)
    shootouts: pd.DataFrame    # penalty shootout outcomes for knockout-stage simulation


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _build_rename_map(former_names: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp, str]]:
    """Return a list of (former_name, start, end, current_name) tuples."""
    fn = former_names.copy()
    fn["start_date"] = pd.to_datetime(fn["start_date"], errors="coerce")
    fn["end_date"] = pd.to_datetime(fn["end_date"], errors="coerce")
    fn["end_date"] = fn["end_date"].fillna(pd.Timestamp("2099-12-31"))
    return list(fn[["former", "start_date", "end_date", "current"]].itertuples(index=False, name=None))


def _normalize_team_names(df: pd.DataFrame, former_names: pd.DataFrame) -> pd.DataFrame:
    """Map historical country names to their current equivalents based on match date."""
    rules = _build_rename_map(former_names)
    out = df.copy()
    for col in ("home_team", "away_team"):
        for former, start, end, current in rules:
            mask = (out[col] == former) & (out["date"] >= start) & (out["date"] <= end)
            out.loc[mask, col] = current
    return out


def _derive_outcome(df: pd.DataFrame) -> pd.Series:
    """Three-class label from the home team's perspective."""
    diff = df["home_score"] - df["away_score"]
    return pd.Series(
        pd.Categorical(
            ["home_win" if d > 0 else "away_win" if d < 0 else "draw" for d in diff],
            categories=[OUTCOME_HOME_WIN, OUTCOME_DRAW, OUTCOME_AWAY_WIN],
        ),
        index=df.index,
        name="outcome",
    )


def load_raw(config: dict | None = None, normalize_names: bool = True) -> FootballData:
    """Load all raw CSVs and return a :class:`FootballData` bundle."""
    config = config or load_config()
    paths = config["paths"]

    results = pd.read_csv(_resolve(paths["results_csv"]), parse_dates=["date"])
    shootouts = pd.read_csv(_resolve(paths["shootouts_csv"]), parse_dates=["date"])
    former_names = pd.read_csv(_resolve(paths["former_names_csv"]))

    if normalize_names:
        results = _normalize_team_names(results, former_names)
        shootouts = _normalize_team_names(shootouts, former_names)

    results = results.sort_values("date", kind="stable").reset_index(drop=True)

    played_mask = results["home_score"].notna() & results["away_score"].notna()
    matches = results.loc[played_mask].copy()
    matches["home_score"] = matches["home_score"].astype(int)
    matches["away_score"] = matches["away_score"].astype(int)
    matches["outcome"] = _derive_outcome(matches)

    fixtures = results.loc[~played_mask].copy()

    return FootballData(matches=matches, fixtures=fixtures, shootouts=shootouts)
