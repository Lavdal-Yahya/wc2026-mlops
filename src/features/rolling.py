"""Per-team rolling form features.

For each match we want, *strictly before kick-off*, summary statistics over
each side's last ``window`` matches:

- ``form_*_winrate``   — wins / (wins+draws+losses) over the window
- ``form_*_drawrate``  — draws / window
- ``form_*_ppg``       — points per game (3/1/0)
- ``form_*_gs``        — mean goals scored
- ``form_*_gc``        — mean goals conceded
- ``rest_*_days``      — days since the team's previous match (proxy for fatigue)

Implementation: one chronological pass, per-team deques of (gs, gc, points).
``home_team``/``away_team`` are just labels — the feature is from each team's
own match history regardless of which side they played on.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

WINDOW = 10


def _outcome_points(team_gs: int, team_gc: int) -> int:
    if team_gs > team_gc:
        return 3
    if team_gs == team_gc:
        return 1
    return 0


def compute_rolling_features(matches: pd.DataFrame, *, window: int = WINDOW) -> pd.DataFrame:
    """Return one row per match with rolling-form columns for home and away sides."""
    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date", kind="stable")

    n = len(matches)
    cols = [
        "form_home_winrate", "form_home_drawrate", "form_home_ppg",
        "form_home_gs", "form_home_gc", "rest_home_days",
        "form_away_winrate", "form_away_drawrate", "form_away_ppg",
        "form_away_gs", "form_away_gc", "rest_away_days",
    ]
    out = {c: np.full(n, np.nan, dtype=float) for c in cols}

    # state per team: deque of (gs, gc, points) over last `window` matches, + last date
    history: dict[str, deque] = {}
    last_date: dict[str, pd.Timestamp] = {}

    home = matches["home_team"].to_numpy()
    away = matches["away_team"].to_numpy()
    hs = matches["home_score"].to_numpy()
    as_ = matches["away_score"].to_numpy()
    dates = matches["date"].to_numpy()

    def snapshot(team: str, dt) -> tuple[float, float, float, float, float, float]:
        dq = history.get(team)
        if not dq:
            return (np.nan,) * 5 + (np.nan,)
        wins = sum(1 for gs_, gc_, _ in dq if gs_ > gc_)
        draws = sum(1 for gs_, gc_, _ in dq if gs_ == gc_)
        gs_mean = np.mean([gs_ for gs_, _, _ in dq])
        gc_mean = np.mean([gc_ for _, gc_, _ in dq])
        ppg = np.mean([p for _, _, p in dq])
        m = len(dq)
        rest = (pd.Timestamp(dt) - last_date[team]).days if team in last_date else np.nan
        return (wins / m, draws / m, ppg, gs_mean, gc_mean, rest)

    for i in range(n):
        h, a = home[i], away[i]
        (out["form_home_winrate"][i], out["form_home_drawrate"][i], out["form_home_ppg"][i],
         out["form_home_gs"][i], out["form_home_gc"][i], out["rest_home_days"][i]) = snapshot(h, dates[i])
        (out["form_away_winrate"][i], out["form_away_drawrate"][i], out["form_away_ppg"][i],
         out["form_away_gs"][i], out["form_away_gc"][i], out["rest_away_days"][i]) = snapshot(a, dates[i])

        # Update state — after reading features, so it's leakage-safe.
        history.setdefault(h, deque(maxlen=window)).append((int(hs[i]), int(as_[i]), _outcome_points(int(hs[i]), int(as_[i]))))
        history.setdefault(a, deque(maxlen=window)).append((int(as_[i]), int(hs[i]), _outcome_points(int(as_[i]), int(hs[i]))))
        last_date[h] = pd.Timestamp(dates[i])
        last_date[a] = pd.Timestamp(dates[i])

    return pd.DataFrame(out, index=matches.index)
