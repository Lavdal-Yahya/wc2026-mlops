"""Head-to-head features.

For each match, summarize the *prior* meetings between the two specific teams,
regardless of which side each played on previously:

- ``h2h_total``           — number of prior meetings
- ``h2h_home_winrate``    — share of those meetings the current home team won
- ``h2h_draw_rate``       — share that were draws
- ``h2h_goal_diff_mean``  — mean (current_home_goals - current_away_goals) over those meetings

The "home" perspective is the *current* match's home team — we look up their
historical record against the current away team and re-orient each prior
meeting's scoreline to match that frame.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def compute_h2h_features(matches: pd.DataFrame) -> pd.DataFrame:
    if not matches["date"].is_monotonic_increasing:
        matches = matches.sort_values("date", kind="stable")

    n = len(matches)
    total = np.zeros(n, dtype=float)
    home_winrate = np.full(n, np.nan, dtype=float)
    draw_rate = np.full(n, np.nan, dtype=float)
    gd_mean = np.full(n, np.nan, dtype=float)

    # state per unordered pair: (n_meetings, wins_by_first_team_in_key, draws, sum_goal_diff_in_key_orientation)
    state: dict[tuple[str, str], tuple[int, int, int, int]] = defaultdict(lambda: (0, 0, 0, 0))

    home = matches["home_team"].to_numpy()
    away = matches["away_team"].to_numpy()
    hs = matches["home_score"].to_numpy()
    as_ = matches["away_score"].to_numpy()

    for i in range(n):
        h, a = home[i], away[i]
        key = _pair_key(h, a)
        nm, wins_first, draws, gd_first = state[key]
        total[i] = nm
        if nm > 0:
            # Re-orient stored stats to the current home team's frame.
            if key[0] == h:
                home_wins = wins_first
                home_gd_sum = gd_first
            else:
                home_wins = nm - wins_first - draws
                home_gd_sum = -gd_first
            home_winrate[i] = home_wins / nm
            draw_rate[i] = draws / nm
            gd_mean[i] = home_gd_sum / nm

        # Update state.
        gd = int(hs[i]) - int(as_[i])
        gd_in_key = gd if key[0] == h else -gd
        if gd > 0:
            won_first = 1 if key[0] == h else 0
        elif gd < 0:
            won_first = 1 if key[0] == a else 0
        else:
            won_first = 0
        is_draw = 1 if gd == 0 else 0
        state[key] = (nm + 1, wins_first + won_first, draws + is_draw, gd_first + gd_in_key)

    return pd.DataFrame(
        {
            "h2h_total": total,
            "h2h_home_winrate": home_winrate,
            "h2h_draw_rate": draw_rate,
            "h2h_goal_diff_mean": gd_mean,
        },
        index=matches.index,
    )
