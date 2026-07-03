"""Stateful replay of historical matches up to a snapshot date.

Produces a frozen world-state (Elo, rolling form deques, last-played dates, H2H
counters) used by the tournament simulator to build per-match feature rows
without re-running the full feature pipeline.

Inside the tournament we update **Elo** (via outcome-only updates) and
**last_played**, but freeze rolling form and H2H. The WC is short (~7 matches
per team max) so within-tournament drift of those features is small relative to
the simulation noise, and a full update would require sampling synthetic
scorelines we don't model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.elo import INITIAL_RATING, K_BY_CATEGORY, categorize_tournament, expected_score
from src.models.common import FEATURE_COLS


def _points(team_gs: int, team_gc: int) -> int:
    if team_gs > team_gc:
        return 3
    if team_gs == team_gc:
        return 1
    return 0


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


@dataclass
class SimulationState:
    """Mutable snapshot of all per-team / per-pair feature state."""
    window: int = 10
    elo: dict[str, float] = field(default_factory=dict)
    history: dict[str, deque] = field(default_factory=dict)
    last_played: dict[str, pd.Timestamp] = field(default_factory=dict)
    # (n_meetings, wins_by_first_team_in_key, draws, sum_goal_diff_in_key_orientation)
    h2h: dict[tuple[str, str], tuple[int, int, int, int]] = field(default_factory=dict)

    def get_elo(self, team: str) -> float:
        return self.elo.get(team, INITIAL_RATING)

    def update_full(self, home: str, away: str, date: pd.Timestamp,
                    tournament: str, hs: int, as_: int) -> None:
        """Replay a historical match: update Elo, rolling, last_played, H2H."""
        r_h = self.get_elo(home)
        r_a = self.get_elo(away)
        e_h = expected_score(r_h, r_a)
        if hs > as_:
            s_h = 1.0
        elif hs < as_:
            s_h = 0.0
        else:
            s_h = 0.5
        k = K_BY_CATEGORY[categorize_tournament(tournament)]
        delta = k * (s_h - e_h)
        self.elo[home] = r_h + delta
        self.elo[away] = r_a - delta

        self.history.setdefault(home, deque(maxlen=self.window)).append((hs, as_, _points(hs, as_)))
        self.history.setdefault(away, deque(maxlen=self.window)).append((as_, hs, _points(as_, hs)))
        self.last_played[home] = date
        self.last_played[away] = date

        key = _pair_key(home, away)
        n, wf, drws, gdf = self.h2h.get(key, (0, 0, 0, 0))
        gd = hs - as_
        gd_in_key = gd if key[0] == home else -gd
        if gd > 0:
            won_first = 1 if key[0] == home else 0
        elif gd < 0:
            won_first = 1 if key[0] == away else 0
        else:
            won_first = 0
        is_draw = 1 if gd == 0 else 0
        self.h2h[key] = (n + 1, wf + won_first, drws + is_draw, gdf + gd_in_key)

    def update_elo_outcome(self, home: str, away: str, date: pd.Timestamp,
                            tournament: str, outcome: str) -> None:
        """Lightweight update for simulated matches: Elo + last_played only."""
        r_h = self.get_elo(home)
        r_a = self.get_elo(away)
        e_h = expected_score(r_h, r_a)
        s_h = 1.0 if outcome == "home_win" else 0.0 if outcome == "away_win" else 0.5
        k = K_BY_CATEGORY[categorize_tournament(tournament)]
        delta = k * (s_h - e_h)
        self.elo[home] = r_h + delta
        self.elo[away] = r_a - delta
        self.last_played[home] = date
        self.last_played[away] = date

    # ---- feature row builder --------------------------------------------------

    def _rolling_snapshot(self, team: str, dt: pd.Timestamp) -> dict:
        dq = self.history.get(team)
        if not dq:
            return {"winrate": np.nan, "drawrate": np.nan, "ppg": np.nan,
                    "gs": np.nan, "gc": np.nan, "rest": np.nan}
        m = len(dq)
        wins = sum(1 for gs_, gc_, _ in dq if gs_ > gc_)
        draws = sum(1 for gs_, gc_, _ in dq if gs_ == gc_)
        return {
            "winrate": wins / m,
            "drawrate": draws / m,
            "ppg": float(np.mean([p for _, _, p in dq])),
            "gs": float(np.mean([gs_ for gs_, _, _ in dq])),
            "gc": float(np.mean([gc_ for _, gc_, _ in dq])),
            "rest": (pd.Timestamp(dt) - self.last_played[team]).days
                     if team in self.last_played else np.nan,
        }

    def _h2h_snapshot(self, home: str, away: str) -> dict:
        key = _pair_key(home, away)
        n, wf, drws, gdf = self.h2h.get(key, (0, 0, 0, 0))
        if n == 0:
            return {"total": 0.0, "home_winrate": np.nan, "draw_rate": np.nan, "gd_mean": np.nan}
        if key[0] == home:
            home_wins = wf
            gd_sum = gdf
        else:
            home_wins = n - wf - drws
            gd_sum = -gdf
        return {
            "total": float(n),
            "home_winrate": home_wins / n,
            "draw_rate": drws / n,
            "gd_mean": gd_sum / n,
        }

    def feature_row(self, home: str, away: str, date: pd.Timestamp,
                    tournament: str, neutral: bool) -> dict:
        """Return one row of 25 features in the same order as FEATURE_COLS."""
        r_h = self.get_elo(home)
        r_a = self.get_elo(away)
        e_h = expected_score(r_h, r_a)
        h_form = self._rolling_snapshot(home, date)
        a_form = self._rolling_snapshot(away, date)
        h2h = self._h2h_snapshot(home, away)
        cat = categorize_tournament(tournament)
        row = {
            "elo_home": r_h, "elo_away": r_a,
            "elo_diff": r_h - r_a, "elo_expected_home": e_h,
            "form_home_winrate": h_form["winrate"], "form_home_drawrate": h_form["drawrate"],
            "form_home_ppg": h_form["ppg"], "form_home_gs": h_form["gs"],
            "form_home_gc": h_form["gc"], "rest_home_days": h_form["rest"],
            "form_away_winrate": a_form["winrate"], "form_away_drawrate": a_form["drawrate"],
            "form_away_ppg": a_form["ppg"], "form_away_gs": a_form["gs"],
            "form_away_gc": a_form["gc"], "rest_away_days": a_form["rest"],
            "h2h_total": h2h["total"], "h2h_home_winrate": h2h["home_winrate"],
            "h2h_draw_rate": h2h["draw_rate"], "h2h_goal_diff_mean": h2h["gd_mean"],
            "is_friendly": int(cat == "friendly"),
            "is_qualification": int(cat == "qualification"),
            "is_competitive": int(cat == "competitive"),
            "is_world_cup": int(cat == "world_cup"),
            "neutral_int": int(neutral),
        }
        # Sanity: row keys match FEATURE_COLS order
        assert list(row.keys()) == FEATURE_COLS, "feature_row keys drifted from FEATURE_COLS"
        return row


def replay_history(matches: pd.DataFrame, *, end_date: pd.Timestamp | None = None) -> SimulationState:
    """Build a SimulationState by replaying all matches up to (and including) end_date."""
    state = SimulationState()
    df = matches.sort_values("date", kind="stable")
    if end_date is not None:
        df = df[df["date"] <= end_date]
    for _, row in df.iterrows():
        state.update_full(
            home=row["home_team"], away=row["away_team"], date=row["date"],
            tournament=row["tournament"], hs=int(row["home_score"]), as_=int(row["away_score"]),
        )
    return state
