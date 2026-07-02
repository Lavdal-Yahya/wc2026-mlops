"""WC 2026 tournament simulator.

One simulation = (1) sample outcomes for the remaining group-stage fixtures,
(2) compute group standings, (3) build the 32-team bracket (group winners +
runners-up + best 8 third-placed teams), (4) play seeded single-elimination
knockouts (draws resolved by an even coin flip — penalty shootouts), (5) report
which round each team reached and the champion.

Simplifications (documented in DECISIONS.md):
- Elo updates between simulated WC matches; rolling form and H2H are frozen at
  WC-start state (within-tournament drift is tiny over ≤7 matches and we don't
  model scorelines).
- Group tiebreakers: points → pre-WC Elo → alphabetical. We don't track
  simulated goal difference because the model predicts outcome class only.
- 32-team bracket uses a *seeded* pyramid layout, not the official FIFA slot
  assignment (which is publicly defined but adds bookkeeping that doesn't
  change the per-team champion probability when seeding is fair).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.metrics import CLASSES
from src.models.common import FEATURE_COLS
from src.simulation.state import SimulationState

# Seeded R32 pairing (zero-indexed seeds) that keeps top seeds apart through to the SF.
# Pairs are ordered so that subsequent rounds are just adjacent-pair winners:
# R16 = winners of pairs (0,1), (2,3), ...; QF = (R16 0,1), (R16 2,3), ...
R32_PAIRS = [
    (0, 31), (15, 16), (7, 24), (8, 23), (3, 28), (12, 19), (4, 27), (11, 20),
    (1, 30), (14, 17), (6, 25), (9, 22), (2, 29), (13, 18), (5, 26), (10, 21),
]


@dataclass
class TournamentResult:
    champion: str
    final: tuple[str, str]            # the two finalists
    rounds_reached: dict[str, str]    # team -> deepest round name ('group','R32','R16','QF','SF','F','W')

ROUND_ORDER = ["group", "R32", "R16", "QF", "SF", "F", "W"]


def _outcome_from_proba(rng: np.random.Generator, proba_row: np.ndarray) -> str:
    """Sample one outcome label from a 3-vector aligned with CLASSES."""
    return str(rng.choice(CLASSES, p=proba_row))


def _knockout_winner(rng: np.random.Generator, proba_row: np.ndarray, home: str, away: str) -> str:
    """Sample a winner from a knockout match. Draws resolved 50/50 (penalty shootout)."""
    o = _outcome_from_proba(rng, proba_row)
    if o == "home_win":
        return home
    if o == "away_win":
        return away
    return home if rng.random() < 0.5 else away


def _batch_features(state: SimulationState, matches: list[tuple], dt: pd.Timestamp) -> pd.DataFrame:
    """Build a feature DataFrame for a batch of (home, away, tournament, neutral) tuples."""
    rows = [state.feature_row(h, a, dt, t, n) for (h, a, t, n) in matches]
    return pd.DataFrame(rows, columns=FEATURE_COLS)


def simulate_one(
    *,
    state: SimulationState,
    pipe,
    unplayed_group_fixtures: pd.DataFrame,
    played_group_results: pd.DataFrame,
    groups: dict[str, list[str]],
    rng: np.random.Generator,
    wc_date: pd.Timestamp,
) -> TournamentResult:
    """Run one full tournament simulation. Mutates a local copy of state — caller passes a fresh one."""
    rounds_reached: dict[str, str] = {team: "group" for g in groups.values() for team in g}

    # ----- 1. Group stage -----
    points: dict[str, int] = defaultdict(int)

    # Played matches: just count their points.
    for _, r in played_group_results.iterrows():
        h, a, hs, as_ = r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"])
        if hs > as_:
            points[h] += 3
        elif hs < as_:
            points[a] += 3
        else:
            points[h] += 1
            points[a] += 1

    # Simulate unplayed group matches in date order. Batch features per date for speed.
    upg = unplayed_group_fixtures.sort_values("date")
    matches_list = list(upg[["home_team", "away_team", "tournament", "neutral", "date"]].itertuples(index=False, name=None))
    if matches_list:
        feats_list = []
        for h, a, t, n, d in matches_list:
            feats_list.append(state.feature_row(h, a, d, t, bool(n)))
        X = pd.DataFrame(feats_list, columns=FEATURE_COLS)
        proba = pipe.predict_proba(X)
        for i, (h, a, t, _, d) in enumerate(matches_list):
            o = _outcome_from_proba(rng, proba[i])
            if o == "home_win":
                points[h] += 3
            elif o == "away_win":
                points[a] += 3
            else:
                points[h] += 1
                points[a] += 1
            state.update_elo_outcome(h, a, d, t, o)

    # ----- 2. Group standings -----
    # Sort each group by (points desc, Elo desc, name asc).
    winners, runners_up, thirds = [], [], []
    for g_id, teams in groups.items():
        ranked = sorted(teams, key=lambda t: (-points[t], -state.get_elo(t), t))
        winners.append(ranked[0])
        runners_up.append(ranked[1])
        thirds.append(ranked[2])

    # Best 8 third-placed teams.
    thirds_sorted = sorted(thirds, key=lambda t: (-points[t], -state.get_elo(t), t))
    best_thirds = thirds_sorted[:8]

    # ----- 3. Seeded R32 bracket -----
    # Tier 1: group winners (by points then Elo), tier 2: runners-up, tier 3: best thirds.
    winners_sorted = sorted(winners, key=lambda t: (-points[t], -state.get_elo(t), t))
    runners_sorted = sorted(runners_up, key=lambda t: (-points[t], -state.get_elo(t), t))
    thirds_seeded = best_thirds  # already sorted
    seeded_32: list[str] = winners_sorted + runners_sorted + thirds_seeded  # length 32

    for t in seeded_32:
        rounds_reached[t] = "R32"

    # ----- 4. Knockouts -----
    knockout_neutral = True
    knockout_tournament = "FIFA World Cup"

    def play_round(round_seeds: list[str], round_name: str) -> list[str]:
        pairs = list(zip(round_seeds[0::2], round_seeds[1::2]))
        matches = [(h, a, knockout_tournament, knockout_neutral) for h, a in pairs]
        X = _batch_features(state, matches, wc_date)
        proba = pipe.predict_proba(X)
        winners_round: list[str] = []
        for i, (h, a) in enumerate(pairs):
            w = _knockout_winner(rng, proba[i], h, a)
            winners_round.append(w)
            # Update Elo with an "implied" outcome (winner home? we treated h as home for simulation).
            o = "home_win" if w == h else "away_win"
            state.update_elo_outcome(h, a, wc_date, knockout_tournament, o)
        for w in winners_round:
            rounds_reached[w] = round_name
        return winners_round

    # R32 layout from R32_PAIRS indices.
    r32_seeds_ordered: list[str] = []
    for hi, ai in R32_PAIRS:
        r32_seeds_ordered.append(seeded_32[hi])
        r32_seeds_ordered.append(seeded_32[ai])

    r16_winners = play_round(r32_seeds_ordered, "R16")
    qf_winners = play_round(r16_winners, "QF")
    sf_winners = play_round(qf_winners, "SF")
    finalists = play_round(sf_winners, "F")
    champ_list = play_round(finalists, "W")
    champion = champ_list[0]
    rounds_reached[champion] = "W"

    return TournamentResult(
        champion=champion,
        final=tuple(finalists),  # type: ignore[arg-type]
        rounds_reached=rounds_reached,
    )


def detect_groups(fixtures_full: pd.DataFrame) -> dict[str, list[str]]:
    """From the 72 WC 2026 fixtures, derive the 12 groups (A..L)."""
    opponents: dict[str, set[str]] = defaultdict(set)
    for _, r in fixtures_full.iterrows():
        opponents[r["home_team"]].add(r["away_team"])
        opponents[r["away_team"]].add(r["home_team"])

    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    idx = 0
    for team in sorted(opponents):
        if team in seen:
            continue
        g = sorted({team} | opponents[team])
        groups[chr(ord("A") + idx)] = g
        seen.update(g)
        idx += 1
    return groups
