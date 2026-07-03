"""Run N Monte Carlo tournament simulations and aggregate per-team statistics.

Usage:
    .venv/bin/python -m src.simulation.runner [N]

Default N = 20000. Outputs:
- reports/wc2026_champion_probs.csv     team, champ_rate, finalist_rate, sf_rate, qf_rate, r16_rate, r32_rate
- reports/wc2026_groups.json            detected groups A..L
"""
from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.load import load_raw
from src.simulation.state import replay_history
from src.simulation.tournament import detect_groups, simulate_one
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.seeds import set_global_seed

DEFAULT_N = 20_000
WC_START = pd.Timestamp("2026-06-11")
SNAPSHOT_DATE = pd.Timestamp("2026-06-13")  # last fully-played day

REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "models" / "lr_baseline.joblib"

ROUND_PROMOTIONS = {
    "group": [],
    "R32": ["r32"],
    "R16": ["r32", "r16"],
    "QF":  ["r32", "r16", "qf"],
    "SF":  ["r32", "r16", "qf", "sf"],
    "F":   ["r32", "r16", "qf", "sf", "finalist"],
    "W":   ["r32", "r16", "qf", "sf", "finalist", "champ"],
}


def main(n_sims: int = DEFAULT_N) -> None:
    set_global_seed(42)
    REPORT_DIR.mkdir(exist_ok=True)

    pipe = joblib.load(MODEL_PATH)

    bundle = load_raw()
    cfg = load_config()  # noqa: F841 — reserved for future config-driven simulation params

    # 1. Build snapshot state at WC start (history + all matches up to SNAPSHOT_DATE,
    #    which includes the 8 WC 2026 matches already played).
    base_state = replay_history(bundle.matches, end_date=SNAPSHOT_DATE)

    # 2. Group fixtures: combine matches (played WC 2026) + fixtures (unplayed WC 2026).
    all_wc = pd.concat([bundle.matches, bundle.fixtures], ignore_index=True)
    wc26 = all_wc[(all_wc["tournament"] == "FIFA World Cup")
                  & (all_wc["date"] >= "2026-01-01")].copy()
    played = wc26[wc26["home_score"].notna()].copy()
    unplayed = wc26[wc26["home_score"].isna()].copy()
    print(f"Played: {len(played)} | Unplayed: {len(unplayed)} | Total fixtures: {len(wc26)}")

    groups = detect_groups(wc26)
    with open(REPORT_DIR / "wc2026_groups.json", "w") as f:
        json.dump(groups, f, indent=2)

    teams = sorted({t for g in groups.values() for t in g})
    assert len(teams) == 48, f"expected 48 teams, got {len(teams)}"

    # 3. Run N simulations.
    counters = {key: Counter() for key in ["champ", "finalist", "sf", "qf", "r16", "r32"]}
    rng = np.random.default_rng(42)

    for i in tqdm(range(n_sims), desc=f"Simulating {n_sims:,} tournaments"):
        state = copy.deepcopy(base_state)
        # New RNG per simulation, derived from the master generator.
        sim_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        result = simulate_one(
            state=state, pipe=pipe,
            unplayed_group_fixtures=unplayed,
            played_group_results=played,
            groups=groups, rng=sim_rng, wc_date=WC_START,
        )
        for team, deepest in result.rounds_reached.items():
            for level in ROUND_PROMOTIONS[deepest]:
                counters[level][team] += 1

    # 4. Aggregate.
    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "champ_rate":    counters["champ"][t]    / n_sims,
            "finalist_rate": counters["finalist"][t] / n_sims,
            "sf_rate":       counters["sf"][t]       / n_sims,
            "qf_rate":       counters["qf"][t]       / n_sims,
            "r16_rate":      counters["r16"][t]      / n_sims,
            "r32_rate":      counters["r32"][t]      / n_sims,
        })
    out = pd.DataFrame(rows).sort_values("champ_rate", ascending=False).reset_index(drop=True)
    out.to_csv(REPORT_DIR / "wc2026_champion_probs.csv", index=False)

    print("\nTop 15 champion probabilities:")
    print(out.head(15).to_string(index=False))
    print(f"\nSum of champ rates: {out['champ_rate'].sum():.4f} (should be 1.0)")
    print(f"Wrote {REPORT_DIR / 'wc2026_champion_probs.csv'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    main(n)
