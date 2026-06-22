"""Warm cold-start teams into the training frames before a live run.

Pulls each named team's recent fixtures across ALL competitions (incl. their
already-played WC2026 group games) and APPENDS to the parquets, so a team the
6-source backfill missed (e.g. New Zealand, an OFC side) gets real ratings + the
players get an EWM history. Then re-run train.py.

  python warmup.py "New Zealand" "Egypt"
"""
from __future__ import annotations

import sys

import pandas as pd

import config
import api_football as af

SEASONS = [2023, 2024, 2025, 2026]


def main(names):
    teams, players = [], []
    for name in names:
        tid = af.team_id(name)
        if not tid:
            print(f"  {name}: no team id found, skipped", flush=True)
            continue
        seen, fx = set(), []
        for sn in SEASONS:
            for f in af.team_fixtures(tid, sn):
                if f["fixture"]["id"] not in seen:
                    seen.add(f["fixture"]["id"])
                    fx.append(f)
        teams += [af._team_row(f) for f in fx]
        players += [r for f in fx for r in af._player_rows(f)]
        print(f"  {name} (id {tid}): {len(fx)} fixtures", flush=True)

    tm = pd.read_parquet(config.DATA_DIR / "team_matches.parquet")
    pm = pd.read_parquet(config.DATA_DIR / "player_matches.parquet")
    tm = pd.concat([tm, pd.DataFrame(teams)], ignore_index=True).drop_duplicates("match_id")
    pm = pd.concat([pm, pd.DataFrame(players)], ignore_index=True) \
           .drop_duplicates(["match_id", "player"])
    tm.to_parquet(config.DATA_DIR / "team_matches.parquet", index=False)
    pm.to_parquet(config.DATA_DIR / "player_matches.parquet", index=False)
    print(f"frames now: {len(tm)} team-matches, {len(pm)} player-matches", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["New Zealand", "Egypt"])
