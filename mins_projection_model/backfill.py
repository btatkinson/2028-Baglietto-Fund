"""One-shot historical backfill: pull config.INTL_SOURCES into cached frames.

Idempotent — api_football caches every response to disk, so an interrupted run
resumes for free and a re-run costs ~0 requests. Writes the combined frames to
DATA_DIR/{team_matches,player_matches}.parquet for the trainer to consume.
"""
from __future__ import annotations

import time

import pandas as pd

import config
import api_football as af


def main(sources=None, with_possession=True):
    sources = sources or config.INTL_SOURCES
    teams, players = [], []
    for lg, sn in sources:
        t0 = time.time()
        tf = af.team_match_frame(lg, sn, with_possession=with_possession)
        pf = af.player_match_frame(lg, sn)
        teams.append(tf)
        players.append(pf)
        print(f"  ({lg},{sn}): {len(tf):4d} team-matches, {len(pf):5d} player-matches "
              f"[{time.time() - t0:.0f}s]", flush=True)

    tm = pd.concat(teams, ignore_index=True).drop_duplicates("match_id")
    pm = pd.concat(players, ignore_index=True)
    tm.to_parquet(config.DATA_DIR / "team_matches.parquet", index=False)
    pm.to_parquet(config.DATA_DIR / "player_matches.parquet", index=False)
    print(f"TOTAL: {len(tm)} team-matches, {len(pm)} player-matches -> {config.DATA_DIR}",
          flush=True)
    return tm, pm


if __name__ == "__main__":
    main()
