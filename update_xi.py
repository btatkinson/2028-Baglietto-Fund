#!/usr/bin/env python3
"""Lightweight XI refresh — re-check lineups for the soonest kickoff(s) ONLY, without
re-running the full pipeline.

run.py re-captures bet365 (dozens of BetsAPI calls) + pulls Underdog/PrizePicks just to
end up re-checking the lineup. This skips all of that: it reuses the cached bet365 capture
(out/board_state.pkl, written by the last real run) and pings only API-Football's
`fixtures/lineups` for the next few games — typically 1-3 calls — then re-prices the board
cache so /scorers flips to "✓ XI confirmed" the moment the XI lands (the manual-XI handoff
in board.rebuild takes over automatically).

  python update_xi.py            # check the soonest game, no Kalshi pull
  python update_xi.py -n 2       # soonest 2 kickoffs
  python update_xi.py --kalshi   # also re-pull the live Kalshi book (a few more calls)
  python update_xi.py --watch 90 # poll every 90s, stop once all soonest are confirmed

Needs APISPORTS_KEY (same as the model). Reuses everything else from the cache.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

import config


def _soonest_matches(b365, n):
    """The n soonest not-yet-started matches in the cached capture, by kickoff."""
    df = b365.assign(_ko=pd.to_numeric(b365.get("kickoff"), errors="coerce"))
    ko = df.dropna(subset=["_ko"]).groupby("match")["_ko"].min().sort_values()
    now = time.time()
    upcoming = ko[ko > now - 600]                      # drop games already underway (10m grace)
    return list(upcoming.index[:n]), ko


def _load_b365():
    """Cached bet365 capture from the board state pickle, or the latest saved capture."""
    import board
    st = board.load_state()
    if st and st.get("b365") is not None and len(st["b365"]):
        return st["b365"], st
    import capture as cap
    return cap.load_latest_flat(), (st or {})


def _merge_proj(old, new):
    """Splice the freshly-projected matches into the cached proj, replacing ONLY the matches
    `new` actually returned a fixture for — so re-checking the soonest game doesn't wipe the
    proj for other already-confirmed board games, and a failed lookup never clobbers good data."""
    from projection import mkey
    old = old or {"cells": {}, "fixtures": [], "n_players": 0}
    fresh_mks = {mkey(f.get("match")) for f in new.get("fixtures", [])}
    fresh_mks.discard(None)
    if not fresh_mks:
        return old                                     # nothing usable came back — keep the cache
    cells = {k: v for k, v in (old.get("cells") or {}).items() if k[0] not in fresh_mks}
    cells.update(new.get("cells") or {})
    fixtures = [f for f in (old.get("fixtures") or []) if mkey(f.get("match")) not in fresh_mks]
    fixtures += new.get("fixtures", [])
    return {"cells": cells, "fixtures": fixtures,
            "n_players": len({(k[0], k[1]) for k in cells})}


def refresh(n=1, kalshi=False):
    """One XI check. Returns the list of (match, lineup_kind) for the soonest n. Pings only
    API-Football lineups (+ Kalshi if asked); reuses the cached bet365 capture."""
    import projection
    import scorers
    import board

    b365, st = _load_b365()
    if b365 is None or not len(b365):
        print("No cached bet365 capture — run `python run.py` once first.")
        return []
    soon, _ = _soonest_matches(b365, n)
    if not soon:
        print("No upcoming games in the cached capture (all started?).")
        return []

    # project ONLY the soonest matches → only those lineups get pinged
    subset = b365[b365["match"].isin(soon)]
    fresh = projection.project(subset)
    if fresh.get("note"):
        print(f"projection: {fresh['note']}")
    proj = _merge_proj(st.get("proj"), fresh)          # splice into the cached full-slate proj

    kinds = {f.get("match"): (f.get("lineup_kind"), f.get("status"), f.get("note"))
             for f in fresh.get("fixtures", [])}
    out = []
    for m in soon:
        kind, status, note = kinds.get(m, (None, None, None))
        flag = "✓ CONFIRMED" if kind == "confirmed" else (
            "… projected (XI not out)" if kind == "projected" else "— no lineup yet")
        print(f"  {m:30} [{status or '?'}]  {flag}" + (f"  ({note})" if note and kind != 'confirmed' else ""))
        out.append((m, kind))

    # update the cache so the live board re-prices off the new proj; keep the existing
    # Kalshi book unless asked to re-pull it.
    board.save_state(b365, proj, st.get("krows"))
    if kalshi:
        try:
            board.rebuild(refresh_kalshi=True)         # re-pull live Kalshi → set_book
            print("  Kalshi book refreshed.")
        except Exception as e:
            print(f"  Kalshi refresh skipped: {e}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=1, help="how many of the soonest kickoffs to check (default 1)")
    ap.add_argument("--kalshi", action="store_true", help="also re-pull the live Kalshi book")
    ap.add_argument("--watch", type=int, metavar="SECS",
                    help="poll every SECS seconds, stop once all checked games are confirmed")
    args = ap.parse_args(argv)

    while True:
        print(time.strftime("%H:%M:%S") + "  checking XI…")
        res = refresh(n=args.n, kalshi=args.kalshi)
        if not args.watch:
            break
        if res and all(k == "confirmed" for _, k in res):
            print("All checked games confirmed — done.")
            break
        time.sleep(max(15, args.watch))


if __name__ == "__main__":
    main()
