#!/usr/bin/env python3
"""main.py — one-button market-making review.

Captures the latest bet365 prices, blends them with your pasted Underdog/PrizePicks
lines (book-normalized de-vig on goals/assists), logs the snapshot for calibration,
and prints fair lines + low-vig Kalshi quotes for the key players in each REMAINING
game today — for you to eyeball before posting.

  python main.py                 # capture fresh, gated blend weight (~0.6 until fit)
  python main.py --w-ud 0.5      # force a 50/50 UD/365 blend
  python main.py --no-scrape     # reuse the last capture (no BetsAPI call)
  python main.py --all           # every captured game, not just today's remaining
  python main.py --kalshi        # overlay Kalshi order book (not wired yet)
  python main.py --top 8         # more players per game
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

import config
import capture as cap
import match as matcher
import calibration
import sources
from sources import parse_underdog, parse_prizepicks

sys.stdout.reconfigure(encoding="utf-8")        # WC names (ć, ñ, …) vs cp1252 console

MARKETS = ["Goals", "Assists", "Goals + assists", "Shots", "Shots on target", "Saves"]
HALF_SPREAD = 0.03                              # your edge: half the bid/ask width


def _quote(p, hs=HALF_SPREAD):
    return max(round(100 * (p - hs)), 1), min(round(100 * (p + hs)), 99)


def _remaining_today(b365):
    if "kickoff" not in b365.columns:
        return None
    now = datetime.now(timezone.utc)
    ko = pd.to_numeric(b365["kickoff"], errors="coerce")
    day = pd.to_datetime(ko, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return b365[(ko > now.timestamp()) & (day == now.strftime("%Y-%m-%d"))]


def _ko_label(b365, game):
    if "kickoff" not in b365.columns:
        return ""
    v = pd.to_numeric(b365.loc[b365["match"] == game, "kickoff"], errors="coerce").dropna()
    return datetime.fromtimestamp(int(v.iloc[0]), tz=timezone.utc).strftime("  ·  ko %H:%M UTC") if len(v) else ""


def _method_tag(m):
    m = str(m)
    return "gs-devig" if "gs-devig" in m else ("devig" if "devig" in m else "1way")


def main(argv=None):
    ap = argparse.ArgumentParser(description="One-button market-making review.")
    ap.add_argument("--w-ud", type=float, default=None,
                    help="UD weight in the UD/365 blend (default: gated/fitted ~0.6)")
    ap.add_argument("--no-scrape", action="store_true", help="reuse the last bet365 capture")
    ap.add_argument("--no-refresh", action="store_true", help="skip the live UD/PP pull (use the pasted files)")
    ap.add_argument("--all", action="store_true", help="all captured games, not just today's remaining")
    ap.add_argument("--kalshi", action="store_true", help="overlay Kalshi order book (not wired yet)")
    ap.add_argument("--top", type=int, default=6, help="key players per game")
    args = ap.parse_args(argv)

    if not args.no_refresh:
        st = sources.refresh_dfs()
        print(f"DFS refresh: UD {st['underdog']} · PP {st['prizepicks']}")
    ud, _ = parse_underdog(config.UNDERDOG_JSON)
    pp, _ = parse_prizepicks(config.PRIZEPICKS_JSON)
    b365 = cap.load_latest_flat() if args.no_scrape else cap.capture()
    if not len(b365):
        sys.exit("No bet365 prices available (capture failed or empty).")

    w = args.w_ud if args.w_ud is not None else calibration.blend_weight()
    df = matcher.join(ud, pp, b365, w_ud=w)
    try:
        added, _ = calibration.log_snapshot(ud, pp, b365)
    except Exception:
        added = 0

    pool = b365 if args.all else _remaining_today(b365)
    if pool is None or not len(pool):
        pool = b365
    games = list(dict.fromkeys(pool["match"].dropna())) if "match" in pool.columns else []

    scope = "all captured" if args.all else "today, remaining"
    tag = "forced" if args.w_ud is not None else "gated default"
    print(f"\nBlend {w:.0%} UD / {1 - w:.0%} 365 ({tag})  ·  {len(games)} game(s) [{scope}]  ·  logged +{added}")
    if args.kalshi:
        print("[--kalshi] order-book overlay not wired yet — needs your Kalshi API key + a ticker map.")
    print("=" * 74)

    order = {s: i for i, s in enumerate(MARKETS)}
    for game in games:
        sub = df[(df["match"] == game) & df["stat"].isin(MARKETS) & df["prob_ud"].notna()]
        if not len(sub):
            continue
        goals = sub[sub["stat"] == "Goals"].sort_values("prob_ud", ascending=False)
        stars = list(dict.fromkeys(goals["name"]))[:args.top]
        if not stars:
            continue
        print(f"\n▌ {game}{_ko_label(b365, game)}")
        for name in stars:
            print(f"  {name}")
            rows = sub[sub["name"] == name].assign(_o=lambda d: d["stat"].map(order)).sort_values("_o")
            for _, r in rows.iterrows():
                line = r["ud_line"] if pd.notna(r["ud_line"]) else r["pp_line"]
                lo, hi = _quote(r["prob_ud"])
                print(f"     {r['stat']:16s} o{line:<4g} {r['prob_ud']:4.0%}   YES {lo}¢/{hi}¢   [{_method_tag(r['method'])}]")
    print()


if __name__ == "__main__":
    main()
