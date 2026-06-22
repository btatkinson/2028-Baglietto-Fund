#!/usr/bin/env python3
"""
run.py — the one button.

  1. parse data/underdog.json + data/prizepicks.json (you paste these)
  2. capture bet365 WC player props for the next 96h via BetsAPI
     (or fall back to the latest saved capture if no token)
  3. fuzzy-match across all three sources
  4. compute implied edge from bet365 and write out/report.html (sorted)

    cd wc_props
    python run.py                  # capture fresh bet365 lines, then match + report
    python run.py --no-scrape      # reuse last capture (debug fast, no BetsAPI calls)
    python run.py --min-edge 0.03  # console preview: only |edge| >= 3%
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from datetime import datetime, timezone

import config
import capture as cap
import match as matcher
import report as rpt
import sources
from sources import parse_underdog, parse_prizepicks


def pipeline(scrape: bool = True, min_edge: float = 0.0, refresh: bool = True):
    """Run the full pipeline. Returns (report_path, summary_text).
    Raises FileNotFoundError if the pasted DFS payloads are missing."""
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    if refresh:
        st = sources.refresh_dfs()                  # pull live UD/PP into the paste files
        log(f"DFS refresh: UD {st['underdog']} · PP {st['prizepicks']}")

    for p in (config.UNDERDOG_JSON, config.PRIZEPICKS_JSON):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} — paste the raw API JSON there first.")

    ud, n_alt = parse_underdog(config.UNDERDOG_JSON)
    pp, n_flash = parse_prizepicks(config.PRIZEPICKS_JSON)
    log(f"Underdog:   {len(ud):4d} lines  ({n_alt} alt collapsed)")
    log(f"PrizePicks: {len(pp):4d} lines  ({n_flash} demons/goblins excluded)")

    if not scrape:
        log("--no-scrape: reusing latest saved bet365 capture.")
        b365 = cap.load_latest_flat()
    elif not config.BETSAPI_TOKEN:
        log("No BETSAPI_TOKEN — reusing latest saved capture (edge needs bet365).")
        b365 = cap.load_latest_flat()
    else:
        log("Capturing bet365 WC props (next %dh)…" % config.WINDOW_HOURS)
        b365 = cap.capture()
    if not len(b365):
        log("  (no bet365 rows found; report will show DFS gaps only)")
    b365_players = b365[["player", "stat"]].drop_duplicates().shape[0] if len(b365) else 0
    log(f"bet365:     {len(b365):4d} prop rows  ({b365_players} player-stat ladders)")

    df = matcher.join(ud, pp, b365)
    if df.attrs.get("margins"):
        lines.append("One-way margins: " + df.attrs["margins"])
    n_edge = int((df["best_edge"].fillna(-9) > 0).sum())
    log(f"Matched/joined: {len(df):4d} rows  ({int(df['has_b365'].sum())} bet365-backed, "
        f"{n_edge} with positive edge)")

    # snapshot every source's raw prices for later devig/weight calibration
    try:
        import calibration
        added, total = calibration.log_snapshot(ud, pp, b365)
        log(f"Calibration log: +{added} price rows (total {total})")
    except Exception as e:
        log(f"Calibration log skipped: {e}")

    import lineups as lineup_mod
    lus = lineup_mod.generate(df)
    for plat in ("ud", "pp"):
        info = lus.get(plat) or {}
        log(f"Lineups [{(info.get('structure') or {}).get('label', plat)}]: "
            f"{len(info.get('lineups') or [])} built from {info.get('n_legs', 0)} qualified legs")

    meta = {
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_ud": len(ud), "n_pp": len(pp), "n_b365_players": b365_players,
        "margins": df.attrs.get("margins", ""),
    }
    path = rpt.write_report(df, meta, lus)
    log(f"Report -> {path}")

    prev = df[df["best_edge"].fillna(-9).abs() >= min_edge].head(20)
    cols = [c for c in ["name", "stat", "pp_line", "ud_line", "b365_lines",
                        "prob_pp", "side_pp", "edge_pp", "best_edge"] if c in prev.columns]
    if len(prev):
        import pandas as pd
        with pd.option_context("display.width", 200, "display.max_colwidth", 22):
            preview = prev[cols].to_string(index=False)
        print("\nTop edges:\n" + preview)
        lines.append("\nTop edges:\n" + preview)

    return path, "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape", action=argparse.BooleanOptionalAction, default=True,
                    help="capture fresh bet365 lines (default). Use --no-scrape to reuse "
                         "the last capture without re-hitting BetsAPI while debugging.")
    ap.add_argument("--refresh", action=argparse.BooleanOptionalAction, default=True,
                    help="pull live Underdog/PrizePicks lines into the paste files first "
                         "(default). Use --no-refresh for fully offline iteration.")
    ap.add_argument("--min-edge", type=float, default=0.0, help="console preview filter on |edge|")
    ap.add_argument("--open", action="store_true", help="open the report in a browser when done")
    args = ap.parse_args(argv)

    try:
        path, _ = pipeline(scrape=args.scrape, min_edge=args.min_edge, refresh=args.refresh)
    except FileNotFoundError as e:
        sys.exit(str(e))

    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
