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


def pipeline(scrape: bool = True, min_edge: float = 0.0, refresh: bool = True,
             trade: bool = False):
    """Run the full pipeline. Returns (report_path, summary_text).
    Raises FileNotFoundError if the pasted DFS payloads are missing.
    `trade=True` (run.py --trade) enables the live Kalshi order step with per-order approval."""
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

    # independent model fair value from each confirmed XI (empty until ~1h pre-KO)
    try:
        import projection
        proj = projection.project(b365)
        avail = [f["match"] for f in proj["fixtures"] if f.get("available")]
        projd = [f["match"] for f in proj["fixtures"] if f.get("lineup_kind") == "projected"]
        log(f"Projection: {proj['n_players']} players priced across {len(avail)} confirmed XI"
            + (f" ({', '.join(avail)})" if avail else "")
            + (f"; {len(projd)} projected XI ({', '.join(projd)})" if projd else "")
            + ("; no XI released yet" if not avail and not projd else "")
            + (f" — {proj['note']}" if proj.get("note") else ""))
    except Exception as e:
        proj = None
        log(f"Projection skipped: {e}")

    df = matcher.join(ud, pp, b365, proj=proj)
    if df.attrs.get("margins"):
        lines.append("One-way margins: " + df.attrs["margins"])
    n_edge = int((df["best_edge"].fillna(-9) > 0).sum())
    log(f"Matched/joined: {len(df):4d} rows  ({int(df['has_b365'].sum())} bet365-backed, "
        f"{n_edge} with positive edge)")

    # snapshot every source's raw prices (+ the model projection) for later
    # devig/weight calibration and reliability scoring against outcomes
    try:
        import calibration
        added, total = calibration.log_snapshot(ud, pp, b365, joined=df)
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

    krows = None
    try:
        import scorers
        # LLM injury/news veto: gated on the master switch (WCP_NEWS_ENABLED) AND a key.
        # Off by default — it spends Opus + web-search requests per act-on player.
        annotate = None
        if config.CLAUDE_API_KEY and config.NEWS_ENABLED:
            import news
            annotate = news.annotate
        # Kalshi take/make highlight on the board (post-XI + key only); fetched once here and
        # the rows returned so kalshi.html below reuses them — no second live call.
        sp, sc, note, krows, games = scorers.write(b365, proj, annotate=annotate,
                                                   kalshi=bool(config.KALSHI_KEY_ID), ud=ud)
        log(f"Scorer/assist board -> {sp}  (+ {sc.name})")
        if note:
            log(note)
        # cache the inputs so webapp's interactive /scorers can re-price minutes + serve
        # order tickets without re-running the model (board.py reads this pickle).
        try:
            import board
            board.save_state(b365, proj, krows, ud=ud)
        except Exception as e:
            log(f"Board state cache skipped: {e}")
    except Exception as e:
        games = None
        log(f"Scorer board skipped: {e}")

    # Kalshi overlay page, rendered from the same fetch the board used.
    if krows is not None:
        try:
            import kalshi_overlay
            kp = kalshi_overlay.write_html(krows)
            n_take = sum(1 for r in krows if r["take_yes"] or r["take_no"])
            log(f"Kalshi overlay -> {kp}  ({n_take} takeable of {len(krows)} overlaps)")
        except Exception as e:
            log(f"Kalshi overlay skipped: {e}")

    # LIVE order placement — only with --trade, a key set, and confirmed XIs. Every order is
    # shown and approved individually at the prompt; nothing is sent without a keystroke.
    if trade and games is not None and config.KALSHI_KEY_ID:
        try:
            import kalshi_orders
            orders = kalshi_orders.build_orders(games)
            kalshi_orders.approve_and_place(orders)
        except Exception as e:
            log(f"Kalshi order step skipped: {e}")
    elif trade and not config.KALSHI_KEY_ID:
        log("--trade given but KALSHI_API key not set — no order step.")

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
    ap.add_argument("--trade", action="store_true",
                    help="LIVE: build Kalshi orders from the confirmed-XI board and approve "
                         "each one at a y/N prompt before it is sent. Needs KALSHI_API set.")
    ap.add_argument("--bankroll", type=float, help="override WCP_KALSHI_BANKROLL for this run")
    ap.add_argument("--max-risk", type=float, help="override WCP_KALSHI_MAX_RISK ($/market)")
    args = ap.parse_args(argv)

    if args.bankroll is not None:
        config.KALSHI_BANKROLL = args.bankroll
    if args.max_risk is not None:
        config.KALSHI_MAX_RISK = args.max_risk

    try:
        path, _ = pipeline(scrape=args.scrape, min_edge=args.min_edge, refresh=args.refresh,
                           trade=args.trade)
    except FileNotFoundError as e:
        sys.exit(str(e))

    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
