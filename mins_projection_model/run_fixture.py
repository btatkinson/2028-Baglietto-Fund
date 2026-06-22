"""Price a live fixture end-to-end from the confirmed XI.

  python run_fixture.py <fixture_id>

Pulls the confirmed lineups (API-Football, uncached), prices each team's roster
with roster-normalized minutes, and prints the projected expected count per stat
plus sample low-vig Kalshi quotes. Re-run if the XI isn't out yet (~1h pre-KO).
"""
from __future__ import annotations

import sys

import api_football as af
import predict
import pricing

sys.stdout.reconfigure(encoding="utf-8")   # WC names (ć, ñ, …) vs cp1252 console

SHOW = ["passes", "shots", "sot", "tackles", "saves", "ga"]


def _print_team(team, opp, out):
    s = next(iter(out.values()))
    print(f"\n=== {team} (vs {opp})   exp team SoT {s['exp_team_sot']:.1f} / opp SoT "
          f"{s['exp_opp_sot']:.1f} ===")
    print(f"{'player':21s}{'pos':4s}{'min':>4s}" + "".join(f"{x:>8s}" for x in SHOW))
    for p, d in sorted(out.items(), key=lambda kv: -kv[1]["minutes_mean"]):
        if d["minutes_mean"] < 1:
            continue
        cells = "".join(f"{d['exp'][s]:>8.1f}" for s in SHOW)
        print(f"{p[:20]:21s}{'':4s}{d['minutes_mean']:4.0f}{cells}")

    # sample markets: each keeper's saves, plus the top passer's passes
    print(" sample Kalshi quotes (fair line / YES bid-ask @2c):")
    gk = [(p, d) for p, d in out.items() if d["minutes_mean"] > 1 and d["exp"]["saves"] > 0.8]
    top_pass = sorted(out.items(), key=lambda kv: -kv[1]["exp"]["passes"])[:1]
    for p, d in gk + top_pass:
        for stat in (["saves"] if d in [x[1] for x in gk] else ["passes"]):
            pr = d["price"][stat]
            line = pr["fair_line"]
            pp = pricing.prob_over(pr["mean"], pr["var"], line)
            bid, ask = pricing.quote(pp)
            print(f"   {p[:22]:23s} {stat:7s} line {line:>4} | P(over) {pp:4.0%} | "
                  f"YES {bid}¢-{ask}¢")


def main(fixture_id):
    meta = af.fixture_meta(fixture_id)
    print(f"{meta['home']} v {meta['away']}  ({meta['league']}, status {meta['status']})")
    lus = af.lineup(fixture_id)
    if len(lus) < 2 or any(len(v) < 11 for v in lus.values()):
        print("\nConfirmed XI not released yet — re-run ~1h before kickoff.")
        return
    home, away = meta["home"], meta["away"]
    for team, opp in [(home, away), (away, home)]:
        out = predict.price_lineup(team, opp, lus.get(team, []),
                                   context={"neutral": meta["neutral"]})
        _print_team(team, opp, out)


if __name__ == "__main__":
    main(int(sys.argv[1]))
