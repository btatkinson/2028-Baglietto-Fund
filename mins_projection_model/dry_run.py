"""Pre-XI FALLBACK dry run — diagnose the projection before lineups drop.

  python dry_run.py "Portugal" "Uzbekistan" --date 2026-06-23 --home-g 2.61 --away-g 0.55

No confirmed XI is needed: each team's most-recently-fielded XI (predict.recent_xi)
stands in for the lineup, the market Team-Total goals override the Goals-Elo, and we
print the full per-player breakdown — prior minutes, leak-free EWM form, model
rates90, normalized minutes, and expected counts — so a bad projection is traceable
to its input (stale form, wrong minutes, missing player, or market not applied).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import api_football as af          # noqa: E402
import predict                     # noqa: E402
from features import RATE_STATS    # noqa: E402
from normalize import norm_country  # noqa: E402  (parent)

SHOW = ["passes", "shots", "sot", "tackles", "saves", "goals", "assists", "ga"]


def _resolve(home, away, date):
    """(api_home, api_away, status) from the date listing matched on country."""
    want = {norm_country(home), norm_country(away)}
    for f in af.fixtures_on(date):
        t = f["teams"]
        if {norm_country(t["home"]["name"]), norm_country(t["away"]["name"])} == want:
            return t["home"]["name"], t["away"]["name"], f["fixture"]["status"]["short"]
    return home, away, "?"


def _print_team(team, opp, lineup, asof, mkt_team_g, mkt_opp_g, ewm):
    ctx = {"neutral": True}
    if mkt_team_g is not None:
        ctx["market_team_g"] = mkt_team_g
    if mkt_opp_g is not None:
        ctx["market_opp_g"] = mkt_opp_g
    out = predict.price_lineup(team, opp, lineup, context=ctx)
    s = next(iter(out.values()))
    print(f"\n=== {team} (vs {opp}) — fallback XI as of {asof} ===")
    print(f"  script: poss_share {s['exp_team_share']:.2f} | exp_team_g {s['exp_team_g']:.2f} "
          f"(mkt {mkt_team_g}) | exp_opp_g {s['exp_opp_g']:.2f} (mkt {mkt_opp_g}) | "
          f"exp_team_sot {s['exp_team_sot']:.1f} | exp_opp_sot {s['exp_opp_sot']:.1f}")
    hdr = f"{'player':20s}{'pos':4s}{'min':>4s}{'±sd':>4s}" + "".join(f"{x[:5]:>7s}" for x in SHOW)
    print(hdr)
    for p, d in sorted(out.items(), key=lambda kv: -kv[1]["minutes_mean"]):
        if d["minutes_mean"] < 1:
            continue
        cells = "".join(f"{d['exp'][s]:>7.2f}" for s in SHOW)
        print(f"{p[:19]:20s}{'':4s}{d['minutes_mean']:4.0f}{d['minutes_sd']:4.0f}{cells}")
    # form check: top passer's EWM vs projected, to catch stale/empty form
    if ewm:
        top = max(out.items(), key=lambda kv: kv[1]["exp"]["passes"])
        nm = top[0]
        e = ewm.get(nm, {})
        print(f"  form check [{nm[:22]}]: ewm_passes90 {e.get('passes')} "
              f"-> rate90 {top[1]['rates90']['passes']:.1f}, "
              f"ewm_shots90 {e.get('shots')} -> rate90 {top[1]['rates90']['shots']:.2f}")


def _ewm_table():
    """{player: {stat: last leak-free ewm per-90}} for the form check."""
    poss, goals, sot, shots, corners, heads, prior, last_ewm, disp = predict._models()
    out = {}
    for s in RATE_STATS:
        for player, v in last_ewm[s].items():
            out.setdefault(player, {})[s] = None if v != v else round(float(v), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home"); ap.add_argument("away")
    ap.add_argument("--date", required=True)
    ap.add_argument("--home-g", type=float, default=None)
    ap.add_argument("--away-g", type=float, default=None)
    args = ap.parse_args()

    api_home, api_away, status = _resolve(args.home, args.away, args.date)
    print(f"{api_home} v {api_away}  ({args.date}, status {status})")
    hx, h_asof = predict.recent_xi(api_home)
    ax, a_asof = predict.recent_xi(api_away)
    for nm, xi, asof in ((api_home, hx, h_asof), (api_away, ax, a_asof)):
        print(f"  {nm}: fallback XI {len([p for p in xi if p[2]])} starters "
              f"+ {len([p for p in xi if not p[2]])} bench, last fielded {asof}"
              + ("  ⚠ NONE — team absent from data" if not xi else ""))
    ewm = _ewm_table()
    if hx:
        _print_team(api_home, api_away, hx, h_asof, args.home_g, args.away_g, ewm)
    if ax:
        _print_team(api_away, api_home, ax, a_asof, args.away_g, args.home_g, ewm)


if __name__ == "__main__":
    main()
