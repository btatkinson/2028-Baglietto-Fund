"""project_cli.py — subprocess bridge from wc_props to the minutes/rate model.

The model is a self-contained sub-project that does `import config` against ITS OWN
config.py; the parent (wc_props) has a same-named config module, so importing the
model in-process collides. We therefore expose the model to the parent as a tiny
JSON-in/JSON-out subprocess that runs with this directory on sys.path[0].

  stdin  : {"half_spread": 0.03, "fixtures": [ <fixture request>, ... ]}
  stdout : {"fixtures": [ <fixture result>, ... ]}

fixture request:
  match            "Home v Away" label from bet365 (echoed back for the join)
  home, away       bet365 team labels (matched to API-Football by country)
  date             "YYYY-MM-DD" used to find the API-Football fixture id
  neutral          bool (WC = true)
  fixture_id       optional; skips the date lookup if supplied
  market_home_g    optional bet365 Team-Total expected goals for the HOME side
  market_away_g    optional ditto for the AWAY side (overrides the Goals-Elo, which
                   can't price cross-confederation mismatches)
  book_totals      optional {nname: {model_stat: implied_match_total}} — the books'
                   counting lines, for the minutes posterior. model_stat in
                   {passes, shots, sot, tackles, saves}.
  lines            optional {nname: {model_stat: threshold}} — DFS/Kalshi lines to
                   price (adds p_over + a YES bid/ask per priced stat).

fixture result:
  match, fixture_id, status, available (bool — false until the XI is released)
  players: [{nname, player, team, opp, pos, is_starter, minutes_mean, minutes_sd,
             exp:{stat:count}, price:{stat:{mean,var,fair_line[,line,p_over,...]}}}]

The book↔model NAME SEAM is closed here: the model's lineup carries accented
API-Football names (Aurélien Tchouaméni), while book_totals/lines arrive keyed by
the book's normalized name (norm_name). We match each lineup player's norm_name to
those dicts, and echo the nname back so the parent can re-join on the same key.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

# this dir first (so `import config`/`predict` hit the MODEL's modules), parent
# last (so `from normalize import ...` resolves to wc_props/normalize.py, which the
# model dir doesn't have — and which only depends on the stdlib, no config clash).
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import api_football as af          # noqa: E402
import predict                     # noqa: E402
from features import RATE_STATS    # noqa: E402
from normalize import norm_name, norm_country   # noqa: E402  (parent module)


@functools.lru_cache(maxsize=8)
def _fixtures_on(day):
    """Memoized so a whole-slate batch hits the date listing once per day."""
    return af.fixtures_on(day)


def _resolve_fixture(req):
    """(fixture_id, status, {country: af_team_name}) for one request, or (None,..)."""
    home_c, away_c = norm_country(req["home"]), norm_country(req["away"])
    if req.get("fixture_id"):
        meta = af.fixture_meta(req["fixture_id"])
        names = {norm_country(meta["home"]): meta["home"],
                 norm_country(meta["away"]): meta["away"]}
        return req["fixture_id"], meta["status"], names
    want = {home_c, away_c}
    for f in _fixtures_on(req["date"]):
        t = f["teams"]
        cs = {norm_country(t["home"]["name"]), norm_country(t["away"]["name"])}
        if cs == want:
            names = {norm_country(t["home"]["name"]): t["home"]["name"],
                     norm_country(t["away"]["name"]): t["away"]["name"]}
            return f["fixture"]["id"], f["fixture"]["status"]["short"], names
    return None, None, {}


def _by_nname(lineup, keyed):
    """Re-key a {nname: {...}} dict onto the lineup's API-Football names."""
    if not keyed:
        return {}
    out = {}
    for player, _pos, _starter in lineup:
        d = keyed.get(norm_name(player))
        if d:
            out[player] = d
    return out


def _price_team(team, opp, lineup, neutral, mkt_team_g, mkt_opp_g,
                book_totals, lines, half_spread):
    ctx = {"neutral": neutral}
    if mkt_team_g is not None:
        ctx["market_team_g"] = mkt_team_g
    if mkt_opp_g is not None:
        ctx["market_opp_g"] = mkt_opp_g
    rows = predict.price_lineup(
        team, opp, lineup, context=ctx,
        book_totals=_by_nname(lineup, book_totals) or None,
        lines=_by_nname(lineup, lines) or None,
        half_spread=half_spread)
    players = []
    for player, d in rows.items():
        price = {s: {k: v for k, v in d["price"][s].items()} for s in RATE_STATS}
        players.append({
            "nname": norm_name(player), "player": player,
            "team": team, "opp": opp,
            "pos": None, "is_starter": d["p_start"] >= 1.0,
            "minutes_mean": d["minutes_mean"], "minutes_sd": d["minutes_sd"],
            "exp": {s: d["exp"][s] for s in RATE_STATS},
            "price": price,
        })
    return players


def project_fixture(req, half_spread):
    out = {"match": req.get("match"), "fixture_id": req.get("fixture_id"),
           "status": None, "available": False, "lineup_kind": None, "players": []}
    fid, status, af_names = _resolve_fixture(req)
    out["fixture_id"], out["status"] = fid, status
    if fid is None:
        out["note"] = "fixture not found on API-Football"
        return out
    lus = af.lineup(fid)
    confirmed = len(lus) >= 2 and all(len(v) >= 11 for v in lus.values())
    if confirmed:
        lineups, kind = lus, "confirmed"
    else:
        # pre-XI FALLBACK: each team's most-recently-fielded XI stands in for the
        # lineup so we can still project minutes (recent_xi) before the official
        # sheet drops. Marked 'projected' so the parent keeps it out of the priced
        # report / calibration and only surfaces it as a soft minutes estimate.
        lineups, asof = {}, {}
        for c in (norm_country(req["home"]), norm_country(req["away"])):
            nm = af_names.get(c)
            xi, a = predict.recent_xi(nm) if nm else ([], None)
            if xi:
                lineups[nm], asof[nm] = xi, a
        if len(lineups) < 2:
            out["note"] = "confirmed XI not released yet; no fallback XI in data"
            return out
        kind = "projected"
        out["note"] = ("projected from most-recent XI ("
                       + ", ".join(f"{k}: {v}" for k, v in asof.items()) + ")")
    out["lineup_kind"] = kind

    # market goals are keyed to the bet365 sides; map each to the matching country
    mkt_by_country = {}
    if req.get("market_home_g") is not None:
        mkt_by_country[norm_country(req["home"])] = req["market_home_g"]
    if req.get("market_away_g") is not None:
        mkt_by_country[norm_country(req["away"])] = req["market_away_g"]

    book_totals, lines = req.get("book_totals") or {}, req.get("lines") or {}
    teams = list(lineups.keys())
    for team in teams:
        opp = next(t for t in teams if t != team)
        mkt_t = mkt_by_country.get(norm_country(team))
        mkt_o = mkt_by_country.get(norm_country(opp))
        out["players"] += _price_team(
            team, opp, lineups[team], req.get("neutral", True), mkt_t, mkt_o,
            book_totals, lines, half_spread)
    out["available"] = confirmed
    return out


def main():
    req = json.loads(sys.stdin.read() or "{}")
    half_spread = float(req.get("half_spread", 0.02))
    results = []
    for fx in req.get("fixtures", []):
        try:
            results.append(project_fixture(fx, half_spread))
        except Exception as e:  # one bad fixture shouldn't sink the batch
            results.append({"match": fx.get("match"), "available": False,
                            "error": f"{type(e).__name__}: {e}"})
    sys.stdout.write(json.dumps({"fixtures": results}))


if __name__ == "__main__":
    main()
