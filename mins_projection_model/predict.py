"""Inference contract consumed one-way by wc_props — a working fair-value engine.

predict(player, position, context)  -> per-player {prior minutes, rates90 per stat}.
price_lineup(team, opp, lineup, ...) -> the XI+subs with minutes NORMALIZED to the
    11*match_minutes roster budget and a priced expected count per stat. That
    normalized (minutes x rate) is the Poisson mean wc_props turns into a fair
    Kalshi rules line to quote around. Models load once and cache.

context = {team, opponent, neutral?, is_starter?, match_minutes?, book_sd?}
"""
from __future__ import annotations

import functools

import pandas as pd

import config
import rate_model
import minutes_model
import pricing
from features import player_ewm, RATE_STATS
from team_strength import load_elos


@functools.lru_cache(maxsize=1)
def _models():
    poss, goals, sot = load_elos()
    heads = {s: rate_model.load(s) for s in RATE_STATS}
    prior = minutes_model.load_prior()
    disp = pricing.load_dispersion()
    pm = pd.read_parquet(config.DATA_DIR / "player_matches.parquet").sort_values(["player", "date"])
    if "ga" not in pm.columns:
        pm["ga"] = pm["goals"].fillna(0) + pm["assists"].fillna(0)
    last_ewm = {}                                  # most-recent leak-free EWM per stat
    for s in RATE_STATS:
        e = pm.assign(_e=player_ewm(pm, stat=s)).dropna(subset=["_e"])
        last_ewm[s] = e.groupby("player")["_e"].last().to_dict()
    return poss, goals, sot, heads, prior, last_ewm, disp


def _script(poss, goals, sot, team, opp, neutral=True, mkt_team_g=None, mkt_opp_g=None):
    """Script (context) features for the rate heads. The Goals-Elo can't price
    cross-confederation mismatches (France/Iraq came out 0.87/0.80), so when the
    market's expected goals are supplied (bet365 Team Total Goals) we override the
    Elo's. NOTE: the heads were trained on Elo-scale exp_*_g (~0.6–1.3), so a tree
    won't extrapolate a 3.3 linearly — the override corrects direction/level within
    range; the per-team goalscorer anchor (devig) is what actually sets goal levels."""
    share = poss.expected(team, opp, neutral)
    eg_t, eg_o = goals.expected(team, opp, neutral)
    es_t, es_o = sot.expected(team, opp, neutral)
    if mkt_team_g is not None:
        eg_t = float(mkt_team_g)
    if mkt_opp_g is not None:
        eg_o = float(mkt_opp_g)
    return {"exp_team_share": share, "exp_team_g": eg_t, "exp_opp_g": eg_o,
            "exp_team_sot": es_t, "exp_opp_sot": es_o}


def predict(player: str, position: str, context: dict) -> dict:
    poss, goals, sot, heads, prior, last_ewm, _disp = _models()
    team, opp, neutral = context["team"], context["opponent"], context.get("neutral", True)
    script = _script(poss, goals, sot, team, opp, neutral,
                     context.get("market_team_g"), context.get("market_opp_g"))
    row = {"position": position or "NA", "source": "intl", **script}
    for s in RATE_STATS:
        row[f"ewm_{s}90"] = last_ewm[s].get(player, float("nan"))
    rates = {s: rate_model.predict_rate(heads[s], row, stat=s) for s in RATE_STATS}
    starter = context.get("is_starter", True)
    pmean, psd, p_full = minutes_model.prior_minutes(
        prior, position, starter, abs(script["exp_team_g"] - script["exp_opp_g"]))
    return {"p_start": 1.0 if starter else 0.0, "minutes_mean": pmean, "minutes_sd": psd,
            "p_full90": p_full, "rates90": rates, **script}


def price_lineup(team, opp, lineup, context=None, book_totals=None,
                 lines=None, half_spread=0.02):
    """lineup: [(player, position, is_starter), ...].
    book_totals: optional {player: {stat: book_total}} — reconcile minutes off the
        books' counting lines ACROSS the listed stats.
    lines: optional {player: {stat: line}} — the Kalshi/DFS thresholds to price;
        each priced stat then carries p_over + a YES bid/ask at `half_spread`.
    Returns {player: dict} with roster-normalized minutes, expected count per stat
    in d['exp'], and the full pricing (mean/var/fair_line[/p_over/quote]) in
    d['price']."""
    poss, goals, sot, heads, prior, last_ewm, disp = _models()
    ctx = dict(context or {})
    ctx["team"], ctx["opponent"] = team, opp
    match_minutes = ctx.get("match_minutes", 90)
    rows = {}
    for player, pos, starter in lineup:
        d = predict(player, pos, dict(ctx, is_starter=starter))
        bt = (book_totals or {}).get(player)
        if bt:                                      # books' counting lines -> implied minutes
            our = {s: d["rates90"].get(s) for s in bt}
            book_min = minutes_model.reconcile_minutes(bt, our)
            d["minutes_mean"], d["minutes_sd"] = minutes_model.posterior_minutes(
                d["minutes_mean"], d["minutes_sd"], book_min, ctx.get("book_sd", 12.0))
        rows[player] = d
    bench = [p for p, d in rows.items() if d["p_start"] < 1]
    if bench:                                   # not all listed subs appear (~5 of ~12)
        appear = min(minutes_model.EXP_SUBS_USED / len(bench), 1.0)
        for p in bench:
            rows[p]["minutes_mean"] *= appear
    raw = {p: d["minutes_mean"] for p, d in rows.items()}
    norm = minutes_model.normalize_roster_minutes(raw, match_minutes=match_minutes)
    for p, d in rows.items():
        d["minutes_raw"], d["minutes_mean"] = raw[p], norm[p]
        d["exp"] = {s: r * norm[p] / 90.0 for s, r in d["rates90"].items()}
        pl = (lines or {}).get(p, {})
        d["price"] = {s: pricing.price_market(d["rates90"][s], norm[p], d["minutes_sd"],
                                              disp.get(s, 1.0), line=pl.get(s),
                                              half_spread=half_spread)
                      for s in RATE_STATS}
    return rows
