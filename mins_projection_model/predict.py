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
from features import player_ewm, player_minutes_ewm, RATE_STATS
from team_strength import load_elos


@functools.lru_cache(maxsize=1)
def _models():
    poss, goals, sot, shots, corners = load_elos()
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
    min_ewm, min_n = player_minutes_ewm(pm)        # per-player bench minutes signal
    last_min_ewm = {p: (float(v), int(min_n.get(p, 0)))
                    for p, v in min_ewm.items() if pd.notna(v)}
    return poss, goals, sot, shots, corners, heads, prior, last_ewm, disp, last_min_ewm


def _script(poss, goals, sot, shots, corners, team, opp, neutral=True,
            mkt_team_g=None, mkt_opp_g=None):
    """Script (context) features for the rate heads, from the GLOBAL ratings
    (possession share, goals, SoT, total shots, corners — team & opp each). The
    market's expected goals (bet365 Team Total Goals), when supplied, override the
    rating's goals; the heads are now trained on the global ratings' realistic range
    so the override interpolates rather than extrapolates. NaN for shots/corners if
    those ratings are absent (old elos.json)."""
    nan = float("nan")
    eg_t, eg_o = goals.expected(team, opp, neutral)
    es_t, es_o = sot.expected(team, opp, neutral)
    sh_t, sh_o = shots.expected(team, opp, neutral) if shots else (nan, nan)
    co_t, co_o = corners.expected(team, opp, neutral) if corners else (nan, nan)
    if mkt_team_g is not None:
        eg_t = float(mkt_team_g)
    if mkt_opp_g is not None:
        eg_o = float(mkt_opp_g)
    return {"exp_team_share": poss.expected(team, opp, neutral),
            "exp_team_g": eg_t, "exp_opp_g": eg_o,
            "exp_team_sot": es_t, "exp_opp_sot": es_o,
            "exp_team_shots": sh_t, "exp_opp_shots": sh_o,
            "exp_team_corners": co_t, "exp_opp_corners": co_o}


def predict(player: str, position: str, context: dict) -> dict:
    poss, goals, sot, shots, corners, heads, prior, last_ewm, _disp, last_min_ewm = _models()
    team, opp, neutral = context["team"], context["opponent"], context.get("neutral", True)
    script = _script(poss, goals, sot, shots, corners, team, opp, neutral,
                     context.get("market_team_g"), context.get("market_opp_g"))
    row = {"position": position or "NA", "source": "intl", **script}
    for s in RATE_STATS:
        row[f"ewm_{s}90"] = last_ewm[s].get(player, float("nan"))
    rates = {s: rate_model.predict_rate(heads[s], row, stat=s) for s in RATE_STATS}
    starter = context.get("is_starter", True)
    sub_ewm, ewm_n = last_min_ewm.get(player) or (None, 0)     # per-player bench signal
    pmean, psd, p_full = minutes_model.prior_minutes(
        prior, position, starter, abs(script["exp_team_g"] - script["exp_opp_g"]),
        sub_min_ewm=sub_ewm, ewm_count=ewm_n)
    return {"p_start": 1.0 if starter else 0.0, "minutes_mean": pmean, "minutes_sd": psd,
            "p_full90": p_full, "rates90": rates, **script}


@functools.lru_cache(maxsize=1)
def _player_matches_sorted():
    pm = pd.read_parquet(config.DATA_DIR / "player_matches.parquet")
    return pm.sort_values(["team", "date"])


def recent_xi(team: str):
    """Pre-XI FALLBACK lineup for a team: the XI + bench it most recently fielded
    (from player_matches.parquet). Returns ([(player, pos, is_starter)], asof_date)
    — empty list if the team is absent. `team` must be the API-Football name (so it
    matches the parquet + the Elo lookup); resolve it from af.fixtures_on upstream.
    This is only as current as the data — warm up stale teams first (warmup.py)."""
    pm = _player_matches_sorted()
    t = pm[pm["team"] == team]
    if not len(t):
        return [], None
    asof = t["date"].max()
    last_mid = t.loc[t["date"] == asof, "match_id"].iloc[0]
    g = t[t["match_id"] == last_mid]
    starters = [(r.player, r.position, True) for r in g.itertuples() if r.is_starter]
    subs = [(r.player, r.position, False) for r in g.itertuples() if not r.is_starter]
    return starters + subs, str(asof)[:10]


# Team-coherence anchors: each maps a player's script dict -> the team total that
# the per-player projections of that stat should SUM to. The heads set the per-player
# SHARE; the (now-trustworthy) team rating/market sets the LEVEL — the same coherence
# idea as normalize_roster_minutes, applied to counting totals. goals uses exp_team_g
# (= the market override when supplied, else the rating); saves uses the SoT/goals
# identity (shots on target faced that weren't conceded). Stats with no clean team
# total (passes/tackles/assists/ga) are left unanchored.
# Share of goals that carry an assist. MEASURED at 0.686 over the API-Football
# training data, and it's a LEAGUE CONSTANT, not a team trait: a variance
# decomposition put the real team-to-team spread at ~0 (observed sd 0.078 ≈ pure
# binomial noise sd 0.080; split-half persistence corr −0.45). So we anchor every
# team's assist total to this constant × its goal total — a team-specific ratio
# (historical or model-predicted) would only add noise. Team creativity differences
# live in the per-player assist head (who assists), not this fraction. (Differs from
# devig.py's 0.75, which is the bet365 assist market's different definition/source.)
ASSIST_FRACTION = 0.686
_ANCHOR_TARGETS = {
    "goals": lambda s: s.get("exp_team_g"),
    "assists": lambda s: (None if s.get("exp_team_g") is None
                          else ASSIST_FRACTION * float(s["exp_team_g"])),
    "sot": lambda s: s.get("exp_team_sot"),
    "shots": lambda s: s.get("exp_team_shots"),
    "saves": lambda s: (None if s.get("exp_opp_sot") is None or s.get("exp_opp_g") is None
                        else max(0.0, float(s["exp_opp_sot"]) - float(s["exp_opp_g"]))),
}


def _anchor_rates(rows, norm, anchor_stats):
    """Scale each team's per-player rate90 so the anchored stats' team SUM matches the
    team total — preserving the heads' relative shares. Mutates rows[*]['rates90']."""
    if not rows or not anchor_stats:
        return {}
    script0 = next(iter(rows.values()))
    factors = {}
    for stat in anchor_stats:
        target = _ANCHOR_TARGETS[stat](script0)
        try:
            target = float(target)
        except (TypeError, ValueError):
            continue
        if target != target or target < 0:                  # NaN / negative
            continue
        cur = sum(d["rates90"].get(stat, 0.0) * norm[p] / 90.0 for p, d in rows.items())
        if cur > 1e-9:
            factors[stat] = target / cur
    for d in rows.values():
        for stat, f in factors.items():
            d["rates90"][stat] *= f
    return factors


def price_lineup(team, opp, lineup, context=None, book_totals=None,
                 lines=None, half_spread=0.02, anchor=True):
    """lineup: [(player, position, is_starter), ...].
    book_totals: optional {player: {stat: book_total}} — reconcile minutes off the
        books' counting lines ACROSS the listed stats.
    lines: optional {player: {stat: line}} — the Kalshi/DFS thresholds to price;
        each priced stat then carries p_over + a YES bid/ask at `half_spread`.
    anchor: scale per-player goals/sot/shots/saves rates so each team SUM matches the
        team rating/market total (shares from the heads, level from the rating).
    Returns {player: dict} with roster-normalized minutes, expected count per stat
    in d['exp'], and the full pricing (mean/var/fair_line[/p_over/quote]) in
    d['price']."""
    poss, goals, sot, shots, corners, heads, prior, last_ewm, disp, _min_ewm = _models()
    ctx = dict(context or {})
    ctx["team"], ctx["opponent"] = team, opp
    match_minutes = ctx.get("match_minutes", 90)
    rows = {}
    for player, pos, starter in lineup:
        d = predict(player, pos, dict(ctx, is_starter=starter))
        d["is_gk"] = str(pos or "").strip().upper() in ("G", "GK", "GOALKEEPER")
        if d["is_gk"]:               # keepers don't score/assist — clamp before anchoring
            for s in ("goals", "assists", "ga"):
                if s in d["rates90"]:
                    d["rates90"][s] = 0.0
        bt = (book_totals or {}).get(player)
        if bt:                                      # books' counting lines -> implied minutes
            our = {s: d["rates90"].get(s) for s in bt}
            book_min = minutes_model.reconcile_minutes(bt, our)
            d["minutes_mean"], d["minutes_sd"] = minutes_model.posterior_minutes(
                d["minutes_mean"], d["minutes_sd"], book_min, ctx.get("book_sd", 12.0))
        rows[player] = d
    cap = match_minutes + 8                       # full match incl. stoppage (= roster cap)
    for d in rows.values():                       # keepers: starter plays the full match, backups ~never
        if d.get("is_gk"):
            d["minutes_mean"] = cap if d["p_start"] >= 1 else 0.0
    raw = {p: d["minutes_mean"] for p, d in rows.items()}

    # Minutes are normalized in TWO independent pools, not one. Total field-time is
    # 11 slots x the full match incl. stoppage (= 11*cap). Of the 11 starters, about
    # EXP_SUBS_USED get hooked; the bench collectively covers exactly the minutes those
    # subs are on for — EXP_SUBS_USED x (mean minutes a sub plays once on). That bench
    # budget is FIXED and decoupled from the starters: when the book reconciliation pulls
    # a starter's minutes down, that field-time goes to the OTHER starters' coverage, not
    # to inflating every sub. (A single shared normalization did the latter — a low
    # starter sum scaled the whole bench up toward ~20', which is more than ~5 subs can
    # physically play. Per-bench minutes are now P(sub on) x mean-time = budget/|bench|.)
    starters = {p: v for p, v in raw.items() if rows[p]["p_start"] >= 1}
    bench = {p: v for p, v in raw.items() if rows[p]["p_start"] < 1}
    n_used = min(minutes_model.EXP_SUBS_USED, len(bench))      # short bench => fewer subs
    bench_budget = min(n_used * prior["sub"]["mean"], 11 * cap)
    norm = minutes_model.normalize_roster_minutes(
        starters, team_total=11 * cap - bench_budget, cap=cap, match_minutes=match_minutes)
    if bench:
        norm.update(minutes_model.normalize_roster_minutes(
            bench, team_total=bench_budget, cap=cap, match_minutes=match_minutes))
    # team-coherence: pin each anchored stat's team sum to the team rating/market total
    anchor_stats = [s for s in _ANCHOR_TARGETS if s in RATE_STATS] if anchor else []
    _anchor_rates(rows, norm, anchor_stats)
    for p, d in rows.items():
        d["minutes_raw"], d["minutes_mean"] = raw[p], norm[p]
        d["exp"] = {s: r * norm[p] / 90.0 for s, r in d["rates90"].items()}
        pl = (lines or {}).get(p, {})
        d["price"] = {s: pricing.price_market(d["rates90"][s], norm[p], d["minutes_sd"],
                                              disp.get(s, 1.0), line=pl.get(s),
                                              half_spread=half_spread)
                      for s in RATE_STATS}
    return rows
