"""De-vig the one-way goalscorer / assist books.

bet365 prices "anytime to score" (and "to assist") one-sided per player, so 1/odds
is vig-inflated with no opposing price to de-vig against. But the market lists
EVERY player, so we normalize the whole book: the de-vigged P(score>=1) per player
must imply a total expected goals equal to the match total (the Goals O/U line).
Work in Poisson lambda-space (lambda_i = -ln(1-P_i), so sum lambda_i = expected
goals). Subtract a constant hazard delta from each lambda (floored at 0) so the sum
hits G: P_i = 1-exp(-max(0, lambda_i-delta)). This penalizes longshots most (they
floor to zero), preserves favorites, and can NEVER inflate (de-vig only removes vig)
— unlike a multiplicative margin (over-crushes favorites) or a power method (blows
up any favorite with lambda>1, i.e. P>0.63). Assists anchor to ~0.75*G;
Goals+assists is then the de-vigged union 1-(1-P_goal)(1-P_assist) — internally
consistent by construction.
"""
from __future__ import annotations

import math

import pandas as pd

from normalize import norm_name

ASSIST_FRACTION = 0.75
_DEFAULT_G = 2.6                      # fallback if no match-total line was captured
_GOAL_STATS = ("Goals", "Assists", "Goals + assists")


SOFT_FLOOR = 0.15      # each player keeps >= this fraction of its raw hazard (no zeros)


def two_way_prob(over_mult, under_mult, oneway_overround=None):
    """De-vigged P(over) from Underdog payout multipliers. A multiplier m is a PROFIT multiple
    (a winning pick returns stake×(1+m), so a multiplier <1 is a favourite), hence a side's
    vigged implied probability is 1/(1+m) — NOT 1/m. With BOTH sides posted we divide by the
    overround (io+iu) to strip the built-in margin. With only the over posted there's no
    opposing price to measure the margin: if the caller passes an assumed `oneway_overround`
    we de-vig by it (e.g. 1.5× a balanced book's vig), else a one-sided line returns None."""
    io = 1.0 / (1.0 + over_mult) if over_mult and over_mult > 0 else None
    if io is None:
        return None
    iu = 1.0 / (1.0 + under_mult) if under_mult and under_mult > 0 else None
    if iu is not None:
        return io / (io + iu)
    if oneway_overround and oneway_overround > 0:
        return min(io / oneway_overround, 0.999)
    return None


def _solve(raw, target):
    """raw: {key: book P(>=1)}. SOFT-FLOOR additive-hazard de-vig: in lambda-space
    subtract a constant delta from each lambda so the hazards sum to the team's
    expected goals, but floor each at SOFT_FLOOR * its own raw lambda (not at 0).
    This removes the overround mostly from the longshots while preserving the
    FAVOURITES' share (the diluted-by-40-longshots raw share that a plain
    multiplicative scale wrongly keeps) — and, unlike the plain additive that
    hard-floored at 0, it never zeros a legitimate mid-tier scorer. P_i = 1-exp(-l).

    NB this fixes the DISTRIBUTION (who gets the goals). The absolute LEVEL is set by
    `target` (bet365 Team Total Goals). The goalscorer market / Kalshi price the
    favourites higher — that implies a larger total than the (sharp, two-sided) Team
    Total market, i.e. the goalscorer side is likely the soft one; do NOT tune to it.
    The arbiter is the calibration reliability curve (do these probs match realized
    scoring rates?). Returns {key: de-vigged P}."""
    lam = {k: -math.log(1 - min(p, 0.999)) for k, p in raw.items()}
    base = sum(lam.values())
    if not lam or target <= 0 or base <= target:
        return {k: 1 - math.exp(-v) for k, v in lam.items()}   # no/low vig -> leave
    floor = {k: SOFT_FLOOR * v for k, v in lam.items()}
    s = lambda d: sum(max(floor[k], v - d) for k, v in lam.items())
    lo, hi = 0.0, max(lam.values())
    for _ in range(80):
        mid = (lo + hi) / 2
        if s(mid) > target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return {k: 1 - math.exp(-max(floor[k], v - d)) for k, v in lam.items()}


def _norm_group(g):
    """(group_total, {Goals|Assists: {nname: book P}}) for one normalization group.
    The group total is the captured market `team_total` (a side's expected goals),
    falling back to the match total split if absent (old captures)."""
    tt = pd.to_numeric(g.get("team_total"), errors="coerce").dropna() \
        if "team_total" in g.columns else pd.Series(dtype=float)
    if len(tt):
        total = float(tt.iloc[0])
    else:                                            # old capture: no per-team total
        gt = pd.to_numeric(g.get("match_total"), errors="coerce").dropna() \
            if "match_total" in g.columns else pd.Series(dtype=float)
        total = float(gt.iloc[0]) if len(gt) else _DEFAULT_G
    raw = {"Goals": {}, "Assists": {}}
    for stat in ("Goals", "Assists"):
        for r in g[g["stat"] == stat].itertuples():
            try:
                o = float(r.odds_decimal)
            except (TypeError, ValueError):
                continue
            if o > 1:
                raw[stat][r.nname] = 1.0 / o
    return total, raw


def goalscorer_probs(b365):
    """{(match, nname, stat): de-vigged P} for Goals / Assists / Goals+assists.
    Each team's scorers are normalized against THAT team's market expected goals
    (the captured `team_total` from bet365's Team Total Goals), not the match total
    over both squads — so France's stars anchor to France's ~3.3, not 3.7 diluted
    across Iraq too. Players are split home/away via the captured `player_team`;
    with no team split (old captures) we fall back to one per-match group on the
    match total."""
    out = {}
    if b365 is None or not len(b365) or "stat" not in getattr(b365, "columns", []):
        return out
    df = b365.copy()
    df["nname"] = df["player"].map(norm_name)
    has_team = "player_team" in df.columns and df["player_team"].notna().any()
    # group key per side; players with no captured team (or matches missing the team
    # markets) fall back to one per-match group anchored to the match total.
    df["_grp"] = df["player_team"].fillna("_match") if has_team else "all"
    for (match, _grp), g in df.groupby(["match", "_grp"]):
        total, raw = _norm_group(g)
        for stat, target in (("Goals", total), ("Assists", ASSIST_FRACTION * total)):
            for nn, p in _solve(raw[stat], target).items():
                out[(match, nn, stat)] = p
        for nn in {nn for (m, nn, _s) in out if m == match} & set(g["nname"]):
            pg, pa = out.get((match, nn, "Goals")), out.get((match, nn, "Assists"))
            if pg is not None and pa is not None:
                out[(match, nn, "Goals + assists")] = 1 - (1 - pg) * (1 - pa)
    return out
