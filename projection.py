"""Bridge: wc_props  ->  the minutes/rate fair-value model.

The model (mins_projection_model/) is a self-contained sub-project with its OWN
config.py, so it can't be imported in-process without a module name clash. We talk
to it as a JSON subprocess (project_cli.py) that runs with the model dir on its
path. This module:

  1. turns the latest bet365 capture into per-fixture requests — market expected
     goals (bet365 Team Total Goals, which the Goals-Elo can't price), and the
     books' counting lines as implied per-player match totals for the minutes
     posterior;
  2. runs the model over the slate (confirmed XI only — empty until ~1h pre-KO);
  3. returns a fair-value cell per (match, normalized-name, canonical stat) as a
     NegBin/Poisson (mean, var) the caller turns into P(over any DFS/Kalshi line).

The book↔model NAME SEAM (accented API-Football 'Aurélien Tchouaméni' vs the book's
'Dembele') is closed by keying everything on normalize.norm_name; the model echoes
each player's nname back so we re-join on the same key.

This reads market prices to produce a model fair value; it is not betting advice.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

import config
from normalize import norm_name, norm_country

_MPM_DIR = config.ROOT / "mins_projection_model"
_CLI = _MPM_DIR / "project_cli.py"

# canonical report stat  <->  model rate-head key
STAT_TO_MODEL = {"Shots": "shots", "Shots on target": "sot", "Saves": "saves",
                 "Passes": "passes", "Tackles": "tackles",
                 "Goals": "goals", "Assists": "assists", "Goals + assists": "ga"}
MODEL_TO_STAT = {v: k for k, v in STAT_TO_MODEL.items()}
# stats whose bet365 ladders are reliable enough to back out implied minutes
_BOOK_MINUTE_STATS = {"Shots", "Shots on target", "Saves", "Passes", "Tackles"}


def mkey(match) -> str | None:
    """Order-independent country-pair key for a 'Home v Away' label — the same seam
    calibration._mkey uses, so bet365 / DFS / API-Football labels collapse to one
    key regardless of country aliases or home/away order."""
    cs = sorted({norm_country(p.strip()) for p in str(match).split(" v ") if p.strip()})
    return "|".join(cs) if len(cs) == 2 else None


# --------------------------- probability port ---------------------------
# mirrors mins_projection_model/pricing.py so the parent can read P(over) off the
# returned (mean, var) at whatever DFS/Kalshi line it discovers later.

def prob_over(mean, var, line):
    """P(count > line) for the NegBin/Poisson the model fitted (var floored at mean)."""
    if mean is None or var is None or line is None:
        return None
    try:
        line = float(line)
    except (TypeError, ValueError):
        return None
    if math.isnan(line):
        return None
    n = math.floor(line) + 1
    if n <= 0:
        return 1.0
    if mean <= 0:
        return 0.0
    from scipy.stats import nbinom, poisson
    if var <= mean * 1.0001:
        return float(poisson.sf(n - 1, mean))
    p = mean / var
    r = mean * mean / (var - mean)
    return float(nbinom.sf(n - 1, r, p))


# --------------------------- request building ---------------------------

def _poisson_mean_for_tail(k, p_ge_k):
    """mu with P(X >= k) == p_ge_k for X ~ Poisson(mu); k >= 1. (Same solve as
    capture._poisson_mean_for_tail, duplicated to avoid importing the scraper.)"""
    if k < 1 or not (0.0 < p_ge_k < 1.0):
        return None

    def p_ge(mu):
        term = cdf = math.exp(-mu)
        for i in range(1, k):
            term *= mu / i
            cdf += term
        return 1.0 - cdf

    lo, hi = 1e-6, 25.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_ge(mid) < p_ge_k else (lo, mid)
    return (lo + hi) / 2


def _implied_total(rows):
    """Implied expected count from a player+stat's bet365 'N+' rungs. Each Over rung
    gives P(>= floor(line)+1) ~ (1/odds) deflated by the one-way margin; we solve the
    Poisson mean per rung and take the median (rungs should roughly agree). Rough by
    design — it only feeds the minutes posterior, never a quoted price."""
    means = []
    for r in rows:
        try:
            line, odds = float(r.get("line")), float(r.get("odds_decimal"))
        except (TypeError, ValueError):
            continue
        if math.isnan(line) or odds <= 1.0:
            continue
        side = str(r.get("side") or "").lower()
        p = 1.0 / odds
        if "under" in side:
            p = 1.0 - p
        p = p / (1.0 + config.ONEWAY_MARGIN)        # strip assumed one-way vig
        mu = _poisson_mean_for_tail(math.floor(line) + 1, min(max(p, 1e-4), 0.999))
        if mu is not None:
            means.append(mu)
    if not means:
        return None
    means.sort()
    return means[len(means) // 2]


def _book_totals(g: pd.DataFrame) -> dict:
    """{nname: {model_stat: implied_total}} for one match's bet365 counting lines."""
    out: dict[str, dict] = {}
    for (player, stat), grp in g.groupby(["player", "stat"]):
        if stat not in _BOOK_MINUTE_STATS:
            continue
        tot = _implied_total(grp.to_dict("records"))
        if tot is not None:
            out.setdefault(norm_name(player), {})[STAT_TO_MODEL[stat]] = round(tot, 3)
    return out


def _date_of(g: pd.DataFrame) -> str | None:
    ko = pd.to_numeric(g.get("kickoff"), errors="coerce").dropna() \
        if "kickoff" in g.columns else pd.Series(dtype=float)
    if not len(ko):
        return None
    return datetime.fromtimestamp(int(ko.iloc[0]), tz=timezone.utc).strftime("%Y-%m-%d")


def _market_goals(g: pd.DataFrame):
    """(home_g, away_g) market expected goals from the captured Team Total Goals."""
    def side_g(side):
        if "player_team" not in g.columns or "team_total" not in g.columns:
            return None
        s = pd.to_numeric(g.loc[g["player_team"] == side, "team_total"],
                          errors="coerce").dropna()
        return float(s.iloc[0]) if len(s) else None
    return side_g("home"), side_g("away")


def build_requests(b365: pd.DataFrame) -> list[dict]:
    """One project_cli fixture request per captured match (needs home/away/date)."""
    reqs = []
    if b365 is None or not len(b365) or "match" not in b365.columns:
        return reqs
    for match, g in b365.groupby("match"):
        homes = g["home"].dropna().unique() if "home" in g.columns else []
        aways = g["away"].dropna().unique() if "away" in g.columns else []
        date = _date_of(g)
        if not len(homes) or not len(aways) or not date:
            continue
        hg, ag = _market_goals(g)
        reqs.append({
            "match": str(match), "home": str(homes[0]), "away": str(aways[0]),
            "date": date, "neutral": True,
            "market_home_g": hg, "market_away_g": ag,
            "book_totals": _book_totals(g),
        })
    return reqs


# --------------------------- run + collect ---------------------------

def _run_cli(payload: dict, timeout=180) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_CLI)], cwd=str(_MPM_DIR),
        input=json.dumps(payload), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"project_cli exited {proc.returncode}: "
                           f"{(proc.stderr or '')[-500:]}")
    return json.loads(proc.stdout or "{}")


def project(b365: pd.DataFrame, half_spread=0.03) -> dict:
    """Run the fair-value model over the captured slate.

    Returns {"cells": {(mkey, nname, canonical_stat): {mean, var, fair_line}},
             "fixtures": [per-fixture status], "n_players": int}. Empty cells when
    no API key, the CLI is missing, or no XI is out yet — callers fall back to the
    book/UD blend. Cells are keyed by the order-independent country-pair `mkey`, so
    a bet365- or DFS-sourced row finds its projection regardless of label spelling."""
    empty = {"cells": {}, "fixtures": [], "n_players": 0}
    if not config.ROOT.joinpath("mins_projection_model").exists() or not _CLI.exists():
        return empty
    import os
    if not os.environ.get("APISPORTS_KEY"):
        return dict(empty, note="APISPORTS_KEY not set — projection skipped")
    reqs = build_requests(b365)
    if not reqs:
        return empty
    try:
        resp = _run_cli({"half_spread": half_spread, "fixtures": reqs})
    except Exception as e:
        return dict(empty, note=f"projection failed: {e}")

    cells, fixtures, n = {}, [], 0
    for fx in resp.get("fixtures", []):
        fixtures.append({k: fx.get(k) for k in
                         ("match", "fixture_id", "status", "available",
                          "lineup_kind", "note", "error")})
        projected = fx.get("lineup_kind") not in (None, "confirmed")
        mk = mkey(fx.get("match"))
        if mk is None:
            continue
        for p in fx.get("players", []):
            nn = p.get("nname")
            if not nn:
                continue
            n += 1
            for mstat, pr in (p.get("price") or {}).items():
                stat = MODEL_TO_STAT.get(mstat)
                if not stat or pr.get("mean") is None:
                    continue
                cells[(mk, nn, stat)] = {
                    "mean": pr["mean"], "var": pr["var"],
                    "fair_line": pr.get("fair_line"),
                    "minutes": p.get("minutes_mean"),
                    "team": p.get("team"),          # API-Football team (for side assignment)
                    "player": p.get("player"),      # display name for model-only players
                    "projected": projected}         # True = pre-XI fallback (recent XI)
    return {"cells": cells, "fixtures": fixtures, "n_players": n}
