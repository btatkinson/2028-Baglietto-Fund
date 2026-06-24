"""Append-only price log for devig + blend-weight calibration.

Every run snapshots all three sources' RAW prices (bet365 odds, Underdog
multipliers, PrizePicks lines) in long format with a capture timestamp, so we can
later (a) join realized outcomes from API-Football and fit the one-way devig
margin per stat + the UD/365 blend weight, and (b) watch line movement over time.
Raw prices are stored (not implied probs) so the fitter can test alternative devig
methods after the fact. Append-only on purpose — repeated snapshots ARE the
line-movement signal.

  prices.parquet columns:
    captured_at source match player nname stat line side
    odds_decimal over_mult under_mult kind
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config
from normalize import norm_name, norm_country

_PATH = config.DATA_DIR / "calibration" / "prices.parquet"
_COLS = ["captured_at", "source", "match", "player", "nname", "stat", "line",
         "side", "odds_decimal", "over_mult", "under_mult", "kind", "p_model"]
_NUM = ["line", "odds_decimal", "over_mult", "under_mult", "p_model"]
_STR = ["captured_at", "source", "match", "player", "nname", "stat", "side", "kind"]


def _b365_rows(b365):
    if b365 is None or not len(b365):
        return pd.DataFrame()
    r = b365[["match", "player", "stat", "line", "side", "odds_decimal", "kind"]].copy()
    r["source"], r["over_mult"], r["under_mult"] = "b365", np.nan, np.nan
    return r


def _ud_rows(ud):
    if ud is None or not len(ud):
        return pd.DataFrame()
    r = ud.rename(columns={"name": "player"})[
        ["player", "stat", "line", "over_mult", "under_mult"]].copy()
    r["source"], r["match"], r["side"] = "ud", "", None
    r["odds_decimal"], r["kind"] = np.nan, "ud"
    return r


def _pp_rows(pp):
    if pp is None or not len(pp):
        return pd.DataFrame()
    r = pp.rename(columns={"name": "player"}).copy()
    r["match"] = (r["team"].fillna("").astype(str) + " v " + r["opp"].fillna("").astype(str))
    r = r[["player", "stat", "line", "match"]].copy()
    r["source"], r["side"], r["odds_decimal"] = "pp", None, np.nan
    r["over_mult"], r["under_mult"], r["kind"] = np.nan, np.nan, "pp"
    return r


def _model_rows(joined):
    """Model fair-value rows from the joined edge frame: the projection's P(over) at
    the line the row displays (PP else UD). Stored as source='model' with the
    probability in `p_model` (it's already a fair prob, so there's nothing to devig),
    so reliability() scores the projection against realized outcomes alongside the
    books. Rows without a model projection are skipped."""
    if joined is None or not len(joined) or "prob_model" not in getattr(joined, "columns", []):
        return pd.DataFrame()
    d = joined.copy()
    d["p_model"] = pd.to_numeric(d["prob_model"], errors="coerce")
    d = d[d["p_model"].notna()]
    if not len(d):
        return pd.DataFrame()
    line = d["pp_line"].where(d["pp_line"].notna(), d["ud_line"])
    r = pd.DataFrame({"player": d["name"], "match": d["match"], "stat": d["stat"],
                      "line": pd.to_numeric(line, errors="coerce"), "p_model": d["p_model"]})
    r = r.dropna(subset=["line"])
    r["source"], r["side"], r["odds_decimal"] = "model", "over", np.nan
    r["over_mult"], r["under_mult"], r["kind"] = np.nan, np.nan, "model"
    return r


def log_snapshot(ud, pp, b365, captured_at=None, joined=None):
    """Append one timestamped price snapshot across all sources (plus the model
    projection, if the joined edge frame is supplied). Returns (added, total)."""
    captured_at = captured_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [p for p in (_b365_rows(b365), _ud_rows(ud), _pp_rows(pp), _model_rows(joined))
             if len(p)]
    if not parts:
        return 0, len(load_log())
    snap = pd.concat(parts, ignore_index=True)
    snap["captured_at"] = captured_at
    snap["nname"] = snap["player"].map(norm_name)
    snap = snap.reindex(columns=_COLS)
    for c in _NUM:
        snap[c] = pd.to_numeric(snap[c], errors="coerce")
    for c in _STR:
        snap[c] = snap[c].astype("string")

    _PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([load_log(), snap], ignore_index=True) if _PATH.exists() else snap
    out.to_parquet(_PATH, index=False)
    return len(snap), len(out)


def load_log():
    return pd.read_parquet(_PATH) if _PATH.exists() else pd.DataFrame(columns=_COLS)


# ----------------------- outcome resolution + fit -----------------------

_FIT = config.DATA_DIR / "calibration" / "fit.json"
_API_BASE = "https://v3.football.api-sports.io"
MIN_SAMPLES = 300

# canonical log stat -> realized per-player fields (summed)
REAL_STAT = {"Goals": ("goals",), "Assists": ("assists",), "Goals + assists": ("goals", "assists"),
             "Shots": ("shots",), "Shots on target": ("sot",), "Tackles": ("tackles",),
             "Saves": ("saves",), "Passes": ("passes",)}


def _mkey(match):
    """Canonical fixture key from a 'Home v Away' string — normalized country pair,
    order-independent — so BetsAPI/DFS/API-Football naming all collapse to one key."""
    cs = sorted({norm_country(p.strip()) for p in str(match).split(" v ") if p.strip()})
    return "|".join(cs) if len(cs) == 2 else None


def _af_get(path, **params):
    key = os.environ.get("APISPORTS_KEY")
    if not key:
        raise RuntimeError("APISPORTS_KEY not set (expected in .env).")
    return requests.get(f"{_API_BASE}/{path}", headers={"x-apisports-key": key},
                        params=params, timeout=30).json()


def _player_counts(fixture_id):
    """Realized per-player counts for one finished fixture (both teams)."""
    rows = []
    for team in _af_get("fixtures/players", fixture=fixture_id).get("response", []):
        for p in team.get("players", []):
            st = (p.get("statistics") or [{}])[0]
            g, sh, go = st.get("games") or {}, st.get("shots") or {}, st.get("goals") or {}
            pa, tk = st.get("passes") or {}, st.get("tackles") or {}
            rows.append({"player": p["player"]["name"], "minutes": g.get("minutes") or 0,
                         "passes": pa.get("total") or 0, "shots": sh.get("total") or 0,
                         "sot": sh.get("on") or 0, "tackles": tk.get("total") or 0,
                         "saves": go.get("saves") or 0, "goals": go.get("total") or 0,
                         "assists": go.get("assists") or 0})
    return rows


def resolve_outcomes(dates=None, log=None) -> pd.DataFrame:
    """Realized per-player counts for FINISHED fixtures on the capture dates (and
    the next day). Returns [date, match, nname, player, stat, realized, minutes]."""
    log = load_log() if log is None else log
    if not len(log):
        return pd.DataFrame()
    if dates is None:
        d = pd.to_datetime(log["captured_at"].astype(str).str[:10], errors="coerce").dropna()
        dates = sorted({x.strftime("%Y-%m-%d") for x in d} |
                       {(x + pd.Timedelta(days=1)).strftime("%Y-%m-%d") for x in d})
    rows = []
    for day in dates:
        for f in _af_get("fixtures", date=day).get("response", []):
            if f["fixture"]["status"]["short"] not in ("FT", "AET", "PEN"):
                continue
            match = f"{f['teams']['home']['name']} v {f['teams']['away']['name']}"
            for pr in _player_counts(f["fixture"]["id"]):
                if not pr["minutes"]:
                    continue
                base = {"date": day, "match": match, "player": pr["player"],
                        "nname": norm_name(pr["player"]), "minutes": pr["minutes"]}
                for stat, fields in REAL_STAT.items():
                    rows.append({**base, "stat": stat,
                                 "realized": sum(pr.get(x) or 0 for x in fields)})
    return pd.DataFrame(rows)


def _raw_p(row):
    """Raw implied P(over line) for one logged row, BEFORE devig margin (so the
    fitter can calibrate the margin)."""
    if pd.isna(row["line"]):
        return None
    if row["source"] == "b365":
        o = row["odds_decimal"]
        if pd.isna(o) or o <= 1:
            return None
        return 1 - 1 / o if "under" in str(row["side"] or "").lower() else 1 / o
    if row["source"] == "ud":
        om, um = row["over_mult"], row["under_mult"]
        if pd.notna(om) and pd.notna(um) and om > 0 and um > 0:
            return (1 / om) / (1 / om + 1 / um)          # de-vig the two multipliers
        return 1 / om if pd.notna(om) and om > 0 else None
    if row["source"] == "model":                         # already a fair P(over line)
        pm = row.get("p_model")
        return float(pm) if pd.notna(pm) else None
    return None                                          # pp: line only, no price


def attach_outcomes(log, outcomes):
    """FIXTURE-AWARE join: realized counts joined on (match_key, nname, stat) so a
    prop only ever matches its OWN fixture's outcome. Rows without a resolvable
    match (e.g. Underdog, which logs no match) are dropped here and re-paired by
    (nname, stat, line) in the weight fit. Adds raw_p + hit."""
    if outcomes is None or not len(outcomes):
        return pd.DataFrame()
    o = outcomes.copy()
    o["mkey"] = o["match"].map(_mkey)
    o = o.dropna(subset=["mkey"]).drop_duplicates(["mkey", "nname", "stat"])[
        ["mkey", "nname", "stat", "realized", "minutes"]]
    L = log.copy()
    L["mkey"] = L["match"].map(_mkey)
    m = L.dropna(subset=["mkey"]).merge(o, on=["mkey", "nname", "stat"], how="inner")
    m = m[m["minutes"] > 0].copy()
    m["raw_p"] = m.apply(_raw_p, axis=1)
    m["hit"] = (m["realized"] > m["line"]).astype(float)
    return m.dropna(subset=["raw_p", "line"])


def reliability(joined, bins=5):
    out = {}
    for src, g in joined.groupby("source"):
        g = g.dropna(subset=["raw_p"])
        if len(g) < bins:
            continue
        q = pd.qcut(g["raw_p"], bins, duplicates="drop")
        r = g.groupby(q, observed=True).agg(implied=("raw_p", "mean"),
                                            actual=("hit", "mean"), n=("hit", "size"))
        out[src] = [{"implied": round(a, 3), "actual": round(b, 3), "n": int(c)}
                    for a, b, c in zip(r["implied"], r["actual"], r["n"])]
    return out


def _fit_margins(joined):
    """One-way devig margin per stat (method of moments on bet365 rows):
    pick m so mean(raw_p/(1+m)) == realized hit rate."""
    out = {}
    for stat, g in joined[joined["source"] == "b365"].groupby("stat"):
        mh = g["hit"].mean()
        if mh > 0:
            out[stat] = round(max(g["raw_p"].mean() / mh - 1, 0.0), 3)
    return out


def _fit_blend_weight(joined, log):
    """UD/365 weight minimizing log-loss of w*p_ud+(1-w)*p_b365 vs realized hits,
    over props priced by both books at the same line. bet365 hits come from the
    fixture-aware join; UD prices are re-paired from the raw log by (nname,stat,line)."""
    b = joined[joined["source"] == "b365"][["nname", "stat", "line", "raw_p", "hit"]] \
        .rename(columns={"raw_p": "p_b"})
    u = log[log["source"] == "ud"].copy()
    u["p_u"] = u.apply(_raw_p, axis=1)
    u = u.dropna(subset=["p_u", "line"])[["nname", "stat", "line", "p_u"]]
    pair = b.merge(u, on=["nname", "stat", "line"])
    if len(pair) < 50:
        return None
    y = pair["hit"].to_numpy()
    best_w, best_ll = config.BLEND_W_UD, 1e9
    for w in np.linspace(0, 1, 21):
        p = np.clip(w * pair["p_u"].to_numpy() + (1 - w) * pair["p_b"].to_numpy(), 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if ll < best_ll:
            best_ll, best_w = ll, round(float(w), 2)
    return best_w


def calibrate(min_samples=MIN_SAMPLES):
    """Resolve outcomes for the logged prices, fit one-way margins + the UD/365
    blend weight. Falls back to config.BLEND_W_UD (0.6) until min_samples outcomes
    exist. Persists fit.json; returns the summary."""
    log = load_log()
    summ = {"n_outcomes": 0, "blend_w_ud": config.BLEND_W_UD, "fitted": False, "default": True}
    if len(log):
        out = resolve_outcomes(log=log)
        if len(out):
            joined = attach_outcomes(log, out)
            summ["n_outcomes"] = int(len(joined))
            if len(joined):
                summ["margins"] = _fit_margins(joined)
                summ["reliability"] = reliability(joined)
            if len(joined) >= min_samples:
                w = _fit_blend_weight(joined, log)
                if w is not None:
                    summ.update(blend_w_ud=w, fitted=True, default=False)
    _FIT.parent.mkdir(parents=True, exist_ok=True)
    _FIT.write_text(json.dumps(summ, indent=2), encoding="utf-8")
    return summ


def blend_weight():
    """The UD weight to use NOW: the fitted value once we have >= MIN_SAMPLES
    outcomes, otherwise the 0.6 default."""
    if _FIT.exists():
        d = json.loads(_FIT.read_text(encoding="utf-8"))
        if d.get("fitted") and d.get("n_outcomes", 0) >= MIN_SAMPLES:
            return float(d["blend_w_ud"])
    return config.BLEND_W_UD
