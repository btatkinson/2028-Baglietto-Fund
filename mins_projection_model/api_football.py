"""API-Football adapter — the single data backbone for the minutes model.

Replaces the FBref/StatsBomb/RDS stitching: one paid source (api-sports.io) gives
national-team goals + possession AND per-player minutes/passes/position, covering
friendlies, qualifiers, Nations League and tournaments. We shape it into the two
frames the existing code already consumes:

  team_match_frame(...)   -> team_strength.build_ratings   (goals + possession)
  player_match_frame(...) -> features.player_ewm + rate_model (passes/minutes/pos)

Every response is cached to config.CACHE_DIR as JSON, so rebuilding a frame after
the first pull costs zero requests. Free tier = seasons 2022-2024; Pro = +2025-26.
Request budget: ~1 + N(statistics) + N(players) per league-season, N = #fixtures.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

import config

_FINISHED = {"FT", "AET", "PEN"}
_SESSION = requests.Session()
_SESSION.headers["x-apisports-key"] = config.APISPORTS_KEY or ""


def _cache_path(path, params):
    stem = path.strip("/").replace("/", "_")
    tail = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
    return config.CACHE_DIR / f"{stem}__{tail}.json"


def _get(path, **params):
    """GET an endpoint, caching the JSON. Raises on an API `errors` payload."""
    cp = _cache_path(path, params)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    if not config.APISPORTS_KEY:
        raise RuntimeError("APISPORTS_KEY not set (expected in wc_props/.env).")
    r = _SESSION.get(f"{config.API_BASE}/{path}", params=params, timeout=30)
    j = r.json()
    errs = j.get("errors")
    if errs:                      # dict {} is falsy; non-empty means a real error
        raise RuntimeError(f"{path} {params} -> {errs}")
    cp.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.2)              # gentle on the per-minute rate limit
    return j


def get_fixtures(league, season, finished_only=True, max_fixtures=None):
    resp = _get("fixtures", league=league, season=season).get("response", [])
    if finished_only:
        resp = [f for f in resp if f["fixture"]["status"]["short"] in _FINISHED]
    resp.sort(key=lambda f: f["fixture"]["date"])
    return resp[:max_fixtures] if max_fixtures else resp


_STAT_KEYS = {"Ball Possession": "poss", "Shots on Goal": "sot", "Total Shots": "shots",
              "Corner Kicks": "corners"}


def _num(v):
    if v is None:
        return None
    try:
        return float(str(v).rstrip("%"))
    except (TypeError, ValueError):
        return None


def _team_stats(fixture_id):
    """{team_id: {poss, sot, shots, corners}} from one cached fixtures/statistics call."""
    out = {}
    for team in _get("fixtures/statistics", fixture=fixture_id).get("response", []):
        vals = {}
        for s in team.get("statistics", []):
            key = _STAT_KEYS.get(s.get("type"))
            if key:
                vals[key] = _num(s.get("value"))
        out[team["team"]["id"]] = vals
    return out


def _team_row(f, neutral=True, with_possession=True) -> dict:
    """One team_match row (date/home/away/goals + possession + SoT + total shots +
    corners) from a fixture. Total shots & corners come free from the same cached
    statistics call; they feed the global shots/corners ratings (script features)."""
    fx, teams, goals = f["fixture"], f["teams"], f["goals"]
    hs = aw = {}
    if with_possession:
        ts = _team_stats(fx["id"])
        hs, aw = ts.get(teams["home"]["id"], {}), ts.get(teams["away"]["id"], {})
    return {"date": fx["date"][:10], "match_id": fx["id"],
            "home": teams["home"]["name"], "away": teams["away"]["name"],
            "home_g": goals["home"], "away_g": goals["away"],
            "home_poss": hs.get("poss"), "away_poss": aw.get("poss"),
            "home_sot": hs.get("sot"), "away_sot": aw.get("sot"),
            "home_shots": hs.get("shots"), "away_shots": aw.get("shots"),
            "home_corners": hs.get("corners"), "away_corners": aw.get("corners"),
            "neutral": neutral}


def _player_rows(f, source="intl") -> list:
    """player_match rows (minutes/position + counting stats) for both teams."""
    fid, date = f["fixture"]["id"], f["fixture"]["date"][:10]
    rows = []
    for team in _get("fixtures/players", fixture=fid).get("response", []):
        tname = team["team"]["name"]
        for p in team.get("players", []):
            st = (p.get("statistics") or [{}])[0]
            g, sh, go = st.get("games") or {}, st.get("shots") or {}, st.get("goals") or {}
            pa, tk = st.get("passes") or {}, st.get("tackles") or {}
            rows.append({
                "date": date, "match_id": fid, "team": tname, "player": p["player"]["name"],
                "position": g.get("position"), "minutes": g.get("minutes") or 0,
                "is_starter": g.get("substitute") is False,
                "passes": pa.get("total") or 0, "shots": sh.get("total") or 0,
                "sot": sh.get("on") or 0, "tackles": tk.get("total") or 0,
                "saves": go.get("saves") or 0, "goals": go.get("total") or 0,
                "assists": go.get("assists") or 0, "source": source})
    return rows


def team_match_frame(league, season, neutral=True, with_possession=True,
                     max_fixtures=None) -> pd.DataFrame:
    rows = [_team_row(f, neutral, with_possession)
            for f in get_fixtures(league, season, max_fixtures=max_fixtures)]
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def player_match_frame(league, season, source="intl", max_fixtures=None) -> pd.DataFrame:
    return pd.DataFrame([r for f in get_fixtures(league, season, max_fixtures=max_fixtures)
                         for r in _player_rows(f, source)])


def team_fixtures(team_id, season, finished_only=True):
    """Finished fixtures for ONE team in a season (all competitions). Used to warm
    a cold-start national team into the ratings + player EWM."""
    resp = _get("fixtures", team=team_id, season=season).get("response", [])
    if finished_only:
        resp = [f for f in resp if f["fixture"]["status"]["short"] in _FINISHED]
    return sorted(resp, key=lambda f: f["fixture"]["date"])


def team_id(name):
    """Resolve to the SENIOR MEN'S national team. The raw search returns clubs,
    youth (U17/U20/U23) and women's (' W') sides too, and the first hit is often
    wrong (e.g. 'Paraguay' -> Club Guarani, 'New Zealand' -> 'New Zealand W'), so
    rank: national > men's > senior > exact-name."""
    r = _get("teams", search=name).get("response", [])
    if not r:
        return None
    nl = name.lower().strip()

    def score(x):
        t = x["team"]
        nm = t.get("name") or ""
        return (bool(t.get("national")),                  # a national team
                not nm.endswith(" W"),                    # men's, not women's
                not re.search(r"\bU\d{2}\b", nm),          # senior, not U17/20/23
                nm.lower() == nl)                          # exact name match
    return max(r, key=score)["team"]["id"]


def build_intl(sources=None, **kw):
    """Concatenate team + player frames across config.INTL_SOURCES (or `sources`)."""
    sources = sources or config.INTL_SOURCES
    tf = pd.concat([team_match_frame(lg, sn, **kw) for lg, sn in sources], ignore_index=True)
    pf = pd.concat([player_match_frame(lg, sn, **{k: v for k, v in kw.items() if k == "max_fixtures"})
                    for lg, sn in sources], ignore_index=True)
    return tf, pf


# ---- live (uncached) endpoints for an upcoming fixture ----

def _get_nocache(path, **params):
    """Like _get but NEVER cached — for live data (lineups, fixture status) that
    changes as kickoff approaches."""
    if not config.APISPORTS_KEY:
        raise RuntimeError("APISPORTS_KEY not set (expected in wc_props/.env).")
    j = _SESSION.get(f"{config.API_BASE}/{path}", params=params, timeout=30).json()
    if j.get("errors"):
        raise RuntimeError(f"{path} {params} -> {j['errors']}")
    return j


def fixtures_on(day):
    """All fixtures (any status) kicking off on `day` (YYYY-MM-DD), uncached. Each
    item is the raw API-Football fixture object; the caller resolves the one it
    wants by team name. Used by the wc_props bridge to find a confirmed XI's
    fixture_id from a bet365 'Home v Away' label without hardcoding ids."""
    return _get_nocache("fixtures", date=day).get("response", [])


def fixture_meta(fixture_id):
    f = _get_nocache("fixtures", id=fixture_id)["response"][0]
    return {"home": f["teams"]["home"]["name"], "away": f["teams"]["away"]["name"],
            "league": f["league"]["name"], "status": f["fixture"]["status"]["short"],
            "neutral": True}          # WC matches are neutral-site


def lineup(fixture_id):
    """Confirmed XI -> {team_name: [(player, pos, is_starter)]}. Empty until the XI
    is released (~1h pre-kickoff). pos is G/D/M/F, matching the rate model."""
    out = {}
    for t in _get_nocache("fixtures/lineups", fixture=fixture_id).get("response", []):
        starters = [(p["player"]["name"], p["player"].get("pos"), True)
                    for p in (t.get("startXI") or [])]
        subs = [(p["player"]["name"], p["player"].get("pos"), False)
                for p in (t.get("substitutes") or [])]
        out[t["team"]["name"]] = starters + subs
    return out


if __name__ == "__main__":
    print(team_match_frame(1, 2022, max_fixtures=5).to_string())
