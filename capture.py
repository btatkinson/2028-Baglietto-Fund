"""bet365 capture via BetsAPI — the priced reference for the edge model.

Chain: /v1/bet365/league (find WC league_id) -> /v1/bet365/upcoming -> /v4/bet365/prematch
Returns a flat DataFrame of player-prop rows:
    [match, home, away, player, market_name, stat, side, line, odds_decimal, market_id]
and saves raw JSON per match under data/{today}/betsapi/raw/.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import config
from normalize import norm_stat

HOST = "https://api.b365api.com"
_NPLUS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")


def _parse_line(o):
    """bet365 player-stat props are 'N+' (P(stat >= N) = over N-0.5). The
    threshold can live in header ('2+'), in handicap ('67+'), or as a clean
    float handicap ('0.5', saves) which is already the over line."""
    for field in (o.get("header"), o.get("handicap")):
        if field:
            mm = _NPLUS.match(str(field))
            if mm:
                return float(mm.group(1)) - 0.5
    h = o.get("handicap")
    try:
        return float(h) if h not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _get(path, params):
    r = requests.get(f"{HOST}{path}", params=dict(params, token=config.BETSAPI_TOKEN), timeout=30)
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"{path} [{r.status_code}] non-JSON: {r.text[:200]}")
    if r.status_code != 200 or j.get("success") != 1:
        raise RuntimeError(f"{path} [{r.status_code}] success={j.get('success')}: "
                           f"{json.dumps(j.get('error') or j)[:200]}  "
                           "(token/permission error = Bet365 API not in your plan tier)")
    return j


def _teamname(x):
    return x.get("name", "") if isinstance(x, dict) else str(x or "")


def _level(name: str) -> str:
    n = name.lower()
    if n.startswith(("player ", "goalkeeper ")):
        return "player"
    if n.startswith("team "):
        return "team"
    if n.startswith("match "):
        return "match"
    return "other"


def _collect_markets(obj):
    out = []
    if isinstance(obj, dict):
        if "name" in obj and isinstance(obj.get("odds"), list):
            out.append(obj)
        for v in obj.values():
            out += _collect_markets(v)
    elif isinstance(obj, list):
        for it in obj:
            out += _collect_markets(it)
    return out


def _dedupe(markets):
    best = {}
    for m in markets:
        odds = m.get("odds") or []
        if not odds:
            continue
        key = m.get("id") or m.get("name")
        if key not in best or len(odds) > len(best[key].get("odds") or []):
            best[key] = m
    return list(best.values())


def _discover_league():
    if config.WC_LEAGUE_ID:
        return config.WC_LEAGUE_ID
    res = _get("/v1/bet365/league", {"sport_id": config.SOCCER_SPORT_ID}).get("results") or []
    wc = [l for l in res if "world cup" in str(l.get("name", "")).lower()
          and not any(x in str(l.get("name", "")).lower()
                      for x in ("women", "u17", "u20", "qualif", "club"))]
    if not wc:
        raise RuntimeError("No World Cup league via /bet365/league. Set WCP_WC_LEAGUE_ID.")
    return wc[0].get("id")


def capture() -> pd.DataFrame:
    if not config.BETSAPI_TOKEN:
        raise RuntimeError("BETSAPI_TOKEN not set — cannot capture bet365 props.")
    lid = _discover_league()
    matches = []
    for page in range(1, 5):
        res = _get("/v1/bet365/upcoming",
                   {"sport_id": config.SOCCER_SPORT_ID, "league_id": lid, "page": page}).get("results") or []
        if not res:
            break
        matches += res

    now = datetime.now(timezone.utc).timestamp()
    hot = []
    for m in matches:
        try:
            t = int(m.get("time"))
        except (TypeError, ValueError):
            continue
        if now <= t <= now + config.WINDOW_HOURS * 3600:
            hot.append(m)
    hot = hot[: config.MAX_DETAIL]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rawdir = config.DATA_DIR / date.today().isoformat() / "betsapi" / "raw"
    rawdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in hot:
        fi = m.get("id")
        home, away = _teamname(m.get("home")), _teamname(m.get("away"))
        try:
            payload = _get("/v4/bet365/prematch", {"FI": fi})
        except RuntimeError as e:
            print(f"   ! {fi} {home} v {away}: {e}")
            continue
        (rawdir / f"{fi}_{stamp}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        results = payload.get("results") or []
        if not results:
            continue
        for mk in _dedupe(_collect_markets(results[0])):
            mname = mk.get("name", "")
            if _level(mname) != "player":
                continue
            stat = norm_stat(mname)
            for o in mk.get("odds") or []:
                try:
                    odd = float(o.get("odds"))
                except (TypeError, ValueError):
                    continue
                line = _parse_line(o)
                rows.append({
                    "match": f"{home} v {away}", "home": home, "away": away,
                    "player": o.get("name"), "market_name": mname, "stat": stat,
                    "side": (o.get("header") or "").strip() or None,
                    "line": line, "odds_decimal": odd, "market_id": mk.get("id"),
                    "kind": "ou" if "over/under" in mname.lower() else "nplus",
                })
        print(f"   {fi} {home} v {away}: captured")
        time.sleep(0.4)

    df = pd.DataFrame(rows, columns=["match", "home", "away", "player", "market_name",
                                     "stat", "side", "line", "odds_decimal", "market_id", "kind"])
    if len(df):
        fdir = config.DATA_DIR / date.today().isoformat() / "betsapi" / "flat"
        fdir.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(fdir / f"props_{stamp}.parquet", index=False)
        except Exception:
            df.to_csv(fdir / f"props_{stamp}.csv", index=False)
    return df


def load_latest_flat() -> pd.DataFrame:
    """Fallback: load the most recent flattened capture if no token / no live pull."""
    base = config.DATA_DIR
    files = sorted(base.glob("*/betsapi/flat/props_*.parquet")) + \
            sorted(base.glob("*/betsapi/flat/props_*.csv"))
    if not files:
        return pd.DataFrame()
    f = files[-1]
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
