"""bet365 capture via BetsAPI — the priced reference for the edge model.

Chain: /v1/bet365/league (find WC league_id) -> /v1/bet365/upcoming -> /v4/bet365/prematch
Returns a flat DataFrame of player-prop rows:
    [match, home, away, player, market_name, stat, side, line, odds_decimal, market_id]
and saves raw JSON per match under data/{today}/betsapi/raw/.
"""
from __future__ import annotations

import json
import math
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import config
from normalize import norm_stat, norm_country

HOST = "https://api.b365api.com"
_NPLUS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*\+\s*$")

# bet365 bundles score / assist / score-or-assist as three HEADERS inside one
# "Player to Score or Assist" market — the header IS the sub-stat, not a line.
_SCORE_ASSIST = {"score": "Goals", "assist": "Assists", "score or assist": "Goals + assists"}


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


def _poisson_mean_for_tail(k, p_ge_k):
    """Solve mu so that P(X >= k) == p_ge_k for X ~ Poisson(mu) (k >= 1)."""
    if k < 1 or not (0.0 < p_ge_k < 1.0):
        return None

    def p_ge(mu):                       # P(X >= k) = 1 - sum_{i<k} e^-mu mu^i/i!
        term, cdf = math.exp(-mu), math.exp(-mu)
        for i in range(1, k):
            term *= mu / i
            cdf += term
        return 1.0 - cdf

    lo, hi = 1e-6, 25.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_ge(mid) < p_ge_k else (lo, mid)
    return (lo + hi) / 2


def _mean_from_ou_ladder(pairs):
    """Expected count (~ Poisson mean) from an Over/Under ladder. `pairs` is
    {line: {"over": odd, "under": odd}}. De-vig each two-sided pair to a fair
    P(over) and back out the Poisson mean that reproduces it; pick the pair nearest
    a coin-flip (least tail-sensitive). Fall back to the bare line (Over nearest
    1.90) if no two-sided pair exists. bet365 posts the SAME goals line (usually
    2.5) for every match and encodes strength in the PRICE, so the bare line badly
    understates lopsided games (France-Iraq: line 2.5 but Over @ 1.33 => mean ~3.7)."""
    best_mu, best_gap = None, 1e9
    fallback, fb_gap = None, 1e9
    for line, pq in pairs.items():
        ov, un = pq.get("over"), pq.get("under")
        if ov is None:
            continue
        if un is None:                  # one-sided: last-resort fallback (line as mean)
            if abs(ov - 1.90) < fb_gap:
                fb_gap, fallback = abs(ov - 1.90), line
            continue
        po, pu = 1.0 / ov, 1.0 / un
        fair_over = po / (po + pu)       # de-vigged P(total > line)
        mu = _poisson_mean_for_tail(int(math.floor(line)) + 1, fair_over)
        if mu is None:
            continue
        gap = abs(fair_over - 0.5)       # nearest a coin-flip = most reliable
        if gap < best_gap:
            best_gap, best_mu = gap, mu
    return best_mu if best_mu is not None else fallback


def _match_total_goals(markets):
    """Expected total match goals from the main 'Goals Over/Under' market."""
    pairs = {}                          # line -> {"over": odd, "under": odd}
    for m in markets:
        if m.get("name") != "Goals Over/Under":
            continue
        for o in m.get("odds") or []:
            side = str(o.get("header", "")).lower()
            if side not in ("over", "under"):
                continue
            try:
                line, od = float(o.get("name")), float(o.get("odds"))   # line is in 'name'
            except (TypeError, ValueError):
                continue
            if od > 1:
                pairs.setdefault(line, {})[side] = od
    return _mean_from_ou_ladder(pairs)


_OU_HCP = re.compile(r"^\s*(over|under)\s+(\d+(?:\.\d+)?)\s*$", re.I)


def _team_total_goals(markets):
    """(home_g, away_g) expected goals per side from 'Team Total Goals'. Each team's
    O/U ladder is under header '1' (home) / '2' (away), with the line+side in the
    `handicap` field ('Over 1.5'). This is the market split we anchor scorers to —
    far better than the Goals-Elo, which can't price cross-confederation mismatches."""
    pairs = {"1": {}, "2": {}}          # team header -> {line: {over/under: odd}}
    for m in markets:
        if m.get("name") != "Team Total Goals":
            continue
        for o in m.get("odds") or []:
            hdr = str(o.get("header", "")).strip()
            if hdr not in pairs:
                continue
            mm = _OU_HCP.match(str(o.get("handicap") or ""))
            if not mm:
                continue
            try:
                od = float(o.get("odds"))
            except (TypeError, ValueError):
                continue
            if od > 1:
                pairs[hdr].setdefault(float(mm.group(2)), {})[mm.group(1).lower()] = od
    return _mean_from_ou_ladder(pairs["1"]), _mean_from_ou_ladder(pairs["2"])


def _player_teams(markets, home, away):
    """{player_name: 'home'|'away'} from 'Team Goalscorer'. Within the 'First'
    header the roster is listed as: home players, 'No Goalscorer (HOME)', away
    players, 'No Goalscorer (AWAY)' — so each 'No Goalscorer (TEAM)' marker closes
    the block of players that precede it."""
    out, pending = {}, []
    sides = {norm_country(home): "home", norm_country(away): "away"}
    for m in markets:
        if m.get("name") != "Team Goalscorer":
            continue
        for o in m.get("odds") or []:
            if str(o.get("header", "")) != "First":
                continue
            nm = str(o.get("name") or "")
            low = nm.lower()
            if low.startswith("no goalscorer"):
                inside = nm[nm.find("(") + 1:nm.find(")")] if "(" in nm else ""
                side = sides.get(norm_country(inside))
                if side:
                    for p in pending:
                        out[p] = side
                pending = []
            elif nm:
                pending.append(nm)
        break                            # one Team Goalscorer market is enough
    return out


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

    rows, gmap, komap = [], {}, {}
    for m in hot:
        fi = m.get("id")
        home, away = _teamname(m.get("home")), _teamname(m.get("away"))
        komap[f"{home} v {away}"] = m.get("time")
        try:
            payload = _get("/v4/bet365/prematch", {"FI": fi})
        except RuntimeError as e:
            print(f"   ! {fi} {home} v {away}: {e}")
            continue
        (rawdir / f"{fi}_{stamp}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        results = payload.get("results") or []
        if not results:
            continue
        markets = _dedupe(_collect_markets(results[0]))
        gmap[f"{home} v {away}"] = _match_total_goals(markets)
        hg, ag = _team_total_goals(markets)               # market expected goals per side
        pteams = _player_teams(markets, home, away)       # {player_name: 'home'|'away'}
        for mk in markets:
            mname = mk.get("name", "")
            if _level(mname) != "player":
                continue
            stat = norm_stat(mname)
            for o in mk.get("odds") or []:
                try:
                    odd = float(o.get("odds"))
                except (TypeError, ValueError):
                    continue
                header = (o.get("header") or "").strip()
                sub = _SCORE_ASSIST.get(header.lower())
                if sub:                       # one market, three sub-bets keyed by header
                    row_stat, line, side = sub, 0.5, "over"
                else:
                    row_stat = stat
                    line = _parse_line(o)
                    if line is None and row_stat in ("Goals + assists", "Goals", "Assists"):
                        line = 0.5            # binary "to score/assist" = over 0.5
                    side = header or None
                pteam = pteams.get(o.get("name"))
                rows.append({
                    "match": f"{home} v {away}", "home": home, "away": away,
                    "player": o.get("name"), "market_name": mname, "stat": row_stat,
                    "side": side,
                    "line": line, "odds_decimal": odd, "market_id": mk.get("id"),
                    "kind": "ou" if "over/under" in mname.lower() else "nplus",
                    "player_team": pteam,
                    "team_total": hg if pteam == "home" else (ag if pteam == "away" else None),
                })
        print(f"   {fi} {home} v {away}: captured")
        time.sleep(0.4)

    df = pd.DataFrame(rows, columns=["match", "home", "away", "player", "market_name",
                                     "stat", "side", "line", "odds_decimal", "market_id", "kind",
                                     "player_team", "team_total"])
    df["match_total"] = df["match"].map(gmap)
    df["kickoff"] = df["match"].map(komap)        # epoch seconds, for today/remaining filters
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
