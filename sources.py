"""Parsers for the two manually-pasted DFS payloads (Underdog, PrizePicks).

Parsing logic carried over from line_gap.py, using the shared normalize module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

import config
from normalize import norm_stat, norm_name

UD_SPORTS = {"", "SOCCER", "FIFA"}

# PrizePicks plain GET → 403 (Cloudflare); a browser UA gets the JSON. UD is fine
# either way but we send the UA to both.
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
       "Accept": "application/json"}


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _fetch_json(url: str) -> dict:
    r = requests.get(url, headers=_UA, timeout=30)
    r.raise_for_status()
    return r.json()


def refresh_underdog(dest=None) -> int:
    """GET fresh Underdog soccer lines → dest. Returns #lines written, or 0 if the
    pull was EMPTY (no live UD slate) — in which case the existing paste is KEPT, so
    a quiet slate never clobbers a good file. Raises on network/HTTP error."""
    dest = Path(dest or config.UNDERDOG_JSON)
    j = _fetch_json(config.UNDERDOG_URL)
    n = len(j.get("over_under_lines") or [])
    if n:
        dest.write_text(json.dumps(j), encoding="utf-8")
    return n


def refresh_prizepicks(dest=None) -> int:
    """GET fresh PrizePicks projections → dest (browser UA required). Returns
    #projections written, or 0 (kept existing) if the pull had no `data`."""
    dest = Path(dest or config.PRIZEPICKS_JSON)
    j = _fetch_json(config.PRIZEPICKS_URL)
    n = len(j.get("data") or [])
    if n:
        dest.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return n


def refresh_dfs() -> dict:
    """Refresh both paste files in place. Sources are independent: a failure or an
    empty pull on one is logged and leaves THAT file untouched (never clobbers a
    good paste). Returns {source: status string}."""
    out = {}
    for name, fn in (("underdog", refresh_underdog), ("prizepicks", refresh_prizepicks)):
        try:
            n = fn()
            out[name] = f"{n} fresh" if n else "empty — kept paste"
        except Exception as e:
            out[name] = f"{type(e).__name__} — kept paste"
    return out


def parse_underdog(path) -> tuple[pd.DataFrame, int]:
    """-> (df[name, stat, line, over_mult, under_mult], n_alt_collapsed)."""
    payload = _load(path)
    lines = payload.get("over_under_lines")
    if not isinstance(lines, list):
        raise ValueError("underdog.json: no 'over_under_lines' array — wrong payload?")

    players = {p["id"]: p for p in payload.get("players", [])}
    apps = {a["id"]: a for a in payload.get("appearances", [])}

    grouped: dict[tuple[str, str], list[dict]] = {}
    for l in lines:
        ou = l.get("over_under") or {}
        ast = ou.get("appearance_stat") or {}
        app = apps.get(ast.get("appearance_id"))
        player = players.get(app["player_id"]) if app else None
        if player and str(player.get("sport_id", "")).upper() not in UD_SPORTS:
            continue
        display_stat = ast.get("display_stat") or ast.get("stat") or ""
        if player:
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        else:
            title = (ou.get("title") or "").strip()
            name = title[:-len(display_stat)].strip() if (display_stat and title.endswith(display_stat)) else title
        try:
            line = float(l.get("stat_value"))
        except (TypeError, ValueError):
            continue
        if not name:
            continue
        stat = norm_stat(ast.get("stat") or display_stat)
        over_mult = under_mult = None
        for o in l.get("options") or []:
            try:
                m = float(o.get("payout_multiplier"))
            except (TypeError, ValueError):
                continue
            if o.get("choice") == "higher":
                over_mult = m
            elif o.get("choice") == "lower":
                under_mult = m
        grouped.setdefault((norm_name(name), stat), []).append({
            "name": name, "stat": stat, "line": line,
            "over_mult": over_mult, "under_mult": under_mult,
            "n_options": len(l.get("options") or []),
        })

    rows, n_alt, ud_ladders = [], 0, {}
    for key, cands in grouped.items():
        mains = [c for c in cands if c["n_options"] == 2] or cands
        mains.sort(key=lambda c: c["line"])
        rows.append(mains[(len(mains) - 1) // 2])  # median main line (DFS target)
        n_alt += len(cands) - 1
        # full multiplier-bearing ladder (every posted line) for UD-implied prob
        ladder = [{"line": c["line"], "over_mult": c["over_mult"],
                   "under_mult": c["under_mult"]}
                  for c in cands
                  if c["over_mult"] is not None and c["under_mult"] is not None]
        if ladder:
            ud_ladders[key] = ladder            # key is (norm_name(name), stat)
    df = pd.DataFrame(rows, columns=["name", "stat", "line", "over_mult", "under_mult"])
    df.attrs["ud_ladders"] = ud_ladders
    return df, n_alt


def parse_prizepicks(path) -> tuple[pd.DataFrame, int]:
    """-> (df[name, team, opp, stat, line], n_demons_goblins_excluded)."""
    payload = _load(path)
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("prizepicks.json: no 'data' array — wrong payload?")

    players = {inc["id"]: inc.get("attributes", {})
               for inc in payload.get("included", []) if inc.get("type") == "new_player"}

    rows, n_flash, seen = [], 0, set()
    for d in data:
        if d.get("type") != "projection":
            continue
        a = d.get("attributes") or {}
        if str(a.get("odds_type", "standard")).lower() != "standard":
            n_flash += 1
            continue
        pid = (((d.get("relationships") or {}).get("new_player") or {}).get("data") or {}).get("id")
        p = players.get(pid, {})
        name = p.get("display_name") or p.get("name")
        try:
            line = float(a.get("line_score"))
        except (TypeError, ValueError):
            continue
        if not name:
            continue
        stat = norm_stat(a.get("stat_type") or a.get("stat_display_name"))
        key = (norm_name(name), stat)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"name": name, "team": p.get("team", ""),
                     "opp": a.get("description", ""), "stat": stat, "line": line})
    df = pd.DataFrame(rows, columns=["name", "team", "opp", "stat", "line"])
    return df, n_flash
