"""Parsers for the two manually-pasted DFS payloads (Underdog, PrizePicks).

Parsing logic carried over from line_gap.py, using the shared normalize module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from normalize import norm_stat, norm_name

UD_SPORTS = {"", "SOCCER", "FIFA"}


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


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

    rows, n_alt = [], 0
    for cands in grouped.values():
        mains = [c for c in cands if c["n_options"] == 2] or cands
        mains.sort(key=lambda c: c["line"])
        rows.append(mains[(len(mains) - 1) // 2])  # median main line
        n_alt += len(cands) - 1
    df = pd.DataFrame(rows, columns=["name", "stat", "line", "over_mult", "under_mult"])
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
