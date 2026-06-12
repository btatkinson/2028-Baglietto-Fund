"""Name + stat normalization shared across all sources.

Lifted from line_gap.py (the surname-anchored fuzzy matcher) and extended with
bet365 market-name -> canonical-stat mappings so all four sources land on the
same stat vocabulary.
"""
from __future__ import annotations

import re
import unicodedata

# canonical stat names are the values; every source's raw label maps in here
STAT_MAP = {
    # --- DFS (Underdog / PrizePicks) raw labels ---
    "shots": "Shots",
    "shots_attempted": "Shots",
    "shots_on_target": "Shots on target",
    "shots_on_goal": "Shots on target",
    "sog": "Shots on target",
    "passes": "Passes",
    "passes_attempted": "Passes",
    "pass_attempts": "Passes",
    "passes_completed": "Passes completed",
    "completed_passes": "Passes completed",
    "goals": "Goals",
    "assists": "Assists",
    "goals_plus_assists": "Goals + assists",
    "goals_assists": "Goals + assists",
    "goals_and_assists": "Goals + assists",
    "saves": "Saves",
    "goalie_saves": "Saves",
    "goalkeeper_saves": "Saves",
    "goals_allowed": "Goals allowed",
    "goals_against": "Goals allowed",
    "goals_conceded": "Goals allowed",
    "tackles": "Tackles",
    "tackles_won": "Tackles won",
    "fouls": "Fouls committed",
    "fouls_committed": "Fouls committed",
    "fouls_drawn": "Fouls drawn",
    "fouls_suffered": "Fouls drawn",
    "clearances": "Clearances",
    "interceptions": "Interceptions",
    "crosses": "Crosses",
    "crosses_attempted": "Crosses",
    "shots_assisted": "Shot assists",
    "shot_assists": "Shot assists",
    "chances_created": "Chances created",
    "fantasy_points": "Fantasy points*",
    "fantasy_score": "Fantasy points*",
    # --- bet365 (BetsAPI) market names, lowercased+underscored ---
    "player_shots": "Shots",
    "player_shots_over_under": "Shots",
    "player_shots_on_target": "Shots on target",
    "player_shots_on_target_over_under": "Shots on target",
    "player_shots_on_goal": "Shots on target",
    "player_tackles": "Tackles",
    "player_passes": "Passes",
    "player_assists": "Assists",
    "player_fouls_committed": "Fouls committed",
    "player_to_be_fouled": "Fouls drawn",
    "player_offsides": "Offsides",
    # goalkeeper saves comes through as "goalkeeper_saves" -> already mapped
    # specialized bet365 markets (headed shots, outside box) intentionally fall
    # through unmapped so they don't false-match a DFS stat.
}

_DROP_TOKENS = {"jr", "sr", "ii", "iii", "iv"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn")


def norm_stat(raw: str | None) -> str | None:
    if not raw:
        return None
    key = strip_accents(str(raw).lower())
    key = "".join(c if c.isalnum() else "_" for c in key).strip("_")
    while "__" in key:
        key = key.replace("__", "_")
    # period scope handling (Underdog) — full regulation is comparable; halves labelled
    prefix = ""
    if key.startswith("period_1_2_"):
        key = key[len("period_1_2_"):]
    else:
        m = re.match(r"period_(\d+)_(?!\d)", key)
        if m:
            prefix = {"1": "1H ", "2": "2H "}.get(m.group(1), f"P{m.group(1)} ")
            key = key[m.end():]
    mapped = STAT_MAP.get(key)
    return prefix + mapped if mapped else prefix + str(raw).strip()


def norm_name(raw: str | None) -> str:
    if not raw:
        return ""
    s = strip_accents(str(raw).lower())
    for ch in ".'\u2019-":
        s = s.replace(ch, " ")
    toks = [t for t in s.split() if t not in _DROP_TOKENS]
    return " ".join(toks)


def names_compatible(a: str, b: str) -> bool:
    """Token-aware match: every token of the shorter name must match a token of
    the longer; single letters match as initials; last tokens (surname) agree.
    Also matches when the token multisets are identical regardless of order, to
    handle family-name-first vs given-name-first (e.g. 'Son Heung-Min' on DFS
    vs 'Heung-Min Son' on bet365)."""
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return False
    if sorted(ta) == sorted(tb):          # order-invariant exact (handles KR reversal)
        return True
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if short[-1] != long_[-1]:
        return False
    pool = list(long_)
    for t in short:
        for i, u in enumerate(pool):
            if u == t or (len(t) == 1 and u.startswith(t)) or (len(u) == 1 and t.startswith(u)):
                pool.pop(i)
                break
        else:
            return False
    return True


def surname(nname: str) -> str:
    toks = nname.split()
    return toks[-1] if toks else ""


# ------------------------------------------------------------------
# country canonicalization (sources spell teams differently:
# bet365 "South Korea" vs PP "Korea Rep", "USA" vs "United States", etc.)
# ------------------------------------------------------------------

_COUNTRY_ALIASES = {
    "united states": "usa", "united states of america": "usa", "usa": "usa",
    "korea rep": "south korea", "korea republic": "south korea",
    "south korea": "south korea", "korea": "south korea",
    "korea dpr": "north korea", "dpr korea": "north korea", "north korea": "north korea",
    "bosnia": "bosnia", "bosnia herzegovina": "bosnia",
    "bosnia and herzegovina": "bosnia",
    "czech republic": "czechia", "czechia": "czechia",
    "turkey": "turkiye", "turkiye": "turkiye",
    "holland": "netherlands", "netherlands": "netherlands",
    "ir iran": "iran", "iran": "iran",
    "cote d ivoire": "ivory coast", "ivory coast": "ivory coast",
    "cabo verde": "cape verde", "cape verde": "cape verde",
    "united arab emirates": "uae", "uae": "uae",
    "congo dr": "dr congo", "dr congo": "dr congo",
    "republic of ireland": "ireland", "ireland": "ireland",
    "curacao": "curacao",
}

_COUNTRY_DISPLAY = {
    "usa": "USA", "south korea": "South Korea", "north korea": "North Korea",
    "bosnia": "Bosnia", "czechia": "Czechia", "turkiye": "Türkiye",
    "uae": "UAE", "dr congo": "DR Congo", "ivory coast": "Ivory Coast",
}


def norm_country(raw: str | None) -> str:
    """Canonical lowercase key for a country/team name."""
    if not raw:
        return ""
    s = strip_accents(str(raw).lower())
    s = "".join(c if c.isalnum() else " " for c in s)
    s = " ".join(s.split())
    return _COUNTRY_ALIASES.get(s, s)


def country_display(canon: str) -> str:
    return _COUNTRY_DISPLAY.get(canon, canon.title())
