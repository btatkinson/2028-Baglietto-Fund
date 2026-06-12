"""Cross-source join.

1) Match Underdog <-> PrizePicks (surname-anchored fuzzy, from line_gap).
2) Attach the bet365 ladder for each (player, stat) via the same fuzzy matcher.
3) Compute implied edge from bet365 at each DFS line.
Returns a DataFrame ready for the report, sorted by best edge.
"""
from __future__ import annotations

import pandas as pd

import edge as edge_model
from normalize import (norm_name, names_compatible, surname,
                       norm_country, country_display)


def _countries_of(*vals):
    """Extract canonical country keys from labels like 'South Korea v Czechia',
    'Korea Rep', or PP combo strings like 'Czechia/Korea Rep'."""
    out = []
    for v in vals:
        if not v:
            continue
        for part in str(v).replace(" v ", "/").split("/"):
            c = norm_country(part)
            if c and c not in out:
                out.append(c)
    return out


def _fixture_registry(b365: pd.DataFrame) -> dict:
    """canonical country -> canonical fixture label, built from bet365's match
    names (authoritative fixtures, home-v-away order preserved)."""
    reg = {}
    if not len(b365) or "match" not in b365.columns:
        return reg
    for mlabel in pd.Series(b365["match"]).dropna().unique():
        parts = [p.strip() for p in str(mlabel).split(" v ")]
        canons = [norm_country(p) for p in parts if p.strip()]
        if len(canons) == 2 and all(canons):
            label = f"{country_display(canons[0])} v {country_display(canons[1])}"
            for c in canons:
                reg.setdefault(c, label)
    return reg


def _canonical_match(registry: dict, b365_match, team, opp) -> str:
    countries = _countries_of(b365_match) or _countries_of(team, opp)
    for c in countries:
        if c in registry:
            return registry[c]
    if len(countries) == 2:
        a, b = sorted(country_display(c) for c in countries)
        return f"{a} v {b}"
    return ""


def _match_dfs(ud: pd.DataFrame, pp: pd.DataFrame):
    pp = pp.copy()
    pp["nname"] = pp["name"].map(norm_name)
    pp["surname"] = pp["nname"].map(surname)
    pp_exact = {(r.nname, r.stat): i for i, r in zip(pp.index, pp.itertuples())}
    pp_by_sur: dict[tuple[str, str], list[int]] = {}
    for i, r in zip(pp.index, pp.itertuples()):
        pp_by_sur.setdefault((r.surname, r.stat), []).append(i)

    used, records = set(), []
    for r in ud.itertuples():
        un = norm_name(r.name)
        idx = pp_exact.get((un, r.stat))
        if idx is None or idx in used:
            cands = [i for i in pp_by_sur.get((surname(un), r.stat), [])
                     if i not in used and names_compatible(un, pp.at[i, "nname"])]
            idx = cands[0] if len(cands) == 1 else None
        if idx is not None:
            used.add(idx)
            h = pp.loc[idx]
            records.append({"name": h["name"], "team": h["team"], "opp": h["opp"],
                            "stat": r.stat, "pp_line": h["line"], "ud_line": r.line,
                            "ud_o_mult": r.over_mult, "ud_u_mult": r.under_mult})
        else:
            records.append({"name": r.name, "team": "", "opp": "", "stat": r.stat,
                            "pp_line": None, "ud_line": r.line,
                            "ud_o_mult": r.over_mult, "ud_u_mult": r.under_mult})
    for i in pp.index:
        if i not in used:
            h = pp.loc[i]
            records.append({"name": h["name"], "team": h["team"], "opp": h["opp"],
                            "stat": h["stat"], "pp_line": h["line"], "ud_line": None,
                            "ud_o_mult": None, "ud_u_mult": None})
    return records


def _index_bet365(b: pd.DataFrame):
    """(token, stat) -> {nname: rows}. Each player is registered under BOTH its
    first and last token, so a reversed-order DFS name (KR family-first vs
    bet365 given-first) can still find the bucket."""
    if not len(b):
        return {}
    b = b.copy()
    b["nname"] = b["player"].map(norm_name)
    idx: dict[tuple[str, str], dict[str, list]] = {}
    for (nname, stat), grp in b.groupby(["nname", "stat"]):
        toks = nname.split()
        if not toks:
            continue
        rows = grp.to_dict("records")
        for key_tok in {toks[0], toks[-1]}:
            idx.setdefault((key_tok, stat), {})[nname] = rows
    return idx


def _find_rows(b365_idx, name, stat):
    """Return (bet365 rows for this player+stat, match label) or (None, None)."""
    nn = norm_name(name)
    toks = nn.split()
    if not toks:
        return None, None
    bucket = {}
    for key_tok in {toks[0], toks[-1]}:
        bucket.update(b365_idx.get((key_tok, stat)) or {})
    if not bucket:
        return None, None
    if nn in bucket:
        rows = bucket[nn]
    else:
        compat = [v for k, v in bucket.items() if names_compatible(nn, k)]
        if len(compat) != 1:
            return None, None
        rows = compat[0]
    return rows, (rows[0].get("match") if rows else None)


def join(ud: pd.DataFrame, pp: pd.DataFrame, b365: pd.DataFrame) -> pd.DataFrame:
    records = _match_dfs(ud, pp)
    if len(b365) and "kind" not in b365.columns:
        b365 = b365.copy()
        b365["kind"] = "nplus"   # old captures: treat everything as one-way
    b365_idx = _index_bet365(b365)

    # ---- per-stat one-way margin calibration ----
    # 1) measured: same player+line has both a devig pair and an N+ price
    measured = edge_model.measure_margins(b365)            # {stat: (m, n)}
    # 2) neutral: pure one-way stats — pick m so median P(over DFS line) = 0.5
    raw_by_stat: dict[str, list] = {}
    rows_cache: dict[int, tuple] = {}
    for i, rec in enumerate(records):
        rows_b, b365_match = _find_rows(b365_idx, rec["name"], rec["stat"])
        rows_cache[i] = (rows_b, b365_match)
        if not rows_b or rec["stat"] in measured:
            continue
        lad0 = edge_model.build_ladder(rows_b, margin=0.0)
        if not lad0 or any(meth == "devig" for _, _, meth in lad0):
            continue   # neutrality only for ladders with no two-sided anchor
        line = rec["pp_line"] if rec["pp_line"] is not None else rec["ud_line"]
        p, _ = edge_model.prob_over_at(lad0, line)
        if p is not None:
            raw_by_stat.setdefault(rec["stat"], []).append(p)
    neutral = {}
    for stat, ps in raw_by_stat.items():
        m = edge_model.neutral_margin(ps)
        if m is not None:
            neutral[stat] = (m, len(ps))

    def resolve_margin(stat):
        if stat in measured:
            return measured[stat][0], "meas"
        if stat in neutral:
            return neutral[stat][0], "neut"
        return None, ""    # build_ladder falls back to config.ONEWAY_MARGIN

    margin_bits = [f"{s}: m={m:.0%} (meas, n={n})" for s, (m, n) in sorted(measured.items())] + \
                  [f"{s}: m={m:.0%} (neutral, n={n})" for s, (m, n) in sorted(neutral.items())]
    if margin_bits:
        print("One-way margins per stat: " + " · ".join(margin_bits))

    # ---- evaluate with calibrated margins ----
    fixtures = _fixture_registry(b365)
    out = []
    for i, rec in enumerate(records):
        rows_b, b365_match = rows_cache[i]
        margin, src = resolve_margin(rec["stat"])
        ladder = edge_model.build_ladder(rows_b, margin=margin, margin_src=src) if rows_b else None
        row = dict(rec)
        # standardized matchup label (country aliases resolved to one fixture)
        row["match"] = _canonical_match(fixtures, b365_match, rec.get("team"), rec.get("opp"))
        row["gap"] = (round(rec["pp_line"] - rec["ud_line"], 2)
                      if rec["pp_line"] is not None and rec["ud_line"] is not None else None)
        row["has_b365"] = bool(ladder)
        row["b365_lines"] = (f"{ladder[0][0]:g}–{ladder[-1][0]:g}" if ladder else "")
        best_edge = None
        for plat, line in (("pp", rec["pp_line"]), ("ud", rec["ud_line"])):
            ev = edge_model.evaluate(ladder, line) if ladder else None
            if not ev:
                row[f"prob_{plat}"] = None
                row[f"edge_{plat}"] = None
                row[f"side_{plat}"] = None
                continue
            row[f"prob_{plat}"] = ev["prob_over"]
            row[f"edge_{plat}"] = ev["edge"]
            row[f"side_{plat}"] = ev["side"]
            row["method"] = ev["method"]
            if best_edge is None or ev["edge"] > best_edge:
                best_edge = ev["edge"]
        row["best_edge"] = best_edge
        out.append(row)

    cols = ["name", "team", "opp", "match", "stat", "pp_line", "ud_line", "gap",
            "has_b365", "b365_lines", "prob_pp", "edge_pp", "side_pp",
            "prob_ud", "edge_ud", "side_ud", "method", "best_edge",
            "ud_o_mult", "ud_u_mult"]
    df = pd.DataFrame(out, columns=cols)
    df.attrs["margins"] = " · ".join(margin_bits)
    # sort: bet365-backed edges first (desc), then by |gap| for the rest
    df["_sortkey"] = df["best_edge"].fillna(-9)
    df["_gapabs"] = df["gap"].abs().fillna(-1)
    df = df.sort_values(["_sortkey", "_gapabs"], ascending=False).drop(columns=["_sortkey", "_gapabs"])
    return df.reset_index(drop=True)
