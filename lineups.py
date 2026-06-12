"""Auto-generate lineups from the edge table.

Rules (per Blake's playbook):
  * each qualified leg is used up to MAX_LEG_USES times (default 2)
  * no two legs are ever paired together in more than one lineup
  * no two legs from the same match in one lineup (avoids unmodeled
    correlation — especially shot overs in the same game)
  * UD and PP lineups are built independently from each platform's own
    positive-edge legs at that platform's cushioned breakeven

EV per lineup is exact: Poisson-binomial over the structure's payout tiers,
with UD leg multipliers applied to the all-correct tier (UD disqualifies a
losing pick's multiplier, so partial tiers use base payouts — conservative).
"""
from __future__ import annotations

import pandas as pd

import config

STRUCTURES = {
    "ud_power2": {"label": "UD 2-Pick Power (3.5x)", "size": 2, "tiers": {2: 3.5}},
    "ud_power3": {"label": "UD 3-Pick Power (6.5x)", "size": 3, "tiers": {3: 6.5}},
    "ud_flex5":  {"label": "UD 5-Pick Flex (10/2.5)", "size": 5, "tiers": {5: 10.0, 4: 2.5}},
    "ud_flex6":  {"label": "UD 6-Pick Flex (25/2.6/0.25)", "size": 6,
                  "tiers": {6: 25.0, 5: 2.6, 4: 0.25}},
    "pp_power3": {"label": "PP 3-Pick Power (6x)", "size": 3, "tiers": {3: 6.0}},
    "pp_flex4":  {"label": "PP 4-Pick Flex (6/1.5)", "size": 4, "tiers": {4: 6.0, 3: 1.5}},
    "pp_flex5":  {"label": "PP 5-Pick Flex (10/2/0.4)", "size": 5,
                  "tiers": {5: 10.0, 4: 2.0, 3: 0.4}},
}


def _isnum(x):
    try:
        f = float(x)
        return f == f
    except (TypeError, ValueError):
        return False


def _legs_from_df(df: pd.DataFrame, plat: str):
    """Qualified legs for a platform: positive (cushioned) edge rows."""
    legs = []
    line_col = "pp_line" if plat == "pp" else "ud_line"
    for i, r in df.iterrows():
        edge = r.get(f"edge_{plat}")
        if not _isnum(edge) or float(edge) <= 0:
            continue
        side, p_over = r.get(f"side_{plat}"), r.get(f"prob_{plat}")
        if not side or not _isnum(p_over) or not _isnum(r.get(line_col)):
            continue
        p = float(p_over) if side == "OVER" else 1.0 - float(p_over)
        mult = r.get("ud_mult") if plat == "ud" else 1.0
        mult = float(mult) if _isnum(mult) else 1.0
        legs.append({
            "id": f"{r['name']}|{r['stat']}",
            "name": r["name"], "stat": r["stat"], "side": side,
            "line": float(r[line_col]), "edge": float(edge), "p": p,
            "mult": mult, "match": str(r.get("match") or f"_unknown_{i}"),
        })
    legs.sort(key=lambda l: -l["edge"])
    return legs


def build_lineups(legs, size, max_uses=2, max_lineups=25, attempts=40):
    """Greedy partial block design with randomized restarts: highest-edge legs
    first, rotating to the least-used leg each round so everyone gets their
    second use with fresh partners. Constraints: per-leg use cap, no repeated
    pair, distinct matches. Greedy can dead-end where a feasible packing
    exists, so we run several jittered attempts and keep the best
    (most lineups, then highest summed edge)."""
    import random
    rng = random.Random(0)

    def attempt(jitter, uses_first):
        uses = {l["id"]: 0 for l in legs}
        used_pairs: set[frozenset] = set()
        out = []
        noise = {l["id"]: rng.random() * jitter for l in legs}
        while len(out) < max_lineups:
            cands = [l for l in legs if uses[l["id"]] < max_uses]
            if len(cands) < size:
                break
            if uses_first:   # spread uses evenly (fresh legs first)
                cands.sort(key=lambda x: (uses[x["id"]], -(x["edge"] + noise[x["id"]])))
            else:            # pure edge order (lets tight overlapping designs form)
                cands.sort(key=lambda x: -(x["edge"] + noise[x["id"]]))
            lineup = []
            for l in cands:
                if len(lineup) == size:
                    break
                if any(m["match"] == l["match"] for m in lineup):
                    continue
                if any(frozenset((l["id"], m["id"])) in used_pairs for m in lineup):
                    continue
                lineup.append(l)
            if len(lineup) < size:
                break
            for i in range(size):
                uses[lineup[i]["id"]] += 1
                for j in range(i + 1, size):
                    used_pairs.add(frozenset((lineup[i]["id"], lineup[j]["id"])))
            out.append(lineup)
        return out

    best = attempt(0.0, True)
    for k in range(attempts - 1):
        cand = attempt(0.05, uses_first=(k % 2 == 0))
        if (len(cand), sum(l["edge"] for lu in cand for l in lu)) > \
           (len(best), sum(l["edge"] for lu in best for l in lu)):
            best = cand
    return best


def _poisson_binomial(ps):
    dist = [1.0]
    for p in ps:
        nd = [0.0] * (len(dist) + 1)
        for k, v in enumerate(dist):
            nd[k] += v * (1.0 - p)
            nd[k + 1] += v * p
        dist = nd
    return dist


def entry_ev(legs, tiers):
    """(EV multiple, P(all correct)). Exact Poisson-binomial over tiers; UD leg
    multipliers applied to the all-correct tier only (losing picks' multipliers
    are disqualified by UD, so this is exact for powers, conservative for flex)."""
    dist = _poisson_binomial([l["p"] for l in legs])
    ev = sum(tiers.get(k, 0.0) * pr for k, pr in enumerate(dist))
    mult_all = 1.0
    for l in legs:
        mult_all *= l["mult"]
    n = len(legs)
    ev += tiers.get(n, 0.0) * dist[n] * (mult_all - 1.0)
    return ev, dist[n]


def generate(df: pd.DataFrame) -> dict:
    """{'ud': {...}, 'pp': {...}} with structure, lineups (legs+ev+p_sweep), n_legs."""
    out = {}
    for plat, skey in (("ud", config.UD_STRUCTURE), ("pp", config.PP_STRUCTURE)):
        s = STRUCTURES.get(skey) or STRUCTURES["ud_power3" if plat == "ud" else "pp_flex5"]
        legs = _legs_from_df(df, plat)
        built = build_lineups(legs, s["size"], config.MAX_LEG_USES)
        lineups = []
        for lu in built:
            ev, p_sweep = entry_ev(lu, s["tiers"])
            lineups.append({"legs": lu, "ev": ev, "p_sweep": p_sweep})
        lineups.sort(key=lambda x: -x["ev"])
        out[plat] = {"structure": s, "lineups": lineups, "n_legs": len(legs)}
    return out
