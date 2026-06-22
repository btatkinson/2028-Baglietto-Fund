"""Edge model.

bet365 is the only priced source for these stats, so it's the reference. For a
player+stat we build an implied P(over line) ladder from bet365's odds, de-vigging
Over/Under pairs where both exist (else using the raw 1/odds with a flag), then
interpolate to the DFS line. Edge = favourable-side probability - breakeven.

ASSUMPTIONS (validate against real data, then tune):
  * Two-sided lines are de-vigged proportionally (basic, standard method).
  * One-sided "Over X" ladders can't be de-vigged (no opposing price), so we
    deflate the raw implied prob by an assumed margin: p=(1/odds)/(1+ONEWAY_MARGIN).
    Flagged method="1way/mNN" so you can see (and discount) them.
  * P(over) is treated as monotonically decreasing in the line for interpolation.
  * BREAKEVEN (config) is the DFS pick'em per-leg breakeven; 0.5 = naive even money.

This computes implied edge from market prices; it is not betting advice.
"""
from __future__ import annotations

import math

import config


def _devig_pair(over_odds, under_odds):
    po, pu = 1.0 / over_odds, 1.0 / under_odds
    return po / (po + pu)


def build_ladder(rows, margin=None, margin_src="") -> list[tuple[float, float, str]]:
    """rows: dicts with side, line, odds_decimal, kind ('ou' two-sided | 'nplus').
    Per line we prefer a two-sided 'ou' price (real de-vig); 'nplus' only fills
    lines/sides not supplied by 'ou'. One-way rungs are deflated by `margin`
    (default config.ONEWAY_MARGIN); margin_src tags where the margin came from
    ('meas' = measured from devig pairs, 'neut' = neutrality-calibrated).
    Returns sorted [(line, p_over, method)]."""
    if margin is None:
        margin = config.ONEWAY_MARGIN
    by_line: dict[float, dict] = {}
    # two-sided rows first so they win ties over one-way rungs at the same line
    for r in sorted(rows, key=lambda r: 0 if str(r.get("kind")) == "ou" else 1):
        line = r.get("line")
        if line is None:
            continue
        try:
            line = float(line)
        except (TypeError, ValueError):
            continue
        if math.isnan(line):
            continue
        side = str(r.get("side") or "").lower()
        try:
            odds = float(r.get("odds_decimal"))
        except (TypeError, ValueError):
            continue
        if odds <= 1.0:
            continue
        kind = "ou" if str(r.get("kind")) == "ou" else "nplus"
        slot = by_line.setdefault(line, {"src": kind})
        if slot["src"] == "ou" and kind == "nplus":
            continue  # don't let a one-way rung overwrite a two-sided line
        slot["under" if "under" in side else "over"] = odds
        if kind == "ou":
            slot["src"] = "ou"

    ladder = []
    tag = f"1way/m{int(round(margin * 100))}" + (f"/{margin_src}" if margin_src else "")
    for line, slot in by_line.items():
        if "over" in slot and "under" in slot:
            ladder.append((line, _devig_pair(slot["over"], slot["under"]), "devig"))
        elif "over" in slot:
            p = (1.0 / slot["over"]) / (1.0 + margin)
            ladder.append((line, min(max(p, 0.001), 0.999), tag))
        elif "under" in slot:
            pu = (1.0 / slot["under"]) / (1.0 + margin)
            ladder.append((line, min(max(1.0 - pu, 0.001), 0.999), tag))
    ladder.sort(key=lambda t: t[0])
    return ladder


def measure_margins(b365_df):
    """Empirically measure the one-way vig per stat: wherever the SAME player+line
    has both a two-sided devig pair AND a one-way N+ price, the one-way margin is
    directly observable as m = (1/odds_1way) / p_devig - 1.
    Returns {stat: (median_margin, n_pairs)}."""
    if b365_df is None or not len(b365_df):
        return {}
    samples: dict[str, list[float]] = {}
    cols = b365_df.columns
    if "kind" not in cols:
        return {}
    grouped: dict[tuple, dict] = {}
    for r in b365_df.to_dict("records"):
        line = r.get("line")
        try:
            line = float(line)
        except (TypeError, ValueError):
            continue
        if math.isnan(line):
            continue
        try:
            odds = float(r.get("odds_decimal"))
        except (TypeError, ValueError):
            continue
        if odds <= 1.0:
            continue
        key = (r.get("stat"), r.get("player"), line)
        slot = grouped.setdefault(key, {})
        side = str(r.get("side") or "").lower()
        if str(r.get("kind")) == "ou":
            slot["ou_under" if "under" in side else "ou_over"] = odds
        elif "under" not in side:
            slot["np_over"] = odds
    for (stat, _, _), slot in grouped.items():
        if "ou_over" in slot and "ou_under" in slot and "np_over" in slot:
            p = _devig_pair(slot["ou_over"], slot["ou_under"])
            if p > 0:
                m = (1.0 / slot["np_over"]) / p - 1.0
                samples.setdefault(stat, []).append(m)
    out = {}
    for stat, ms in samples.items():
        ms.sort()
        med = ms[len(ms) // 2]
        out[stat] = (min(max(med, 0.0), 0.60), len(ms))
    return out


def neutral_margin(raw_probs):
    """Given raw (margin=0) P(over DFS line) samples for a stat, choose the margin
    that makes the MEDIAN deflated prob exactly 0.5 — balanced over/under recs by
    construction. p_deflated = raw/(1+m), so m = median(raw)/0.5 - 1.
    Cap is higher than for measured vig because this margin also absorbs
    rung-vs-DFS-line extrapolation distance (e.g. passes milestone rungs)."""
    ps = sorted(p for p in raw_probs if p is not None)
    if len(ps) < 5:
        return None
    med = ps[len(ps) // 2]
    return min(max(med / 0.5 - 1.0, 0.0), 1.50)


def prob_over_at(ladder, dfs_line: float):
    """Interpolate P(over dfs_line) from the ladder. Returns (prob, method)."""
    if not ladder or dfs_line is None or (isinstance(dfs_line, float) and math.isnan(dfs_line)):
        return None, "none"
    for line, p, method in ladder:
        if abs(line - dfs_line) < 1e-9:
            return p, f"exact/{method}"
    below = [t for t in ladder if t[0] < dfs_line]
    above = [t for t in ladder if t[0] > dfs_line]
    if below and above:
        (l0, p0, m0), (l1, p1, m1) = below[-1], above[0]
        frac = (dfs_line - l0) / (l1 - l0)
        src = m0 if m0 == m1 else f"{m0}|{m1}"
        return p0 + frac * (p1 - p0), f"interp[{src}]"
    if not below and not above:
        return None, "none"
    # outside ladder range -> clamp to nearest endpoint (flagged, with source)
    n_line, n_p, n_m = above[0] if above else below[-1]
    return n_p, f"extrap[{n_m}]"


def evaluate(ladder, dfs_line: float, over_mult=1.0, under_mult=1.0, breakeven=None):
    """Return edge info for a DFS line given a bet365 ladder.

    breakeven: the platform/structure per-leg breakeven (config.BREAKEVEN_UD /
    BREAKEVEN_PP). over_mult/under_mult are the platform's payout multipliers
    per side; None = side not offered (never recommended). Multipliers stack
    multiplicatively into the entry payout, so the required probability is
    breakeven/mult, plus config.EDGE_CUSHION as insurance against the bet365
    de-vig model being wrong.
    """
    if breakeven is None:
        breakeven = 0.5
    p_over, method = prob_over_at(ladder, dfs_line)
    if p_over is None:
        return None
    p_over = min(max(p_over, 0.0), 1.0)
    # tiered cushion: soft tier for neutrality-calibrated or extrapolated probs
    cushion = (config.EDGE_CUSHION_SOFT
               if ("neut" in method or "extrap" in method)
               else config.EDGE_CUSHION)

    cands = []
    for side, p_side, mult in (("OVER", p_over, over_mult),
                               ("UNDER", 1.0 - p_over, under_mult)):
        if mult is None:
            continue                      # side not offered on the platform
        try:
            m = float(mult)
        except (TypeError, ValueError):
            m = 1.0
        if m != m or m <= 0:              # NaN / nonsense -> neutral
            m = 1.0
        be = min(breakeven / m + cushion, 0.999)
        cands.append((side, p_side, p_side - be, be, m))
    if not cands:
        return None
    side, p_side, edge, be, m = max(cands, key=lambda c: c[2])
    return {
        "prob_over": round(p_over, 4),
        "side": side,
        "edge": round(edge, 4),
        "breakeven": round(be, 4),
        "mult": m,
        "method": method,
        "n_lines": len(ladder),
    }


def build_ud_ladder(rungs, skip_symmetric=False):
    """Underdog-implied P(over line) ladder from UD's own option multipliers.

    `rungs`: list of {line, over_mult, under_mult} (the alt-line set that
    parse_underdog now preserves in df.attrs['ud_ladders']). UD's higher/lower
    payout multipliers are de-vigged exactly like a two-sided bet365 price — the
    cheaper-paying side is the more likely one. This is a *relative* read (UD
    multipliers aren't true decimal odds), so it captures UD's lean, not a
    guaranteed-calibrated absolute; validate the level against realized outcomes
    (reliability curve) before trusting it as a maker fair.

      * asymmetric multipliers -> de-vigged lean          (method 'ud-devig')
      * symmetric multipliers   -> 0.5 anchor at that line (method 'ud-anchor';
        carries only UD's line-placement opinion). UD is sharp, so a symmetric
        line really does say "~coin flip here"; skip_symmetric drops these if you
        decide the anchors are noise.

    Returns sorted [(line, p_over, method)]."""
    out, seen = [], set()
    for r in rungs or []:
        line = r.get("line")
        try:
            line = float(line)
        except (TypeError, ValueError):
            continue
        if line != line or line in seen:        # NaN / duplicate line
            continue
        try:
            om, um = float(r.get("over_mult")), float(r.get("under_mult"))
        except (TypeError, ValueError):
            continue
        if om <= 0 or um <= 0 or om != om or um != um:
            continue
        if abs(om - um) < 1e-9:                  # symmetric -> no price signal
            if skip_symmetric:
                continue
            out.append((line, 0.5, "ud-anchor"))
        else:
            out.append((line, _devig_pair(om, um), "ud-devig"))
        seen.add(line)
    out.sort(key=lambda t: t[0])
    return out


def blend_ladders(b365, ud, w_ud=0.6):
    """Blend a bet365 ladder and a UD-implied ladder into one [(line, p, method)].

    w_ud is the weight on UD (default 0.6 — UD treated as the sharper book).
    Either side may be empty/None; if one is, the other is returned unchanged.
    Blending happens at the probability level on the UNION of both ladders'
    lines: each source is interpolated to every line via prob_over_at, then
    weighted-averaged. The method tag records both legs and how each was derived,
    so a 'neut'/'extrap' sub-method still trips evaluate()'s softer cushion."""
    b365, ud = b365 or [], ud or []
    if not b365 and not ud:
        return []
    if not ud:
        return list(b365)
    if not b365:
        return list(ud)
    w_ud = min(max(float(w_ud), 0.0), 1.0)
    w_b = 1.0 - w_ud
    lines = sorted({l for l, _, _ in b365} | {l for l, _, _ in ud})
    out = []
    for line in lines:
        pb, mb = prob_over_at(b365, line)
        pu, mu = prob_over_at(ud, line)
        if pb is None and pu is None:
            continue
        if pb is None:
            out.append((line, pu, f"ud-only[{mu}]"))
        elif pu is None:
            out.append((line, pb, f"365-only[{mb}]"))
        else:
            out.append((line, w_b * pb + w_ud * pu,
                        f"blend{int(round(w_ud * 100))}u[365:{mb}|ud:{mu}]"))
    return out
