"""Goals & assists board — the Kalshi-facing view (Kalshi lists goals, assists and
G+A; nothing else), built straight off the bet365 capture + the model projection.

For the next few kickoffs only (you post after lineups anyway) it shows, per side:

  * the de-vigged team total — bet365's Team Total Goals for goals, 0.75x that for
    assists (the assist anchor devig uses) — nice and big; and
  * every player's HAZARD (lambda = -ln(1-P)) and anytime % beneath it. The hazards
    SUM to the team total by construction, so you can add the column on a calculator
    and land back on the big number. P = 1 - exp(-lambda).

Players bet365 doesn't price fall back to the minutes/rate model's expected count as
their hazard (tagged 'model'); these are shown below the de-vig block, outside the
checksum, since the de-vigged book players already account for the whole team total.

Writes out/scorers.html (eyeball) and out/scorers.csv (flat, for the Kalshi diff).
"""
from __future__ import annotations

import html
import json
import math
import time
from datetime import datetime, timezone

import pandas as pd

import config
import devig
from normalize import norm_name, norm_country, names_compatible

N_GAMES = 3                       # only the soonest few — you post after lineups
_HAZ_STATS = ("Goals", "Assists")
_FULL_MATCH = 98.0                # minutes at/above which a player is a nailed starter
                                  # (match length + stoppage = the minutes-model cap);
                                  # the minutes tilt is a no-op when every listed player
                                  # clears this, so a clean all-starters board is untouched


def _kickoff(g: pd.DataFrame):
    ko = pd.to_numeric(g.get("kickoff"), errors="coerce").dropna() \
        if "kickoff" in g.columns else pd.Series(dtype=float)
    return float(ko.iloc[0]) if len(ko) else None


def _model_players(match, proj):
    """{nname: {'player','team_country','Goals':(mean,var),'Assists':(mean,var)}} for
    one match from the projection cells (empty if no XI / no projection)."""
    out: dict = {}
    if not proj:
        return out
    try:
        from projection import mkey
    except Exception:
        return out
    mk = mkey(match)
    if mk is None:
        return out
    for (cmk, nn, stat), cell in (proj.get("cells") or {}).items():
        if cmk != mk or stat not in _HAZ_STATS:
            continue
        rec = out.setdefault(nn, {"player": cell.get("player") or nn.title(),
                                  "team_country": norm_country(cell.get("team")),
                                  "minutes": cell.get("minutes")})
        rec[stat] = (cell.get("mean"), cell.get("var"))
    return out


def _model_anytime(mean, var):
    if mean is None:
        return None
    try:
        from projection import prob_over
        p = prob_over(mean, var, 0.5)
    except Exception:
        p = None
    if p is None and mean is not None:        # Poisson fallback if scipy/port absent
        p = 1 - math.exp(-float(mean))
    return p


def _ed1(a, b):
    """True if strings a, b are within edit distance 1 (one insert/delete/substitute).
    Cheap one-pass check used only to bridge near-identical name tokens."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la          # ensure a is the shorter/equal
    i = j = edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i, j = i + 1, j + 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1                            # advance the longer
            if la == lb:
                i += 1                        # substitution (equal length)
    return True


_ARABIC_ARTICLE = {"al", "el", "bin", "ibn", "ben", "abu"}


def _arabic_tokens(name):
    """norm_name tokens with Arabic article prefixes glued to the following token, so
    'al amin' / 'al-amin' -> 'alamin' and compare equal to a one-word 'Almanai'."""
    toks = name.split()
    out, i = [], 0
    while i < len(toks):
        if toks[i] in _ARABIC_ARTICLE and i + 1 < len(toks):
            out.append(toks[i] + toks[i + 1])
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return out


def _tok_match(x, y):
    """Two name tokens are the same person-token: equal, off-by-one (Hasan/Hassan), one an
    initial of the other, or a nickname PREFIX of the other (Thelo/Thelonious, Alex/Alexander,
    Manu/Manuel). The prefix arm needs >=3 shared leading chars so it doesn't over-merge, and
    the per-team unique-candidate guard at every call site backstops a wrong collision."""
    if x == y or _ed1(x, y):
        return True
    if (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)):
        return True
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    return len(short) >= 3 and long_.startswith(short)


def _cover(short, long_):
    """Every token of `short` greedily matches a distinct token of `long_` — allowing a
    short token to match the concatenation of two adjacent long tokens ('Abdulaziz' ~
    'Abdel Aziz'). `long_` may carry extra middle/last names. Requires >=2 tokens so
    single given names can't loosely collide."""
    if len(short) < 2:
        return False
    pool = list(long_)
    for t in short:
        hit = -1
        for i in range(len(pool)):
            if _tok_match(t, pool[i]):
                hit = i
                break
            if i + 1 < len(pool) and _tok_match(t, pool[i] + pool[i + 1]):
                pool[i:i + 2] = [pool[i] + pool[i + 1]]     # collapse the pair, now matched
                hit = i
                break
        if hit < 0:
            return False
        pool.pop(hit)
    return True


_ALIASES = None


def _name_aliases() -> dict:
    """{norm_name: canonical norm_name} from config.NAME_ALIASES_JSON, cached. For seams no
    string matcher can bridge (different given names for one player). Missing/bad file → {}."""
    global _ALIASES
    if _ALIASES is None:
        try:
            raw = json.loads(config.NAME_ALIASES_JSON.read_text(encoding="utf-8"))
            _ALIASES = {norm_name(k): norm_name(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            _ALIASES = {}
    return _ALIASES


def _names_compatible_loose(a, b):
    """names_compatible, plus tolerance for the book↔API-Football↔Kalshi spelling seams
    the strict matcher misses: one-letter variants ('Meschack'/'Meschak', 'Hasan'/
    'Hassan'), nickname prefixes ('Thelo'/'Thelonious'), extra name parts ('Jassem Gaber'
    vs 'Jassem Gaber Abdulsallam'), and Arabic article joins ('Mohamed Al Manai' vs
    'Mohamed Naceur Almanai'). A hand-maintained alias map covers different-given-name seams
    no matcher can bridge ('Kouadio Kone'/'Manu Kone'). Token-cover based and >=2-token
    gated; the unique-candidate guard at every call site backstops a wrong collision."""
    al = _name_aliases()
    if al.get(a, a) == al.get(b, b):                 # alias-equal (covers exact too)
        return True
    if names_compatible(a, b):
        return True
    ta, tb = _arabic_tokens(a), _arabic_tokens(b)
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if len(short) == 1 and len(long_) >= 2:
        # a MONONYM (Brazil's 'Rayan', 'Vinicius', 'Ronaldinho') vs a fuller name: match on the
        # given OR family token. Safe only because every call site keeps a UNIQUE candidate
        # within one team/match, so a bare first name can't collide across the slate.
        return _tok_match(short[0], long_[0]) or _tok_match(short[0], long_[-1])
    return _cover(short, long_)


def _raw_hazards(gs):
    """{nname: {'Goals': raw_h, 'Assists': raw_h}} of RAW (vigged) bet365 anytime hazards
    for one side. The anytime score/assist book voids if a player doesn't feature, so each
    listed price is P(scores | he plays) — i.e. a 'given he features' number. We take the
    shortest odds per (player, stat) (= the anytime 1+ line, highest prob) and convert to
    hazard h = -ln(1-1/odds). These are the un-anchored inputs to _price_by_minutes."""
    out = {}
    for stat in ("Goals", "Assists"):
        for nn, gg in gs[gs["stat"] == stat].groupby("nname"):
            o = pd.to_numeric(gg["odds_decimal"], errors="coerce").dropna()
            if len(o):
                p = min(1.0 / float(o.min()), 0.999)        # min odds = anytime 1+ line
                out.setdefault(nn, {})[stat] = -math.log(1 - p)
    return out


def _waterfill(rows, base, facs, caps, target):
    """Distribute the shortfall (target − Σbase) onto players who still have headroom below
    their MINUTES-SCALED ceiling, weighted by expected minutes, so high-minute starters soak it
    up and a player we peg at 11' can barely take any — he's never priced as if he played 70'.
    Returns {id(r): allocated hazard}. Iterative because the per-player ceiling binds: each pass
    adds deficit·factor/Σfactor, clips at the ceiling, and re-spreads the residual among players
    that still have room. If total headroom is exhausted, the sum falls short (conservative —
    better than inflating a fringe player to force the total)."""
    alloc = dict(base)
    deficit = target - sum(base.values())
    for _ in range(50):                                     # converges fast; cap as a backstop
        if deficit <= 1e-9:
            break
        elig = [r for r in rows if facs[id(r)] > 0 and alloc[id(r)] < caps[id(r)] - 1e-12]
        w = sum(facs[id(r)] for r in elig)
        if not elig or w <= 0:
            break                                           # no minutes-headroom left → fall short
        moved = 0.0
        for r in elig:
            add = min(deficit * facs[id(r)] / w, caps[id(r)] - alloc[id(r)])
            alloc[id(r)] += add
            moved += add
        deficit -= moved
        if moved <= 1e-12:
            break
    return alloc


def _two_sided(om, um):
    return (om is not None and um is not None and not pd.isna(om) and not pd.isna(um)
            and om > 0 and um > 0)


def _ud_hazards(ud):
    """{(nname, 'Goals'|'Assists'): hazard} from Underdog's ANYTIME (line 0.5) over/under
    multipliers. Two-sided lines de-vig directly (1/(1+m) implieds normalized by the overround).
    ONE-SIDED lines (only 'higher' posted) are used too — they still imply a price — de-vigged
    by an assumed overround = UD_ONEWAY_VIG_MULT × the slate's MEASURED balanced two-way vig
    (so a lone multiplier isn't taken at full margin). UD voids on a no-play, so these are
    'given he features' hazards h = -ln(1-P), the same basis as the bet365 anytime book."""
    out = {}
    if ud is None or not len(ud):
        return out
    cols = [c for c in ("name", "stat", "line", "over_mult", "under_mult") if c in ud.columns]
    if "name" not in cols or "stat" not in cols:
        return out
    recs = []
    for rec in ud[cols].to_dict("records"):
        line = rec.get("line")
        if (rec.get("stat") in ("Goals", "Assists") and line is not None
                and not pd.isna(line) and abs(float(line) - 0.5) <= 1e-6):
            recs.append(rec)
    if not recs:
        return out
    # measure the balanced two-way overround from the lines that HAVE both sides, then assume a
    # one-sided line carries UD_ONEWAY_VIG_MULT× that vig (fallback to the config baseline).
    rounds = [1.0 / (1.0 + rec["over_mult"]) + 1.0 / (1.0 + rec["under_mult"])
              for rec in recs if _two_sided(rec.get("over_mult"), rec.get("under_mult"))]
    # need a few two-sided lines to trust a measured vig; UD posts almost all G/A one-sided, so
    # in practice this is usually the config baseline.
    base = float(pd.Series(rounds).median()) if len(rounds) >= 3 else config.UD_BASELINE_OVERROUND
    oneway_round = 1.0 + config.UD_ONEWAY_VIG_MULT * (base - 1.0)
    for rec in recs:
        om, um = rec.get("over_mult"), rec.get("under_mult")
        p = devig.two_way_prob(om, um, oneway_overround=(None if _two_sided(om, um) else oneway_round))
        if p is None or not (0 < p < 1):
            continue
        out[(norm_name(rec.get("name")), rec["stat"])] = -math.log(1 - p)
    return out


def _blend_haz(h_b365, h_ud):
    """65/35 bet365/Underdog blend of two anytime hazards (config weights). Uses whichever is
    present when only one book covers the leg, so UD's thin coverage just sharpens the players
    it does price without dropping anyone bet365 lists."""
    if h_b365 is None and h_ud is None:
        return None
    if h_ud is None:
        return h_b365
    if h_b365 is None:
        return h_ud
    return config.BOARD_BLEND_W_B365 * h_b365 + config.BOARD_BLEND_W_UD * h_ud


def _price_by_minutes(players, tt_goals, tt_assists, confirmed=False):
    """Price each book leg from its (blended bet365+UD) RAW vigged hazard, shrunk by expected
    minutes and reconciled to the market team total. Baseline per player = raw × minutes-factor
    ('given he features' price, scaled to how long we expect him on the pitch). Then:

      * Σbase ≥ team_total  → scale every baseline DOWN by k = team_total/Σbase (the normal
        case: the one-way book over-sums, so we shrink to the sharp two-sided total). k ≤ 1 so
        no player exceeds his minutes-adjusted hazard, let alone his raw price.
      * Σbase < team_total  → WATER-FILL the shortfall (see _waterfill): hand the missing hazard
        to players with headroom below their MINUTES-SCALED ceiling, weighted by minutes, so
        high-minute starters absorb it and a player we peg at 11' stays near his 11/90 share
        instead of being inflated back toward full. The ceiling = raw × (1−(1−factor)²): a nailed
        starter (factor≈1) can fill up to his raw price; a fringe 11' player can barely move. If
        the available headroom is exhausted the sum falls short (conservative — we don't inflate
        a fringe player just to hit the total).

    This replaces the old uniform raw×factor×k (capped at raw): that inflated EVERY player by k
    when k>1, including low-minute ones. Minutes factor: matched → min(minutes/full-match, 1);
    UNMATCHED → 1 pre-XI (don't penalize a star we lack data for) but 0 post-XI (the confirmed
    sheet is exhaustive, so a listed player it omits isn't playing). Mutates rows in place."""
    def fac(r):
        m = r.get("minutes")
        if m is None or pd.isna(m):
            return 0.0 if confirmed else 1.0
        return min(max(float(m) / _FULL_MATCH, 0.0), 1.0)
    for rawk, hzk, pk, target in (("g_rawh", "g_haz", "g_p", tt_goals),
                                  ("a_rawh", "a_haz", "a_p", tt_assists)):
        rows = [r for r in players if r.get(rawk)]
        facs = {id(r): fac(r) for r in rows}
        base = {id(r): r[rawk] * facs[id(r)] for r in rows}    # minutes-honest baseline hazard
        s = sum(base.values())
        if target and s > 0 and s >= target:                  # over-sums → uniform down-scale
            k = target / s
            alloc = {i: v * k for i, v in base.items()}
        elif target and s > 0:                                # under-sums → water-fill onto minutes
            caps = {id(r): r[rawk] * (1 - (1 - facs[id(r)]) ** 2) for r in rows}  # minutes-scaled
            alloc = _waterfill(rows, base, facs, caps, target)
        else:                                                 # no target / no hazard → baseline
            alloc = base
        for r in rows:
            h = alloc[id(r)]
            r[hzk] = h
            r[pk] = 1 - math.exp(-h)
    for r in players:                  # G+A union recomputed from the priced legs
        if r.get("g_p") is not None and r.get("a_p") is not None:
            r["ga_p"] = 1 - (1 - r["g_p"]) * (1 - r["a_p"])


def _apply_overrides(rows, mk, overrides):
    """Stamp manual minutes overrides onto rows whose (mkey, norm_name) matches. Keyed by
    the string 'mk|nname' (the same key board.py persists). Mutates rows; called right
    before _price_by_minutes so the override drives the re-price, not just the display."""
    if not overrides or mk is None:
        return
    for r in rows:
        m = overrides.get(f"{mk}|{norm_name(r['player'])}")
        if m is not None:
            r["minutes"] = m
            r["min_override"] = True


def build(b365: pd.DataFrame, proj=None, overrides=None, force_confirmed=None,
          ud=None) -> list[dict]:
    """Per-game structure for the soonest N_GAMES, each with two team panels of
    goal/assist hazards. Each book leg is priced from bet365's raw anytime hazard,
    shrunk by expected minutes and scaled to the market team total (see
    _price_by_minutes), so the hazards sum to ~the team total with nailed starters near
    their raw price and bench/rotation risks deflated.

    `overrides` ({'mk|nname': minutes}) replaces a player's model minutes with a
    hand-set value before pricing — e.g. a surprise sub the model under-allocated.
    `force_confirmed` (set of mkeys) marks those matches confirmed even when the model
    couldn't read the XI (API-Football lineup lag) — the manual-XI escape hatch; combine
    with `overrides` carrying that XI's minutes so unlisted book players price to 0."""
    if b365 is None or not len(b365) or "stat" not in getattr(b365, "columns", []):
        return []
    try:
        from projection import mkey as _mkey
    except Exception:
        def _mkey(_m):
            return None
    df = b365.copy()
    df["nname"] = df["player"].map(norm_name)
    ud_haz = _ud_hazards(ud)                              # (nname, stat) -> UD anytime hazard
    kinds = {f.get("match"): f.get("lineup_kind")
             for f in (proj or {}).get("fixtures", [])}     # confirmed | projected | None

    # soonest games first; keep only those with any goal/assist market
    order = []
    for match, g in df.groupby("match"):
        if not str(match).strip():
            continue
        if not ((g["stat"] == "Goals") | (g["stat"] == "Assists")).any():
            continue
        order.append((_kickoff(g) if _kickoff(g) is not None else math.inf, match, g))
    order.sort(key=lambda t: t[0])

    games = []
    for ko, match, g in order[:N_GAMES]:
        home = g["home"].dropna().iloc[0] if g.get("home") is not None and g["home"].notna().any() else ""
        away = g["away"].dropna().iloc[0] if g.get("away") is not None and g["away"].notna().any() else ""
        model = _model_players(match, proj)
        mkm = _mkey(match)
        manual_conf = bool(force_confirmed and mkm in force_confirmed)
        is_conf = (kinds.get(match) == "confirmed") or manual_conf
        sides = []
        for side, tlabel in (("home", home), ("away", away)):
            gs = g[g["player_team"] == side]
            if not len(gs):
                continue
            tt = pd.to_numeric(gs.get("team_total"), errors="coerce").dropna()
            tt_goals = float(tt.iloc[0]) if len(tt) else None
            tt_assists = devig.ASSIST_FRACTION * tt_goals if tt_goals is not None else None
            name_of = {r.nname: r.player for r in gs.itertuples()}      # display spelling

            raw = _raw_hazards(gs)                                       # raw vigged hazards
            book_players, book_nn, book_by_nn = [], set(gs["nname"]), {}
            for nn in book_nn:
                # blend bet365 + Underdog anytime hazards (65/35) per leg, where UD covers it
                grh = _blend_haz(raw.get(nn, {}).get("Goals"), ud_haz.get((nn, "Goals")))
                arh = _blend_haz(raw.get(nn, {}).get("Assists"), ud_haz.get((nn, "Assists")))
                if grh is None and arh is None:
                    continue
                row = {
                    "player": name_of.get(nn, nn.title()), "source": "bet365",
                    "minutes": (model.get(nn) or {}).get("minutes"),     # exact-name minutes
                    "g_rawh": grh, "a_rawh": arh,                        # priced below
                    "g_haz": None, "g_p": None, "a_haz": None, "a_p": None, "ga_p": None}
                book_players.append(row)
                book_by_nn[nn] = row

            # model fallback: priced by the model, not by bet365 on this side. Before
            # listing one as its own (amber) row, try to attach its minutes to a
            # bet365 player that still has no projection but whose name is compatible
            # (the book↔API-Football spelling seam, e.g. 'Luis Suarez' vs 'Luis Javier
            # Suarez'); highest-minute model players get first claim. Only a UNIQUE
            # compatible, still-unprojected book row is filled — ambiguous stays amber.
            tcountry = norm_country(tlabel)
            model_players = []
            for nn, rec in sorted(model.items(),
                                  key=lambda kv: -(kv[1].get("minutes") or 0)):
                if nn in book_nn or rec.get("team_country") != tcountry:
                    continue
                gmean, gvar = rec.get("Goals", (None, None))
                amean, avar = rec.get("Assists", (None, None))
                gp, ap = _model_anytime(gmean, gvar), _model_anytime(amean, avar)
                if not gp and not ap and not rec.get("minutes"):
                    continue                 # nothing to show (e.g. backup keeper: 0 min, 0 hazard)
                cand = [bnn for bnn, row in book_by_nn.items()
                        if row.get("minutes") is None and _names_compatible_loose(nn, bnn)]
                if len(cand) == 1:
                    book_by_nn[cand[0]]["minutes"] = rec.get("minutes")
                    continue
                gap = None if (gp is None or ap is None) else 1 - (1 - gp) * (1 - ap)
                model_players.append({
                    "player": rec.get("player", nn.title()), "source": "model",
                    "minutes": rec.get("minutes"),
                    "g_haz": gmean, "g_p": gp,
                    "a_haz": amean, "a_p": ap, "ga_p": gap})

            # minutes are now fully attached (incl. the name-seam fills above); apply any
            # manual override last (it wins over the model), then price the raw book by
            # expected minutes + team total before summing/sorting. Post-XI, book players
            # still unmatched are treated as not in the squad (factor 0).
            _apply_overrides(book_players, mkm, overrides)
            _apply_overrides(model_players, mkm, overrides)
            # a MANUAL XI is exhaustive: any book player not in it (no override) isn't
            # playing, so drop the model's projected minutes that leaked onto the row →
            # factor 0 under confirmed. (Real model-confirmed XIs are already exhaustive.)
            if manual_conf:
                for r in book_players:
                    if not r.get("min_override"):
                        r["minutes"] = None
            _price_by_minutes(book_players, tt_goals, tt_assists, confirmed=is_conf)
            book_players.sort(key=lambda r: (r["g_haz"] is None, -(r["g_haz"] or 0)))
            model_players.sort(key=lambda r: (r["g_haz"] is None, -(r["g_haz"] or 0)))
            sides.append({
                "team": tlabel, "tt_goals": tt_goals, "tt_assists": tt_assists,
                "book": book_players, "model": model_players,
                "sum_min": sum(r["minutes"] for r in (book_players + model_players)
                               if r.get("minutes") is not None),
                "sum_g": sum(r["g_haz"] for r in book_players if r["g_haz"]),
                "sum_a": sum(r["a_haz"] for r in book_players if r["a_haz"])})
        if sides:
            games.append({"match": match, "kickoff": ko, "sides": sides,
                          "lineup_kind": "confirmed" if is_conf else kinds.get(match),
                          "manual_xi": manual_conf})
    return games


# ----------------------------------- render -----------------------------------

def _pct(x):
    return "" if x is None or pd.isna(x) else f"{float(x) * 100:.0f}%"


def _num(x):
    return "" if x is None or pd.isna(x) else f"{float(x):.2f}"


def _min(x):
    return "" if x is None or pd.isna(x) else f"{float(x):.0f}"


def _kalshi_fee_c(p, rate=None):
    """Kalshi trading fee in CENTS for one contract at prob p (rate*p*(1-p) of $1,
    rounded up to the cent, the way Kalshi bills it). Symmetric in p, so YES and NO
    carry the same fee. `rate` defaults to the taker rate; pass the maker rate for a rest."""
    return math.ceil(100 * (config.KALSHI_FEE_RATE if rate is None else rate) * p * (1 - p))


def _accept(p, cushion=None, rate=None):
    """(acceptable YES ¢, acceptable NO ¢) for a fair YES prob p: the most we'd PAY for each
    side and still clear `cushion` edge after the Kalshi fee. Take a side iff the market offers
    it at/below its number. A side is None when it floors below 1¢ (too thin to ever pay).
    `cushion` defaults to the take cushion; pass (cushion+liquidity premium) for a resting make.
    `rate` overrides the fee rate (maker rate for a resting order)."""
    if p is None or pd.isna(p):
        return None, None
    cushion = config.KALSHI_EDGE_CUSHION if cushion is None else cushion
    buf = 100 * cushion + _kalshi_fee_c(p, rate)
    def clip(c):
        return 99 if c > 99 else (None if c < 1 else int(c))
    return clip(round(100 * p - buf)), clip(round(100 * (1 - p) - buf))


def _maker(p, show, cushion=None, rate=None):
    """(acceptable YES ¢, acceptable NO ¢) honoring the post-XI gate AND the noise floor:
    nothing is quoted pre-XI, and a market whose YES prob is below KALSHI_MIN_QUOTE_PROB
    (a 2'-cameo player) is dropped entirely so we never rest an order on it. Default is the
    TAKE threshold; pass cushion=(CUSHION+LIQUIDITY_PREMIUM), rate=MAKER for the make ceiling."""
    if not show or p is None or pd.isna(p) or p < config.KALSHI_MIN_QUOTE_PROB:
        return None, None
    return _accept(p, cushion, rate)


def _ov_title(ov):
    """Tooltip for a highlighted maker cell: the action verdict + the live Kalshi book."""
    bits = []
    if ov.get("take_yes"):
        bits.append(f"TAKE YES @{ov['k_yes_ask']} +{ov['edge_yes']}c")
    if ov.get("take_no"):
        bits.append(f"TAKE NO @{ov['k_no_ask']} +{ov['edge_no']}c")
    if ov.get("make_yes") and not ov.get("take_yes"):
        bits.append(f"make YES bid {ov['acc_yes']}")
    if ov.get("make_no") and not ov.get("take_no"):
        bits.append(f"make NO bid {ov['acc_no']}")
    live = (f"K yes {ov['k_yes_bid']}/{ov['k_yes_ask']} · no {ov['k_no_bid']}/{ov['k_no_ask']} · "
            f"OI {ov['k_oi']:.0f}")
    return "  ".join(bits) + ("  ·  " if bits else "") + live


# take-edge magnitude buckets (¢ of profit beyond our half-spread + fee), so the board can
# shout the fat pitches: 3 = big (>=6¢), 2 = good (3-5¢), 1 = thin (1-2¢), 0 = none.
_EDGE_BIG, _EDGE_GOOD = 6, 3


def _edge_tier(e):
    if e is None or e < 1:
        return 0
    return 3 if e >= _EDGE_BIG else (2 if e >= _EDGE_GOOD else 1)


def _row_best_edge(kx):
    """Largest takeable edge (¢) across a row's G/A/G+A verdicts, or None."""
    edges = []
    for v in (kx or {}).values():
        if v.get("take_yes") and v.get("edge_yes") is not None:
            edges.append(v["edge_yes"])
        if v.get("take_no") and v.get("edge_no") is not None:
            edges.append(v["edge_no"])
    return max(edges) if edges else None


def _mk_cell(p, show, ov=None, interactive=False, match=None, player=None, market=None):
    """Compact 'YES/NO' acceptable-maker-price cell in ¢. '·' until the XI is confirmed
    (pre-XI minutes are too soft to rest a real Kalshi order on); blank for a noise
    market below the quote floor. When a live Kalshi verdict (`ov`) is supplied, the
    SPECIFIC side that's actionable is highlighted — green = TAKE (cross the live book),
    amber = make (improve it) — so a two-sided make shows both numbers lit. Hover the cell
    for the action + live bid/ask + OI. In `interactive` mode each lit side is clickable
    (carries match/player/market/side) to open a Kalshi order ticket."""
    if not show:
        return "<td class='num mk'>·</td>"
    ay, an = _maker(p, True)
    if ay is None and an is None:
        return "<td class='num mk'></td>"
    ov = ov or {}
    def side(c, take, make, sname):
        s = "—" if c is None else str(c)
        cls = "ks take" if take else ("ks make" if make else None)
        if cls is None:
            return s
        inner = s
        if take:                                   # heat-tier the take + show its ¢ edge
            edge = ov.get(f"edge_{sname}")
            t = _edge_tier(edge)
            if t:
                cls += f" e{t}"
            if edge:
                inner = f"{s}<sup class='eg'>+{edge}</sup>"
        if interactive and ov.get("ticker"):
            data = (f" data-match=\"{html.escape(str(match))}\""
                    f" data-player=\"{html.escape(str(player))}\""
                    f" data-market=\"{market}\" data-side=\"{sname}\"")
            return f"<span class='{cls} order-cell'{data}>{inner}</span>"
        return f"<span class='{cls}'>{inner}</span>"
    yes = side(ay, ov.get("take_yes"), ov.get("make_yes"), "yes")
    no = side(an, ov.get("take_no"), ov.get("make_no"), "no")
    hot = any(ov.get(k) for k in ("take_yes", "take_no", "make_yes", "make_no"))
    cls = "num mk kcell" if hot else "num mk"
    t = f" title='{html.escape(_ov_title(ov))}'" if hot else ""
    return f"<td class='{cls}'{t}>{yes}<span class='sl'>/</span>{no}</td>"


def _attach_overlay(games, krows):
    """Join live Kalshi overlay rows onto the board's book rows (keyed match × player ×
    market), so the maker cells can highlight takes/makes. Sets row['kalshi'] = {market:
    verdict}. Mutates games in place."""
    by = {}
    for r in (krows or []):
        by[(r["match"], norm_name(r["player"]), r["market"])] = r
    for gm in games:
        for s in gm.get("sides", []):
            for row in s.get("book", []) + s.get("model", []):     # model rows quote too now
                nn = norm_name(row["player"])
                kx = {mk: by[(gm["match"], nn, mk)]
                      for mk in ("G", "A", "G+A") if (gm["match"], nn, mk) in by}
                if kx:
                    row["kalshi"] = kx


def _when(ko):
    if not ko or math.isinf(ko):
        return ""
    dt = datetime.fromtimestamp(int(ko), tz=timezone.utc).strftime("%a %d %b %H:%M UTC")
    dh = (ko - time.time()) / 3600
    rel = "started" if dh < 0 else (f"in {dh:.0f}h" if dh >= 1 else f"in {dh * 60:.0f}m")
    return f"{dt} · {rel}"


def _news_chip(r):
    """Inline injury/news chip for a row (LLM veto). Only doubtful/out are shown — starts
    /unknown stay silent to keep the board clean. 'conflict' (out/doubtful while we project
    a near-full match) is the loudest case. Links to the cited source when present."""
    n = r.get("news")
    if not n or n.get("status") not in ("doubtful", "out"):
        return ""
    label = "OUT" if n["status"] == "out" else "doubtful"
    cls = "nflag " + ("nout" if n["status"] == "out" else "ndbt")
    if r.get("news_conflict"):
        cls += " nconf"
    reason = n.get("reason") or ""
    conf = n.get("confidence") or ""
    title = html.escape((f"{reason} ({conf} confidence)" if reason else f"{conf} confidence").strip())
    url = n.get("source_url") or ""
    inner = f"⚠ {label}"
    if url.startswith("http"):
        return f" <a class='{cls}' href='{html.escape(url)}' target='_blank' title='{title}'>{inner}</a>"
    return f" <span class='{cls}' title='{title}'>{inner}</span>"


def _min_cell(r, match, interactive):
    """Minutes cell. Static = plain number; interactive = an editable input carrying
    match/player so a commit POSTs a manual override, with an × to clear when one is set."""
    val = _min(r.get("minutes"))
    if not interactive:
        return f"<td class='num dim'>{val}</td>"
    ovr = bool(r.get("min_override"))
    xbtn = "<span class='ovx' title='clear manual override'>×</span>" if ovr else ""
    cls = "num dim minc" + (" ovon" if ovr else "")
    return (f"<td class='{cls}' data-match=\"{html.escape(str(match))}\" "
            f"data-player=\"{html.escape(str(r['player']))}\">"
            f"<input class='mininp' type='number' min='0' max='120' step='1' value='{val}'>{xbtn}</td>")


def _rows_html(players, with_total, sum_g, sum_a, tt_goals, tt_assists, confirmed=False,
               interactive=False, match=None):
    trs = []
    for r in players:
        # maker prices: post-XI, not news-suppressed. Includes MODEL-est rows (bet365 only
        # prices ~10-11 anytime scorers, but Kalshi lists ~20) so a bench player bet365
        # skipped still gets a quote off our model fair value; the noise floor keeps it to
        # the worth-trading roster. Model rows render amber to flag they lack a book anchor.
        mk = confirmed and not r.get("news_suppress")
        kx = (r.get("kalshi") or {}) if mk else {}
        # a row earns the green marker when any of its markets is takeable on live Kalshi;
        # the bar + name pill scale with the row's BIGGEST take edge so fat pitches pop
        best_edge = _row_best_edge(kx)
        etier = _edge_tier(best_edge)
        rowcls = ["mdl"] if r["source"] == "model" else []
        if best_edge is not None:
            rowcls += ["krow", f"e{etier}"]
        cls = f" class='{' '.join(rowcls)}'" if rowcls else ""
        tag = " <span class='tag'>model</span>" if r["source"] == "model" else ""
        ehot = f" <span class='ehot e{etier}'>+{best_edge}¢</span>" if best_edge is not None else ""
        pl = r["player"]
        trs.append(
            f"<tr{cls}><td class='p'>{html.escape(str(pl))}{tag}{ehot}{_news_chip(r)}</td>"
            f"{_min_cell(r, match, interactive)}"
            f"<td class='num'>{_num(r['g_haz'])}</td><td class='num'>{_pct(r['g_p'])}</td>"
            f"<td class='num'>{_num(r['a_haz'])}</td><td class='num'>{_pct(r['a_p'])}</td>"
            f"<td class='num gap'>{_pct(r['ga_p'])}</td>"
            f"{_mk_cell(r['g_p'], mk, kx.get('G'), interactive, match, pl, 'G')}"
            f"{_mk_cell(r['a_p'], mk, kx.get('A'), interactive, match, pl, 'A')}"
            f"{_mk_cell(r['ga_p'], mk, kx.get('G+A'), interactive, match, pl, 'G+A')}</tr>")
    if with_total:
        chk = "✓" if (tt_goals is not None and abs(sum_g - tt_goals) < 0.02) else ""
        trs.append(
            f"<tr class='tot'><td>Σ hazard {chk}</td><td></td>"
            f"<td class='num'>{_num(sum_g)}</td><td></td>"
            f"<td class='num'>{_num(sum_a)}</td><td></td><td></td>"
            f"<td></td><td></td><td></td></tr>")
    return "".join(trs)


def _panel(side, confirmed=False, interactive=False, match=None):
    head = ("<tr><th>Player</th><th class='num'>Min</th><th class='num'>G λ</th><th class='num'>G%</th>"
            "<th class='num'>A λ</th><th class='num'>A%</th><th class='num gap'>G+A%</th>"
            "<th class='num mk'>G Y/N</th><th class='num mk'>A Y/N</th><th class='num mk'>G+A Y/N</th></tr>")
    body = _rows_html(side["book"], True, side["sum_g"], side["sum_a"],
                      side["tt_goals"], side["tt_assists"], confirmed, interactive, match)
    if side["model"]:
        body += ("<tr class='sep'><td colspan='10'>not priced by bet365 — model est.</td></tr>"
                 + _rows_html(side["model"], False, 0, 0, None, None, confirmed, interactive, match))
    smin = (f"<span class='smin'>Σmin {side['sum_min']:.0f}</span>"
            if side.get("sum_min") else "")
    return (f"<div class='team'>"
            f"<div class='th'><span class='tn'>{html.escape(str(side['team']))}</span>"
            f"<span class='big'>xG {_num(side['tt_goals'])}</span>"
            f"<span class='big xa'>xA {_num(side['tt_assists'])}</span>{smin}</div>"
            f"<table><thead>{head}</thead><tbody>{body}</tbody></table></div>")


_BOARD_CSS = """
 body{margin:0;background:#0f1115;color:#e6e9ef;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;padding:18px 22px}
 h1{font-size:18px;margin:0 0 4px}
 h2{font-size:15px;margin:26px 0 8px;color:#cfe0ff;display:flex;gap:12px;align-items:baseline}
 .ko{font-size:12px;color:#8b93a7;font-weight:400}
 .badge{font-size:11px;font-weight:700;padding:1px 8px;border-radius:9px}
 .badge.proj{color:#231700;background:#e0a458}
 .badge.conf{color:#04210f;background:#7ee0a0}
 .badge.man{color:#04210f;background:#8fd0ff}
 .smin{font-size:12px;color:#8b93a7;font-variant-numeric:tabular-nums}
 .teams{display:flex;gap:26px;flex-wrap:wrap}
 .team{min-width:340px}
 .th{display:flex;align-items:baseline;gap:14px;margin-bottom:4px}
 .tn{font-size:16px;font-weight:700}
 .big{font-size:22px;font-weight:700;color:#7ee0a0;font-variant-numeric:tabular-nums}
 .big.xa{color:#9fc6ff}
 table{border-collapse:collapse}
 th,td{padding:4px 10px;border-bottom:1px solid #20242e;text-align:left;white-space:nowrap}
 th{font-size:11px;color:#8b93a7;font-weight:600}
 td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
 td.dim,.dim{color:#8b93a7}
 .gap{color:#cdb6ff} .p{font-weight:600}
 .tot td{border-top:2px solid #3a4150;border-bottom:none;font-weight:700;color:#7ee0a0}
 .tot td.num:nth-child(5){color:#9fc6ff}
 .sep td{color:#e0a458;font-size:11px;font-style:italic;border-bottom:none;padding-top:8px}
 tr.mdl td{color:#e0a458} tr.mdl td.p{color:#e8b878} tr.mdl td.dim{color:#b9883f}
 tr.mdl td.mk{color:#c79a55} tr.mdl td.mk .sl{color:#6e5526}
 .tag{font-size:9px;color:#e0a458;border:1px solid #6e5526;border-radius:3px;padding:0 3px;margin-left:5px}
 td.mk,th.mk{color:#7ee0a0;border-left:1px solid #20242e}
 th.mk{color:#5f8f72} td.mk .sl{color:#3a4150;margin:0 1px}
 td.kcell{cursor:help}
 .ks{border-radius:3px;padding:0 3px}
 .ks.take{background:#1c5236;color:#aef7c8;font-weight:700}
 .ks.take.e2{background:#1c7a47;color:#e2fff0}
 .ks.take.e3{background:#2bd277;color:#04210f}
 .ks.make{background:#4a3a16;color:#f0c478}
 sup.eg{font-size:8px;font-weight:700;margin-left:1px;opacity:.85}
 .ehot{font-size:10px;font-weight:800;border-radius:4px;padding:0 5px;vertical-align:middle;font-variant-numeric:tabular-nums}
 .ehot.e1{background:#244b38;color:#9ad9b4}
 .ehot.e2{background:#1c7a47;color:#d7ffe7}
 .ehot.e3{background:#2bd277;color:#04210f;box-shadow:0 0 0 2px rgba(43,210,119,.4)}
 tr.krow.e1 td.p{box-shadow:inset 3px 0 0 #3f8f64}
 tr.krow.e2 td.p{box-shadow:inset 4px 0 0 #2bb56a}
 tr.krow.e3 td.p{box-shadow:inset 5px 0 0 #2bd277}
 .nflag{font-size:9px;font-weight:700;border-radius:3px;padding:0 4px;margin-left:6px;text-decoration:none;white-space:nowrap}
 a.nflag:hover{text-decoration:underline}
 .nflag.ndbt{color:#231700;background:#e0a458}
 .nflag.nout{color:#2a0c0c;background:#e07a7a}
 .nflag.nconf{box-shadow:0 0 0 2px #ff5d5d}
"""

# Extra styling active only on the live /scorers board (editable minutes + clickable order cells).
_INTERACTIVE_CSS = """
 .minc{position:relative}
 .mininp{width:38px;background:#171a21;color:#e6e9ef;border:1px solid #2a3140;border-radius:4px;
         padding:1px 3px;font:inherit;text-align:right;font-variant-numeric:tabular-nums}
 .mininp:focus{outline:none;border-color:#1f6feb}
 td.minc.ovon .mininp{border-color:#7ee0a0;color:#9ff0bd}
 .ovx{cursor:pointer;color:#e07a7a;font-weight:700;margin-left:4px}
 .ovx:hover{color:#ff9b9b}
 .order-cell{cursor:pointer;text-decoration:underline dotted}
 .order-cell:hover{filter:brightness(1.25)}
 .xibtn{margin-left:8px;background:#2a3140;color:#cfe0ff;border:1px solid #3a4150;border-radius:6px;
        padding:1px 8px;font-size:11px;cursor:pointer}
 .xibtn:hover{background:#3a4150}
 .toolbar{display:flex;gap:10px;align-items:center;margin:6px 0 10px}
 .tbtn{background:#1f6feb;color:#fff;border:0;padding:6px 12px;border-radius:7px;font-size:13px;cursor:pointer}
 .tbtn.sec{background:#2a3140}
 .tnote{font-size:12px;color:#8b93a7}
 #ovl{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:center;justify-content:center;z-index:50}
 #ovl.on{display:flex}
 .modal{background:#161a22;border:1px solid #2a3140;border-radius:12px;padding:20px 22px;min-width:340px;max-width:440px;box-shadow:0 10px 40px rgba(0,0,0,.5)}
 .modal h3{margin:0 0 4px;font-size:16px}
 .modal .sub{font-size:12px;color:#8b93a7;margin-bottom:12px}
 .modal .row{display:flex;justify-content:space-between;align-items:center;margin:8px 0;font-size:13px}
 .modal label{color:#8b93a7}
 .modal input.f{width:90px;background:#0f1115;color:#e6e9ef;border:1px solid #2a3140;border-radius:6px;padding:5px 7px;text-align:right;font:inherit}
 .modal .book{font-size:12px;color:#8b93a7;font-variant-numeric:tabular-nums;margin:8px 0}
 .modal .acts{display:flex;gap:10px;margin-top:16px}
 .modal .place{flex:1;background:#1c8b4f;color:#fff;border:0;padding:10px;border-radius:8px;font-size:14px;font-weight:700;cursor:pointer}
 .modal .place:disabled{opacity:.5;cursor:not-allowed}
 .modal .cancel{background:#2a3140;color:#e6e9ef;border:0;padding:10px 14px;border-radius:8px;cursor:pointer}
 .modal .status{font-size:12px;margin-top:12px;min-height:16px}
 .modal .status.ok{color:#7ee0a0} .modal .status.err{color:#f08a96}
"""

_MODAL_HTML = """
<div id="ovl"><div class="modal">
  <h3 id="m_title">Kalshi order</h3>
  <div class="sub" id="m_sub"></div>
  <div class="row"><label>Ticker</label><span id="m_ticker" style="font-variant-numeric:tabular-nums"></span></div>
  <div class="row"><label>Price (¢)</label><input class="f" id="m_price" type="number" min="1" max="99" step="1"></div>
  <div class="row"><label>Contracts</label><input class="f" id="m_count" type="number" min="1" step="1"></div>
  <div class="row"><label>Risk / payout</label><span id="m_rp"></span></div>
  <div class="row"><label>Edge (fair)</label><span id="m_edge"></span></div>
  <div class="row"><label>Expires</label><span id="m_exp"></span></div>
  <div class="book" id="m_book"></div>
  <div class="acts">
    <button class="place" id="m_place">Place order</button>
    <button class="cancel" id="m_cancel">Cancel</button>
  </div>
  <div class="status" id="m_status"></div>
</div></div>
"""

_INTERACTIVE_JS = r"""
<script>
const $ = id => document.getElementById(id);
let TICKET = null;

async function postForm(url, data){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  return r;
}
async function swapBoard(url, data){
  const r = await postForm(url, data);
  const txt = await r.text();
  if(r.ok){ $('board').innerHTML = txt; }
  else { alert('Update failed: ' + txt); }
}

// --- minutes overrides (event-delegated, survives board re-render) ---
document.addEventListener('change', e => {
  if(!e.target.classList || !e.target.classList.contains('mininp')) return;
  const td = e.target.closest('.minc');
  const v = e.target.value.trim();
  swapBoard('/board/minutes', {match: td.dataset.match, player: td.dataset.player,
                               minutes: v === '' ? null : parseFloat(v)});
});
document.addEventListener('keydown', e => {
  if(e.target.classList && e.target.classList.contains('mininp') && e.key === 'Enter'){ e.target.blur(); }
});

// --- clicks: clear-override (×), refresh-kalshi, open order ticket ---
document.addEventListener('click', e => {
  const x = e.target.closest('.ovx');
  if(x){ const td = x.closest('.minc');
         swapBoard('/board/minutes', {match: td.dataset.match, player: td.dataset.player, minutes: null});
         return; }
  if(e.target.id === 'k_refresh'){ swapBoard('/board/kalshi-refresh', {}); return; }
  if(e.target.classList && e.target.classList.contains('xibtn')){
    swapBoard('/board/clear-xi', {match: e.target.dataset.match}); return; }
  const oc = e.target.closest('.order-cell');
  if(oc){ openTicket(oc.dataset); return; }
});

async function openTicket(ds){
  $('m_status').textContent=''; $('m_status').className='status';
  $('m_title').textContent='Loading…'; $('m_sub').textContent='';
  $('m_place').disabled=true; $('ovl').classList.add('on');
  const q = new URLSearchParams({match:ds.match, player:ds.player, market:ds.market, side:ds.side});
  try{
    const r = await fetch('/board/ticket?' + q.toString());
    const t = await r.json();
    if(!r.ok || t.error){ $('m_title').textContent='Unavailable'; $('m_sub').textContent = t.error || ('HTTP '+r.status); return; }
    TICKET = t;
    $('m_title').textContent = `${t.player} — ${t.market} ${t.side.toUpperCase()}`;
    const modeTag = t.post_only ? `${t.mode} · resting (post-only, maker)` : `${t.mode} · crossing (taker)`;
    $('m_sub').innerHTML = `${t.match} · ${modeTag}` +
      (t.source === 'model' ? ` · <b style="color:#e0a458">⚠ MODEL est — no bet365 anchor, softer</b>` : '') +
      (t.tradable ? '' : ` · <b style="color:#e0a458">⚠ view-only — XI not confirmed (projected minutes)</b>`);
    $('m_ticker').textContent = t.ticker;
    $('m_price').value = t.price_c;
    $('m_count').value = t.count;
    $('m_edge').textContent = `+${t.edge_c}¢ (fair ${Math.round(t.fair*100)}%)`;
    $('m_exp').textContent = t.exp_str;
    $('m_book').textContent = `live: YES ${t.k_yes_bid??'·'}/${t.k_yes_ask??'·'}  ·  NO ${t.k_no_bid??'·'}/${t.k_no_ask??'·'}  ·  OI ${t.k_oi??0}`;
    recalcRP();
    // pre-XI is view-only: show the edge but block placement until the lineup is confirmed
    $('m_place').disabled = !t.tradable;
    if(!t.tradable){ setStatus('view-only until XI confirmed — no order can be placed yet',''); }
  }catch(err){ $('m_title').textContent='Error'; $('m_sub').textContent=String(err); }
}
function recalcRP(){
  const p = parseInt($('m_price').value||'0'), c = parseInt($('m_count').value||'0');
  const risk = (p*c/100), pay = c;
  $('m_rp').textContent = `$${risk.toFixed(2)} / $${pay.toFixed(2)}`;
}
['m_price','m_count'].forEach(id => document.addEventListener('input', e => { if(e.target.id===id) recalcRP(); }));
$('m_cancel').addEventListener('click', () => $('ovl').classList.remove('on'));
$('ovl').addEventListener('click', e => { if(e.target.id==='ovl') $('ovl').classList.remove('on'); });

$('m_place').addEventListener('click', async () => {
  if(!TICKET) return;
  const price_c = parseInt($('m_price').value||'0'), count = parseInt($('m_count').value||'0');
  if(!(price_c>=1 && price_c<=99) || !(count>=1)){ setStatus('price 1-99¢ and ≥1 contract','err'); return; }
  $('m_place').disabled=true; setStatus('placing…','');
  try{
    const r = await postForm('/kalshi/place', {ticker:TICKET.ticker, side:TICKET.side,
                  price_c, count, expiration_ts:TICKET.expiration_ts, post_only:TICKET.post_only});
    const j = await r.json();
    if(r.ok && j.order_id){ setStatus('✓ placed · order '+j.order_id,'ok'); }
    else { setStatus('✗ '+(j.error||('HTTP '+r.status)),'err'); $('m_place').disabled=false; }
  }catch(err){ setStatus('✗ '+err,'err'); $('m_place').disabled=false; }
});
function setStatus(msg,cls){ const s=$('m_status'); s.textContent=msg; s.className='status '+(cls||''); }
</script>
"""


def _sections(games, interactive=False):
    """The per-game <section> blocks (the swappable board body)."""
    blocks = []
    for gm in games:
        kind = gm.get("lineup_kind")
        panels = "".join(_panel(s, confirmed=(kind == "confirmed"),
                                interactive=interactive, match=gm["match"]) for s in gm["sides"])
        manual = gm.get("manual_xi")
        if kind == "projected":
            badge = "<span class='badge proj'>⚠ no confirmed XI — projected minutes</span>"
        elif manual:
            badge = "<span class='badge man'>✋ manual XI</span>"
            if interactive:
                badge += (f"<button class='xibtn' data-match=\"{html.escape(str(gm['match']))}\""
                          f" title='drop the manual XI and hand back to the API/model'>clear</button>")
        elif kind == "confirmed":
            badge = "<span class='badge conf'>✓ XI confirmed</span>"
        else:
            badge = ""
        blocks.append(
            f"<section><h2>{html.escape(str(gm['match']))}"
            f"<span class='ko'>{_when(gm['kickoff'])}</span>{badge}</h2>"
            f"<div class='teams'>{panels}</div></section>")
    return "".join(blocks) or "<p>No goal/assist markets in the next games.</p>"


def build_html(games, interactive=False) -> str:
    """The full board page as a string. `interactive=True` adds editable minutes cells,
    clickable order-ticket cells, and the supporting modal + JS (served live by webapp);
    the static CLI file uses interactive=False."""
    body = _sections(games, interactive)
    extra_css = _INTERACTIVE_CSS if interactive else ""
    if interactive:
        toolbar = ("<div class='toolbar'>"
                   "<button class='tbtn' id='k_refresh'>↻ Refresh Kalshi book</button>"
                   "<span class='tnote'>Edit a Min cell to override; click a lit Y/N price to open a Kalshi order ticket.</span>"
                   "</div>")
        modal, script = _MODAL_HTML, _INTERACTIVE_JS
    else:
        toolbar = modal = script = ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Goals &amp; assists board</title>
<style>{_BOARD_CSS}{extra_css}</style></head><body>
<h1>Goals &amp; assists board</h1>{toolbar}
<div class="ko" style="margin-bottom:6px">Soonest {N_GAMES} kickoffs · G λ / A λ are hazards (Σ = the team total); % = 1-e^(-λ). xG = bet365 de-vigged Team Total Goals, xA = 0.75·xG.<br>
<b>White rows = bet365</b> (trust this) · <span style="color:#e0a458"><b>amber rows = model fallback</b></span> (no bet365 price — softer). Min = model expected minutes. Book legs start from bet365's raw anytime hazard (which assumes each listed player features, so the book over-sums), shrink by expected minutes, then scale to the team total — a nailed starter keeps ~his raw price; bench/rotation risks deflate. No player is pushed above his raw bet365 price.<br>
<span style="color:#7ee0a0"><b>G/A/G+A Y/N</b></span> = acceptable Kalshi <b>TAKE</b> price in ¢ (YES/NO): fair − {_pct(config.KALSHI_EDGE_CUSHION)} cushion − taker fee — cross the book if it offers a side ≤ this number. A <b>make</b> rests lower still (− an extra {_pct(config.KALSHI_LIQUIDITY_PREMIUM)} liquidity premium, cheaper maker fee), pennying the best bid; the order ticket prices it for you. Shown only once the XI is confirmed (“·” = awaiting XI); markets under {_pct(config.KALSHI_MIN_QUOTE_PROB)} fair are noise and left blank.<br>
<span style="color:#e0a458"><b>⚠ doubtful</b></span> / <span style="color:#e07a7a"><b>⚠ OUT</b></span> = LLM injury/news check (hover for reason, click for source); a red outline = flagged while we project a near-full match. OUT <b>suppresses</b> that player's maker quote — never moves a price.<br>
The <b>side</b> (YES or NO) that's live on Kalshi is lit in each Y/N cell: <span style="background:#1c5236;color:#aef7c8;padding:1px 5px;border-radius:3px"><b>green</b></span> = <b>TAKE</b> (Kalshi offers it ≤ our price — cross it) · <span style="background:#4a3a16;color:#f0c478;padding:1px 5px;border-radius:3px"><b>amber</b></span> = <b>make</b> (resting our price improves the book; both sides lit = a two-sided make). Green bar on the name = a takeable market. Hover a lit cell for the action + live bid/ask + OI. Shown only with a Kalshi key set and the XI confirmed.<br>
<b>Edge size</b> — attack the brightest first: each takeable cell shows its <b>+¢ edge</b> and the name carries a pill of the row's <b>biggest</b> take edge, color-graded <span style="background:#244b38;color:#9ad9b4;padding:1px 5px;border-radius:4px"><b>+thin</b></span> (1–2¢) · <span style="background:#1c7a47;color:#d7ffe7;padding:1px 5px;border-radius:4px"><b>+good</b></span> (3–5¢) · <span style="background:#2bd277;color:#04210f;padding:1px 5px;border-radius:4px"><b>+BIG</b></span> (≥6¢). The name bar thickens with it too.</div>
<div id="board">{body}</div>{modal}{script}
</body></html>"""


def write_html(games, path=None):
    path = path or (config.OUT_DIR / "scorers.html")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(games, interactive=False), encoding="utf-8")
    return path


def write_csv(games, path=None):
    path = path or (config.OUT_DIR / "scorers.csv")
    def rnd(x, n):
        return None if x is None or pd.isna(x) else round(float(x), n)
    rows = []
    for gm in games:
        # emit acceptable maker prices once we have ANY lineup (projected or confirmed) so the
        # Kalshi overlay can show take/make verdicts pre-XI (VIEW-ONLY — order placement stays
        # gated to a confirmed XI in board.ticket/place + kalshi_orders).
        mk = gm.get("lineup_kind") in ("confirmed", "projected")
        for s in gm["sides"]:
            for r in s["book"] + s["model"]:
                # news says out → suppress the quote here too, so the Kalshi overlay can't
                # rest an order on a player we believe isn't playing. Model rows quote off
                # their own fair value (no bet365 anchor) so Kalshi-listed bench players covered.
                show = mk and not r.get("news_suppress")
                gy, gn = _maker(r["g_p"], show)
                ay, an = _maker(r["a_p"], show)
                gay, gan = _maker(r["ga_p"], show)
                nf = r.get("news") or {}
                rows.append({
                    "match": gm["match"], "team": s["team"], "source": r["source"],
                    "player": r["player"], "minutes": rnd(r.get("minutes"), 1),
                    "tt_goals": rnd(s["tt_goals"], 3), "tt_assists": rnd(s["tt_assists"], 3),
                    "goal_hazard": rnd(r["g_haz"], 4), "goal_pct": rnd(r["g_p"], 4),
                    "assist_hazard": rnd(r["a_haz"], 4), "assist_pct": rnd(r["a_p"], 4),
                    "ga_pct": rnd(r["ga_p"], 4),
                    "goal_yes_c": gy, "goal_no_c": gn, "assist_yes_c": ay, "assist_no_c": an,
                    "ga_yes_c": gay, "ga_no_c": gan,
                    "news_status": nf.get("status"), "news_reason": nf.get("reason"),
                    "news_source": nf.get("source_url")})
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["match", "team", "source", "player", "minutes", "tt_goals",
                                "tt_assists", "goal_hazard", "goal_pct",
                                "assist_hazard", "assist_pct", "ga_pct",
                                "goal_yes_c", "goal_no_c", "assist_yes_c", "assist_no_c",
                                "ga_yes_c", "ga_no_c",
                                "news_status", "news_reason", "news_source"]).to_csv(path, index=False)
    return path


def write(b365: pd.DataFrame, proj=None, annotate=None, kalshi=False, overrides=None,
          force_confirmed=None, ud=None):
    """Build the board, optionally run an annotator over the games (e.g. news.annotate),
    then render HTML + CSV. When `kalshi` is set and an XI is confirmed, also pull the live
    Kalshi overlay once, highlight takes/makes on the board, and return its rows so the
    caller can render kalshi.html from the SAME fetch (no double call).

    Returns (html_path, csv_path, note, kalshi_rows-or-None, games). `games` carries the
    attached live verdicts + kickoff and is what kalshi_orders sizes from."""
    games = build(b365, proj, overrides=overrides, force_confirmed=force_confirmed, ud=ud)
    note = annotate(games) if annotate else None
    csv_path = write_csv(games)               # write csv first so the overlay can read it
    krows = None
    # attach the live overlay once we have ANY lineup (projected or confirmed) so edges show
    # pre-XI — VIEW-ONLY; trading stays gated to a confirmed XI downstream.
    if kalshi and any(gm.get("lineup_kind") in ("confirmed", "projected") for gm in games):
        try:
            import kalshi_overlay
            krows = kalshi_overlay.build(csv_path)
            _attach_overlay(games, krows)
        except Exception as e:
            print(f"  scorers: Kalshi highlight skipped: {e}")
    return write_html(games), csv_path, note, krows, games
