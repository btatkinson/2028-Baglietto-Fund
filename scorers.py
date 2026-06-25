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
    """Two name tokens are the same person-token: equal, off-by-one (Hasan/Hassan), or
    one an initial of the other."""
    return (x == y or _ed1(x, y)
            or (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)))


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


def _names_compatible_loose(a, b):
    """names_compatible, plus tolerance for the book↔API-Football↔Kalshi spelling seams
    the strict matcher misses: one-letter variants ('Meschack'/'Meschak', 'Hasan'/
    'Hassan'), extra name parts ('Jassem Gaber' vs 'Jassem Gaber Abdulsallam'), and
    Arabic article joins ('Mohamed Al Manai' vs 'Mohamed Naceur Almanai'). Token-cover
    based and >=2-token gated; the unique-candidate guard at every call site is the
    backstop against a wrong collision."""
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


def _price_by_minutes(players, tt_goals, tt_assists, confirmed=False):
    """Price each book leg from its RAW vigged hazard via: raw × minutes-factor × k, where
    k = team_total / Σ(raw × factor), and every player is capped at his own raw hazard.

    The raw book over-sums vs the (sharp, two-sided) team total — every listed player is
    priced as if he features, so summed it implies ~5-6 goals, not ~1.4. Shrinking each by
    expected minutes removes that 'everyone plays' artifact and lands the sum ≈ the team
    total; the gentle k then scales onto the total exactly, stripping the residual one-way
    juice. In the normal case k ≤ 1 (a down-scale), so with factor ≤ 1 too NO player can
    exceed his raw 365 price — the de-vig invariant holds for free. A nailed starter
    (factor ≈ 1) therefore keeps ~his raw price; rotation/bench risks deflate by minutes.

    The cap at raw matters only in the rare k > 1 case (a high-hazard starter benched, so
    Σshrunk dips below the total): it prevents inflating anyone past their 365 price, and
    Σ simply falls short of the total — correct and conservative.

    Minutes factor: a matched player gets min(minutes/full-match, 1). An UNMATCHED player
    (no projection) is factor 1 pre-XI (can't tell a star we lack data for from a bench
    body — don't penalize) but factor 0 post-XI (the confirmed sheet carries the whole
    squad, so a listed player it omits isn't playing). Mutates rows in place."""
    def fac(r):
        m = r.get("minutes")
        if m is None or pd.isna(m):
            return 0.0 if confirmed else 1.0
        return min(max(float(m) / _FULL_MATCH, 0.0), 1.0)
    for rawk, hzk, pk, target in (("g_rawh", "g_haz", "g_p", tt_goals),
                                  ("a_rawh", "a_haz", "a_p", tt_assists)):
        rows = [r for r in players if r.get(rawk)]
        shrunk = {id(r): r[rawk] * fac(r) for r in rows}
        s = sum(shrunk.values())
        k = (target / s) if (target and s > 0) else 1.0
        for r in rows:
            h = min(shrunk[id(r)] * k, r[rawk])             # never above the raw 365 price
            r[hzk] = h
            r[pk] = 1 - math.exp(-h)
    for r in players:                  # G+A union recomputed from the priced legs
        if r.get("g_p") is not None and r.get("a_p") is not None:
            r["ga_p"] = 1 - (1 - r["g_p"]) * (1 - r["a_p"])


def build(b365: pd.DataFrame, proj=None) -> list[dict]:
    """Per-game structure for the soonest N_GAMES, each with two team panels of
    goal/assist hazards. Each book leg is priced from bet365's raw anytime hazard,
    shrunk by expected minutes and scaled to the market team total (see
    _price_by_minutes), so the hazards sum to ~the team total with nailed starters near
    their raw price and bench/rotation risks deflated."""
    if b365 is None or not len(b365) or "stat" not in getattr(b365, "columns", []):
        return []
    df = b365.copy()
    df["nname"] = df["player"].map(norm_name)
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
                grh = raw.get(nn, {}).get("Goals")
                arh = raw.get(nn, {}).get("Assists")
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

            # minutes are now fully attached (incl. the name-seam fills above), so price
            # the raw book by expected minutes + team total before summing/sorting. Post-XI,
            # book players still unmatched are treated as not in the squad (factor 0).
            _price_by_minutes(book_players, tt_goals, tt_assists,
                              confirmed=(kinds.get(match) == "confirmed"))
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
                          "lineup_kind": kinds.get(match)})
    return games


# ----------------------------------- render -----------------------------------

def _pct(x):
    return "" if x is None or pd.isna(x) else f"{float(x) * 100:.0f}%"


def _num(x):
    return "" if x is None or pd.isna(x) else f"{float(x):.2f}"


def _min(x):
    return "" if x is None or pd.isna(x) else f"{float(x):.0f}"


def _kalshi_fee_c(p):
    """Kalshi trading fee in CENTS for one contract at prob p (rate*p*(1-p) of $1,
    rounded up to the cent, the way Kalshi bills it). Symmetric in p, so YES and NO
    carry the same fee."""
    return math.ceil(100 * config.KALSHI_FEE_RATE * p * (1 - p))


def _accept(p):
    """(acceptable YES ¢, acceptable NO ¢) for a fair YES prob p: the most we'd PAY for
    each side and still clear a fixed half-spread edge after the Kalshi fee. Take a side
    iff the market offers it at/below its number. A side is None when it floors below 1¢
    (too thin to ever pay)."""
    if p is None or pd.isna(p):
        return None, None
    buf = 100 * config.KALSHI_MAKER_HALF_SPREAD + _kalshi_fee_c(p)
    def clip(c):
        return 99 if c > 99 else (None if c < 1 else int(c))
    return clip(round(100 * p - buf)), clip(round(100 * (1 - p) - buf))


def _maker(p, show):
    """(acceptable YES ¢, acceptable NO ¢) honoring the post-XI gate AND the noise floor:
    nothing is quoted pre-XI, and a market whose YES prob is below KALSHI_MIN_QUOTE_PROB
    (a 2'-cameo player) is dropped entirely so we never rest an order on it."""
    if not show or p is None or pd.isna(p) or p < config.KALSHI_MIN_QUOTE_PROB:
        return None, None
    return _accept(p)


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


def _mk_cell(p, show, ov=None):
    """Compact 'YES/NO' acceptable-maker-price cell in ¢. '·' until the XI is confirmed
    (pre-XI minutes are too soft to rest a real Kalshi order on); blank for a noise
    market below the quote floor. When a live Kalshi verdict (`ov`) is supplied, the
    SPECIFIC side that's actionable is highlighted — green = TAKE (cross the live book),
    amber = make (improve it) — so a two-sided make shows both numbers lit. Hover the cell
    for the action + live bid/ask + OI."""
    if not show:
        return "<td class='num mk'>·</td>"
    ay, an = _maker(p, True)
    if ay is None and an is None:
        return "<td class='num mk'></td>"
    ov = ov or {}
    def side(c, take, make):
        s = "—" if c is None else str(c)
        if take:
            return f"<span class='ks take'>{s}</span>"
        if make:
            return f"<span class='ks make'>{s}</span>"
        return s
    yes = side(ay, ov.get("take_yes"), ov.get("make_yes"))
    no = side(an, ov.get("take_no"), ov.get("make_no"))
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
            for row in s.get("book", []):
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


def _rows_html(players, with_total, sum_g, sum_a, tt_goals, tt_assists, confirmed=False):
    trs = []
    for r in players:
        # maker prices: post-XI book rows only, and never on a player the news says is out
        mk = confirmed and r["source"] == "bet365" and not r.get("news_suppress")
        kx = (r.get("kalshi") or {}) if mk else {}
        # a row earns the green marker when any of its markets is takeable on live Kalshi
        take_row = any(v.get("take_yes") or v.get("take_no") for v in kx.values())
        rowcls = ["mdl"] if r["source"] == "model" else []
        if take_row:
            rowcls.append("krow")
        cls = f" class='{' '.join(rowcls)}'" if rowcls else ""
        tag = " <span class='tag'>model</span>" if r["source"] == "model" else ""
        trs.append(
            f"<tr{cls}><td class='p'>{html.escape(str(r['player']))}{tag}{_news_chip(r)}</td>"
            f"<td class='num dim'>{_min(r.get('minutes'))}</td>"
            f"<td class='num'>{_num(r['g_haz'])}</td><td class='num'>{_pct(r['g_p'])}</td>"
            f"<td class='num'>{_num(r['a_haz'])}</td><td class='num'>{_pct(r['a_p'])}</td>"
            f"<td class='num gap'>{_pct(r['ga_p'])}</td>"
            f"{_mk_cell(r['g_p'], mk, kx.get('G'))}{_mk_cell(r['a_p'], mk, kx.get('A'))}"
            f"{_mk_cell(r['ga_p'], mk, kx.get('G+A'))}</tr>")
    if with_total:
        chk = "✓" if (tt_goals is not None and abs(sum_g - tt_goals) < 0.02) else ""
        trs.append(
            f"<tr class='tot'><td>Σ hazard {chk}</td><td></td>"
            f"<td class='num'>{_num(sum_g)}</td><td></td>"
            f"<td class='num'>{_num(sum_a)}</td><td></td><td></td>"
            f"<td></td><td></td><td></td></tr>")
    return "".join(trs)


def _panel(side, confirmed=False):
    head = ("<tr><th>Player</th><th class='num'>Min</th><th class='num'>G λ</th><th class='num'>G%</th>"
            "<th class='num'>A λ</th><th class='num'>A%</th><th class='num gap'>G+A%</th>"
            "<th class='num mk'>G Y/N</th><th class='num mk'>A Y/N</th><th class='num mk'>G+A Y/N</th></tr>")
    body = _rows_html(side["book"], True, side["sum_g"], side["sum_a"],
                      side["tt_goals"], side["tt_assists"], confirmed)
    if side["model"]:
        body += ("<tr class='sep'><td colspan='10'>not priced by bet365 — model est.</td></tr>"
                 + _rows_html(side["model"], False, 0, 0, None, None, confirmed))
    smin = (f"<span class='smin'>Σmin {side['sum_min']:.0f}</span>"
            if side.get("sum_min") else "")
    return (f"<div class='team'>"
            f"<div class='th'><span class='tn'>{html.escape(str(side['team']))}</span>"
            f"<span class='big'>xG {_num(side['tt_goals'])}</span>"
            f"<span class='big xa'>xA {_num(side['tt_assists'])}</span>{smin}</div>"
            f"<table><thead>{head}</thead><tbody>{body}</tbody></table></div>")


def write_html(games, path=None):
    path = path or (config.OUT_DIR / "scorers.html")
    blocks = []
    for gm in games:
        kind = gm.get("lineup_kind")
        panels = "".join(_panel(s, confirmed=(kind == "confirmed")) for s in gm["sides"])
        if kind == "projected":
            badge = "<span class='badge proj'>⚠ no confirmed XI — projected minutes</span>"
        elif kind == "confirmed":
            badge = "<span class='badge conf'>✓ XI confirmed</span>"
        else:
            badge = ""
        blocks.append(
            f"<section><h2>{html.escape(str(gm['match']))}"
            f"<span class='ko'>{_when(gm['kickoff'])}</span>{badge}</h2>"
            f"<div class='teams'>{panels}</div></section>")
    body = "".join(blocks) or "<p>No goal/assist markets in the next games.</p>"
    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>Goals &amp; assists board</title>
<style>
 body{{margin:0;background:#0f1115;color:#e6e9ef;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;padding:18px 22px}}
 h1{{font-size:18px;margin:0 0 4px}}
 h2{{font-size:15px;margin:26px 0 8px;color:#cfe0ff;display:flex;gap:12px;align-items:baseline}}
 .ko{{font-size:12px;color:#8b93a7;font-weight:400}}
 .badge{{font-size:11px;font-weight:700;padding:1px 8px;border-radius:9px}}
 .badge.proj{{color:#231700;background:#e0a458}}
 .badge.conf{{color:#04210f;background:#7ee0a0}}
 .smin{{font-size:12px;color:#8b93a7;font-variant-numeric:tabular-nums}}
 .teams{{display:flex;gap:26px;flex-wrap:wrap}}
 .team{{min-width:340px}}
 .th{{display:flex;align-items:baseline;gap:14px;margin-bottom:4px}}
 .tn{{font-size:16px;font-weight:700}}
 .big{{font-size:22px;font-weight:700;color:#7ee0a0;font-variant-numeric:tabular-nums}}
 .big.xa{{color:#9fc6ff}}
 table{{border-collapse:collapse}}
 th,td{{padding:4px 10px;border-bottom:1px solid #20242e;text-align:left;white-space:nowrap}}
 th{{font-size:11px;color:#8b93a7;font-weight:600}}
 td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}}
 td.dim,.dim{{color:#8b93a7}}
 .gap{{color:#cdb6ff}} .p{{font-weight:600}}
 .tot td{{border-top:2px solid #3a4150;border-bottom:none;font-weight:700;color:#7ee0a0}}
 .tot td.num:nth-child(5){{color:#9fc6ff}}
 .sep td{{color:#e0a458;font-size:11px;font-style:italic;border-bottom:none;padding-top:8px}}
 tr.mdl td{{color:#e0a458}} tr.mdl td.p{{color:#e8b878}} tr.mdl td.dim{{color:#b9883f}}
 .tag{{font-size:9px;color:#e0a458;border:1px solid #6e5526;border-radius:3px;padding:0 3px;margin-left:5px}}
 td.mk,th.mk{{color:#7ee0a0;border-left:1px solid #20242e}}
 th.mk{{color:#5f8f72}} td.mk .sl{{color:#3a4150;margin:0 1px}}
 td.kcell{{cursor:help}}
 .ks{{border-radius:3px;padding:0 3px}}
 .ks.take{{background:#1c5236;color:#aef7c8;font-weight:700}}
 .ks.make{{background:#4a3a16;color:#f0c478}}
 tr.krow td.p{{box-shadow:inset 3px 0 0 #7ee0a0}}
 .nflag{{font-size:9px;font-weight:700;border-radius:3px;padding:0 4px;margin-left:6px;text-decoration:none;white-space:nowrap}}
 a.nflag:hover{{text-decoration:underline}}
 .nflag.ndbt{{color:#231700;background:#e0a458}}
 .nflag.nout{{color:#2a0c0c;background:#e07a7a}}
 .nflag.nconf{{box-shadow:0 0 0 2px #ff5d5d}}
</style></head><body>
<h1>Goals &amp; assists board</h1>
<div class="ko" style="margin-bottom:6px">Soonest {N_GAMES} kickoffs · G λ / A λ are hazards (Σ = the team total); % = 1-e^(-λ). xG = bet365 de-vigged Team Total Goals, xA = 0.75·xG.<br>
<b>White rows = bet365</b> (trust this) · <span style="color:#e0a458"><b>amber rows = model fallback</b></span> (no bet365 price — softer). Min = model expected minutes. Book legs start from bet365's raw anytime hazard (which assumes each listed player features, so the book over-sums), shrink by expected minutes, then scale to the team total — a nailed starter keeps ~his raw price; bench/rotation risks deflate. No player is pushed above his raw bet365 price.<br>
<span style="color:#7ee0a0"><b>G/A/G+A Y/N</b></span> = acceptable Kalshi <b>maker</b> price in ¢ (YES/NO): the most to pay each side after a {_pct(config.KALSHI_MAKER_HALF_SPREAD)} half-spread + fee — buy a side if the book offers it ≤ its number. Shown only once the XI is confirmed (“·” = awaiting XI); markets under {_pct(config.KALSHI_MIN_QUOTE_PROB)} fair are noise and left blank.<br>
<span style="color:#e0a458"><b>⚠ doubtful</b></span> / <span style="color:#e07a7a"><b>⚠ OUT</b></span> = LLM injury/news check (hover for reason, click for source); a red outline = flagged while we project a near-full match. OUT <b>suppresses</b> that player's maker quote — never moves a price.<br>
The <b>side</b> (YES or NO) that's live on Kalshi is lit in each Y/N cell: <span style="background:#1c5236;color:#aef7c8;padding:1px 5px;border-radius:3px"><b>green</b></span> = <b>TAKE</b> (Kalshi offers it ≤ our price — cross it) · <span style="background:#4a3a16;color:#f0c478;padding:1px 5px;border-radius:3px"><b>amber</b></span> = <b>make</b> (resting our price improves the book; both sides lit = a two-sided make). Green bar on the name = a takeable market. Hover a lit cell for the action + live bid/ask + OI. Shown only with a Kalshi key set and the XI confirmed.</div>
{body}
</body></html>"""
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")
    return path


def write_csv(games, path=None):
    path = path or (config.OUT_DIR / "scorers.csv")
    def rnd(x, n):
        return None if x is None or pd.isna(x) else round(float(x), n)
    rows = []
    for gm in games:
        mk = gm.get("lineup_kind") == "confirmed"          # maker prices only post-XI
        for s in gm["sides"]:
            for r in s["book"] + s["model"]:
                # news says out → suppress the quote here too, so the Kalshi overlay can't
                # rest an order on a player we believe isn't playing
                show = mk and r["source"] == "bet365" and not r.get("news_suppress")
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


def write(b365: pd.DataFrame, proj=None, annotate=None, kalshi=False):
    """Build the board, optionally run an annotator over the games (e.g. news.annotate),
    then render HTML + CSV. When `kalshi` is set and an XI is confirmed, also pull the live
    Kalshi overlay once, highlight takes/makes on the board, and return its rows so the
    caller can render kalshi.html from the SAME fetch (no double call).

    Returns (html_path, csv_path, note, kalshi_rows-or-None, games). `games` carries the
    attached live verdicts + kickoff and is what kalshi_orders sizes from."""
    games = build(b365, proj)
    note = annotate(games) if annotate else None
    csv_path = write_csv(games)               # write csv first so the overlay can read it
    krows = None
    if kalshi and any(gm.get("lineup_kind") == "confirmed" for gm in games):
        try:
            import kalshi_overlay
            krows = kalshi_overlay.build(csv_path)
            _attach_overlay(games, krows)
        except Exception as e:
            print(f"  scorers: Kalshi highlight skipped: {e}")
    return write_html(games), csv_path, note, krows, games
