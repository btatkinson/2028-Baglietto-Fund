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


def _haz(p):
    """Hazard (Poisson lambda) for an anytime probability: lambda = -ln(1-P)."""
    if p is None or pd.isna(p):
        return None
    p = min(max(float(p), 0.0), 0.999)
    return -math.log(1 - p)


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


def build(b365: pd.DataFrame, proj=None) -> list[dict]:
    """Per-game structure for the soonest N_GAMES, each with two team panels of
    goal/assist hazards that sum to the de-vigged team total."""
    if b365 is None or not len(b365) or "stat" not in getattr(b365, "columns", []):
        return []
    df = b365.copy()
    df["nname"] = df["player"].map(norm_name)
    probs = devig.goalscorer_probs(b365)        # {(match, nname, stat): de-vigged P}
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

            book_players, book_nn, book_by_nn = [], set(gs["nname"]), {}
            for nn in book_nn:
                gp = probs.get((match, nn, "Goals"))
                ap = probs.get((match, nn, "Assists"))
                gap = probs.get((match, nn, "Goals + assists"))
                if gp is None and ap is None:
                    continue
                row = {
                    "player": name_of.get(nn, nn.title()), "source": "bet365",
                    "minutes": (model.get(nn) or {}).get("minutes"),     # exact-name minutes
                    "g_haz": _haz(gp), "g_p": gp,
                    "a_haz": _haz(ap), "a_p": ap, "ga_p": gap}
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
                        if row.get("minutes") is None and names_compatible(nn, bnn)]
                if len(cand) == 1:
                    book_by_nn[cand[0]]["minutes"] = rec.get("minutes")
                    continue
                gap = None if (gp is None or ap is None) else 1 - (1 - gp) * (1 - ap)
                model_players.append({
                    "player": rec.get("player", nn.title()), "source": "model",
                    "minutes": rec.get("minutes"),
                    "g_haz": gmean, "g_p": gp,
                    "a_haz": amean, "a_p": ap, "ga_p": gap})

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


def _when(ko):
    if not ko or math.isinf(ko):
        return ""
    dt = datetime.fromtimestamp(int(ko), tz=timezone.utc).strftime("%a %d %b %H:%M UTC")
    dh = (ko - time.time()) / 3600
    rel = "started" if dh < 0 else (f"in {dh:.0f}h" if dh >= 1 else f"in {dh * 60:.0f}m")
    return f"{dt} · {rel}"


def _rows_html(players, with_total, sum_g, sum_a, tt_goals, tt_assists):
    trs = []
    for r in players:
        cls = " class='mdl'" if r["source"] == "model" else ""
        tag = " <span class='tag'>model</span>" if r["source"] == "model" else ""
        trs.append(
            f"<tr{cls}><td class='p'>{html.escape(str(r['player']))}{tag}</td>"
            f"<td class='num dim'>{_min(r.get('minutes'))}</td>"
            f"<td class='num'>{_num(r['g_haz'])}</td><td class='num'>{_pct(r['g_p'])}</td>"
            f"<td class='num'>{_num(r['a_haz'])}</td><td class='num'>{_pct(r['a_p'])}</td>"
            f"<td class='num gap'>{_pct(r['ga_p'])}</td></tr>")
    if with_total:
        chk = "✓" if (tt_goals is not None and abs(sum_g - tt_goals) < 0.02) else ""
        trs.append(
            f"<tr class='tot'><td>Σ hazard {chk}</td><td></td>"
            f"<td class='num'>{_num(sum_g)}</td><td></td>"
            f"<td class='num'>{_num(sum_a)}</td><td></td><td></td></tr>")
    return "".join(trs)


def _panel(side):
    head = ("<tr><th>Player</th><th class='num'>Min</th><th class='num'>G λ</th><th class='num'>G%</th>"
            "<th class='num'>A λ</th><th class='num'>A%</th><th class='num gap'>G+A%</th></tr>")
    body = _rows_html(side["book"], True, side["sum_g"], side["sum_a"],
                      side["tt_goals"], side["tt_assists"])
    if side["model"]:
        body += ("<tr class='sep'><td colspan='7'>not priced by bet365 — model est.</td></tr>"
                 + _rows_html(side["model"], False, 0, 0, None, None))
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
        panels = "".join(_panel(s) for s in gm["sides"])
        kind = gm.get("lineup_kind")
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
</style></head><body>
<h1>Goals &amp; assists board</h1>
<div class="ko" style="margin-bottom:6px">Soonest {N_GAMES} kickoffs · G λ / A λ are hazards (Σ = the team total); % = 1-e^(-λ). xG = bet365 de-vigged Team Total Goals, xA = 0.75·xG.<br>
<b>White rows = bet365 de-vig</b> (trust this) · <span style="color:#e0a458"><b>amber rows = model fallback</b></span> (no bet365 price — softer). Min = model expected minutes.</div>
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
        for s in gm["sides"]:
            for r in s["book"] + s["model"]:
                rows.append({
                    "match": gm["match"], "team": s["team"], "source": r["source"],
                    "player": r["player"], "minutes": rnd(r.get("minutes"), 1),
                    "tt_goals": rnd(s["tt_goals"], 3), "tt_assists": rnd(s["tt_assists"], 3),
                    "goal_hazard": rnd(r["g_haz"], 4), "goal_pct": rnd(r["g_p"], 4),
                    "assist_hazard": rnd(r["a_haz"], 4), "assist_pct": rnd(r["a_p"], 4),
                    "ga_pct": rnd(r["ga_p"], 4)})
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["match", "team", "source", "player", "minutes", "tt_goals",
                                "tt_assists", "goal_hazard", "goal_pct",
                                "assist_hazard", "assist_pct", "ga_pct"]).to_csv(path, index=False)
    return path


def write(b365: pd.DataFrame, proj=None):
    games = build(b365, proj)
    return write_html(games), write_csv(games)
