"""Overlay live Kalshi per-match prices against our acceptable maker prices.

READ-ONLY decision support: pulls Kalshi's per-match Score-or-Assist (G+A) and Assist
markets, joins each listed player to our board (out/scorers.csv), and flags where Kalshi
is offering a side at/below our acceptable price — i.e. a take. It never places orders;
you submit trades yourself.

Kalshi per-match series (each event = one match, event_ticker e.g. KXWCGOAL-26JUN24BIHQAT):
  KXWCGOAL  Goalscorer       -> our Goals  (goal_yes_c / goal_no_c)
  KXWCAST   Assists          -> our Assists (assist_yes_c / assist_no_c)
  KXWCSOA   Score or Assist  -> our G+A    (ga_yes_c / ga_no_c)
KXWCGOAL & KXWCAST carry several lines per player (1+, 2+, 3+); we keep only the ANYTIME
line — floor_strike == 0.5 ("at least 1") — to match our anytime hazards. KXWCSOA is a
single binary (floor_strike None). (KXWCPLAYERGOALS is tournament-long, not per-match, so
it's intentionally ignored.) Kalshi adds players post-XI and through the day, so this is a
snapshot — re-run it to pick up newly-listed players.

Live best bid/ask come from the market's *_dollars fields (the int-cent fields are null).
"""
from __future__ import annotations

import pandas as pd

import config
import kalshi
from normalize import norm_name, norm_country
from scorers import _names_compatible_loose

# (Kalshi series, our acceptable-YES col, our acceptable-NO col, label)
MARKETS = [
    ("KXWCGOAL", "goal_yes_c", "goal_no_c", "G"),
    ("KXWCAST", "assist_yes_c", "assist_no_c", "A"),
    ("KXWCSOA", "ga_yes_c", "ga_no_c", "G+A"),
]


def _cents(v):
    try:
        return round(float(v) * 100)
    except (TypeError, ValueError):
        return None


def _pair_key(a, b):
    cs = sorted([norm_country(a), norm_country(b)])
    return "|".join(cs) if len(cs) == 2 and all(cs) else None


def fetch_series(series: str) -> dict:
    """{country_pair_key: {nname: {player, yes_bid, yes_ask, no_bid, no_ask, last, oi,
    ticker}}} for one per-match Kalshi series (cents)."""
    out: dict[str, dict] = {}
    events = kalshi.events(series_ticker=series, status="open", limit=200).get("events", [])
    for e in events:
        teams = str(e.get("title", "")).split(":")[0]
        if " vs " not in teams:
            continue
        home, away = teams.split(" vs ", 1)
        key = _pair_key(home, away)
        if not key:
            continue
        players = {}
        for m in kalshi.markets(event_ticker=e["event_ticker"], limit=200).get("markets", []):
            fs = m.get("floor_strike")
            if fs is not None and abs(float(fs) - 0.5) > 1e-6:
                continue                          # keep only anytime: binary (None) or the 1+ line
            sub = m.get("yes_sub_title") or m.get("no_sub_title") or ""
            name = sub.split(":")[0].strip()      # strip the ': 1+' threshold suffix
            if not name:
                continue
            players[norm_name(name)] = {
                "player": name, "ticker": m.get("ticker"),
                "yes_bid": _cents(m.get("yes_bid_dollars")), "yes_ask": _cents(m.get("yes_ask_dollars")),
                "no_bid": _cents(m.get("no_bid_dollars")), "no_ask": _cents(m.get("no_ask_dollars")),
                "last": _cents(m.get("last_price_dollars")),
                "oi": float(m.get("open_interest_fp") or 0)}
        out[key] = players
    return out


def _find(players: dict, nn: str):
    """Match our normalized name to a Kalshi player dict — exact, then loose (the same
    one-letter/extra-token fuzz the board uses for the bet365↔API-Football seam)."""
    if nn in players:
        return players[nn]
    cand = [v for k, v in players.items() if _names_compatible_loose(nn, k)]
    return cand[0] if len(cand) == 1 else None


def build(scorers_csv=None) -> list[dict]:
    """Join our acceptable prices to live Kalshi and flag takes. One row per
    (player, market) where both sides exist."""
    path = scorers_csv or (config.OUT_DIR / "scorers.csv")
    df = pd.read_csv(path)
    have = [m for m in MARKETS if m[1] in df.columns and m[2] in df.columns]
    if not have:
        raise RuntimeError(f"{path} has no maker-price columns — run a post-XI capture first.")
    books = {series: fetch_series(series) for series, *_ in have}

    rows = []
    for r in df.itertuples():
        parts = [p for p in str(r.match).split(" v ") if p.strip()]
        key = _pair_key(parts[0], parts[1]) if len(parts) == 2 else None
        if not key:
            continue
        nn = norm_name(r.player)
        for series, ycol, ncol, label in have:
            acc_yes, acc_no = getattr(r, ycol, None), getattr(r, ncol, None)
            if pd.isna(acc_yes) and pd.isna(acc_no):
                continue                          # we don't quote this market for this player
            kp = _find(books.get(series, {}).get(key, {}), nn)
            if not kp:
                continue                          # Kalshi doesn't list this player here
            acc_yes = None if pd.isna(acc_yes) else int(acc_yes)
            acc_no = None if pd.isna(acc_no) else int(acc_no)
            yb, ya, nb, na = kp["yes_bid"], kp["yes_ask"], kp["no_bid"], kp["no_ask"]
            # TAKE: Kalshi already offers the side at/below our acceptable price -> cross it.
            take_yes = acc_yes is not None and ya is not None and ya <= acc_yes
            take_no = acc_no is not None and na is not None and na <= acc_no
            # MAKE: not a take, but resting a bid at our acceptable price would IMPROVE the
            # book (above the current best bid) while staying below the ask — post & wait.
            make_yes = (not take_yes and acc_yes is not None
                        and (yb is None or acc_yes > yb) and (ya is None or acc_yes < ya))
            make_no = (not take_no and acc_no is not None
                       and (nb is None or acc_no > nb) and (na is None or acc_no < na))
            rows.append({
                "match": r.match, "player": r.player, "market": label,
                "acc_yes": acc_yes, "acc_no": acc_no,
                "k_yes_bid": yb, "k_yes_ask": ya, "k_no_bid": nb, "k_no_ask": na,
                "k_last": kp["last"], "k_oi": kp["oi"], "ticker": kp["ticker"],
                "take_yes": take_yes, "take_no": take_no,
                "make_yes": make_yes, "make_no": make_no,
                "edge_yes": (acc_yes - ya) if take_yes else None,
                "edge_no": (acc_no - na) if take_no else None})
    return rows


def _action(r) -> str:
    """Plain-text action verdict for one overlay row (take beats make)."""
    bits = []
    if r["take_yes"]:
        bits.append(f"TAKE YES @{r['k_yes_ask']} +{r['edge_yes']}¢")
    if r["take_no"]:
        bits.append(f"TAKE NO @{r['k_no_ask']} +{r['edge_no']}¢")
    if r["make_yes"] and not r["take_yes"]:
        bits.append(f"make YES bid {r['acc_yes']}")
    if r["make_no"] and not r["take_no"]:
        bits.append(f"make NO bid {r['acc_no']}")
    return "  ".join(bits)


def print_overlay(rows: list[dict]):
    if not rows:
        print("No (player×market) overlaps between our quotes and live Kalshi markets.")
        return
    takes = [r for r in rows if r["take_yes"] or r["take_no"]]
    print(f"{'match':22} {'player':20} {'mkt':4} {'ourY/N':>8} {'K yes b/a':>10} "
          f"{'K no b/a':>9} {'OI':>7}  action")
    rank = lambda r: (not (r["take_yes"] or r["take_no"]), -max(r["edge_yes"] or 0, r["edge_no"] or 0))
    for r in sorted(rows, key=rank):
        oy = "" if r["acc_yes"] is None else r["acc_yes"]
        on = "" if r["acc_no"] is None else r["acc_no"]
        print(f"{r['match'][:22]:22} {r['player'][:20]:20} {r['market']:4} "
              f"{str(oy)+'/'+str(on):>8} {str(r['k_yes_bid'])+'/'+str(r['k_yes_ask']):>10} "
              f"{str(r['k_no_bid'])+'/'+str(r['k_no_ask']):>9} {r['k_oi']:>7.0f}  {_action(r)}")
    makes = [r for r in rows if (r["make_yes"] or r["make_no"]) and not (r["take_yes"] or r["take_no"])]
    print(f"\n{len(takes)} takeable · {len(makes)} make-only · {len(rows)} overlapping markets.")


# ----------------------------------- HTML -----------------------------------

def _c(x):
    return "" if x is None else str(x)


def write_html(rows: list[dict], path=None):
    """Kalshi overlay board (out/kalshi.html), styled like scorers.html. Grouped by match;
    takes highlighted green, make-only amber. READ-ONLY — you place the orders."""
    import html as _h
    from datetime import datetime, timezone
    from pathlib import Path
    path = Path(path) if path else (config.OUT_DIR / "kalshi.html")
    by_match: dict[str, list] = {}
    for r in rows:
        by_match.setdefault(r["match"], []).append(r)
    rank = lambda r: (not (r["take_yes"] or r["take_no"]), -max(r["edge_yes"] or 0, r["edge_no"] or 0))

    sections = []
    n_take = n_make = 0
    for match, rs in by_match.items():
        trs = []
        for r in sorted(rs, key=rank):
            take = r["take_yes"] or r["take_no"]
            make = (r["make_yes"] or r["make_no"]) and not take
            n_take += bool(take)
            n_make += bool(make)
            cls = "take" if take else ("make" if make else "")
            act = _h.escape(_action(r)) or "—"
            our = f"{_c(r['acc_yes'])}<span class='sl'>/</span>{_c(r['acc_no'])}"
            ky = f"{_c(r['k_yes_bid'])}<span class='sl'>/</span>{_c(r['k_yes_ask'])}"
            kn = f"{_c(r['k_no_bid'])}<span class='sl'>/</span>{_c(r['k_no_ask'])}"
            trs.append(
                f"<tr class='{cls}'><td class='p'>{_h.escape(str(r['player']))}</td>"
                f"<td>{r['market']}</td><td class='num'>{our}</td>"
                f"<td class='num'>{ky}</td><td class='num'>{kn}</td>"
                f"<td class='num dim'>{_c(r['k_last'])}</td><td class='num dim'>{r['k_oi']:.0f}</td>"
                f"<td class='act'>{act}</td></tr>")
        head = ("<tr><th>Player</th><th>Mkt</th><th class='num'>our Y/N</th>"
                "<th class='num'>Kalshi yes b/a</th><th class='num'>Kalshi no b/a</th>"
                "<th class='num'>last</th><th class='num'>OI</th><th>action</th></tr>")
        sections.append(f"<section><h2>{_h.escape(str(match))}</h2>"
                        f"<table><thead>{head}</thead><tbody>{''.join(trs)}</tbody></table></section>")
    body = "".join(sections) or "<p>No overlapping Kalshi markets (await post-XI listings).</p>"
    stamp = datetime.now(timezone.utc).strftime("%a %d %b %H:%M UTC")
    out = f"""<!doctype html><html><head><meta charset="utf-8"><title>Kalshi overlay</title>
<style>
 body{{margin:0;background:#0f1115;color:#e6e9ef;font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;padding:18px 22px}}
 h1{{font-size:18px;margin:0 0 2px}} h2{{font-size:15px;margin:24px 0 6px;color:#cfe0ff}}
 .sub{{font-size:12px;color:#8b93a7;margin-bottom:6px}}
 table{{border-collapse:collapse}} th,td{{padding:4px 10px;border-bottom:1px solid #20242e;text-align:left;white-space:nowrap}}
 th{{font-size:11px;color:#8b93a7;font-weight:600}}
 td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums}} td.dim,.dim{{color:#8b93a7}}
 .p{{font-weight:600}} .sl{{color:#3a4150;margin:0 1px}} .act{{font-weight:600}}
 tr.take td{{background:#0e2a18}} tr.take td.act{{color:#7ee0a0}}
 tr.make td.act{{color:#e0a458}}
</style></head><body>
<h1>Kalshi overlay</h1>
<div class="sub">{stamp} · <b style="color:#7ee0a0">TAKE</b> = Kalshi offers a side ≤ our acceptable price (cross it) · <span style="color:#e0a458"><b>make</b></span> = rest a bid at our price to improve the book · our Y/N = acceptable maker price. READ-ONLY — place orders yourself.</div>
{body}
</body></html>"""
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")
    return path


if __name__ == "__main__":
    rows = build()
    print_overlay(rows)
    print("HTML ->", write_html(rows))
