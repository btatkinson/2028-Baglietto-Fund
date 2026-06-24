"""Render the matched/edge table to a single self-contained HTML file."""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

import config


def _fmt_pct(x):
    return "" if x is None or pd.isna(x) else f"{x * 100:+.1f}%"


def _num(x):
    try:
        f = float(x)
        return f == f
    except (TypeError, ValueError):
        return False


def _fmt_prob(x):
    return "" if x is None or pd.isna(x) else f"{x * 100:.0f}%"


def _fmt_num(x):
    return "" if x is None or pd.isna(x) else f"{x:g}"


def _best_play(r):
    """Pick the platform/side with the best edge for the headline cell."""
    cands = []
    if r.get("edge_pp") is not None and not pd.isna(r.get("edge_pp")):
        cands.append(("PP", r["pp_line"], r["side_pp"], r["edge_pp"], None))
    if r.get("edge_ud") is not None and not pd.isna(r.get("edge_ud")):
        cands.append(("UD", r["ud_line"], r["side_ud"], r["edge_ud"], r.get("ud_mult")))
    if not cands:
        return "", None
    plat, line, side, edge, mult = max(cands, key=lambda c: c[3])
    tag = ""
    if mult is not None and not pd.isna(mult) and abs(float(mult) - 1.0) > 1e-9:
        tag = f" @{float(mult):g}x"
    return f"{side} {line:g} ({plat}{tag})", edge


def _lineups_html(lineups) -> str:
    if not lineups:
        return ""
    cols = []
    for plat in ("ud", "pp"):
        info = lineups.get(plat)
        if not info:
            continue
        s, lus = info["structure"], info["lineups"]
        cards = []
        for i, lu in enumerate(lus, 1):
            legs = "".join(
                f"<div class='leg'><span>{html.escape(l['name'])} — {html.escape(l['stat'])}"
                f" <b>{l['side']} {l['line']:g}</b>"
                + (f" @{l['mult']:g}x" if abs(l['mult'] - 1.0) > 1e-9 else "")
                + f"</span><span class='legedge'>{l['edge']*100:+.1f}%</span></div>"
                for l in lu["legs"])
            cards.append(
                f"<div class='card'><div class='cardhead'>#{i}"
                f"<span class='cardmeta'>EV {lu['ev']:.2f}x · sweep {lu['p_sweep']*100:.0f}%</span></div>"
                f"{legs}</div>")
        cols.append(f"<div class='lcol'><h3>{html.escape(s['label'])} "
                    f"<span class='hintc'>({info['n_legs']} qualified legs)</span></h3>"
                    + ("".join(cards) or "<div class='hintc'>not enough qualified legs</div>")
                    + "</div>")
    if not cols:
        return ""
    return ("<details class='lineups' open><summary>Suggested lineups "
            "<span class='hintc'>(each leg ≤2 uses, no repeated pairings, no same-game legs)</span>"
            "</summary><div class='lwrap'>" + "".join(cols) + "</div></details>")


def _play_str(side, line, plat, mult=None):
    if not side or line is None or pd.isna(line):
        return ""
    tag = ""
    if mult is not None and not pd.isna(mult) and abs(float(mult) - 1.0) > 1e-9:
        tag = f" @{float(mult):g}x"
    return f"{side} {float(line):g} ({plat}{tag})"


def build_html(df: pd.DataFrame, meta: dict, lineups=None) -> str:
    rows_html = []
    for _, r in df.iterrows():
        play, edge = _best_play(r)
        play_pp = _play_str(r.get("side_pp"), r.get("pp_line"), "PP")
        play_ud = _play_str(r.get("side_ud"), r.get("ud_line"), "UD", r.get("ud_mult"))
        e_pp = "" if not _num(r.get("edge_pp")) else f"{float(r['edge_pp']):.4f}"
        e_ud = "" if not _num(r.get("edge_ud")) else f"{float(r['edge_ud']):.4f}"
        edge_cls = ""
        if edge is not None and not pd.isna(edge):
            edge_cls = "pos" if edge > 0 else ("neg" if edge < 0 else "")
        gap = r.get("gap")
        gap_html = "" if gap is None or pd.isna(gap) else f"{gap:+g}"
        rows_html.append(
            f"<tr data-match=\"{html.escape(str(r.get('match') or ''))}\""
            f" data-play-all=\"{html.escape(play)}\""
            f" data-edge-all=\"{'' if edge is None or pd.isna(edge) else edge}\""
            f" data-play-pp=\"{html.escape(play_pp)}\" data-edge-pp=\"{e_pp}\""
            f" data-play-ud=\"{html.escape(play_ud)}\" data-edge-ud=\"{e_ud}\">"
            f"<td class='player'>{html.escape(str(r['name']))}</td>"
            f"<td class='dim'>{html.escape(str(r.get('team') or ''))}</td>"
            f"<td class='dim'>{html.escape(str(r.get('opp') or ''))}</td>"
            f"<td>{html.escape(str(r['stat']))}</td>"
            f"<td class='num'>{_fmt_num(r.get('pp_line'))}</td>"
            f"<td class='num'>{_fmt_num(r.get('ud_line'))}</td>"
            f"<td class='num gap'>{gap_html}</td>"
            f"<td class='dim small'>{html.escape(str(r.get('b365_lines') or ''))}</td>"
            f"<td class='num'>{_fmt_prob(r.get('prob_pp') if r.get('prob_pp') is not None else r.get('prob_ud'))}</td>"
            f"<td class='num model'>{_fmt_prob(r.get('prob_model'))}</td>"
            f"<td class='num dim small'>{_fmt_num(r.get('fair_model'))}</td>"
            f"<td class='play'>{html.escape(play)}</td>"
            f"<td class='num edge {edge_cls}' data-edge='{'' if edge is None or pd.isna(edge) else edge}'>{_fmt_pct(edge)}</td>"
            f"<td class='dim small'>{html.escape(str(r.get('method') or ''))}</td>"
            "</tr>"
        )

    headers = ["Player", "Team", "Opp", "Stat", "PP", "UD", "Gap",
               "b365 lines", "b365 P(o)", "Model P(o)", "Model fair",
               "Best play", "Edge", "Method"]
    th = "".join(f"<th onclick='sortBy({i})'>{h}</th>" for i, h in enumerate(headers))

    matches = sorted({str(m).strip() for m in df.get("match", pd.Series(dtype=str)).tolist()
                      if str(m).strip()})
    games_html = "".join(
        f'<button class="gamebtn" data-m="{html.escape(m)}" '
        f'onclick="setMatch(this)">{html.escape(m)}</button>'
        for m in matches)

    lineups_html = _lineups_html(lineups)

    meta_line = (f"captured {meta.get('captured_at', '')} · "
                 f"UD {meta.get('n_ud', 0)} · PP {meta.get('n_pp', 0)} · "
                 f"bet365 {meta.get('n_b365_players', 0)} player-stat rows · "
                 f"breakevens UD {config.BREAKEVEN_UD:.1%} / PP {config.BREAKEVEN_PP:.1%} "
                 f"+ {config.EDGE_CUSHION:.0%} cushion ({config.EDGE_CUSHION_SOFT:.0%} for neut/extrap) "
                 f"· window {config.WINDOW_HOURS}h")
    margins_line = meta.get("margins") or ""

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>WC Props — implied edge</title>
<style>
 :root {{ --bg:#0f1115; --card:#171a21; --line:#262b36; --txt:#e6e9ef; --dim:#8b93a7;
          --pos:#1f7a4d; --posbg:#0f3322; --neg:#8a2f3a; }}
 * {{ box-sizing:border-box; }}
 body {{ margin:0; background:var(--bg); color:var(--txt);
         font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
 header {{ padding:18px 22px; border-bottom:1px solid var(--line); }}
 h1 {{ margin:0 0 4px; font-size:18px; }}
 .meta {{ color:var(--dim); font-size:12px; }}
 .controls {{ padding:12px 22px; display:flex; gap:12px; align-items:center; }}
 input[type=text] {{ background:var(--card); border:1px solid var(--line); color:var(--txt);
                     padding:7px 10px; border-radius:8px; min-width:240px; }}
 label {{ color:var(--dim); font-size:13px; display:flex; gap:6px; align-items:center; }}
 .games {{ padding:4px 22px 12px; display:flex; flex-wrap:wrap; gap:8px; }}
 .gamebtn {{ background:var(--card); border:1px solid var(--line); color:var(--dim);
             padding:6px 11px; border-radius:999px; cursor:pointer; font-size:12px; }}
 .gamebtn:hover {{ color:var(--txt); border-color:#3a4150; }}
 .gamebtn.active {{ background:#1f6feb22; border-color:#1f6feb; color:#cfe0ff; font-weight:600; }}
 .platgrp {{ display:flex; gap:0; margin-left:auto; border:1px solid var(--line); border-radius:999px; overflow:hidden; }}
 .platbtn {{ background:var(--card); border:0; color:var(--dim); padding:6px 13px; cursor:pointer; font-size:12px; }}
 .platbtn.active {{ background:#1f6feb22; color:#cfe0ff; font-weight:600; }}
 .lineups {{ margin:6px 22px 14px; border:1px solid var(--line); border-radius:10px; background:var(--card); }}
 .lineups summary {{ padding:10px 14px; cursor:pointer; font-weight:600; }}
 .hintc {{ color:var(--dim); font-weight:400; font-size:12px; }}
 .lwrap {{ display:flex; flex-wrap:wrap; gap:18px; padding:4px 14px 14px; }}
 .lcol {{ flex:1 1 320px; min-width:300px; }}
 .lcol h3 {{ font-size:14px; margin:8px 0; }}
 .card {{ border:1px solid var(--line); border-radius:8px; padding:8px 10px; margin-bottom:8px; background:var(--bg); }}
 .cardhead {{ display:flex; justify-content:space-between; color:var(--dim); font-size:12px; margin-bottom:6px; }}
 .cardmeta {{ color:#cdb46b; }}
 .leg {{ display:flex; justify-content:space-between; gap:10px; font-size:13px; padding:2px 0; }}
 .legedge {{ color:#5ad19a; font-variant-numeric:tabular-nums; }}
 .wrap {{ padding:0 22px 40px; }}
 table {{ border-collapse:collapse; width:100%; }}
 th,td {{ padding:7px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }}
 th {{ position:sticky; top:0; background:var(--card); cursor:pointer; user-select:none;
       font-size:12px; color:var(--dim); }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .player {{ font-weight:600; }}
 .dim {{ color:var(--dim); }} .small {{ font-size:12px; }}
 .play {{ font-weight:600; }}
 .model {{ color:#9fc6ff; }}
 .edge.pos {{ color:#5ad19a; background:var(--posbg); }}
 .edge.neg {{ color:#f08a96; }}
 tr:hover td {{ background:#1c212b; }}
 .gap {{ color:#cdb46b; }}
 footer {{ color:var(--dim); font-size:12px; padding:0 22px 30px; }}
</style></head><body>
<header>
  <h1>World Cup Props — implied edge</h1>
  <div class="meta">{html.escape(meta_line)}</div>
  {f'<div class="meta">one-way margins — {html.escape(margins_line)}</div>' if margins_line else ''}
</header>
<div class="controls">
  <input id="q" type="text" placeholder="filter player / stat / team…" oninput="filt()">
  <label><input type="checkbox" id="onlyb" onchange="filt()"> bet365-backed only</label>
  <label><input type="checkbox" id="onlypos" onchange="filt()"> positive edge only</label>
  <span class="platgrp">
    <button class="platbtn active" data-p="all" onclick="setPlat(this)">All</button>
    <button class="platbtn" data-p="ud" onclick="setPlat(this)">UD only</button>
    <button class="platbtn" data-p="pp" onclick="setPlat(this)">PP only</button>
  </span>
</div>
<div class="games">
  <button id="allbtn" class="gamebtn active" onclick="clearMatch()">All games</button>
  {games_html}
</div>
{lineups_html}
<div class="wrap">
<table id="t"><thead><tr>{th}</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</div>
<footer>
  Edge = bet365 de-vigged P(side) − required probability, where required = platform breakeven
  (UD 3-power / PP 5-flex by default) ÷ leg multiplier + a cushion against bet365 model error.
  Sides Underdog doesn't offer (one-way lines) are never recommended. "Method" notes how the
  bet365 probability was derived (devig = two-sided; 1way/mNN = one-sided deflated by a measured
  or neutrality-calibrated margin; interp/extrap = off-ladder). Lineups: greedy pairing with each
  leg used ≤2×, no repeated pairings, no same-game legs; EV is exact Poisson-binomial over the
  structure's tiers. Implied edge from market prices — not betting advice.
</footer>
<script>
 const t=document.getElementById('t');
 let activeMatch=null, platMode='all';
 function setPlat(btn){{
   platMode=btn.dataset.p;
   document.querySelectorAll('.platbtn').forEach(b=>b.classList.toggle('active', b===btn));
   for(const tr of t.tBodies[0].rows){{
     const play=tr.dataset['play'+cap(platMode)]||'';
     const ev=tr.dataset['edge'+cap(platMode)];
     const pc=tr.querySelector('.play'), ec=tr.querySelector('.edge');
     if(pc) pc.textContent=play;
     if(ec){{
       const f=parseFloat(ev);
       ec.dataset.edge=isNaN(f)?'':f;
       ec.textContent=isNaN(f)?'':((f>=0?'+':'')+(f*100).toFixed(1)+'%');
       ec.classList.toggle('pos', f>0); ec.classList.toggle('neg', f<0);
     }}
   }}
   filt();
 }}
 function cap(s){{ return s==='all'?'All':(s==='ud'?'Ud':'Pp'); }}
 function setMatch(btn){{
   const m=btn.dataset.m;
   activeMatch=(m===activeMatch)?null:m;   // click the active game again to clear
   document.querySelectorAll('.gamebtn').forEach(b=>b.classList.toggle('active', b.dataset.m===activeMatch));
   document.getElementById('allbtn').classList.toggle('active', !activeMatch);
   filt();
 }}
 function clearMatch(){{
   activeMatch=null;
   document.querySelectorAll('.gamebtn').forEach(b=>b.classList.remove('active'));
   document.getElementById('allbtn').classList.add('active');
   filt();
 }}
 function filt(){{
   const q=document.getElementById('q').value.toLowerCase();
   const ob=document.getElementById('onlyb').checked;
   const op=document.getElementById('onlypos').checked;
   for(const tr of t.tBodies[0].rows){{
     const txt=tr.innerText.toLowerCase();
     const ec=tr.querySelector('.edge'); const ev=ec?parseFloat(ec.dataset.edge):NaN;
     let show = txt.includes(q);
     if(platMode!=='all' && !tr.dataset['edge'+cap(platMode)]) show=false;
     if(ob && (isNaN(ev))) show=false;
     if(op && !(ev>0)) show=false;
     if(activeMatch && tr.dataset.match!==activeMatch) show=false;
     tr.style.display = show?'':'none';
   }}
 }}
 let asc={{}};
 function sortBy(i){{
   const rows=[...t.tBodies[0].rows];
   asc[i]=!asc[i];
   rows.sort((a,b)=>{{
     let x=a.cells[i].dataset.edge??a.cells[i].innerText;
     let y=b.cells[i].dataset.edge??b.cells[i].innerText;
     const nx=parseFloat(x), ny=parseFloat(y);
     if(!isNaN(nx)&&!isNaN(ny)) return asc[i]?nx-ny:ny-nx;
     return asc[i]?(''+x).localeCompare(y):(''+y).localeCompare(x);
   }});
   for(const r of rows) t.tBodies[0].appendChild(r);
 }}
</script>
</body></html>"""


def write_report(df: pd.DataFrame, meta: dict, lineups=None):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_HTML.write_text(build_html(df, meta, lineups), encoding="utf-8")
    return config.REPORT_HTML
