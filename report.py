"""Render the matched/edge table to a single self-contained HTML file."""
from __future__ import annotations

import html
from datetime import datetime, timezone

import pandas as pd

import config


def _fmt_pct(x):
    return "" if x is None or pd.isna(x) else f"{x * 100:+.1f}%"


def _fmt_prob(x):
    return "" if x is None or pd.isna(x) else f"{x * 100:.0f}%"


def _fmt_num(x):
    return "" if x is None or pd.isna(x) else f"{x:g}"


def _best_play(r):
    """Pick the platform/side with the best edge for the headline cell."""
    cands = []
    if r.get("edge_pp") is not None and not pd.isna(r.get("edge_pp")):
        cands.append(("PP", r["pp_line"], r["side_pp"], r["edge_pp"]))
    if r.get("edge_ud") is not None and not pd.isna(r.get("edge_ud")):
        cands.append(("UD", r["ud_line"], r["side_ud"], r["edge_ud"]))
    if not cands:
        return "", None
    plat, line, side, edge = max(cands, key=lambda c: c[3])
    return f"{side} {line:g} ({plat})", edge


def build_html(df: pd.DataFrame, meta: dict) -> str:
    rows_html = []
    for _, r in df.iterrows():
        play, edge = _best_play(r)
        play_side = "OVER" if play.startswith("OVER") else ("UNDER" if play.startswith("UNDER") else "")
        edge_cls = ""
        if edge is not None and not pd.isna(edge):
            edge_cls = "pos" if edge > 0 else ("neg" if edge < 0 else "")
        gap = r.get("gap")
        gap_html = "" if gap is None or pd.isna(gap) else f"{gap:+g}"
        rows_html.append(
            f"<tr data-match=\"{html.escape(str(r.get('match') or ''))}\" data-side=\"{play_side}\">"
            f"<td class='player'>{html.escape(str(r['name']))}</td>"
            f"<td class='dim'>{html.escape(str(r.get('team') or ''))}</td>"
            f"<td class='dim'>{html.escape(str(r.get('opp') or ''))}</td>"
            f"<td>{html.escape(str(r['stat']))}</td>"
            f"<td class='num'>{_fmt_num(r.get('pp_line'))}</td>"
            f"<td class='num'>{_fmt_num(r.get('ud_line'))}</td>"
            f"<td class='num gap'>{gap_html}</td>"
            f"<td class='dim small'>{html.escape(str(r.get('b365_lines') or ''))}</td>"
            f"<td class='num'>{_fmt_prob(r.get('prob_pp') if r.get('prob_pp') is not None else r.get('prob_ud'))}</td>"
            f"<td class='play'>{html.escape(play)}</td>"
            f"<td class='num edge {edge_cls}' data-edge='{'' if edge is None or pd.isna(edge) else edge}'>{_fmt_pct(edge)}</td>"
            f"<td class='dim small'>{html.escape(str(r.get('method') or ''))}</td>"
            "</tr>"
        )

    headers = ["Player", "Team", "Opp", "Stat", "PP", "UD", "Gap",
               "b365 lines", "b365 P(o)", "Best play", "Edge", "Method"]
    th = "".join(f"<th onclick='sortBy({i})'>{h}</th>" for i, h in enumerate(headers))

    matches = sorted({str(m).strip() for m in df.get("match", pd.Series(dtype=str)).tolist()
                      if str(m).strip()})
    games_html = "".join(
        f'<button class="gamebtn" data-m="{html.escape(m)}" '
        f'onclick="setMatch(this)">{html.escape(m)}</button>'
        for m in matches)

    meta_line = (f"captured {meta.get('captured_at', '')} · "
                 f"UD {meta.get('n_ud', 0)} · PP {meta.get('n_pp', 0)} · "
                 f"bet365 {meta.get('n_b365_players', 0)} player-stat rows · "
                 f"breakeven {config.BREAKEVEN:.0%} · window {config.WINDOW_HOURS}h")
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
 .wrap {{ padding:0 22px 40px; }}
 table {{ border-collapse:collapse; width:100%; }}
 th,td {{ padding:7px 10px; text-align:left; border-bottom:1px solid var(--line); white-space:nowrap; }}
 th {{ position:sticky; top:0; background:var(--card); cursor:pointer; user-select:none;
       font-size:12px; color:var(--dim); }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .player {{ font-weight:600; }}
 .dim {{ color:var(--dim); }} .small {{ font-size:12px; }}
 .play {{ font-weight:600; }}
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
  <label><input type="checkbox" id="noufnders" onchange="filt()"> hide UNDER plays</label>
</div>
<div class="games">
  <button id="allbtn" class="gamebtn active" onclick="clearMatch()">All games</button>
  {games_html}
</div>
<div class="wrap">
<table id="t"><thead><tr>{th}</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody></table>
</div>
<footer>
  Edge = bet365 de-vigged P(favoured side) − breakeven, evaluated at each DFS line.
  "Method" notes how the bet365 probability was derived (devig = two-sided, raw = one-sided/optimistic,
  interp/extrap = off-ladder). Implied edge from market prices — not betting advice; validate the
  breakeven and de-vig assumptions against how you actually play.
</footer>
<script>
 const t=document.getElementById('t');
 let activeMatch=null;
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
   const nu=document.getElementById('noufnders').checked;
   for(const tr of t.tBodies[0].rows){{
     const txt=tr.innerText.toLowerCase();
     const ec=tr.querySelector('.edge'); const ev=ec?parseFloat(ec.dataset.edge):NaN;
     let show = txt.includes(q);
     if(ob && (isNaN(ev))) show=false;
     if(op && !(ev>0)) show=false;
     if(nu && tr.dataset.side==='UNDER') show=false;
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


def write_report(df: pd.DataFrame, meta: dict):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORT_HTML.write_text(build_html(df, meta), encoding="utf-8")
    return config.REPORT_HTML
