# WC Props — implied-edge pipeline

Paste two JSON files, run one command, get a sorted HTML edge report.

It compares **PrizePicks** and **Underdog** pick'em lines against **bet365's**
actual priced player-stat markets (via BetsAPI) — the only source that carries
the deep World Cup stat tab (tackles, passes, saves, shots, fouls). bet365's
de-vigged odds give an implied probability at each DFS line; the edge is that
probability minus your pick'em breakeven.

## Layout

```
wc_props/
  run.py          # the one button: capture -> match -> report
  config.py       # env keys, paths, window, breakeven
  normalize.py    # name + stat normalization (surname-anchored fuzzy match)
  sources.py      # parse underdog.json / prizepicks.json
  capture.py      # BetsAPI bet365 capture (next 48h) + flatten
  edge.py         # de-vig, implied P(over line), edge
  match.py        # UD<->PP fuzzy match, attach bet365 ladders
  report.py       # self-contained sortable/filterable HTML
  data/           # <- paste underdog.json + prizepicks.json here; captures saved here
  out/report.html # <- the deliverable
```

## Setup

```
pip install -r requirements.txt
cp .env.example .env        # then edit .env and paste your token(s)
```

`.env` (read automatically via python-dotenv):

```
BETSAPI_TOKEN=your_bet365_api_token
# ODDSPAPI_KEY=your_oddspapi_key        # optional (future: goalscorer/Pinnacle)
# WCP_BREAKEVEN=0.5                      # your real pick'em per-leg breakeven
```

No shell `export` needed. If you'd rather not use a `.env`, set it for the
session instead — PowerShell: `$env:BETSAPI_TOKEN = "your_token"`,
bash: `export BETSAPI_TOKEN=your_token`.

Other optional knobs (in `.env` or environment): `WCP_WINDOW_HOURS` (default 48),
`WCP_WC_LEAGUE_ID` (skip auto-detection), `WCP_MAX_DETAIL` (max matches/sweep).

## Use

1. Paste the raw payloads into `data/`:
   - `data/underdog.json` ← `https://api.underdogfantasy.com/beta/v6/over_under_lines?sport_id=SOCCER`
   - `data/prizepicks.json` ← `https://api.prizepicks.com/projections?league_id=241&per_page=500`
2. Run:
   ```
   python run.py            # capture fresh bet365 lines, match, write out/report.html
   python run.py --open     # also open the report
   python run.py --no-scrape    # reuse the last capture (fast iteration, no BetsAPI calls)
   ```
3. Open `out/report.html` — sortable by any column, filter box, "bet365-backed
   only" and "positive edge only" toggles.

## How the edge is computed

For each `(player, stat)` matched across sources, bet365's Over/Under odds are
de-vigged into an implied `P(over line)` ladder; that ladder is interpolated to
the DFS line, and edge = `P(favoured side) − breakeven`. The `method` column
flags how each probability was derived:

- `devig` — two-sided line, de-vigged (most trustworthy)
- `1way/mNN` — one-sided "Over X" line, deflated by an assumed NN% margin
  (set `WCP_ONEWAY_MARGIN`, default 0.20 — bigger = harsher, more conservative)
- `interp` / `extrap` — DFS line fell between / outside bet365's posted lines

These are implied edges from market prices, **not betting advice.** The de-vig
is the basic proportional method and the default breakeven is naive even-money;
both are assumptions to validate against how you actually play. Tune `edge.py`
and `WCP_BREAKEVEN` once you've eyeballed real captures.

## Deploy to Railway (access from anywhere, password-protected)

The repo includes a small Flask app (`webapp.py`) and a `Dockerfile`. Railway
builds the Dockerfile automatically.

1. Push the project to a GitHub repo (the `.gitignore` keeps `.env`, data, and
   reports out).
2. In Railway: **New Project → Deploy from GitHub repo**, pick the repo.
3. In the service **Variables**, set:
   - `APP_PASSWORD` — required; the app refuses to serve without it
   - `BETSAPI_TOKEN` — for live bet365 captures
   - `APP_USER` — optional basic-auth username (default `blake`)
4. Open the generated `*.up.railway.app` URL. Your browser prompts for the
   username/password (HTTP Basic over Railway's TLS).

Using it remotely: `/control` has paste boxes / file uploads for
`underdog.json` and `prizepicks.json` plus a **Run pipeline** button (with a
fresh-scrape checkbox). `/` serves the latest report with all the filters and
game buttons. `/healthz` is an unauthenticated liveness probe.

Note Railway's filesystem is **ephemeral** — uploaded JSONs, captures, and the
report reset on redeploy/restart. That's fine for the refill-and-run workflow;
if you want capture history (line movement) to persist, attach a Railway
Volume and point the app at it via `WCP_DATA_DIR` and `WCP_OUT_DIR`.

## Notes

- bet365 often posts the deep stat tab only near kickoff or in-play, so run
  inside the 48h window (and re-run closer to matches) for full depth.
- Captures are timestamped under `data/{date}/betsapi/`, so repeated runs build
  line-movement history for free.
- Fantasy-points lines are matched but flagged `*` — scoring differs by platform.
