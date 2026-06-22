"""Config for the minutes-projection model (self-contained sub-project).

The whole folder does `import config`; this is that module. It deliberately does
NOT reuse wc_props/config.py — the model is trained offline and only the parent
depends on IT (one-way, via predict.predict). We do borrow the parent's .env so
the API-Football key lives in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # mins_projection_model/
WCP_ROOT = ROOT.parent                           # wc_props/

# Borrow the parent .env (APISPORTS_KEY etc.) if python-dotenv is present.
try:
    from dotenv import load_dotenv
    load_dotenv(WCP_ROOT / ".env")
except ImportError:
    pass

DATA_DIR = Path(os.environ.get("MPM_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("MPM_MODEL_DIR", ROOT / "models"))
CACHE_DIR = DATA_DIR / "apifootball_cache"
for _d in (DATA_DIR, MODEL_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- API-Football (api-sports.io) ---
APISPORTS_KEY = os.environ.get("APISPORTS_KEY")
API_BASE = "https://v3.football.api-sports.io"

# International (league_id, season) pairs to train the national-team ratings on.
# Free tier serves seasons 2022-2024; Pro unlocks 2025-2026. Common league ids:
#   1 World Cup · 4 Euro · 9 Copa America · 6 AFCON · 5 Nations League · 10 Friendlies
#   WC qualifiers: 29 Africa · 30 Asia · 31 N&C America · 32 S.America · 33 Oceania · 34 Europe
INTL_SOURCES = [(1, 2022),                       # World Cup 2022
                (6, 2023),                       # AFCON 2023
                (4, 2024),                       # Euro 2024
                (9, 2024),                       # Copa America 2024
                (5, 2022), (5, 2024)]            # UEFA Nations League 2022-23, 2024-25
# Friendlies add ~1,100 fixtures of Elo continuity but noisier player rates. Pull
# them as a top-up when you want maximal rating coverage:
#   import backfill; backfill.main([(10,2022),(10,2023),(10,2024)])

# --- team-strength Elo hyperparameters (see team_strength.py) ---
POSS_K, POSS_HOME = 0.08, 0.15                   # logit link (possession share)
GOALS_K, GOALS_HOME, GOALS_MEAN = 0.04, 0.20, 1.25   # log link (goals)
# Shots-on-target uses an IDENTITY link (LinearElo): mid-count and ~symmetric, so
# the rating lives in SoT units and updates on the raw residual, not the log one.
SOT_K, SOT_HOME, SOT_MEAN = 0.06, 0.30, 4.3

# --- feature / rate-model knobs ---
EWM_HALFLIFE = 5                                 # matches; player form half-life

# --- live XI source (download_upc_starting_xi.py); endpoints fixed to /api/data/ ---
XI_SOURCE = os.environ.get("MPM_XI_SOURCE", "fotmob")

# --- legacy FBref downloader knobs (superseded by api_football.py; kept so the
#     old download_past_*_fbref.py modules still import without AttributeError) ---
CLUB_LEAGUES: list = []
INTL_LEAGUES: list = []
SEASONS: list = []
