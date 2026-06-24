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
# GOALS: the rating is now a WALK-FORWARD GLOBAL POISSON attack/defence fit
# (team_strength.build_ratings), not the online Elo. The online k=0.04 was both
# less accurate AND over-shrunk, and even at k=0.15 it couldn't separate
# cross-confederation pairs (Portugal-Uzbekistan stuck at 0.95-0.83) because an
# online nudge can't propagate sparse cross-cluster links. Solving ALL prior
# matches jointly each refit does: same data, Portugal-Uzbekistan 0.95 -> 1.78,
# and the confederation hierarchy (AFC weakest) emerges. GOALS_HOME seeds the
# home coefficient name only; the fit learns its own home term. GOALS_MEAN is the
# cold-start anchor exp(mu) for teams with no window history. GOALS_K is retained
# only for any legacy online use; build_ratings no longer reads it for goals.
GOALS_K, GOALS_HOME, GOALS_MEAN = 0.15, 0.20, 1.25   # (GOALS_K legacy/unused for goals)
GOALS_RIDGE = 1e-3                               # L2 on the global Poisson attack/def coefs
# Warm-up before HEAD training: a fixture's rating snapshot is only trustworthy
# once the system has seen enough history. We gate head-training rows on it — it
# takes a tournament cycle or two before sparse-graph ratings carry real signal,
# and training on cold estimates just teaches the heads noise. A snapshot is warm
# when the global fit window has >= GOALS_WARMUP_MIN_MATCHES total AND BOTH teams
# have >= GOALS_WARMUP_TEAM_MATCHES prior matches. Ratings still BUILD on the cold
# early matches (the fit needs them); they're only withheld as training targets.
GOALS_WARMUP_MIN_MATCHES = int(os.environ.get("MPM_GOALS_WARMUP_MIN", "200"))
GOALS_WARMUP_TEAM_MATCHES = int(os.environ.get("MPM_GOALS_WARMUP_TEAM", "5"))
# Shots-on-target uses an IDENTITY link (LinearElo): mid-count and ~symmetric, so
# the rating lives in SoT units and updates on the raw residual, not the log one.
SOT_K, SOT_HOME, SOT_MEAN = 0.06, 0.30, 4.3
# Possession & SoT are now WALK-FORWARD GLOBAL fits too (same rationale as goals):
# possession via ridge least-squares on logit(home share) — one rating per team;
# SoT via ridge LS attack/defence (identity link, reads in SoT units). Ridge alphas
# are on the linear scale (team-dummy design), unlike GOALS_RIDGE (Poisson scale).
POSS_RIDGE = float(os.environ.get("MPM_POSS_RIDGE", "10.0"))
SOT_RIDGE = float(os.environ.get("MPM_SOT_RIDGE", "10.0"))
# Total shots & corners: same global identity attack/defence treatment, added as
# script features (shots drives the player-shots head directly; corners is an
# attacking-pressure proxy). Both come free from the cached statistics payloads.
SHOTS_MEAN, SHOTS_RIDGE = 12.5, float(os.environ.get("MPM_SHOTS_RIDGE", "10.0"))
CORNERS_MEAN, CORNERS_RIDGE = 5.0, float(os.environ.get("MPM_CORNERS_RIDGE", "10.0"))
# K/HOME above are retained only for any legacy online use; build_ratings no longer
# reads POSS_K/SOT_K — the global fits learn their own home term.

# --- feature / rate-model knobs ---
EWM_HALFLIFE = 5                                 # matches; player form half-life

# --- live XI source (download_upc_starting_xi.py); endpoints fixed to /api/data/ ---
XI_SOURCE = os.environ.get("MPM_XI_SOURCE", "fotmob")

# --- legacy FBref downloader knobs (superseded by api_football.py; kept so the
#     old download_past_*_fbref.py modules still import without AttributeError) ---
CLUB_LEAGUES: list = []
INTL_LEAGUES: list = []
SEASONS: list = []
