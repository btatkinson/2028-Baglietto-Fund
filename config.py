"""Central config. Reads from a .env file (preferred) or environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Load .env from the project root if python-dotenv is installed (preferred).
# Falls back silently to plain environment variables if it isn't.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DATA_DIR = Path(os.environ.get("WCP_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(os.environ.get("WCP_OUT_DIR", ROOT / "out"))

# Credentials
BETSAPI_TOKEN = os.environ.get("BETSAPI_TOKEN")        # bet365 stat props (required for edge)
ODDSPAPI_KEY = os.environ.get("ODDSPAPI_KEY")          # optional: goalscorer + Pinnacle

# Capture window / scope
SOCCER_SPORT_ID = 1                                    # BetsAPI soccer
WINDOW_HOURS = int(os.environ.get("WCP_WINDOW_HOURS", "48"))
WC_LEAGUE_ID = os.environ.get("WCP_WC_LEAGUE_ID")      # set to skip auto-detection
MAX_DETAIL = int(os.environ.get("WCP_MAX_DETAIL", "40"))

# Per-platform per-leg breakevens (at 1.0x multiplier), set to the entry
# structure you actually play. Defaults: UD 3-power (6.5x -> 53.6%) and
# PP 5-flex (10x/2x/0.4x -> 54.3%). WCP_BREAKEVEN overrides both if set.
_BE_BOTH = os.environ.get("WCP_BREAKEVEN")
BREAKEVEN_UD = float(os.environ.get("WCP_BREAKEVEN_UD", _BE_BOTH or "0.536"))
BREAKEVEN_PP = float(os.environ.get("WCP_BREAKEVEN_PP", _BE_BOTH or "0.543"))

# Edge cushion: extra required probability on top of the breakeven, as
# insurance against bet365 being wrong / de-vig model error. Applied after
# multiplier scaling: required p = breakeven/mult + cushion.
# Tiered by trust: the base cushion covers devig / measured-margin rows
# (de-vig method ambiguity ~1pp + bet365 prop softness 1-3pp); the SOFT
# cushion applies when the probability came from a neutrality-calibrated
# margin or off-ladder extrapolation, which carry real model risk.
# Base 0.03 puts the effective bars at ~56.6% (UD 3-power) / ~57.3% (PP 5-flex).
EDGE_CUSHION = float(os.environ.get("WCP_EDGE_CUSHION", "0.03"))
EDGE_CUSHION_SOFT = float(os.environ.get("WCP_EDGE_CUSHION_SOFT", "0.05"))

# Lineup generation (structures defined in lineups.STRUCTURES)
UD_STRUCTURE = os.environ.get("WCP_UD_STRUCTURE", "ud_power3")
PP_STRUCTURE = os.environ.get("WCP_PP_STRUCTURE", "pp_flex5")
MAX_LEG_USES = int(os.environ.get("WCP_MAX_LEG_USES", "2"))

# One-way markets (Over quoted with no Under) bake the vig into the single price
# and we can't observe the overround, so we assume one and deflate:
#   p_fair = (1/odds) / (1 + ONEWAY_MARGIN)
# Bigger = more conservative ("terrible vig" haircut on one-sided lines).
ONEWAY_MARGIN = float(os.environ.get("WCP_ONEWAY_MARGIN", "0.20"))

# Manual paste files
UNDERDOG_JSON = DATA_DIR / "underdog.json"
PRIZEPICKS_JSON = DATA_DIR / "prizepicks.json"
REPORT_HTML = OUT_DIR / "report.html"
