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

# Edge model. DFS pick'em breakeven probability PER LEG. 0.5 = naive even-money.
# Real breakeven depends on entry type (e.g. ~0.5 flex 2-pick, higher for power
# plays) — set WCP_BREAKEVEN to match how you actually play.
BREAKEVEN = float(os.environ.get("WCP_BREAKEVEN", "0.5"))

# One-way markets (Over quoted with no Under) bake the vig into the single price
# and we can't observe the overround, so we assume one and deflate:
#   p_fair = (1/odds) / (1 + ONEWAY_MARGIN)
# Bigger = more conservative ("terrible vig" haircut on one-sided lines).
ONEWAY_MARGIN = float(os.environ.get("WCP_ONEWAY_MARGIN", "0.20"))

# Manual paste files
UNDERDOG_JSON = DATA_DIR / "underdog.json"
PRIZEPICKS_JSON = DATA_DIR / "prizepicks.json"
REPORT_HTML = OUT_DIR / "report.html"
