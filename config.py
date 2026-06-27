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

# Claude (Anthropic) API — powers the LLM injury/news check on the scorers board
# (news.py). The numbers pipeline is blind to real-world news (a warmup knock, an
# illness, a late suspension, a "manager hints at rotation"); this is the one signal
# that catches a player we'd otherwise confidently quote a price on and get picked off.
# Used ONLY as a veto/flag — it can warn or suppress a quote, never numerically move a
# price (LLM probabilities aren't calibrated). Scoped tight to post-XI act-on players to
# keep cost + false positives down; no key set ⇒ the check is skipped silently.
CLAUDE_API_KEY = os.environ.get("CLAUDE_API")
CLAUDE_MODEL = os.environ.get("WCP_CLAUDE_MODEL", "claude-opus-4-8")
# Master switch for the news check — OFF by default (it spends real Opus + web-search
# requests per act-on player and can burn through quota fast). Flip WCP_NEWS_ENABLED=1
# to turn it back on; the key can stay in .env meanwhile.
NEWS_ENABLED = os.environ.get("WCP_NEWS_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
NEWS_TOP_PER_SIDE = int(os.environ.get("WCP_NEWS_TOP_PER_SIDE", "6"))  # checked per team, by hazard
NEWS_MAX_PLAYERS = int(os.environ.get("WCP_NEWS_MAX_PLAYERS", "24"))   # hard slate cap (cost)
NEWS_MAX_WORKERS = int(os.environ.get("WCP_NEWS_MAX_WORKERS", "4"))    # parallel API calls
NEWS_CACHE_TTL = int(os.environ.get("WCP_NEWS_CACHE_TTL", "1800"))     # reuse a verdict for 30 min

# Kalshi — READ-ONLY market data + order-book overlay for the post-XI maker board.
# KALSHI_API is the API key id (UUID); KALSHI_RSA is the RSA private key (the base64 DER
# body, or a full PEM) used to sign each request (see kalshi.py). Base URL is overridable
# (prod vs demo). We only ever GET here — no order placement.
KALSHI_KEY_ID = os.environ.get("KALSHI_API")
KALSHI_PRIVATE_KEY = os.environ.get("KALSHI_RSA")
KALSHI_BASE = os.environ.get("WCP_KALSHI_BASE", "https://api.elections.kalshi.com")

# Capture window / scope
SOCCER_SPORT_ID = 1                                    # BetsAPI soccer
# Capture horizon. Default 168h (7 days): WC group games cluster every 3-4 days,
# and DFS posts keeper/prop lines for the *next* fixture well before kickoff
# (Türkiye v USA posted at ~107h out). 96h silently dropped those — anything
# beyond the window has no bet365 anchor, so its DFS props can never show edge.
WINDOW_HOURS = int(os.environ.get("WCP_WINDOW_HOURS", "168"))
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

# Per-stat P(over) correction in the report edges. The bet365-informed shots ladder runs hot —
# we land on the OVER far too often — so shave P(over Shots) down a touch to rebalance the
# over/under split (negative = bring overs down). Applied in edge.evaluate via match.join.
STAT_OVER_ADJ = {"Shots": float(os.environ.get("WCP_SHOTS_OVER_ADJ", "-0.04"))}

# Lineup generation (structures defined in lineups.STRUCTURES)
UD_STRUCTURE = os.environ.get("WCP_UD_STRUCTURE", "ud_power3")
PP_STRUCTURE = os.environ.get("WCP_PP_STRUCTURE", "pp_flex5")
MAX_LEG_USES = int(os.environ.get("WCP_MAX_LEG_USES", "2"))

# Blend weight on the Underdog-implied probability when combining the bet365
# ladder with UD's own multiplier-implied ladder (remainder goes to bet365).
# 0.5 = equal trust (the uninformed-prior default until calibration fits a weight);
# 0 = pure bet365, 1 = pure UD. calibration.blend_weight() overrides this once
# >= MIN_SAMPLES outcomes exist; main.py --w-ud and WCP_BLEND_W_UD override too.
BLEND_W_UD = float(os.environ.get("WCP_BLEND_W_UD", "0.5"))

# One-way markets (Over quoted with no Under) bake the vig into the single price
# and we can't observe the overround, so we assume one and deflate:
#   p_fair = (1/odds) / (1 + ONEWAY_MARGIN)
# Bigger = more conservative ("terrible vig" haircut on one-sided lines).
ONEWAY_MARGIN = float(os.environ.get("WCP_ONEWAY_MARGIN", "0.20"))

# Kalshi take/make pricing for the post-XI board. Required EDGE below the de-vigged fair,
# before fees (in probability):
#   acceptable TAKE price  = fair*100 - 100*CUSHION                  - taker_fee(p)   (¢/side)
#   acceptable MAKE ceiling = fair*100 - 100*(CUSHION + LIQ_PREMIUM) - maker_fee(p)   (¢/side)
# We cross the live book (taker) iff it offers a side at/below the TAKE price; otherwise we
# rest (maker) at penny-the-bid, capped by the MAKE ceiling. CUSHION is the base edge each
# side — NOTE the prior default was a 1% "half-spread", so 3% is materially tighter (fewer
# fills, more edge each); dial WCP_KALSHI_CUSHION to taste. LIQ_PREMIUM is the EXTRA edge
# demanded to REST (improve the book), compensating fill risk + adverse selection.
KALSHI_EDGE_CUSHION = float(os.environ.get("WCP_KALSHI_CUSHION", "0.01"))
KALSHI_LIQUIDITY_PREMIUM = float(os.environ.get("WCP_KALSHI_LIQ_PREMIUM", "0.01"))
# FEE_RATE is Kalshi's taker fee rate; the per-contract fee is rate*p*(1-p) of $1, rounded up
# to the cent. 0.07 is the taker schedule.
KALSHI_FEE_RATE = float(os.environ.get("WCP_KALSHI_FEE_RATE", "0.07"))
# Kalshi's maker fee is currently 25% of the taker fee (and rounds to ~$0 on small size), so a
# RESTING order priced off this is the right ceiling — using the taker rate under-bids makes.
# Used only for the 'make' side of the order ticket; the take side keeps the full taker rate.
KALSHI_MAKER_FEE_RATE = float(os.environ.get("WCP_KALSHI_MAKER_FEE_RATE",
                                             str(0.25 * float(os.environ.get("WCP_KALSHI_FEE_RATE", "0.07")))))
# Don't post a maker quote on noise: skip a market whose fair YES prob is below this (a
# 2'-cameo defender at ~0% shouldn't be quoted). The high-prob (NO) side of a near-cert
# is still suppressed automatically when its acceptable price floors below 1c.
KALSHI_MIN_QUOTE_PROB = float(os.environ.get("WCP_KALSHI_MIN_QUOTE_PROB", "0.03"))

# Live order sizing (kalshi_orders, only reached via `run.py --trade` + per-order approval).
# Fractional-Kelly: capital-at-risk on a side = KELLY_FRACTION * f* * BANKROLL, where the
# full-Kelly fraction f* = edge/(1-price) and edge = our_fair_prob - price_paid. This sizes
# by how mispriced the live market is; the per-market MAX_RISK cap is the hard backstop
# (it binds most on favourites, whose small (1-price) inflates f*). Payout = contracts*$1 =
# risk/price, so longshots pay more and favourites less for the same risk — as intended.
KALSHI_BANKROLL = float(os.environ.get("WCP_KALSHI_BANKROLL", "5000"))
KALSHI_KELLY_FRACTION = float(os.environ.get("WCP_KALSHI_KELLY", "0.5"))
KALSHI_MAX_RISK = float(os.environ.get("WCP_KALSHI_MAX_RISK", "250"))     # cap $ at risk / market
# Per-PLAYER cap = this multiple of MAX_RISK, summed across that player's correlated G/A/G+A
# markets (both sides). 1.4 lets a name's best market run near the per-market cap while
# stopping all three from each loading to full (which would over-concentrate one player).
KALSHI_PLAYER_RISK_MULT = float(os.environ.get("WCP_KALSHI_PLAYER_MULT", "1.4"))
# Optional contract-count cap (≈ max payout / liquidity guard); blank = no cap (only MAX_RISK).
_mc = os.environ.get("WCP_KALSHI_MAX_CONTRACTS", "").strip()
KALSHI_MAX_CONTRACTS = int(_mc) if _mc else None

# DFS paste files + the public endpoints they come from (sources.refresh_dfs pulls
# these in place). UD's beta endpoint is unauthenticated; PP needs a browser UA
# (plain GET → 403 Cloudflare). Override the URLs via env if the league/sport shifts.
UNDERDOG_JSON = DATA_DIR / "underdog.json"
PRIZEPICKS_JSON = DATA_DIR / "prizepicks.json"
UNDERDOG_URL = os.environ.get(
    "WCP_UNDERDOG_URL", "https://api.underdogfantasy.com/beta/v6/over_under_lines?sport_id=FIFA")
PRIZEPICKS_URL = os.environ.get(
    "WCP_PRIZEPICKS_URL", "https://api.prizepicks.com/projections?league_id=241&per_page=500")
REPORT_HTML = OUT_DIR / "report.html"

# Interactive board (board.py + webapp.py /scorers): the pickle is the last pipeline's
# (b365, proj) so the web process can re-price minutes without re-running the model
# subprocess; the JSON holds manual minutes overrides (persist + auto-expire at kickoff).
BOARD_STATE_PKL = OUT_DIR / "board_state.pkl"
MINUTES_OVERRIDES_JSON = DATA_DIR / "minutes_overrides.json"
# manual-XI escape hatch: forces a match 'confirmed' with a hand-entered starting XI
# (resolved to board norm_names + minutes) when API-Football's lineup feed lags the
# real announcement. Auto-expires at kickoff, same as the minutes overrides.
MANUAL_XI_JSON = DATA_DIR / "manual_xi.json"

# Hand-maintained name aliases for book<->model<->Kalshi seams that NO string matcher can
# bridge — different given names for the same player (bet365 'Kouadio Kone' vs API-Football
# 'Manu Kone', both = Kouadio Manu Kone). Flat JSON map of norm_name -> canonical norm_name;
# the board treats the two as the same player. Extend it as new seams surface.
NAME_ALIASES_JSON = DATA_DIR / "name_aliases.json"

# Scorers/Kalshi board live-price blend: the board combines bet365 and Underdog ANYTIME
# hazards BEFORE the minutes engine prices them. Both are "given he features" numbers (UD
# voids on a no-play, same as the bet365 anytime book), so they blend on the same basis.
# Only players Underdog actually covers for that stat get the UD share; everyone else stays
# 100% bet365. Default 65/35 b365/UD (UD has a sharp internal model but thin coverage).
BOARD_BLEND_W_UD = float(os.environ.get("WCP_BOARD_W_UD", "0.35"))
BOARD_BLEND_W_B365 = 1.0 - BOARD_BLEND_W_UD
# One-sided UD line handling (only 'higher' posted, no opposing side to de-vig against): we
# still use it — it implies a price — but assume it's vigged ~UD_ONEWAY_VIG_MULT× a balanced
# two-way UD book and de-vig by that assumed overround. The balanced overround is measured from
# the slate's two-sided UD lines each run, falling back to UD_BASELINE_OVERROUND when too few.
UD_ONEWAY_VIG_MULT = float(os.environ.get("WCP_UD_ONEWAY_VIG_MULT", "1.5"))
UD_BASELINE_OVERROUND = float(os.environ.get("WCP_UD_BASELINE_OVERROUND", "1.06"))

# Trade ledger (trades.py): every Kalshi order we actually PLACE is appended to the
# JSONL with its full pricing context (our fair, price paid, mode, source, minutes +
# raw hazard assumed, live book) so we can later match each take to the realized
# match result and score our edges. Append-only — never rewritten. resolve() joins
# outcomes and writes the settled view (parquet + a CSV for eyeballing).
TRADES_LOG = DATA_DIR / "trades" / "fills.jsonl"
TRADES_SETTLED = DATA_DIR / "trades" / "settled.parquet"
TRADES_CSV = OUT_DIR / "trades.csv"
