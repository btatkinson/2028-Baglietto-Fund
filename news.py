"""LLM injury/news veto for the scorers board — the one signal the numbers can't see.

The whole pipeline (Elos → rate heads → minutes prior → de-vig → Kalshi maker quote)
is blind to real-world news: a warmup knock, an illness, a late suspension, a "manager
hints at rotation". That's exactly where we'd confidently rest a price that's miles off
and get picked off — the worst failure mode for a maker. This module closes that gap.

Design (deliberate, see config.CLAUDE_API_KEY):
  * Scope tight — only post-XI rows we'd actually act on (a real, non-noise maker quote),
    top N per side by hazard, hard-capped per slate. Keeps cost and false positives down.
  * Veto/flag, never a price input — the LLM's probabilities aren't calibrated, so its
    verdict can WARN a quote ("⚠ doubtful — Afif knock") or SUPPRESS it (status=out), but
    it never numerically moves the fair value. Same-day knocks and late withdrawals after
    the XI is posted are the high-value catch: that's when a Kalshi order is already resting.
  * Structured + sourced — per player {status, confidence, reason, source_url}; the model
    must search the web first and cite a URL, and is told to return `unknown` (not guess)
    when it can't find specific recent news. Hallucination/staleness guardrail.
  * Cross-check vs our minutes — the loudest flag is "out/doubtful" while we project a near
    full match (a player we'd quote big on); rendered with extra emphasis (conflict).

Verdicts are written onto the scorers row dicts in place (`news`, `news_conflict`,
`news_suppress`) for scorers.py to render. Cached to data/news_cache.json so re-runs (the
normal "re-run to pick up newly-listed players" flow) don't re-pay for unchanged players.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import config
from normalize import norm_name

# Above this fair anytime prob we treat a confirmed-XI book player as one we'd quote, so
# worth a news check (mirrors KALSHI_MIN_QUOTE_PROB — below it we never rest an order).
_ACT_MIN_PROB = config.KALSHI_MIN_QUOTE_PROB
_FULL_MATCH_MIN = 80.0            # "we project a near-full match" → an out/doubtful here is loud

_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["starts", "doubtful", "out", "unknown"],
                   "description": "Will the player be available and start THIS match?"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string",
                   "description": "One concise line (e.g. 'knock in training, manager said doubtful'). Empty string if unknown."},
        "source_url": {"type": "string",
                       "description": "URL of the single most relevant source. Empty string only if status is unknown."},
    },
    "required": ["status", "confidence", "reason", "source_url"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a football (soccer) team-news researcher for a World Cup betting maker. "
    "Given ONE named player and a specific upcoming match, search the web for the most "
    "recent, reliable news on that player's availability for THAT match, then judge whether "
    "he will START, is DOUBTFUL, is OUT (injury, illness, suspension, or simply not selected), "
    "or is UNKNOWN.\n"
    "Rules:\n"
    "- Search first. Prefer beat reporters, official club/national-team accounts, and major "
    "outlets; weight sources dated close to the match over older ones.\n"
    "- If you cannot find specific, recent news about THIS player and match, return "
    "status='unknown' with an empty reason and source_url. Do NOT guess from reputation, "
    "form, or general priors — silence is a valid, useful answer.\n"
    "- Be conservative about 'out'/'doubtful': only when a credible source actually says so.\n"
    "- Keep reason to one concise line, and include the single most relevant source URL.\n"
    "Finish by calling report_status exactly once with your verdict."
)

_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 5},
    {"name": "report_status",
     "description": "Record your final availability verdict for the player. Call exactly once.",
     "strict": True, "input_schema": _STATUS_SCHEMA},
]


def _kickoff_str(ko) -> str:
    try:
        return datetime.fromtimestamp(int(ko), tz=timezone.utc).strftime("%a %d %b %Y %H:%M UTC")
    except (TypeError, ValueError, OverflowError):
        return "soon"


def _query(client, player, team, home, away, kostr, minutes) -> dict | None:
    """One player → verdict dict, or None on failure. Lets the model web-search freely
    (server-side) and read its final structured verdict off the report_status tool call."""
    mins = f"~{minutes:.0f}" if isinstance(minutes, (int, float)) else "a full"
    prompt = (
        f"Match: {home} vs {away}, kickoff {kostr}.\n"
        f"Player: {player} ({team}).\n"
        f"Our model projects he plays {mins} minutes, so we treat him as a likely starter "
        f"and may quote a price on him. Find the latest team-news on his availability for "
        f"THIS match and report it."
    )
    messages = [{"role": "user", "content": prompt}]
    for _ in range(6):                       # bounded: search rounds + the final verdict
        try:
            resp = client.messages.create(
                model=config.CLAUDE_MODEL, max_tokens=2500,
                thinking={"type": "adaptive"}, output_config={"effort": "low"},
                system=_SYSTEM, tools=_TOOLS, tool_choice={"type": "auto"},
                messages=messages)
        except Exception as e:               # network/rate/refusal — fail soft, no flag
            print(f"    news: API error for {player}: {e}")
            return None
        for b in resp.content:
            if b.type == "tool_use" and b.name == "report_status":
                return dict(b.input)
        if resp.stop_reason == "pause_turn":  # hit the server tool-iteration cap — resume
            messages.append({"role": "assistant", "content": resp.content})
            continue
        if resp.stop_reason == "end_turn":    # answered without the tool — nudge once
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": "Now call report_status with your verdict."})
            continue
        break                                 # unexpected stop (e.g. refusal) — give up
    return None


# ------------------------------- cache ---------------------------------------

def _cache_path():
    return config.DATA_DIR / "news_cache.json"


def _load_cache() -> dict:
    try:
        return json.loads(_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict):
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path().write_text(json.dumps(cache, indent=0), encoding="utf-8")
    except OSError:
        pass


# ----------------------------- selection -------------------------------------

def _act_on_players(games) -> list[dict]:
    """The post-XI rows worth a news check: confirmed-XI book players with a real
    (non-noise) maker quote, top NEWS_TOP_PER_SIDE per side by hazard, capped per slate.
    Each item carries the row dict itself so verdicts can be written back in place."""
    picks = []
    for gm in games:
        if gm.get("lineup_kind") != "confirmed":     # we only quote — and act — post-XI
            continue
        teams = [s.get("team") for s in gm.get("sides", [])]
        home = teams[0] if len(teams) > 0 else ""
        away = teams[1] if len(teams) > 1 else ""
        for s in gm.get("sides", []):
            team = s.get("team", "")
            opp = away if team == home else home
            cands = []
            for r in s.get("book", []):
                if r.get("source") != "bet365":
                    continue
                best = max(r.get("g_p") or 0, r.get("ga_p") or 0)
                if best < _ACT_MIN_PROB:             # noise — we'd never rest an order on it
                    continue
                cands.append((r.get("g_haz") or 0, r))
            cands.sort(key=lambda t: -t[0])
            for _, r in cands[:config.NEWS_TOP_PER_SIDE]:
                picks.append({"match": gm["match"], "kickoff": gm.get("kickoff"),
                              "home": team if team == home else opp,
                              "away": opp if team == home else team,
                              "team": team, "row": r,
                              "player": r["player"], "nname": norm_name(r["player"]),
                              "minutes": r.get("minutes")})
    return picks[:config.NEWS_MAX_PLAYERS]


def _apply(row, flag):
    """Write a verdict onto a scorers row: store it, mark the minutes-conflict, and
    suppress the maker quote outright when the player is reported out."""
    row["news"] = flag
    status = flag.get("status")
    mins = row.get("minutes")
    near_full = isinstance(mins, (int, float)) and mins >= _FULL_MATCH_MIN
    row["news_conflict"] = status in ("out", "doubtful") and near_full
    row["news_suppress"] = status == "out" and flag.get("confidence") in ("medium", "high")


# ------------------------------- entry ---------------------------------------

def annotate(games) -> str | None:
    """Run the news check over the act-on players in `games`, writing verdicts onto the
    row dicts in place. No-ops (and makes no API call) when there's no key or nothing to
    check. Returns a one-line summary for the run log, or None when skipped."""
    if not config.CLAUDE_API_KEY:
        return None
    picks = _act_on_players(games)
    if not picks:
        return None

    try:
        import anthropic
    except ImportError:
        return "news check skipped: `pip install anthropic`"
    client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)

    cache, now = _load_cache(), time.time()

    def resolve(p):
        key = f"{p['match']}|{p['nname']}"
        hit = cache.get(key)
        if hit and (now - hit.get("ts", 0)) < config.NEWS_CACHE_TTL:
            return p, hit["flag"], True
        flag = _query(client, p["player"], p["team"], p["home"], p["away"],
                      _kickoff_str(p["kickoff"]), p["minutes"])
        if flag is not None:
            cache[key] = {"ts": now, "flag": flag}
        return p, flag, False

    flagged = n_cached = 0
    with ThreadPoolExecutor(max_workers=config.NEWS_MAX_WORKERS) as ex:
        for p, flag, cached in ex.map(resolve, picks):
            if cached:
                n_cached += 1
            if flag is None:
                continue
            _apply(p["row"], flag)
            if flag.get("status") in ("out", "doubtful"):
                flagged += 1

    _save_cache(cache)
    return (f"News check: {len(picks)} players ({n_cached} cached), "
            f"{flagged} flagged doubtful/out")
