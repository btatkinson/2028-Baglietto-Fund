"""Interactive scorers-board controller for the live web UI (webapp.py /scorers).

The CLI pipeline (run.py) writes a *static* board. This module turns it interactive without
re-running the model: it caches the last pipeline's (b365, proj) — in memory and as a pickle so
a fresh webapp process or a separate CLI run can pick it up — and re-prices the board in-process
when the user hand-edits a player's minutes. Manual minutes overrides persist to JSON and
auto-expire once their match has kicked off.

It also serves a single Kalshi order ticket: given a clicked (player, market, side) it re-pulls
that one market fresh, sizes it with the existing fractional-Kelly logic, and hands the webapp a
ready-to-place order (expiration = kickoff). No pricing or trading math lives here — it composes
scorers.build / kalshi_overlay.verdict / kalshi_orders._kelly_size / kalshi.create_order.
"""
from __future__ import annotations

import json
import math
import pickle
import time
from datetime import datetime, timezone

import config
import kalshi
import kalshi_overlay
import kalshi_orders
import scorers
import trades
from normalize import norm_name

try:
    from projection import mkey
except Exception:                       # pragma: no cover - projection should import fine
    def mkey(_m):
        return None

# market label -> the per-row probability key on a board book row
_MARKET_PK = {"G": "g_p", "A": "a_p", "G+A": "ga_p"}
# the "nailed starter" placeholder a manual XI assigns; on handoff to a real API XI these
# defer to the model, while any DELIBERATE minute (≠ default, e.g. an early-sub call)
# migrates into the per-player tweak store so it survives on top of the model's XI.
MANUAL_DEFAULT_MIN = 90.0

_STATE: dict | None = None              # {"b365", "proj", "krows", "ts"} cached in-process


# ------------------------------- pipeline state -------------------------------

def save_state(b365, proj, krows=None, ud=None):
    """Cache the inputs needed to re-price the board (in memory + pickle). Called at the end of
    run.pipeline so both a CLI run and the web process can rebuild without re-running the model.
    `ud` is the parsed Underdog frame so the web process can re-blend the 65/35 board price."""
    global _STATE
    _STATE = {"b365": b365, "proj": proj, "krows": krows, "ud": ud, "ts": time.time()}
    try:
        config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.BOARD_STATE_PKL, "wb") as fh:
            pickle.dump(_STATE, fh)
    except Exception as e:
        print(f"  board: could not persist state ({e})")


def load_state() -> dict | None:
    """The cached (b365, proj, krows). In-memory first, else the pickle (different process)."""
    global _STATE
    if _STATE is not None:
        return _STATE
    try:
        with open(config.BOARD_STATE_PKL, "rb") as fh:
            _STATE = pickle.load(fh)
    except Exception:
        return None
    return _STATE


def set_book(krows):
    """Update the cached live Kalshi book (after a fresh overlay pull) and persist it."""
    st = load_state() or {}
    st["krows"] = krows
    globals()["_STATE"] = st
    try:
        with open(config.BOARD_STATE_PKL, "wb") as fh:
            pickle.dump(st, fh)
    except Exception as e:
        print(f"  board: could not persist book ({e})")


# ------------------------------- minutes overrides ----------------------------

def _okey(match, player):
    mk = mkey(match)
    return f"{mk}|{norm_name(player)}" if mk else None


def get_overrides() -> dict:
    """{'mk|nname': minutes} of live overrides, dropping any whose match has kicked off
    (auto-expire). Rewrites the file when it prunes so stale entries don't accumulate."""
    try:
        raw = json.loads(config.MINUTES_OVERRIDES_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    now = time.time()
    live, pruned = {}, False
    for k, v in raw.items():
        ko = v.get("kickoff")
        if ko is not None and ko < now:                 # match started — override is spent
            pruned = True
            continue
        live[k] = v
    if pruned:
        _write_overrides(live)
    return {k: v["minutes"] for k, v in live.items() if v.get("minutes") is not None}


def _read_raw() -> dict:
    try:
        return json.loads(config.MINUTES_OVERRIDES_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_overrides(raw: dict):
    config.MINUTES_OVERRIDES_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.MINUTES_OVERRIDES_JSON.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def set_override(match, player, minutes, kickoff=None):
    """Persist (or clear, if minutes is None) a manual minutes override for one player."""
    key = _okey(match, player)
    if not key:
        return False
    raw = _read_raw()
    if minutes is None:
        raw.pop(key, None)
    else:
        raw[key] = {"minutes": float(minutes), "kickoff": kickoff,
                    "match": str(match), "player": str(player)}
    _write_overrides(raw)
    return True


# ------------------------------- manual XI (force-confirmed) -------------------

def get_manual_xi() -> dict:
    """{mk: {nname: minutes}} of hand-entered XIs, dropping any whose match has kicked
    off (auto-expire). These force the match 'confirmed' AND set each starter's minutes;
    any board player NOT in the XI prices to 0 (post-confirmed factor 0)."""
    try:
        raw = json.loads(config.MANUAL_XI_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    now = time.time()
    live, pruned = {}, False
    for mk, v in raw.items():
        ko = v.get("kickoff")
        if ko is not None and ko < now:
            pruned = True
            continue
        live[mk] = v
    if pruned:
        config.MANUAL_XI_JSON.write_text(json.dumps(live, indent=2), encoding="utf-8")
    return {mk: v.get("minutes", {}) for mk, v in live.items()}


def set_manual_xi(match, minutes_by_nname, kickoff=None):
    """Persist (or clear, if minutes_by_nname is falsy) a manual XI for one match. Keys are
    board norm_names already resolved from the entered surnames."""
    mk = mkey(match)
    if not mk:
        return False
    try:
        raw = json.loads(config.MANUAL_XI_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not minutes_by_nname:
        raw.pop(mk, None)
    else:
        raw[mk] = {"minutes": {k: float(v) for k, v in minutes_by_nname.items()},
                   "kickoff": kickoff, "match": str(match)}
    config.MANUAL_XI_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.MANUAL_XI_JSON.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return True


def _supersede_manual(mks):
    """The API now has a real confirmed XI for these matches → hand back to it: drop each
    manual XI, but first MIGRATE any deliberate (non-default) minute into the per-player
    tweak store so it still layers on top of the model's XI (the default-90 'starts'
    markers are dropped — the model prices those starters itself). Won't clobber a tweak the
    user already set on that player."""
    if not mks:
        return
    try:
        raw = json.loads(config.MANUAL_XI_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    tweaks = _read_raw()
    drop_manual = migrated = False
    for mk in mks:
        entry = raw.pop(mk, None)
        if entry is None:
            continue
        drop_manual = True
        ko, label = entry.get("kickoff"), entry.get("match")
        for nn, mins in (entry.get("minutes") or {}).items():
            if float(mins) == MANUAL_DEFAULT_MIN:
                continue                              # routine starter — let the API price it
            okey = f"{mk}|{nn}"
            if okey not in tweaks:                    # don't overwrite a user tweak
                tweaks[okey] = {"minutes": float(mins), "kickoff": ko,
                                "match": label, "player": nn}
                migrated = True
    if drop_manual:
        config.MANUAL_XI_JSON.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        print(f"  board: manual XI superseded by real API XI for {list(mks)}"
              + (" (deliberate minutes migrated to tweaks)" if migrated else ""))
    if migrated:
        _write_overrides(tweaks)


def clear_manual_xi(match) -> str:
    """Drop the manual XI for one match (hand back to the model) and return the re-priced
    board fragment for the web UI."""
    set_manual_xi(match, None)
    return render_fragment(refresh_kalshi=False)


# ------------------------------- rebuild --------------------------------------

def _recompute_krows(games, cached):
    """Re-derive take/make verdicts from the CACHED live book against the board's freshly
    re-priced acceptable prices (after a minutes edit). Cheap — no Kalshi call."""
    book = {(r["match"], norm_name(r["player"]), r["market"]): r for r in (cached or [])}
    out = []
    for gm in games:
        if gm.get("lineup_kind") not in ("confirmed", "projected"):   # view-only pre-XI too
            continue
        for s in gm.get("sides", []):
            for r in s.get("book", []) + s.get("model", []):     # model-est rows overlay too
                if r.get("news_suppress"):
                    continue
                nn = norm_name(r["player"])
                for market, pk in _MARKET_PK.items():
                    base = book.get((gm["match"], nn, market))
                    if not base:
                        continue
                    acc_yes, acc_no = scorers._maker(r.get(pk), True)
                    kp = {"yes_bid": base["k_yes_bid"], "yes_ask": base["k_yes_ask"],
                          "no_bid": base["k_no_bid"], "no_ask": base["k_no_ask"],
                          "last": base["k_last"], "oi": base["k_oi"], "ticker": base["ticker"]}
                    v = kalshi_overlay.verdict(gm["match"], r["player"], market, acc_yes, acc_no, kp)
                    if v:
                        out.append(v)
    return out


def rebuild(refresh_kalshi=False):
    """Re-price the board from cached state + current minutes overrides, attach the Kalshi
    overlay (fresh pull if refresh_kalshi, else recomputed against the cached book), and return
    the games. Returns [] when no pipeline has run yet (no cached state)."""
    st = load_state()
    if not st or st.get("b365") is None:
        return []
    proj = st.get("proj")
    # a match the model now reports a REAL confirmed XI for supersedes any manual XI we
    # forced earlier (API lineup lag caught up) — hand back to the better data automatically.
    real_confirmed = {mkey(f.get("match")) for f in (proj or {}).get("fixtures", [])
                      if f.get("lineup_kind") == "confirmed"}
    real_confirmed.discard(None)
    manual = get_manual_xi()
    superseded = [mk for mk in manual if mk in real_confirmed]
    if superseded:
        _supersede_manual(superseded)                 # migrate deliberate minutes → tweaks, drop
        manual = {mk: mp for mk, mp in manual.items() if mk not in real_confirmed}
    # manual XI (if any still active) forces those matches confirmed and seeds each starter's
    # minutes; per-player tweak overrides layer on top (so editing a Min cell still wins).
    overrides = {f"{mk}|{nn}": mins for mk, mp in manual.items() for nn, mins in mp.items()}
    overrides.update(get_overrides())
    games = scorers.build(st["b365"], proj, overrides=overrides,
                          force_confirmed=set(manual.keys()), ud=st.get("ud"))
    # attach the live Kalshi overlay once we have ANY lineup (projected or confirmed). Pre-XI
    # is VIEW-ONLY: cells light up so you can see edges, but board.ticket/place gate the actual
    # order to a confirmed XI.
    has_lineup = any(gm.get("lineup_kind") in ("confirmed", "projected") for gm in games)
    if has_lineup and config.KALSHI_KEY_ID:
        if refresh_kalshi:
            try:
                csv_path = scorers.write_csv(games)
                krows = kalshi_overlay.build(csv_path)
                set_book(krows)
                scorers._attach_overlay(games, krows)
            except Exception as e:
                print(f"  board: Kalshi refresh failed ({e})")
        elif st.get("krows"):
            scorers._attach_overlay(games, _recompute_krows(games, st["krows"]))
    return games


def render(interactive=True, refresh_kalshi=False) -> str:
    """Full interactive board HTML for the current state."""
    games = rebuild(refresh_kalshi=refresh_kalshi)
    if not games:
        return ""
    return scorers.build_html(games, interactive=interactive)


def render_fragment(refresh_kalshi=False) -> str:
    """Just the swappable board sections (returned to the JS after a minutes/refresh edit)."""
    games = rebuild(refresh_kalshi=refresh_kalshi)
    return scorers._sections(games, interactive=True)


def apply_minutes(match, player, minutes) -> str:
    """Set/clear a minutes override (resolving the match kickoff so it can auto-expire), then
    return the re-priced board fragment for the web UI to swap in."""
    ko = None
    for gm in rebuild(refresh_kalshi=False):
        if str(gm["match"]) == str(match):
            ko = gm.get("kickoff")
            break
    if ko is not None and math.isinf(ko):
        ko = None
    set_override(match, player, None if minutes is None else float(minutes), kickoff=ko)
    return render_fragment(refresh_kalshi=False)


# ------------------------------- order ticket ---------------------------------

def _find_row(games, match, player):
    nn = norm_name(player)
    for gm in games:
        if str(gm["match"]) != str(match):
            continue
        for s in gm.get("sides", []):
            for r in s.get("book", []) + s.get("model", []):
                if norm_name(r["player"]) == nn:
                    return gm, r
    return None, None


def ticket(match, player, market, side) -> dict:
    """Build a single Kalshi order ticket for a clicked board cell. Re-pulls that one market
    fresh, sizes it with the existing fractional-Kelly logic, expiration = kickoff. Raises
    ValueError with a human message if the cell is no longer actionable."""
    if market not in _MARKET_PK or side not in ("yes", "no"):
        raise ValueError("bad market/side")
    games = rebuild(refresh_kalshi=False)
    gm, row = _find_row(games, match, player)
    if not row:
        raise ValueError("player no longer on the board (re-run the pipeline?)")
    ov = (row.get("kalshi") or {}).get(market)
    if not ov or not ov.get("ticker"):
        raise ValueError("no live Kalshi market for this cell")
    ticker = ov["ticker"]

    q = row.get(_MARKET_PK[market])                      # our fair YES prob for this market
    if q is None:
        raise ValueError("no fair price for this market")
    acc_yes, acc_no = scorers._maker(q, True)

    # fresh single-market snapshot (the cached book may be minutes old)
    try:
        m = kalshi.market(ticker).get("market", {})
        kp = {"yes_bid": kalshi_overlay._cents(m.get("yes_bid_dollars")),
              "yes_ask": kalshi_overlay._cents(m.get("yes_ask_dollars")),
              "no_bid": kalshi_overlay._cents(m.get("no_bid_dollars")),
              "no_ask": kalshi_overlay._cents(m.get("no_ask_dollars")),
              "last": kalshi_overlay._cents(m.get("last_price_dollars")),
              "oi": float(m.get("open_interest_fp") or m.get("open_interest") or 0),
              "ticker": ticker}
    except Exception as e:
        raise ValueError(f"could not read live market ({e})")

    v = kalshi_overlay.verdict(match, player, market, acc_yes, acc_no, kp) or {}
    prob = q if side == "yes" else (1 - q)
    take = v.get(f"take_{side}")
    bid = kp["yes_bid"] if side == "yes" else kp["no_bid"]
    ask = kp["yes_ask"] if side == "yes" else kp["no_ask"]
    if take:
        price_c = ask                                   # cross the live book (taker)
        mode, post_only = "TAKE", False
    else:
        # MAKE: rest a bid that improves the book. Price it off the cheaper MAKER fee, then
        # penny the best bid (best_bid + 1) so we claim top priority while conceding only 1¢
        # of edge — capped at our acceptable price and kept strictly below the ask. post_only
        # so it can never silently cross and pay the taker fee.
        am_yes, am_no = scorers._maker(
            q, True, cushion=config.KALSHI_EDGE_CUSHION + config.KALSHI_LIQUIDITY_PREMIUM,
            rate=config.KALSHI_MAKER_FEE_RATE)
        ceil_c = am_yes if side == "yes" else am_no
        if ceil_c is None:
            raise ValueError("no quotable price on this side right now")
        price_c = (bid + 1) if bid is not None else ceil_c
        price_c = min(price_c, ceil_c)
        if ask is not None:
            price_c = min(price_c, ask - 1)             # never cross
        # discipline: only rest if we can actually IMPROVE the bid within the premium-adjusted
        # ceiling — otherwise the edge is too thin to make (take it or skip).
        if bid is not None and price_c <= bid:
            raise ValueError("make too thin after the liquidity premium — take it or skip")
        mode, post_only = "make", True
    if price_c is None:
        raise ValueError("no quotable price on this side right now")
    price_c = max(1, min(99, int(price_c)))

    sized = kalshi_orders._kelly_size(prob, price_c)
    count = sized[0] if sized else 1                    # editable in the modal regardless
    risk, payout = (sized[1], sized[2]) if sized else (round(count * price_c / 100, 2), float(count))

    ko = gm.get("kickoff")
    has_ko = ko and not math.isinf(ko)
    exp_ts = int(ko) if has_ko else None
    exp_str = (datetime.fromtimestamp(int(ko), tz=timezone.utc).strftime("kickoff %H:%MZ")
               if has_ko else "good-till-cancel")

    # VIEW-ONLY pre-XI: we still compute and return the full edge/sizing so you can see it, but
    # the order is only placeable on a confirmed XI (the UI disables Place; place() enforces it).
    tradable = gm.get("lineup_kind") == "confirmed"
    return {
        "match": str(match), "player": str(player), "market": market, "side": side,
        "ticker": ticker, "mode": mode, "source": row.get("source"), "post_only": post_only,
        "tradable": tradable, "lineup_kind": gm.get("lineup_kind"),
        "price_c": price_c, "count": int(count), "risk": risk, "payout": payout,
        "fair": prob, "edge_c": round((prob - price_c / 100) * 100, 1),
        "expiration_ts": exp_ts, "exp_str": exp_str,
        "k_yes_bid": kp["yes_bid"], "k_yes_ask": kp["yes_ask"],
        "k_no_bid": kp["no_bid"], "k_no_ask": kp["no_ask"], "k_oi": round(kp["oi"])}


def _context_for_ticker(ticker):
    """Best-effort: find the board row + market whose live Kalshi ticker matches `ticker`, so a
    placed order can be logged with its full pricing context (match/player/source/minutes/raw
    hazard/our fair + the book at take time). Returns {} if the board can't be rebuilt or the
    ticker isn't on it — logging then falls back to whatever the caller passed."""
    haz_key = {"G": "g_haz", "A": "a_haz", "G+A": None}
    try:
        games = rebuild(refresh_kalshi=False)
    except Exception:
        return {}
    for gm in games:
        for s in gm.get("sides", []):
            for r in s.get("book", []) + s.get("model", []):
                kx = r.get("kalshi") or {}
                for market, pk in _MARKET_PK.items():
                    ov = kx.get(market)
                    if not ov or ov.get("ticker") != ticker:
                        continue
                    hk = haz_key.get(market)
                    return {
                        "match": gm.get("match"), "player": r.get("player"),
                        "market": market, "source": r.get("source"),
                        "lineup_kind": gm.get("lineup_kind"),
                        "minutes": r.get("minutes"),
                        "hazard": r.get(hk) if hk else None,
                        "q_yes": r.get(pk), "kickoff": gm.get("kickoff"),
                        "k_yes_bid": ov.get("k_yes_bid"), "k_yes_ask": ov.get("k_yes_ask"),
                        "k_no_bid": ov.get("k_no_bid"), "k_no_ask": ov.get("k_no_ask")}
    return {}


def place(ticker, side, price_c, count, expiration_ts=None, post_only=False) -> dict:
    """Place one limit BUY on Kalshi (the only write path the web UI reaches). Validates the
    inputs, builds the same idempotency key the CLI uses, and returns {'order_id': ...}.
    post_only=True (a resting 'make') rejects rather than crosses, so it can't pay taker fees."""
    side = str(side)
    if side not in ("yes", "no"):
        raise ValueError("side must be yes/no")
    price_c, count = int(price_c), int(count)
    if not (1 <= price_c <= 99):
        raise ValueError("price must be 1-99c")
    if count < 1:
        raise ValueError("count must be >= 1")
    # context re-derived from the board by ticker — used to GATE (pre-XI is view-only) and to
    # log the take's full pricing context. Trading is confirmed-XI only: refuse a projected
    # board here too (defense-in-depth behind the disabled Place button in the UI).
    ctx = _context_for_ticker(ticker)
    if ctx.get("lineup_kind") and ctx["lineup_kind"] != "confirmed":
        raise ValueError("order placement is gated to a confirmed XI - the board is view-only "
                         "until the lineup is confirmed")
    coid = kalshi_orders._client_order_id(ticker, side, price_c, expiration_ts)
    kw = {"ticker": ticker, "action": "buy", "side": side, "count": count,
          "type": "limit", "client_order_id": coid, "post_only": bool(post_only)}
    kw["yes_price" if side == "yes" else "no_price"] = price_c
    if expiration_ts:
        kw["expiration_ts"] = int(expiration_ts)
    resp = kalshi.create_order(**kw)
    # v2 returns order_id at the top level; legacy wrapped it in {"order": {...}} — handle both
    oid = resp.get("order_id") or (resp.get("order") or {}).get("order_id")
    # record the take — never let a logging hiccup undo a placed order (lineup_kind isn't a
    # log_fill field, so drop it from the spread).
    log_ctx = {k: v for k, v in ctx.items() if k != "lineup_kind"}
    trades.log_fill(ticker=ticker, side=side, price_c=price_c, count=count, order_id=oid,
                    client_order_id=coid, mode=("make" if post_only else "TAKE"),
                    post_only=post_only, **log_ctx)
    return {"order_id": oid, "client_order_id": coid}
