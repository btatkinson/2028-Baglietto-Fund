"""Live Kalshi order placement - fractional-Kelly sizing, expire-at-kickoff, human-gated.

This is the only module that places real orders. It is reached ONLY from `run.py --trade`,
and EVERY order is shown and individually approved at a `python input()` prompt before it is
sent - nothing rests on Kalshi without a keystroke. It consumes the scorers `games` after the
live overlay is attached (each book row carries our fair prob, the live verdict + book, the
market ticker, and the kickoff), so it sees exactly what the board shows.

Sizing (config.KALSHI_*): for an actionable side we pay `price` (the live ask when crossing a
TAKE, else our resting acceptable price for a make). With our fair prob `q'` for that side, the
edge is `q' - price` and the full-Kelly capital-at-risk fraction is `f* = edge/(1-price)`. We
stake `KELLY_FRACTION * f* * BANKROLL`, capped at `MAX_RISK` per market (and an optional
contract cap). Contracts = floor(risk/price); payout = contracts*$1 = risk/price, so the same
risk buys a bigger payout on a longshot than on a favourite.

Safety rails: post-XI confirmed games only; news-suppressed players are already gone from the
board; `expiration_ts = kickoff` (Kalshi cancels the rest at the whistle); a deterministic
client_order_id makes a re-run idempotent (Kalshi dedupes); orders whose kickoff has passed are
dropped. We never SELL and never place market orders here - buys, limit only.
"""
from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timezone

import config
import kalshi

_EXP_BUFFER = 60          # don't bother placing if kickoff is under a minute away


def _kelly_size(prob, price_c):
    """(contracts, risk$, payout$) for buying a side at price_c¢ with our fair prob `prob`.
    Fractional Kelly capped by MAX_RISK (and MAX_CONTRACTS if set). None if no positive edge."""
    p = price_c / 100.0
    edge = prob - p
    if edge <= 0 or p <= 0 or p >= 1:
        return None
    f = edge / (1 - p)                                   # full-Kelly capital-at-risk fraction
    risk = min(config.KALSHI_KELLY_FRACTION * f * config.KALSHI_BANKROLL, config.KALSHI_MAX_RISK)
    count = int(risk // p)                               # whole contracts at p$ cost each
    if config.KALSHI_MAX_CONTRACTS:
        count = min(count, config.KALSHI_MAX_CONTRACTS)
    if count < 1:
        return None
    return count, round(count * p, 2), round(count * 1.0, 2)


def _client_order_id(ticker, side, price_c, exp):
    """Deterministic idempotency key: same intended order across re-runs → same id, so Kalshi
    won't double-place. Scoped by expiration (= kickoff), so it's naturally per-match."""
    raw = f"{ticker}|{side}|{price_c}|{exp}"
    return "wcp-" + hashlib.sha1(raw.encode()).hexdigest()[:24]


def build_orders(games):
    """Intended buy orders from the attached board. One per actionable (player, market, side):
    cross at the live ask for a TAKE, else rest at our acceptable price for a make."""
    now = time.time()
    orders = []
    for gm in games:
        if gm.get("lineup_kind") != "confirmed":         # quotes/orders are post-XI only
            continue
        ko = gm.get("kickoff")
        exp = int(ko) if (ko and not math.isinf(ko)) else None
        if exp is not None and exp - now < _EXP_BUFFER:  # game basically started - skip
            continue
        for s in gm.get("sides", []):
            for r in s.get("book", []) + s.get("model", []):     # model-est rows quote too
                if r.get("news_suppress"):
                    continue
                kx = r.get("kalshi") or {}
                for mkt, qkey in (("G", "g_p"), ("A", "a_p"), ("G+A", "ga_p")):
                    ov = kx.get(mkt)
                    q = r.get(qkey)
                    if not ov or q is None:
                        continue
                    for side, prob, tkey, mkey, acc_key, ask_key in (
                            ("yes", q, "take_yes", "make_yes", "acc_yes", "k_yes_ask"),
                            ("no", 1 - q, "take_no", "make_no", "acc_no", "k_no_ask")):
                        take, make = ov.get(tkey), ov.get(mkey)
                        if not (take or make):
                            continue
                        price_c = ov.get(ask_key) if take else ov.get(acc_key)  # cross vs rest
                        if price_c is None:
                            continue
                        sized = _kelly_size(prob, int(price_c))
                        if not sized:
                            continue
                        count, risk, payout = sized
                        haz = r.get("g_haz") if mkt == "G" else (r.get("a_haz") if mkt == "A" else None)
                        orders.append({
                            "ticker": ov["ticker"], "side": side, "action": "buy",
                            "price_c": int(price_c), "count": count, "expiration_ts": exp,
                            "client_order_id": _client_order_id(ov["ticker"], side, int(price_c), exp),
                            "match": gm["match"], "player": r["player"], "market": mkt,
                            "kickoff": ko, "mode": "TAKE" if take else "make",
                            "fair": prob, "edge_c": round((prob - price_c / 100.0) * 100, 1),
                            "risk": risk, "payout": payout,
                            # context carried through for the trade ledger (trades.log_fill)
                            "source": r.get("source"), "minutes": r.get("minutes"), "hazard": haz,
                            "q_yes": q, "k_yes_bid": ov.get("k_yes_bid"), "k_yes_ask": ov.get("k_yes_ask"),
                            "k_no_bid": ov.get("k_no_bid"), "k_no_ask": ov.get("k_no_ask")})
    # best price FIRST: biggest edge wins so the juiciest opportunities get the cap budget
    # before any thin make; a TAKE (certain fill) breaks ties over a make (speculative rest).
    orders.sort(key=lambda o: (-o["edge_c"], o["mode"] != "TAKE"))
    return orders


def _resting_state():
    """Read our live Kalshi state for the cumulative cap. Returns
    (resting {(ticker,side): [{order_id,count,price_c,ours}]}, pos_risk {ticker: $ filled},
    ok). ok=False if a read failed (caller warns — don't silently assume zero exposure)."""
    resting, pos, ok = {}, {}, True
    try:
        for o in kalshi.orders(status="resting", limit=1000).get("orders", []):
            if o.get("action") != "buy":
                continue
            side, cnt = o.get("side"), o.get("remaining_count") or 0
            price = o.get("yes_price") if side == "yes" else o.get("no_price")
            if not (side and cnt and price):
                continue
            resting.setdefault((o["ticker"], side), []).append(
                {"order_id": o.get("order_id"), "count": cnt, "price_c": price,
                 "ours": (o.get("client_order_id") or "").startswith("wcp-")})
    except Exception as e:
        ok = False
        print(f"   reconcile: could not read resting orders ({e})")
    try:
        for p in kalshi.positions(limit=1000).get("market_positions", []):
            exp = p.get("market_exposure") or 0      # cents of filled capital-at-risk
            if exp:
                pos[p["ticker"]] = exp / 100.0
    except Exception as e:
        ok = False
        print(f"   reconcile: could not read positions ({e})")
    return resting, pos, ok


def reconcile(orders):
    """Trim each intended order to the cumulative per-market (ticker) MAX_RISK across our
    existing FILLED positions + the resting orders we'll keep, and flag our own stale resting
    order on the same (ticker, side) for cancel-replace (so a re-run refreshes rather than
    stacks). Sets count=0 on anything already at the cap. Returns (orders, ok)."""
    resting, pos, ok = _resting_state()
    intended_sides = {(o["ticker"], o["side"]) for o in orders}
    t2p = {o["ticker"]: (o["match"], o["player"]) for o in orders}    # ticker -> (match, player)
    player_cap = config.KALSHI_MAX_RISK * config.KALSHI_PLAYER_RISK_MULT
    # committed $ per ticker that counts against the cap: filled positions + resting we KEEP
    # (i.e. NOT our own orders on a side we're about to replace).
    committed = dict(pos)
    for (tk, side), lst in resting.items():
        for ro in lst:
            if (tk, side) in intended_sides and ro["ours"]:
                continue                             # cancel-replaced below → frees its room
            committed[tk] = committed.get(tk, 0.0) + ro["count"] * ro["price_c"] / 100.0
    used_t = dict(committed)                         # running $ per ticker (market cap)
    used_p = {}                                      # running $ per player (player cap)
    for tk, v in committed.items():
        pk = t2p.get(tk)
        if pk:
            used_p[pk] = used_p.get(pk, 0.0) + v
    # orders arrive best-edge-first, so the cap budget is spent on the juiciest first.
    for o in orders:
        tk, p, pk = o["ticker"], o["price_c"] / 100.0, (o["match"], o["player"])
        o["cancels"] = [ro["order_id"] for ro in resting.get((tk, o["side"]), []) if ro["ours"]]
        mkt_room = config.KALSHI_MAX_RISK - used_t.get(tk, 0.0)
        plr_room = player_cap - used_p.get(pk, 0.0)
        room = min(mkt_room, plr_room)
        o["cap_room"] = round(room, 2)
        o["cap_by"] = "player" if plr_room < mkt_room else "market"
        o["orig_count"] = o["count"]
        o["count"] = min(o["count"], int(max(room, 0.0) // p))
        o["risk"] = round(o["count"] * p, 2)
        o["payout"] = round(o["count"] * 1.0, 2)
        if o["count"] >= 1:                          # feed both running totals for later orders
            used_t[tk] = used_t.get(tk, 0.0) + o["count"] * p
            used_p[pk] = used_p.get(pk, 0.0) + o["count"] * p
    return orders, ok


def _fmt(o):
    exp = (datetime.fromtimestamp(int(o["kickoff"]), tz=timezone.utc).strftime("%H:%MZ")
           if o.get("kickoff") and not math.isinf(o["kickoff"]) else "GTC")
    s = (f"{o['mode']:4} {o['match']} | {o['player']} {o['market']} "
         f"{o['side'].upper()} @ {o['price_c']}c x{o['count']}  "
         f"risk ${o['risk']:.0f} / pay ${o['payout']:.0f}  "
         f"edge +{o['edge_c']}c (fair {o['fair'] * 100:.0f}%)  exp {exp}")
    extra = []
    if o.get("cancels"):
        extra.append(f"replaces {len(o['cancels'])}")
    if o.get("orig_count") and o["count"] < o["orig_count"]:
        extra.append(f"capped {o['orig_count']}->{o['count']} ({o.get('cap_by', 'market')})")
    return s + (("  [" + ", ".join(extra) + "]") if extra else "")


def _payload(o):
    kw = {"ticker": o["ticker"], "action": "buy", "side": o["side"],
          "count": o["count"], "type": "limit", "client_order_id": o["client_order_id"],
          "expiration_ts": o["expiration_ts"]}
    kw["yes_price" if o["side"] == "yes" else "no_price"] = o["price_c"]
    return kw


def _cancel_stale(o):
    """Cancel our own prior resting order(s) on this (ticker, side) so the new one replaces
    rather than stacks. Returns True if all cancels succeeded (or there were none)."""
    for oid in o.get("cancels", []):
        try:
            kalshi.cancel_order(oid)
            print(f"   canceled stale {oid}")
        except Exception as e:
            print(f"   ERROR canceling {oid} ({e}) - skipping placement to avoid stacking")
            return False
    return True


def approve_and_place(orders):
    """Reconcile against live Kalshi state (cumulative per-market cap + cancel-replace), show
    the placeable orders, take one top-level confirm, then a y/N per order (a=all, q=quit).
    On approval, cancel our stale order on that side first, then place. Returns placed list."""
    if not orders:
        print("\nKalshi orders: none actionable (no takes/makes on the live book).")
        return []
    orders, ok = reconcile(orders)
    placeable = [o for o in orders if o["count"] >= 1]
    capped = len(orders) - len(placeable)
    if not placeable:
        print(f"\nKalshi orders: none placeable after reconcile "
              f"({capped} already at the per-market cap).")
        return []
    if not ok:
        print("\n   WARNING: could not fully read existing Kalshi state — the cumulative "
              "cap may be incomplete. Proceed with extra care.")

    tot_risk = sum(o["risk"] for o in placeable)
    tot_pay = sum(o["payout"] for o in placeable)
    n_cancel = sum(len(o["cancels"]) for o in placeable)
    print(f"\n===== LIVE KALSHI ORDERS - {len(placeable)} placeable "
          f"(sum risk ${tot_risk:.0f}, sum payout ${tot_pay:.0f}; {n_cancel} stale to replace"
          + (f"; {capped} at cap, skipped" if capped else "") + ") =====")
    for i, o in enumerate(placeable, 1):
        print(f"  [{i:2}] {_fmt(o)}")
    if input(f"\nReview each of the {len(placeable)} above? [y/N] ").strip().lower() != "y":
        print("Aborted - nothing sent.")
        return []

    placed, approve_all = [], False
    for i, o in enumerate(placeable, 1):
        print(f"\n[{i}/{len(placeable)}] {_fmt(o)}")
        if not approve_all:
            ans = input("   place this order? [y/N/a=all/q=quit]: ").strip().lower()
            if ans == "q":
                print("   quit - no further orders.")
                break
            if ans == "a":
                approve_all = True
            elif ans != "y":
                print("   skipped.")
                continue
        if not _cancel_stale(o):                     # replace failed → don't stack
            continue
        try:
            resp = kalshi.create_order(**_payload(o))
            oid = resp.get("order_id") or (resp.get("order") or {}).get("order_id")  # v2 top-level
            o["order_id"] = oid
            placed.append(o)
            print(f"   PLACED  order_id={oid}")
            import trades                                  # record the take for result-matching
            trades.log_fill(
                ticker=o["ticker"], match=o["match"], player=o["player"], market=o["market"],
                side=o["side"], price_c=o["price_c"], count=o["count"], order_id=oid,
                client_order_id=o["client_order_id"], mode=o["mode"], source=o.get("source"),
                fair=o.get("fair"), q_yes=o.get("q_yes"), minutes=o.get("minutes"),
                hazard=o.get("hazard"), kickoff=o.get("kickoff"),
                post_only=(o["mode"] != "TAKE"), k_yes_bid=o.get("k_yes_bid"),
                k_yes_ask=o.get("k_yes_ask"), k_no_bid=o.get("k_no_bid"), k_no_ask=o.get("k_no_ask"))
        except Exception as e:
            print(f"   ERROR - not placed: {e}")
    print(f"\nPlaced {len(placed)} of {len(placeable)} "
          f"(sum risk ${sum(o['risk'] for o in placed):.0f}).")
    return placed
