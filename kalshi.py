"""Kalshi trade-API v2 client — READ-ONLY market data for the scorers-board overlay.

Auth is RSA request signing (api-key id + RSA private key). Every request carries:
  KALSHI-ACCESS-KEY        the key id (UUID, config.KALSHI_KEY_ID)
  KALSHI-ACCESS-TIMESTAMP  unix milliseconds
  KALSHI-ACCESS-SIGNATURE  base64( RSA-PSS-SHA256( timestamp + METHOD + path ) )
where `path` is the full request path INCLUDING the /trade-api/v2 prefix and EXCLUDING
the query string. We only ever issue GETs — this module never places or cancels orders
(you submit trades yourself); it exists to read prices and order books for comparison
against our acceptable maker prices.

The private key in .env (KALSHI_RSA) is stored as the bare base64 DER body (no PEM
header) or as a full PEM block; _load_key handles both.
"""
from __future__ import annotations

import base64
import textwrap
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import config

_PREFIX = "/trade-api/v2"
_TIMEOUT = 20
_session = requests.Session()
_key = None


def _load_key():
    """Parse config.KALSHI_PRIVATE_KEY into an RSA private key object (cached). Accepts a
    full PEM, or the bare base64 of the DER (PKCS#1 or PKCS#8) — the form saved in .env."""
    global _key
    if _key is not None:
        return _key
    raw = (config.KALSHI_PRIVATE_KEY or "").strip().strip('"').strip("'")
    if not raw:
        raise RuntimeError("KALSHI_RSA not set in .env")
    if "BEGIN" in raw:
        _key = serialization.load_pem_private_key(raw.encode(), password=None)
        return _key
    der = base64.b64decode(raw)
    try:
        _key = serialization.load_der_private_key(der, password=None)
    except Exception:
        pem = ("-----BEGIN RSA PRIVATE KEY-----\n"
               + "\n".join(textwrap.wrap(raw, 64))
               + "\n-----END RSA PRIVATE KEY-----\n")
        _key = serialization.load_pem_private_key(pem.encode(), password=None)
    return _key


def _headers(method, path):
    if not config.KALSHI_KEY_ID:
        raise RuntimeError("KALSHI_API (key id) not set in .env")
    ts = str(int(time.time() * 1000))
    sig = _load_key().sign(
        (ts + method + path).encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256())
    return {
        "KALSHI-ACCESS-KEY": config.KALSHI_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Accept": "application/json",
    }


def _get(endpoint, params=None):
    path = _PREFIX + endpoint
    r = _session.get(config.KALSHI_BASE + path, params=params,
                     headers=_headers("GET", path), timeout=_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Kalshi GET {endpoint} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def _post(endpoint, body):
    path = _PREFIX + endpoint
    r = _session.post(config.KALSHI_BASE + path, json=body,
                      headers={**_headers("POST", path), "Content-Type": "application/json"},
                      timeout=_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Kalshi POST {endpoint} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def _delete(endpoint):
    path = _PREFIX + endpoint
    r = _session.delete(config.KALSHI_BASE + path, headers=_headers("DELETE", path),
                        timeout=_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Kalshi DELETE {endpoint} -> {r.status_code}: {r.text[:300]}")
    return r.json()


# --------------------------- read-only wrappers ---------------------------

def exchange_status() -> dict:
    """{'exchange_active': bool, 'trading_active': bool} — cheap reachability check."""
    return _get("/exchange/status")


def balance() -> dict:
    """Portfolio cash balance (cents). Auth-required, so a 200 here proves our signing
    works end-to-end."""
    return _get("/portfolio/balance")


def markets(**params) -> dict:
    """One page of markets. Useful params: limit (<=1000), cursor, status ('open'),
    event_ticker, series_ticker, tickers (comma-separated)."""
    return _get("/markets", params or None)


def iter_markets(**params) -> list[dict]:
    """Every market across the cursor pagination (one flat list)."""
    params = dict(params)
    out = []
    while True:
        resp = _get("/markets", params)
        out += resp.get("markets", [])
        cur = resp.get("cursor")
        if not cur or not resp.get("markets"):
            break
        params["cursor"] = cur
    return out


def market(ticker: str) -> dict:
    return _get(f"/markets/{ticker}")


def orderbook(ticker: str, depth: int | None = None) -> dict:
    """Resting YES/NO order book for a market: {'orderbook': {'yes': [[price¢, qty], ...],
    'no': [[price¢, qty], ...]}}. Prices are cents; each side is bids for that side."""
    return _get(f"/markets/{ticker}/orderbook", {"depth": depth} if depth else None)


def events(**params) -> dict:
    """Events (groups of markets). Useful params: series_ticker, status,
    with_nested_markets (bool), limit, cursor."""
    return _get("/events", params or None)


def series_list(**params) -> dict:
    """Series (templates). Useful param: category (e.g. 'Sports')."""
    return _get("/series", params or None)


# --------------------------- WRITE (orders) -------------------------------
# These place/cancel real orders. They are only ever reached from kalshi_orders, and
# only after explicit per-order human approval (run.py --trade). Same RSA signing as GETs.

def create_order(ticker, action, side, count, type="limit", yes_price=None, no_price=None,
                 expiration_ts=None, client_order_id=None, post_only=False) -> dict:
    """Place a limit BUY on the v2 order endpoint (POST /portfolio/events/orders). The legacy
    /portfolio/orders path (side yes/no + yes_price/no_price in cents) was retired in 2026, so
    this translates our yes/no + cents interface to the v2 YES-leg model:

      * v2 expresses every order on the YES leg as bid (buy YES) / ask (sell YES). Selling YES
        is economically buying NO at (1 - price), so:
            buy YES @ p¢  ->  side 'bid', price = p/100 dollars
            buy NO  @ p¢  ->  side 'ask', price = (100 - p)/100 dollars   (sell YES at the complement)
      * price is a fixed-point DOLLAR string; count is a contract string; there is no `type`
        field (time_in_force defines it). We rest good-till-cancel with expiration_time = kickoff.

    The signature is kept (action/type/yes_price/no_price) so callers don't change. action must
    be 'buy' — this module never sells. client_order_id stays the idempotency key."""
    if action != "buy":
        raise RuntimeError(f"create_order: only 'buy' supported, got {action!r}")
    if side == "yes":
        if yes_price is None:
            raise RuntimeError("create_order: yes side needs yes_price")
        v2_side, price_c = "bid", int(yes_price)                # buy YES at its price
    elif side == "no":
        if no_price is None:
            raise RuntimeError("create_order: no side needs no_price")
        v2_side, price_c = "ask", 100 - int(no_price)           # buy NO == sell YES at the complement
    else:
        raise RuntimeError(f"create_order: side must be yes/no, got {side!r}")
    if not (1 <= price_c <= 99):
        raise RuntimeError(f"create_order: YES-leg price {price_c}c out of range")
    body = {
        "ticker": ticker,
        "side": v2_side,
        "count": str(int(count)),
        "price": f"{price_c / 100:.4f}",                        # dollars, YES leg
        "time_in_force": "good_till_canceled",
        "self_trade_prevention_type": "taker_at_cross",
    }
    if post_only:                       # reject (don't take) if it would cross — guarantees maker
        body["post_only"] = True
    if client_order_id:
        body["client_order_id"] = client_order_id
    if expiration_ts:
        body["expiration_time"] = int(expiration_ts)            # was expiration_ts in the legacy API
    return _post("/portfolio/events/orders", body)


def cancel_order(order_id: str) -> dict:
    """Cancel (reduce to 0) a resting order by its Kalshi order_id."""
    return _delete(f"/portfolio/orders/{order_id}")


def orders(**params) -> dict:
    """Our resting orders. Useful params: ticker, status ('resting'), limit, cursor."""
    return _get("/portfolio/orders", params or None)


def positions(**params) -> dict:
    """Our market positions. Useful params: ticker, settlement_status, limit, cursor."""
    return _get("/portfolio/positions", params or None)
