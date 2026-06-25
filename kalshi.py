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
                 expiration_ts=None, client_order_id=None) -> dict:
    """Place an order. side 'yes'|'no'; action 'buy'|'sell'; price in CENTS on the named
    side (yes_price for a yes order, no_price for a no order). expiration_ts (unix seconds)
    makes it good-till-date — we set it to kickoff so resting orders die at the whistle.
    client_order_id is an idempotency key (Kalshi dedupes repeats), so a re-run won't
    double-place the same intended order."""
    body = {"ticker": ticker, "action": action, "side": side,
            "count": int(count), "type": type}
    if client_order_id:
        body["client_order_id"] = client_order_id
    if yes_price is not None:
        body["yes_price"] = int(yes_price)
    if no_price is not None:
        body["no_price"] = int(no_price)
    if expiration_ts:
        body["expiration_ts"] = int(expiration_ts)
    return _post("/portfolio/orders", body)


def cancel_order(order_id: str) -> dict:
    """Cancel (reduce to 0) a resting order by its Kalshi order_id."""
    return _delete(f"/portfolio/orders/{order_id}")


def orders(**params) -> dict:
    """Our resting orders. Useful params: ticker, status ('resting'), limit, cursor."""
    return _get("/portfolio/orders", params or None)


def positions(**params) -> dict:
    """Our market positions. Useful params: ticker, settlement_status, limit, cursor."""
    return _get("/portfolio/positions", params or None)
