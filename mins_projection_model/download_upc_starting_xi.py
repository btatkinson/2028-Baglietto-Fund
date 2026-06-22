"""Upcoming confirmed starting XI.

NOTE: soccerdata 1.9.0 has no usable lineup reader (FotMob class removed,
Sofascore exposes no read_lineup), so this hits the live source's JSON directly.
Confirmed XIs publish ~1h pre-kickoff. Fill in the endpoint for your chosen
source (FotMob match details, or Sofascore event/lineups); the adapter normalises
to: [{match_id, kickoff, team, player, position, is_starter}].
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import requests
import config

_FOTMOB_MATCHES = "https://www.fotmob.com/api/matches"   # ?date=YYYYMMDD
_FOTMOB_DETAILS = "https://www.fotmob.com/api/matchDetails"  # ?matchId=...


def _normalise(details_json) -> list[dict]:
    """TODO: map the source payload -> normalised rows. Lineups live under the
    match-details 'lineup' block; each side has a confirmed flag + players with
    shirt/position. Keep only confirmed XIs; flag predicted ones."""
    raise NotImplementedError("wire the source payload mapping for your XI source")


def upcoming_xis(within_hours: int = 6) -> list[dict]:
    if config.XI_SOURCE != "fotmob":
        raise NotImplementedError(f"add adapter for XI_SOURCE={config.XI_SOURCE}")
    now = datetime.now(timezone.utc)
    day = now.strftime("%Y%m%d")
    matches = requests.get(_FOTMOB_MATCHES, params={"date": day}, timeout=30).json()
    rows = []
    for m in _iter_matches(matches):
        ko = _kickoff(m)
        if ko and now <= ko <= now + timedelta(hours=within_hours):
            det = requests.get(_FOTMOB_DETAILS, params={"matchId": m["id"]}, timeout=30).json()
            rows += _normalise(det)
    return rows


def _iter_matches(payload):           # TODO: source-specific traversal
    raise NotImplementedError
def _kickoff(m):                      # TODO: parse kickoff -> aware datetime
    raise NotImplementedError


if __name__ == "__main__":
    print(upcoming_xis())
