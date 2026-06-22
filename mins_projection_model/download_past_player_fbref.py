"""Past player-match data from FBref (training rows for the rate + minutes models).

Pulls per-MATCH (not season) rows: passes, minutes, position, plus start status
(read_lineup) and substitution timing (read_events, for minutes-given-start).
Run on your machine — FBref isn't reachable from every sandbox, and it rate-limits.
"""
from __future__ import annotations
import pandas as pd
import config


def download(leagues=None, seasons=None, store=True) -> pd.DataFrame:
    import soccerdata as sd
    leagues = leagues or (config.CLUB_LEAGUES + config.INTL_LEAGUES)
    seasons = seasons or config.SEASONS
    fb = sd.FBref(leagues=leagues, seasons=seasons)

    # high-volume + the stat we inv-minutes off; 'summary' carries passes/min/pos
    stats = fb.read_player_match_stats(stat_type="summary")        # MultiIndex frame
    lineups = fb.read_lineup()                                     # is_starter, position
    df = stats.reset_index()
    df["source"] = ["intl" if str(lg).startswith("INT") else "club"
                    for lg in df.get("league", [""] * len(df))]
    if store:
        df.to_parquet(config.DATA_DIR / "fbref_player_match.parquet", index=False)
    return df


if __name__ == "__main__":
    out = download()
    print(f"{len(out)} player-match rows -> data/fbref_player_match.parquet")
