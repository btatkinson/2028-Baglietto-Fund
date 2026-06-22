"""Past team-match data from FBref -> feeds the team-strength Elos and team EWMs.

read_team_match_stats carries possession % and goals per team-match, which is
exactly what PossessionElo and GoalsElo consume. We replay matches in date order
to produce a leak-free pre-match rating snapshot per fixture.
"""
from __future__ import annotations
import pandas as pd
import config
from team_strength import build_ratings  # relocated to team_strength (source-agnostic)


def download(leagues=None, seasons=None, store=True) -> pd.DataFrame:
    import soccerdata as sd
    leagues = leagues or (config.CLUB_LEAGUES + config.INTL_LEAGUES)
    seasons = seasons or config.SEASONS
    fb = sd.FBref(leagues=leagues, seasons=seasons)
    poss = fb.read_team_match_stats(stat_type="possession").reset_index()
    summ = fb.read_team_match_stats(stat_type="schedule").reset_index()    # gf/ga/date
    if store:
        poss.to_parquet(config.DATA_DIR / "fbref_team_possession.parquet", index=False)
        summ.to_parquet(config.DATA_DIR / "fbref_team_schedule.parquet", index=False)
    return summ


if __name__ == "__main__":
    download()
    print("team possession + schedule -> data/")
