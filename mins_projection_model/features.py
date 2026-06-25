"""Assemble the leak-free training frame and the inference feature row.

Feature contract (one row per player-match), consumed by both models:
  position            categorical (GK/CB/FB/DM/CM/AM/W/ST or FBref's codes)
  ewm_pass90          player EWM passes-per-90 (leak-free; club-weighted prior)
  exp_team_share      PossessionElo expected possession share for this team
  exp_team_g          GoalsElo expected goals for this team
  exp_opp_g           GoalsElo expected goals for the opponent
  source              'club' | 'intl'  (domain flag)
  -- minutes model also wants: --
  is_starter, exp_margin (exp_team_g - exp_opp_g), |exp_margin|, stage,
  dead_rubber, days_rest

Build the script features by joining each player-match to the PRE-match rating
snapshot from team_strength.build_ratings (NOT post-match) so nothing leaks.
"""
from __future__ import annotations
import pandas as pd
import config


def player_ewm(player_matches: pd.DataFrame, stat="passes",
               by="player", date_col="date", halflife=None) -> pd.Series:
    """Leak-free EWM of a per-90 stat: shift(1) so the current match is excluded."""
    hl = halflife or config.EWM_HALFLIFE
    pm = player_matches.sort_values([by, date_col]).copy()
    per90 = pm[stat] / (pm["minutes"].clip(lower=1) / 90.0)
    return (per90.groupby(pm[by])
                 .apply(lambda s: s.shift(1).ewm(halflife=hl).mean())
                 .reset_index(level=0, drop=True))


def player_minutes_ewm(player_matches: pd.DataFrame, by="player", date_col="date",
                       halflife=None):
    """Latest EWM of RAW minutes per player — the per-player bench signal. The history
    keeps 0' DNP / unused-sub rows, so this already reflects P(play) x minutes (a deep
    reserve sits near 0, a rotated regular near 90), which is what tells two bench
    players apart. No shift(1): the match being projected is never in `player_matches`,
    so including all observed history is correct at inference (no leak). Returns
    (Series of last EWM minutes, Series of match counts), both indexed by player."""
    hl = halflife or config.EWM_HALFLIFE
    pm = player_matches.sort_values([by, date_col])
    ewm = pm.groupby(by)["minutes"].apply(lambda s: s.ewm(halflife=hl).mean())
    return ewm.groupby(level=0).last(), pm.groupby(by).size()


RATE_STATS = ["passes", "shots", "sot", "tackles", "saves", "goals", "assists", "ga"]

# Team game-script columns attached to every player-match (this team's value, then
# the opponent's). Order MUST match the per-side tuples built in build_training_frame
# and the raters in team_strength.build_ratings. rate_model.SCRIPT imports this.
SCRIPT_FIELDS = ["exp_team_share", "exp_team_g", "exp_opp_g", "exp_team_sot", "exp_opp_sot",
                 "exp_team_shots", "exp_opp_shots", "exp_team_corners", "exp_opp_corners"]


def build_training_frame(player_matches: pd.DataFrame,
                         snaps: pd.DataFrame) -> pd.DataFrame:
    """Leak-free join of each player-match to its team's PRE-match rating snapshot,
    widened to every rate stat. `snaps` (from build_ratings) carries match_id plus
    exp_home_share/exp_home_g/exp_away_g/exp_home_sot/exp_away_sot. Each fixture is
    split into its two sides so a player gets HIS team's expected share/goals/SoT
    and the OPPONENT's expected goals/SoT (the latter drives saves & shots-faced).
    For every stat we attach the leak-free EWM per-90 (a feature) and the per-90
    target. One row per player-match (who played). SCRIPT_FIELDS below lists the
    team/opp script columns built here (shots/corners are NaN for snaps that lack
    them)."""
    nan = float("nan")

    def g(r, name):                       # snap attr or NaN (shots/corners optional)
        return getattr(r, name, nan)

    side = {}
    for r in snaps.itertuples():
        side[(r.match_id, r.home)] = (
            r.exp_home_share, r.exp_home_g, r.exp_away_g, r.exp_home_sot, r.exp_away_sot,
            g(r, "exp_home_shots"), g(r, "exp_away_shots"),
            g(r, "exp_home_corners"), g(r, "exp_away_corners"))
        side[(r.match_id, r.away)] = (
            1.0 - r.exp_home_share, r.exp_away_g, r.exp_home_g, r.exp_away_sot, r.exp_home_sot,
            g(r, "exp_away_shots"), g(r, "exp_home_shots"),
            g(r, "exp_away_corners"), g(r, "exp_home_corners"))

    pm = player_matches.sort_values(["player", "date"]).copy()
    if "ga" not in pm.columns:
        pm["ga"] = pm["goals"].fillna(0) + pm["assists"].fillna(0)

    factor = (pm["minutes"] / 90.0).where(pm["minutes"] > 0)   # NaN target if 0'
    for s in RATE_STATS:
        pm[f"ewm_{s}90"] = player_ewm(pm, stat=s)
        pm[f"{s}90"] = pm[s] / factor

    sc = [side.get((mid, tm), (None,) * len(SCRIPT_FIELDS))
          for mid, tm in zip(pm["match_id"], pm["team"])]
    pm[SCRIPT_FIELDS] = pd.DataFrame(sc, index=pm.index)

    # carry the per-fixture ratings warm-up flag so the trainer can withhold cold rows
    if "warm" in snaps.columns and "match_id" in snaps.columns:
        warm = snaps.drop_duplicates("match_id").set_index("match_id")["warm"]
        pm["warm"] = pm["match_id"].map(warm).fillna(False)
    else:
        pm["warm"] = True

    pm = pm[pm["minutes"] > 0].copy()
    pm["position"] = pm["position"].fillna("NA")          # CatBoost cat feature
    keep = (["position", "source"] + SCRIPT_FIELDS
            + ["player", "team", "date", "match_id", "minutes", "is_starter", "warm"]
            + [f"ewm_{s}90" for s in RATE_STATS] + [f"{s}90" for s in RATE_STATS])
    return pm[keep].reset_index(drop=True)
