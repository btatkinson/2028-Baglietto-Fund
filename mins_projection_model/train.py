"""Train everything from the cached frames: Elos (saved), one rate head per stat,
and the minutes prior. Prints a time-based holdout benchmark of each head vs the
EWM fallback so you can see where context adds signal. Run after backfill.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
import features as feat
import rate_model
import minutes_model
from team_strength import build_ratings, save_elos


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _drop_womens(tm, pm):
    """Women's national teams (' W' suffix) leak into the same api-football pool as
    the men's; pooling them poisons the Goals Elo (USA W top-6 attack, NZ W bottom)
    and the rate heads. Drop them from BOTH frames before fitting."""
    w = lambda s: s.astype("string").str.endswith(" W").fillna(False)
    tm = tm[~(w(tm["home"]) | w(tm["away"]))].copy()
    pm = pm[~w(pm["team"])].copy()
    return tm, pm


def main():
    tm = pd.read_parquet(config.DATA_DIR / "team_matches.parquet")
    pm = pd.read_parquet(config.DATA_DIR / "player_matches.parquet")
    tm, pm = _drop_womens(tm, pm)
    snaps, poss, goals, sot = build_ratings(tm, neutral_col="neutral")
    save_elos(poss, goals, sot)

    frame = feat.build_training_frame(pm, snaps)
    frame = frame[frame["minutes"] >= 20].sort_values("date").reset_index(drop=True)
    k = int(len(frame) * 0.8)
    tr, te = frame.iloc[:k], frame.iloc[k:]
    print(f"frame: {len(frame)} rows, {frame['player'].nunique()} players, "
          f"{frame['date'].min()}..{frame['date'].max()}\n")

    from catboost import CatBoostRegressor, Pool
    print(f"{'stat':9s}{'mean':>7s}{'EWM rmse':>10s}{'model':>8s}{'lift':>7s}")
    for s in rate_model.RATE_STATS:
        tgt, feats = f"{s}90", rate_model.features_for(s)
        trS, teS = tr.dropna(subset=[tgt]), te.dropna(subset=[tgt])
        m = CatBoostRegressor(loss_function="RMSE", depth=6, iterations=600,
                              learning_rate=0.03, verbose=False)
        m.fit(Pool(trS[feats], trS[tgt], cat_features=rate_model.CAT_FEATURES))
        y, pred = teS[tgt].to_numpy(), m.predict(teS[feats])
        ewm = teS[f"ewm_{s}90"].fillna(trS[tgt].mean()).to_numpy()
        rb, rm = _rmse(y, ewm), _rmse(y, pred)
        print(f"{s:9s}{y.mean():7.1f}{rb:10.2f}{rm:8.2f}{(rb - rm) / rb * 100 if rb else 0:6.0f}%")
        rate_model.train(frame, stat=s)            # refit on all rows + persist

    minutes_model.train_prior(frame)
    import pricing
    disp = pricing.calibrate_dispersion(frame)
    print("dispersion phi:", {k: round(v, 1) for k, v in disp.items()})
    print(f"\nsaved elos.json, rate_*.cbm, minutes_prior.json, dispersion.json -> {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
