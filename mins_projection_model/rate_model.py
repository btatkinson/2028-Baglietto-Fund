"""Per-90 rate model: position x game-script -> expected per-90, ONE CatBoost head
per stat (passes/shots/sot/tackles/saves/goals/assists/ga).

Each head's features: position + source (categorical), the player's own leak-free
EWM per-90 of THAT stat (form), and the script — possession share, expected goals
(team & opp) and expected shots-on-target (team & opp). exp_opp_sot is what makes
the saves / shots-faced heads work. Returns the per-90 rate wc_props multiplies by
projected (normalized) minutes to price a counting line.
"""
from __future__ import annotations

import config
from features import RATE_STATS, SCRIPT_FIELDS as SCRIPT

CAT_FEATURES = ["position", "source"]


def features_for(stat: str) -> list:
    """Feature columns for a stat's head: cats first (stable indices), then the
    stat's own EWM, then the shared script block."""
    return CAT_FEATURES + [f"ewm_{stat}90"] + SCRIPT


def train(frame, stat="passes"):
    """Fit CatBoostRegressor(features_for(stat) -> {stat}90). Persist to MODEL_DIR."""
    from catboost import CatBoostRegressor, Pool
    feats, target = features_for(stat), f"{stat}90"
    df = frame.dropna(subset=[target])
    pool = Pool(df[feats], df[target], cat_features=CAT_FEATURES)
    model = CatBoostRegressor(loss_function="RMSE", depth=6, iterations=1200,
                              learning_rate=0.03, verbose=False)
    model.fit(pool)
    model.save_model(str(config.MODEL_DIR / f"rate_{stat}.cbm"))
    return model


def load(stat="passes"):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor()
    m.load_model(str(config.MODEL_DIR / f"rate_{stat}.cbm"))
    return m


def predict_rate(model, feature_row: dict, stat="passes") -> float:
    import pandas as pd
    feats = features_for(stat)
    pred = float(model.predict(pd.DataFrame([{f: feature_row.get(f) for f in feats}]))[0])
    return max(0.0, pred)               # a counting rate can't be negative
