# minutes_projection

Predicts per-90 rates and a minutes distribution for international (World Cup)
players, so `wc_props` can turn the book's counting lines into clean minutes and
price low-count props (G+A) post-lineup. Trained offline; `wc_props` depends on
it one-way via `predict.predict()`.

## Why it exists
A counting prop is `rate x minutes`. Both move with game script, and you can't
separate them from a single line — so we model the **rate** (context-adjusted)
ourselves, read the book's **total**, and back out **minutes**, reconciling
across stats and anchoring to a minutes prior so a rate disagreement on one stat
doesn't masquerade as a minutes difference.

## Layout
- `team_strength.py` — PossessionElo (logit; drives pass rate) + GoalsElo
  (attack/defence, log-link, ~1.25 anchor; drives G+A rate & blow-out signal).
- `download_past_player_fbref.py` — per-match passes/minutes/position (training).
- `download_past_team_fbref.py` — possession + goals -> Elos (`build_ratings`).
- `download_upc_starting_xi.py` — live confirmed XI (FotMob/Sofascore JSON;
  soccerdata 1.9.0 has no usable lineup reader — adapter seam inside).
- `features.py` — leak-free training frame + inference feature row.
- `rate_model.py` — CatBoost: position x script -> per-90.
- `minutes_model.py` — start-conditional prior + multi-stat reconciliation +
  prior x likelihood posterior.
- `predict.py` — the one-way contract wc_props calls.

## Data notes
- **Script feature is model-derived expected goals/possession**, not market odds —
  so training needs no historical odds and train==inference. Market totals are an
  optional inference-time override (you already have an `ODDSPAPI_KEY` slot).
- Train mostly on **club** data (rich); apply to **international** (sparse, more
  extreme mismatches). Fit Elos at the level you query them — national teams for
  WC — and cold-start thin teams with a prior (FIFA rank / squad-club aggregate).
- FBref rate-limits; soccerdata caches locally. Run the downloaders on your box.
