"""Minutes-given-start: a DISTRIBUTION, not a point (the G+A fair integrates a
convex function of minutes, so E[g(m)] != g(E[m])).

Structure (from the design):
  prior   : P(full 90 | start) classifier + conditional minutes dist, from
            position, role/star proxy, |exp_margin|, exp_team_g, stage,
            dead_rubber, days_rest. Post-XI, EWM features are start-conditional.
  update  : the book's counting lines are an observation. Reconcile the implied
            minutes across ALL listed stats given our rates, then combine with
            the prior (prior x likelihood) so a single-stat rate disagreement
            can't dominate. No listed lines => fall back to the prior (no seam).
"""
from __future__ import annotations
import json

import config

EXP_SUBS_USED = 5.0     # avg substitutes a team actually brings on (of a ~12-man bench)


def reconcile_minutes(book_totals: dict, our_rates: dict) -> float | None:
    """argmin_m sum_stat (book_total - (rate90/90)*m)^2, with m in MINUTES.
    => m = S((rate90/90)*total) / S((rate90/90)^2).
    book_totals / our_rates keyed by stat (e.g. {'passes':38,'touches':55});
    rates are per-90, totals are the book's implied match totals."""
    num = den = 0.0
    for stat, rate90 in our_rates.items():
        tot = book_totals.get(stat)
        if tot is None or rate90 is None or rate90 <= 0:
            continue
        r = rate90 / 90.0
        num += r * float(tot)
        den += r * r
    return None if den == 0 else num / den


def posterior_minutes(prior_mean, prior_sd, book_minutes, book_sd):
    """Gaussian prior x likelihood (in minutes). book_minutes from
    reconcile_minutes; if None, return the prior. Returns (mean, sd)."""
    if book_minutes is None:
        return prior_mean, prior_sd
    wp, wb = 1.0 / prior_sd**2, 1.0 / book_sd**2
    mean = (wp * prior_mean + wb * book_minutes) / (wp + wb)
    return mean, (1.0 / (wp + wb)) ** 0.5


def train_prior(frame, match_minutes=90):
    """Fit a start-conditional minutes prior from the training frame (needs
    position, minutes, is_starter, exp_team_g, exp_opp_g). Per-position
    P(full match | start) + mean/sd, a global blowout slope (starters rest as the
    expected goal margin widens), and a sub-appearance distribution. Persist JSON."""
    import numpy as np
    f = frame.copy()
    f["abs_margin"] = (f["exp_team_g"] - f["exp_opp_g"]).abs()
    st = f[f["is_starter"] & (f["minutes"] > 0)]
    by_pos = {}
    for pos, g in st.groupby("position"):
        by_pos[str(pos)] = {"p_full": float((g["minutes"] >= match_minutes).mean()),
                            "mean": float(g["minutes"].mean()),
                            "sd": float(g["minutes"].std() or 8.0)}
    x, y = st["abs_margin"].to_numpy(), st["minutes"].to_numpy()
    slope = float(np.polyfit(x, y, 1)[0]) if len(st) > 50 and x.std() > 1e-6 else 0.0
    sub = f[(~f["is_starter"]) & (f["minutes"] > 0)]
    prior = {
        "starter_by_pos": by_pos,
        "margin_slope": slope, "margin_ref": float(st["abs_margin"].mean()),
        "sub": {"mean": float(sub["minutes"].mean()), "sd": float(sub["minutes"].std() or 10.0)},
        "match_minutes": match_minutes,
    }
    (config.MODEL_DIR / "minutes_prior.json").write_text(json.dumps(prior, indent=2),
                                                         encoding="utf-8")
    return prior


def load_prior():
    return json.loads((config.MODEL_DIR / "minutes_prior.json").read_text(encoding="utf-8"))


def prior_minutes(prior, position, is_starter, abs_margin=0.0):
    """(mean, sd, p_full) for one player from the start-conditional prior."""
    mm = prior["match_minutes"]
    if is_starter:
        base = (prior["starter_by_pos"].get(str(position))
                or {"mean": 80.0, "sd": 15.0, "p_full": 0.5})
        mean = base["mean"] + prior["margin_slope"] * (abs_margin - prior["margin_ref"])
        return min(max(mean, 0.0), mm), base["sd"], base["p_full"]
    s = prior["sub"]
    return s["mean"], s["sd"], 0.0


def normalize_roster_minutes(means, team_total=None, cap=None, match_minutes=90):
    """Scale a team's per-player minutes so they sum to the roster budget
    (11 * match_minutes), holding any player at `cap` (one player can't exceed the
    match length + stoppage) and redistributing the rest. This is the constraint
    that makes pricing several players in the SAME match mutually coherent —
    without it, independently-projected minutes over/under-shoot the 990 budget."""
    team_total = float(team_total if team_total is not None else 11 * match_minutes)
    cap = float(cap if cap is not None else match_minutes + 8)
    m = {k: max(0.0, float(v)) for k, v in means.items()}
    for _ in range(12):
        fixed = sum(v for v in m.values() if v >= cap - 1e-9)
        free = [k for k, v in m.items() if v < cap - 1e-9]
        sfree = sum(m[k] for k in free)
        if not free or sfree <= 0:
            break
        scale = (team_total - fixed) / sfree
        hit_cap = False
        for k in free:
            nv = m[k] * scale
            if nv > cap:
                nv, hit_cap = cap, True
            m[k] = nv
        if abs(scale - 1.0) < 1e-6 and not hit_cap:
            break
    return m
