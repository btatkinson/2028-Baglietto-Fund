"""Pricing layer: expected counts -> P(stat >= line) -> fair line + low-vig quote.

A counting prop is a count X with mean mu = rate90 * minutes/90. Two variance
sources matter for the tails (hence for any line away from the median):
  1. conditional over-dispersion of the stat itself — passes are far more variable
     than Poisson, goals are ~Poisson — a per-stat factor `phi` calibrated from the
     rate model's own residuals (calibrate_dispersion).
  2. MINUTES uncertainty — a starter may be subbed at 60' or play 95' — adds
     (rate/90)^2 * minutes_sd^2 (the convex/Jensen term the minutes model flagged).

We fit a Negative Binomial to (mean, var) (Poisson when var<=mean), then read off
P(over line), the fair (median) line, and a two-sided YES quote at a chosen
half-spread — the market-maker's bid/ask around fair.
"""
from __future__ import annotations

import json
import math

import config


def load_dispersion() -> dict:
    p = config.MODEL_DIR / "dispersion.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def count_mean_var(rate90, minutes_mean, minutes_sd, phi=1.0):
    """(mean, var) of the match count. var floored at mean (NegBin needs var>=mean)."""
    mu = rate90 * minutes_mean / 90.0
    var = phi * mu + (rate90 / 90.0) ** 2 * (minutes_sd ** 2)
    return mu, max(var, mu)


def _params(mean, var):
    """-> ('pois', mu) or ('nb', r, p) for scipy. None if no mass."""
    if mean <= 0:
        return None
    if var <= mean * 1.0001:
        return ("pois", mean)
    p = mean / var                       # var/mean = 1/p
    r = mean * mean / (var - mean)
    return ("nb", r, p)


def sf_ge(mean, var, n):
    """P(X >= n), integer n."""
    from scipy.stats import nbinom, poisson
    if n <= 0:
        return 1.0
    d = _params(mean, var)
    if d is None:
        return 0.0
    return float(poisson.sf(n - 1, d[1]) if d[0] == "pois" else nbinom.sf(n - 1, d[1], d[2]))


def prob_over(mean, var, line):
    """P(X > line). For a half-integer line (2.5) this is P(X >= 3)."""
    return sf_ge(mean, var, math.floor(line) + 1)


def fair_line(mean, var):
    """Half-integer line closest to a coin flip (P(over) = 0.5)."""
    hi = int(mean + 6 * math.sqrt(max(var, 1e-9)) + 5)
    best, best_d = 0.5, 1.0
    for k in range(hi + 1):
        d = abs(prob_over(mean, var, k + 0.5) - 0.5)
        if d < best_d:
            best, best_d = k + 0.5, d
    return best


def quote(p, half_spread=0.02):
    """Two-sided YES quote in cents around fair prob p, clipped to [1, 99]."""
    lo = min(max(round(100 * (p - half_spread)), 1), 99)
    hi = min(max(round(100 * (p + half_spread)), 1), 99)
    return lo, hi


def price_market(rate90, minutes_mean, minutes_sd, phi=1.0, line=None, half_spread=0.02):
    """Full pricing for one player-stat. Always returns mean/var/fair_line; if a
    `line` is given (a Kalshi/DFS threshold), adds P(over) and a YES bid/ask."""
    mu, var = count_mean_var(rate90, minutes_mean, minutes_sd, phi)
    out = {"mean": mu, "var": var, "fair_line": fair_line(mu, var)}
    if line is not None:
        p = prob_over(mu, var, line)
        out.update(line=line, p_over=p)
        out["yes_bid"], out["yes_ask"] = quote(p, half_spread)
    return out


_DISP_MU_FLOOR = 0.05            # Pearson ratio (y-mu)^2/mu is numerically unstable as
                                # mu->0 (a single rare-event hit at mu~1e-4 blows the
                                # mean to ~10); for rare heads (goals/ga) only rows with
                                # a non-negligible expected count carry dispersion signal.
_DISP_MU_CAP = 8.0


def calibrate_dispersion(frame, save=True):
    """Per-stat Pearson dispersion phi = mean((y-mu)^2 / mu) using each fitted
    head's CONDITIONAL mean (so it's the residual over-dispersion, not the raw
    mixed-population one). Counts reconstructed from per-90 x minutes. Only rows with
    mu >= _DISP_MU_FLOOR contribute (near-zero mu gives unstable ratios — that's what
    made ga read phi~9.7); clamped to [1, _DISP_MU_CAP]."""
    import rate_model
    minutes = frame["minutes"].to_numpy()
    out = {}
    for s in rate_model.RATE_STATS:
        rate = rate_model.load(s).predict(frame[rate_model.features_for(s)]).clip(min=0)
        mu = rate * minutes / 90.0
        y = frame[f"{s}90"].to_numpy() * minutes / 90.0      # reconstruct raw count
        ok = mu > _DISP_MU_FLOOR
        phi = float((((y[ok] - mu[ok]) ** 2) / mu[ok]).mean()) if ok.sum() else 1.0
        out[s] = min(max(phi, 1.0), _DISP_MU_CAP)
    if save:
        (config.MODEL_DIR / "dispersion.json").write_text(json.dumps(out, indent=2),
                                                          encoding="utf-8")
    return out
