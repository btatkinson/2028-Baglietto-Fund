"""Online team-strength ratings that feed the rate/minutes 'script' features.

Two separate ratings, because the downstream models want different things:

  PossessionElo : expected possession share, in logit space. Drives PASS rate —
                  passes track possession, which a goals model does NOT capture
                  (a sterile-domination favourite: high possession, modest goals;
                  a counterattacker: the reverse).
  GoalsElo      : separate attack & defence ratings (log link); expected goals
                  per team, anchored at the competition mean (~1.25). Drives the
                  G+A rate and the blow-out signal the minutes model needs.

Both update online and walk-forward, so a fixture's features use only ratings as
of kickoff (leak-free). Fit NATIONAL-team ratings on international history for WC
inference; fit CLUB ratings on club history for the bulk of player rows. Each
player-match row takes the rating of whatever team it played for in that match.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import pandas as pd

import config


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class PossessionElo:
    """E[home share] = logistic(R_home - R_away + home_adv); update on the
    observed home share. Ratings are logit points; zero-sum updates keep the
    pool centred. Keep k small — possession is stable and low-variance."""
    k: float = 0.08
    home_adv: float = 0.15        # logit bump for the home / designated side
    prior: float = 0.0            # unseen team => 0.0 => 50% expected share
    ratings: dict = field(default_factory=dict)

    def rating(self, team) -> float:
        return self.ratings.get(team, self.prior)

    def expected(self, home, away, neutral: bool = False) -> float:
        h = 0.0 if neutral else self.home_adv
        return _logistic(self.rating(home) - self.rating(away) + h)

    def update(self, home, away, home_share: float, neutral: bool = False) -> float:
        """home_share in [0,1] (e.g. 0.58). Returns the PRE-match expected share
        (use that as the leak-free feature, then call this to roll forward)."""
        exp = self.expected(home, away, neutral)
        err = home_share - exp
        self.ratings[home] = self.rating(home) + self.k * err
        self.ratings[away] = self.rating(away) - self.k * err
        return exp


@dataclass
class GoalsElo:
    """Separate attack (A) and defence (D) log-rates:
        E[goals_i] = exp(mu + A_i - D_j + home).
    mu = log(competition mean goals/team). Update A and D on the log-residual of
    observed vs expected goals (one online Poisson-ish step per side). Higher D
    = better defence (suppresses opponent goals)."""
    k: float = 0.04
    home_adv: float = 0.20        # log-goals bump at home
    mu: float = math.log(1.25)
    attack: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)

    def _a(self, t): return self.attack.get(t, 0.0)
    def _d(self, t): return self.defense.get(t, 0.0)

    def expected(self, home, away, neutral: bool = False):
        h = 0.0 if neutral else self.home_adv
        eh = math.exp(self.mu + self._a(home) - self._d(away) + h)
        ea = math.exp(self.mu + self._a(away) - self._d(home))
        return eh, ea

    def update(self, home, away, gh: float, ga: float, neutral: bool = False):
        """Returns PRE-match (E[home goals], E[away goals]); then rolls forward."""
        eh, ea = self.expected(home, away, neutral)
        rh = math.log((gh + 0.5) / (eh + 0.5))     # home scored vs expectation
        ra = math.log((ga + 0.5) / (ea + 0.5))     # away scored vs expectation
        self.attack[home] = self._a(home) + self.k * rh
        self.defense[away] = self._d(away) - self.k * rh   # conceded more => weaker D
        self.attack[away] = self._a(away) + self.k * ra
        self.defense[home] = self._d(home) - self.k * ra
        return eh, ea


@dataclass
class LinearElo:
    """Identity-link attack/defence ratings for a mid-count, ~symmetric per-team
    stat (shots on target, shots, corners):
        E[stat_i] = mu + A_i - D_j (+ home), clamped at 0.
    Unlike GoalsElo's log link (multiplicative, for low counts), the update is on
    the RAW residual (observed - expected), so a rating reads directly in the
    stat's own units (e.g. '+1.5 SoT above average'). Higher D = better defence
    (suppresses the opponent's count)."""
    k: float = 0.06
    home_adv: float = 0.30
    mu: float = 4.3
    attack: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)

    def _a(self, t): return self.attack.get(t, 0.0)
    def _d(self, t): return self.defense.get(t, 0.0)

    def expected(self, home, away, neutral: bool = False):
        h = 0.0 if neutral else self.home_adv
        eh = max(0.0, self.mu + self._a(home) - self._d(away) + h)
        ea = max(0.0, self.mu + self._a(away) - self._d(home))
        return eh, ea

    def update(self, home, away, sh: float, sa: float, neutral: bool = False):
        """Returns PRE-match (E[home stat], E[away stat]); then rolls forward."""
        eh, ea = self.expected(home, away, neutral)
        rh, ra = sh - eh, sa - ea                  # raw (linear) residual
        self.attack[home] = self._a(home) + self.k * rh
        self.defense[away] = self._d(away) - self.k * rh   # allowed more => weaker D
        self.attack[away] = self._a(away) + self.k * ra
        self.defense[home] = self._d(home) - self.k * ra
        return eh, ea


class _GlobalAttackDefense:
    """Walk-forward GLOBAL attack/defence fit for a per-team count stat (Massey/
    Poisson over the whole graph, not an online nudge — so sparse cross-cluster
    links propagate). `link='log'` => Poisson MLE (goals); `link='identity'` =>
    ridge least-squares (SoT/shots/corners, mid-counts where a linear residual
    reads in the stat's own units). `refit_before(cutoff)` fits on matches strictly
    before row `cutoff` (leak-free). Coefs are stored in the GoalsElo/LinearElo sign
    convention (defence positive = suppresses), so the existing containers serve
    inference unchanged. mu is log(mean) for log-link, the raw mean for identity."""

    def __init__(self, tm, date_col, home_col, away_col, h_col, a_col,
                 neutral_col, alpha, home_seed, mean_seed, link="log"):
        self.alpha, self.link = alpha, link
        self.mu = math.log(mean_seed) if link == "log" else float(mean_seed)
        self.home_adv, self.attack, self.defense = home_seed, {}, {}
        self.window_n = 0
        atk, dfn, home, val, midx = [], [], [], [], []
        for i, (_, r) in enumerate(tm.iterrows()):
            neutral = bool(r[neutral_col]) if neutral_col else False
            hadv = 0.0 if neutral else 1.0
            for scorer, conceder, hv, v in ((r[home_col], r[away_col], hadv, r[h_col]),
                                            (r[away_col], r[home_col], 0.0, r[a_col])):
                atk.append(scorer); dfn.append(conceder); home.append(hv)
                val.append(v); midx.append(i)
        self._long = pd.DataFrame({"atk": atk, "dfn": dfn, "home": home,
                                   "v": val, "midx": pd.array(midx)})
        A = pd.get_dummies(self._long["atk"], prefix="a")
        D = pd.get_dummies(self._long["dfn"], prefix="d")
        self._X = pd.concat([A, D, self._long["home"]], axis=1).astype(float)
        self._y = pd.to_numeric(self._long["v"], errors="coerce")
        self._midx = self._long["midx"].to_numpy()

    def refit_before(self, cutoff):
        mask = (self._midx < cutoff) & self._y.notna().to_numpy()
        self.window_n = int(cutoff)
        if int(mask.sum()) < 30:        # too thin to fit — keep the cold-start anchor
            self.attack, self.defense = {}, {}
            return
        if self.link == "log":
            from sklearn.linear_model import PoissonRegressor
            m = PoissonRegressor(alpha=self.alpha, max_iter=3000, fit_intercept=True)
        else:
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=self.alpha, fit_intercept=True)
        m.fit(self._X[mask], self._y[mask])
        coef = dict(zip(self._X.columns, m.coef_))
        self.mu = float(m.intercept_)
        self.home_adv = float(coef.get("home", self.home_adv))
        self.attack = {c[2:]: float(v) for c, v in coef.items() if c.startswith("a_")}
        self.defense = {c[2:]: -float(v) for c, v in coef.items() if c.startswith("d_")}

    def expected(self, home, away, neutral=False):
        h = 0.0 if neutral else self.home_adv
        a_h, a_a = self.attack.get(home, 0.0), self.attack.get(away, 0.0)
        d_h, d_a = self.defense.get(home, 0.0), self.defense.get(away, 0.0)
        if self.link == "log":
            return math.exp(self.mu + a_h - d_a + h), math.exp(self.mu + a_a - d_h)
        return max(0.0, self.mu + a_h - d_a + h), max(0.0, self.mu + a_a - d_h)


class _GlobalShare:
    """Walk-forward GLOBAL possession-share rating (one rating per team), fit by
    ridge least-squares on the logit of the observed home share:
        logit(home_share) = R_home - R_away + home_adv.
    Signed design (home +1 / away -1) so the same fit propagates across the graph.
    Stored in PossessionElo's ratings dict for unchanged inference."""

    def __init__(self, tm, date_col, home_col, away_col, hposs_col,
                 neutral_col, alpha, home_seed):
        self.alpha, self.home_adv, self.ratings, self.window_n = alpha, home_seed, {}, 0
        teams = pd.unique(pd.concat([tm[home_col], tm[away_col]]))
        self._cols = {t: i for i, t in enumerate(teams)}
        rows, y, midx = [], [], []
        import numpy as np
        for i, (_, r) in enumerate(tm.iterrows()):
            p = r.get(hposs_col)
            if pd.isna(p):
                continue
            share = min(max(float(p) / 100.0, 0.01), 0.99)
            row = np.zeros(len(teams) + 1)
            row[self._cols[r[home_col]]] += 1.0
            row[self._cols[r[away_col]]] -= 1.0
            row[-1] = 0.0 if (neutral_col and bool(r[neutral_col])) else 1.0
            rows.append(row); y.append(math.log(share / (1 - share))); midx.append(i)
        self._X = np.array(rows) if rows else np.zeros((0, len(teams) + 1))
        self._y = np.array(y)
        self._midx = np.array(midx)

    def refit_before(self, cutoff):
        from sklearn.linear_model import Ridge
        self.window_n = int(cutoff)
        mask = self._midx < cutoff
        if int(mask.sum()) < 30:
            self.ratings = {}
            return
        m = Ridge(alpha=self.alpha, fit_intercept=False).fit(self._X[mask], self._y[mask])
        self.ratings = {t: float(m.coef_[i]) for t, i in self._cols.items()}
        self.home_adv = float(m.coef_[-1])

    def expected(self, home, away, neutral=False):
        h = 0.0 if neutral else self.home_adv
        return _logistic(self.ratings.get(home, 0.0) - self.ratings.get(away, 0.0) + h)


def seed_prior(elo, team, value):
    """Cold-start a national team whose match history is too thin to converge —
    e.g. map FIFA rank or a squad-club aggregate to a starting rating."""
    if isinstance(elo, PossessionElo):
        elo.ratings[team] = value
    elif isinstance(elo, (GoalsElo, LinearElo)):
        a, d = value if isinstance(value, (tuple, list)) else (value, 0.0)
        elo.attack[team], elo.defense[team] = a, d


def build_ratings(team_matches: pd.DataFrame, date_col="date",
                  home_col="home", away_col="away",
                  hposs_col="home_poss", hg_col="home_g", ag_col="away_g",
                  hsot_col="home_sot", asot_col="away_sot",
                  hshots_col="home_shots", ashots_col="away_shots",
                  hcorners_col="home_corners", acorners_col="away_corners",
                  neutral_col=None, id_col="match_id"):
    """Walk team-matches in date order; emit a pre-match rating row per fixture
    (leak-free) and return the fitted Elos for inference. Source-agnostic — feed
    it FBref or api_football frames; column names are adapter args because each
    source's frame shape differs. `neutral_col`, if given, reads a per-row bool
    (WC/continental finals are neutral-site, so the home bump is dropped).

    Possession (logit share), goals (Poisson) and SoT (identity) are ALL
    WALK-FORWARD GLOBAL fits now, refit per month on the expanding window (leak-free).
    Each snapshot carries `warm` — True once the window has >= GOALS_WARMUP_MIN_MATCHES
    and BOTH teams have >= GOALS_WARMUP_TEAM_MATCHES priors — so the trainer can
    withhold cold, signal-less early rows.

    Returns (snaps, poss, goals, sot, shots, corners), each the FINAL all-matches fit
    in its container (PossessionElo/GoalsElo/LinearElo) for unchanged inference. shots
    & corners are identity attack/defence ratings whose expected values become extra
    script features (shots drives the player-shots head; corners is a pressure proxy).
    Rows missing the shots/corners columns (old captures) yield empty fits, harmlessly."""
    tm = team_matches.sort_values(date_col).reset_index(drop=True)
    have_shots = hshots_col in tm.columns and ashots_col in tm.columns
    have_corn = hcorners_col in tm.columns and acorners_col in tm.columns
    gp = _GlobalShare(tm, date_col, home_col, away_col, hposs_col, neutral_col,
                      config.POSS_RIDGE, config.POSS_HOME)
    gg = _GlobalAttackDefense(tm, date_col, home_col, away_col, hg_col, ag_col,
                              neutral_col, config.GOALS_RIDGE, config.GOALS_HOME,
                              config.GOALS_MEAN, link="log")
    gs = _GlobalAttackDefense(tm, date_col, home_col, away_col, hsot_col, asot_col,
                              neutral_col, config.SOT_RIDGE, config.SOT_HOME,
                              config.SOT_MEAN, link="identity")
    gsh = (_GlobalAttackDefense(tm, date_col, home_col, away_col, hshots_col, ashots_col,
                                neutral_col, config.SHOTS_RIDGE, config.SOT_HOME,
                                config.SHOTS_MEAN, link="identity") if have_shots else None)
    gc = (_GlobalAttackDefense(tm, date_col, home_col, away_col, hcorners_col, acorners_col,
                               neutral_col, config.CORNERS_RIDGE, config.SOT_HOME,
                               config.CORNERS_MEAN, link="identity") if have_corn else None)
    raters = [g for g in (gp, gg, gs, gsh, gc) if g is not None]
    team_min, win_min = config.GOALS_WARMUP_TEAM_MATCHES, config.GOALS_WARMUP_MIN_MATCHES

    snaps, counts, cur_month = [], {}, None
    for i, r in tm.iterrows():
        h, a = r[home_col], r[away_col]
        neutral = bool(r[neutral_col]) if neutral_col else False
        month = str(r[date_col])[:7]
        if month != cur_month:               # refit all on everything before this month
            for g in raters:
                g.refit_before(i)
            cur_month = month
        eh, ea = gg.expected(h, a, neutral)
        esh, esa = gs.expected(h, a, neutral)
        warm = (gg.window_n >= win_min
                and counts.get(h, 0) >= team_min and counts.get(a, 0) >= team_min)
        snap = {"date": r[date_col], "home": h, "away": a,
                "exp_home_share": gp.expected(h, a, neutral),
                "exp_home_g": eh, "exp_away_g": ea,
                "exp_home_sot": esh, "exp_away_sot": esa, "warm": warm}
        if gsh is not None:
            snap["exp_home_shots"], snap["exp_away_shots"] = gsh.expected(h, a, neutral)
        if gc is not None:
            snap["exp_home_corners"], snap["exp_away_corners"] = gc.expected(h, a, neutral)
        if id_col and id_col in tm.columns:
            snap["match_id"] = r[id_col]            # lets build_training_frame join cleanly
        snaps.append(snap)
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    for g in raters:
        g.refit_before(len(tm))               # final fit on ALL matches for inference
    poss = PossessionElo(home_adv=gp.home_adv, ratings=dict(gp.ratings))
    goals = GoalsElo(home_adv=gg.home_adv, mu=gg.mu,
                     attack=dict(gg.attack), defense=dict(gg.defense))
    sot = LinearElo(home_adv=gs.home_adv, mu=gs.mu,
                    attack=dict(gs.attack), defense=dict(gs.defense))
    shots = (LinearElo(home_adv=gsh.home_adv, mu=gsh.mu, attack=dict(gsh.attack),
                       defense=dict(gsh.defense)) if gsh is not None else None)
    corners = (LinearElo(home_adv=gc.home_adv, mu=gc.mu, attack=dict(gc.attack),
                         defense=dict(gc.defense)) if gc is not None else None)
    return pd.DataFrame(snaps), poss, goals, sot, shots, corners


def save_elos(poss, goals, sot, shots=None, corners=None, path=None):
    """Persist the fitted ratings so predict() loads them without replaying history."""
    path = path or (config.MODEL_DIR / "elos.json")
    d = {
        "poss_ratings": poss.ratings, "poss_home": poss.home_adv,
        "attack": goals.attack, "defense": goals.defense,
        "goals_home": goals.home_adv, "mu": goals.mu,
        "sot_attack": sot.attack, "sot_defense": sot.defense,
        "sot_home": sot.home_adv, "sot_mu": sot.mu,
    }
    for name, r in (("shots", shots), ("corners", corners)):
        if r is not None:
            d[f"{name}_attack"], d[f"{name}_defense"] = r.attack, r.defense
            d[f"{name}_home"], d[f"{name}_mu"] = r.home_adv, r.mu
    path.write_text(json.dumps(d), encoding="utf-8")
    return path


def load_elos(path=None):
    """Returns (poss, goals, sot, shots, corners). shots/corners are None for older
    elos.json that predate them (callers treat the script features as absent)."""
    path = path or (config.MODEL_DIR / "elos.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    poss = PossessionElo(home_adv=d["poss_home"], ratings=dict(d["poss_ratings"]))
    goals = GoalsElo(home_adv=d["goals_home"], mu=d["mu"],
                     attack=dict(d["attack"]), defense=dict(d["defense"]))
    sot = LinearElo(home_adv=d["sot_home"], mu=d["sot_mu"],
                    attack=dict(d["sot_attack"]), defense=dict(d["sot_defense"]))

    def _lin(name):
        if f"{name}_attack" not in d:
            return None
        return LinearElo(home_adv=d[f"{name}_home"], mu=d[f"{name}_mu"],
                         attack=dict(d[f"{name}_attack"]), defense=dict(d[f"{name}_defense"]))
    return poss, goals, sot, _lin("shots"), _lin("corners")
