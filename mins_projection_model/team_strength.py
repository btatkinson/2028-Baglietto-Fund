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
                  neutral_col=None, id_col="match_id"):
    """Walk team-matches in date order; emit a pre-match rating row per fixture
    (leak-free) and return the fitted Elos for inference. Source-agnostic — feed
    it FBref or api_football frames; column names are adapter args because each
    source's frame shape differs. `neutral_col`, if given, reads a per-row bool
    (WC/continental finals are neutral-site, so the home bump is dropped).

    Returns (snaps, poss, goals, sot). `sot` is a LinearElo whose expected
    shots-on-target — especially the OPPONENT's — drives the saves/shots heads."""
    poss = PossessionElo(config.POSS_K, config.POSS_HOME)
    goals = GoalsElo(config.GOALS_K, config.GOALS_HOME, math.log(config.GOALS_MEAN))
    sot = LinearElo(config.SOT_K, config.SOT_HOME, config.SOT_MEAN)
    snaps = []
    for _, r in team_matches.sort_values(date_col).iterrows():
        h, a = r[home_col], r[away_col]
        neutral = bool(r[neutral_col]) if neutral_col else False
        exp_share = poss.expected(h, a, neutral)
        eh, ea = goals.expected(h, a, neutral)
        esh, esa = sot.expected(h, a, neutral)
        snap = {"date": r[date_col], "home": h, "away": a,
                "exp_home_share": exp_share, "exp_home_g": eh, "exp_away_g": ea,
                "exp_home_sot": esh, "exp_away_sot": esa}
        if id_col and id_col in team_matches.columns:
            snap["match_id"] = r[id_col]            # lets build_training_frame join cleanly
        snaps.append(snap)
        if pd.notna(r.get(hposs_col)):
            poss.update(h, a, float(r[hposs_col]) / 100.0, neutral)
        if pd.notna(r.get(hg_col)) and pd.notna(r.get(ag_col)):
            goals.update(h, a, float(r[hg_col]), float(r[ag_col]), neutral)
        if pd.notna(r.get(hsot_col)) and pd.notna(r.get(asot_col)):
            sot.update(h, a, float(r[hsot_col]), float(r[asot_col]), neutral)
    return pd.DataFrame(snaps), poss, goals, sot


def save_elos(poss, goals, sot, path=None):
    """Persist the fitted Elos so predict() loads them without replaying history."""
    path = path or (config.MODEL_DIR / "elos.json")
    path.write_text(json.dumps({
        "poss_ratings": poss.ratings, "poss_home": poss.home_adv,
        "attack": goals.attack, "defense": goals.defense,
        "goals_home": goals.home_adv, "mu": goals.mu,
        "sot_attack": sot.attack, "sot_defense": sot.defense,
        "sot_home": sot.home_adv, "sot_mu": sot.mu,
    }), encoding="utf-8")
    return path


def load_elos(path=None):
    path = path or (config.MODEL_DIR / "elos.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    poss = PossessionElo(home_adv=d["poss_home"], ratings=dict(d["poss_ratings"]))
    goals = GoalsElo(home_adv=d["goals_home"], mu=d["mu"],
                     attack=dict(d["attack"]), defense=dict(d["defense"]))
    sot = LinearElo(home_adv=d["sot_home"], mu=d["sot_mu"],
                    attack=dict(d["sot_attack"]), defense=dict(d["sot_defense"]))
    return poss, goals, sot
