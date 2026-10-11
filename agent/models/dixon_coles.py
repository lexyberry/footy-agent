"""Time-weighted Dixon-Coles (1997) model with L2 shrinkage."""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln


def _tau(x, y, lam, mu, rho):
    t = np.ones_like(lam)
    t = np.where((x == 0) & (y == 0), 1 - lam * mu * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lam * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + mu * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-10, None)


@dataclass
class DixonColes:
    half_life_days: float = 365.0
    l2: float = 0.05
    teams: list = field(default_factory=list)
    attack: np.ndarray | None = None
    defence: np.ndarray | None = None
    home_adv: float = 0.25
    rho: float = -0.05
    n_matches: dict = field(default_factory=dict)

    def fit(self, home_ids, away_ids, hg, ag, days_ago, promoted=(), prior_strength=6.0,
            promoted_prior=(-0.25, 0.25), groups=None):
        """promoted: teams with no prior top-flight history. They are shrunk toward
        promoted_prior (attack, defence-weakness); the pull fades as matches accumulate.
        groups: optional int per match selecting its home-advantage term
        (0 = domestic, 1 = European cup), so cup home advantage is estimated separately."""
        home_ids, away_ids = np.asarray(home_ids), np.asarray(away_ids)
        gi = np.zeros(len(home_ids), int) if groups is None else np.asarray(groups, int)
        G = int(gi.max()) + 1 if len(gi) else 1
        hg, ag = np.asarray(hg, float), np.asarray(ag, float)
        w = 0.5 ** (np.asarray(days_ago, float) / self.half_life_days)

        self.teams = sorted(set(home_ids) | set(away_ids))
        idx = {t: i for i, t in enumerate(self.teams)}
        hi = np.array([idx[t] for t in home_ids])
        ai = np.array([idx[t] for t in away_ids])
        n = len(self.teams)
        self.n_matches = {t: int(((home_ids == t) | (away_ids == t)).sum()) for t in self.teams}
        lg_h, lg_a = gammaln(hg + 1), gammaln(ag + 1)

        # per-team prior: mean and strength (strength decays with matches played)
        promoted = set(promoted)
        mu_att = np.array([promoted_prior[0] if t in promoted else 0.0 for t in self.teams])
        mu_def = np.array([promoted_prior[1] if t in promoted else 0.0 for t in self.teams])
        lam_t = np.array([self.l2 + (prior_strength * np.exp(-self.n_matches[t] / 10)
                                     if t in promoted else 0.0) for t in self.teams])

        def unpack(p):
            return p[:n], p[n:2 * n], p[2 * n:2 * n + G], p[2 * n + G]

        def nll(p):
            att, de, home, rho = unpack(p)
            lam = np.exp(home[gi] + att[hi] + de[ai])
            mu = np.exp(att[ai] + de[hi])
            ll = (np.log(_tau(hg, ag, lam, mu, rho))
                  + hg * np.log(lam) - lam - lg_h
                  + ag * np.log(mu) - mu - lg_a)
            pen = (lam_t * ((att - mu_att) ** 2 + (de - mu_def) ** 2)).sum() + 10.0 * att.sum() ** 2
            return -(w * ll).sum() + pen

        def grad(p):
            att, de, home, rho = unpack(p)
            lam = np.exp(home[gi] + att[hi] + de[ai])
            mu = np.exp(att[ai] + de[hi])
            # Poisson parts
            dlam = w * (hg - lam)  # d ll / d log(lam)
            dmu = w * (ag - mu)
            # tau parts
            tau = _tau(hg, ag, lam, mu, rho)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            dlam = dlam + w * np.where(m00, -lam * mu * rho, 0) / tau \
                        + w * np.where(m01, lam * rho, 0) / tau
            dmu = dmu + w * np.where(m00, -lam * mu * rho, 0) / tau \
                      + w * np.where(m10, mu * rho, 0) / tau
            drho = (w * (np.where(m00, -lam * mu, 0) + np.where(m01, lam, 0)
                         + np.where(m10, mu, 0) + np.where(m11, -1, 0)) / tau).sum()
            g_att = np.bincount(hi, dlam, n) + np.bincount(ai, dmu, n)
            g_de = np.bincount(ai, dlam, n) + np.bincount(hi, dmu, n)
            g_home = np.bincount(gi, dlam, G)
            g = -np.concatenate([g_att, g_de, g_home, [drho]])
            g[:n] += 2 * lam_t * (att - mu_att) + 20.0 * att.sum()
            g[n:2 * n] += 2 * lam_t * (de - mu_def)
            return g

        x0 = np.concatenate([np.zeros(2 * n), np.full(G, 0.25), [-0.05]])
        bounds = [(-3, 3)] * (2 * n) + [(-1, 1)] * G + [(-0.3, 0.3)]
        res = minimize(nll, x0, jac=grad, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 3000})
        self.attack, self.defence, homes, self.rho = unpack(res.x)
        self.home_advs = np.array(homes, float)
        self.home_adv = float(homes[0])  # domestic (group 0)
        self._idx = idx
        return self

    def expected_goals(self, home_id, away_id, group=0, neutral=False):
        i, j = self._idx.get(home_id), self._idx.get(away_id)
        # unseen (e.g. newly promoted with no data): use a below-average prior
        a_h = self.attack[i] if i is not None else -0.25
        d_h = self.defence[i] if i is not None else 0.25
        a_a = self.attack[j] if j is not None else -0.25
        d_a = self.defence[j] if j is not None else 0.25
        h = 0.0 if neutral else float(self.home_advs[min(group, len(self.home_advs) - 1)])
        return float(np.exp(h + a_h + d_a)), float(np.exp(a_a + d_h))

    def score_matrix(self, lam, mu, max_goals=10):
        return score_matrix(lam, mu, self.rho, max_goals)


def score_matrix(lam, mu, rho, max_goals=10):
    g = np.arange(max_goals + 1)
    from scipy.stats import poisson
    m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()
