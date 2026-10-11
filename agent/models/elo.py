"""Goal-difference Elo with an ordered-logit map from rating gap to 1X2."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class Elo:
    def __init__(self, k=20, home_adv=60, season_regression=0.33, base=1500):
        self.k, self.home_adv, self.reg, self.base = k, home_adv, season_regression, base
        self.r: dict = {}
        self.cut = np.array([-0.4, 0.4])  # ordered-logit cutpoints
        self.scale = 1 / 150

    def rating(self, t):
        return self.r.get(t, self.base - 100)  # unseen (usually promoted) teams start below average

    def run(self, matches):
        """matches: iterable of (season, home, away, hg, ag) in chronological order.
        Returns pre-match rating gaps (home - away + home_adv) for each match."""
        gaps, season = [], None
        for s, h, a, hg, ag in matches:
            if season is not None and s != season:
                self.r = {t: self.base + (1 - self.reg) * (v - self.base) for t, v in self.r.items()}
            season = s
            gap = self.rating(h) - self.rating(a) + self.home_adv
            gaps.append(gap)
            exp_h = 1 / (1 + 10 ** (-gap / 400))
            res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
            mult = np.log(abs(hg - ag) + 1) + 1 if hg != ag else 1.0
            d = self.k * mult * (res - exp_h)
            self.r[h] = self.rating(h) + d
            self.r[a] = self.rating(a) - d
        return np.array(gaps)

    def fit_outcome_map(self, gaps, outcomes, weights=None):
        """Ordered logit: outcome 0=away,1=draw,2=home."""
        y = np.asarray(outcomes)
        w = np.ones(len(y)) if weights is None else np.asarray(weights)

        def nll(p):
            c1, d, s = p[0], abs(p[1]) + 1e-3, p[2]
            z = gaps * s
            pa = expit(c1 - z)
            pd = expit(c1 + d - z) - pa
            ph = 1 - pa - pd
            pr = np.stack([pa, pd, ph], 1)[np.arange(len(y)), y]
            return -(w * np.log(np.clip(pr, 1e-12, 1))).sum()

        res = minimize(nll, [-0.3, 0.6, 1 / 150], method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
        c1, d, s = res.x
        self.cut = np.array([c1, c1 + abs(d) + 1e-3])
        self.scale = s
        return self

    def probs(self, home, away, neutral=False):
        gap = self.rating(home) - self.rating(away) + (0 if neutral else self.home_adv)
        z = gap * self.scale
        pa = expit(self.cut[0] - z)
        pd = expit(self.cut[1] - z) - pa
        return np.array([1 - pa - pd, pd, pa])  # H, D, A
