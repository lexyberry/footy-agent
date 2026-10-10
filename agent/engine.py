"""Train per-league models and produce base + adjusted predictions."""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .db import FINISHED
from .markets import derived, probs_1x2, rescale_to_1x2, top_scores
from .models.dixon_coles import DixonColes, score_matrix
from .models.elo import Elo


@dataclass
class LeagueModel:
    league_id: int
    as_of: str
    dc: DixonColes
    elo: Elo
    n_train: int


def _parse(ts):
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def train(con, cfg, league_id: int, as_of: str | None = None) -> LeagueModel:
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    rows = con.execute(
        f"""SELECT season, kickoff, home_id, away_id, home_goals, away_goals FROM fixtures
            WHERE league_id=? AND status IN {FINISHED} AND kickoff < ? ORDER BY kickoff""",
        (league_id, as_of)).fetchall()
    if len(rows) < 50:
        raise ValueError(f"league {league_id}: only {len(rows)} finished matches before {as_of}")
    mc = cfg["model"]
    ref = _parse(as_of)
    days = [(ref - _parse(r["kickoff"])).days for r in rows]
    dc = DixonColes(half_life_days=mc["dc_half_life_days"], l2=mc["dc_l2"]).fit(
        [r["home_id"] for r in rows], [r["away_id"] for r in rows],
        [r["home_goals"] for r in rows], [r["away_goals"] for r in rows], days)

    elo = Elo(k=mc["elo_k"], home_adv=mc["elo_home_adv"],
              season_regression=mc["elo_season_regression"])
    gaps = elo.run((r["season"], r["home_id"], r["away_id"], r["home_goals"], r["away_goals"])
                   for r in rows)
    outcomes = [2 if r["home_goals"] > r["away_goals"] else 1 if r["home_goals"] == r["away_goals"]
                else 0 for r in rows]
    w = 0.5 ** (np.array(days) / mc["dc_half_life_days"])
    burn = min(100, len(rows) // 4)  # early Elo gaps are noisy
    elo.fit_outcome_map(gaps[burn:], outcomes[burn:], w[burn:])
    return LeagueModel(league_id, as_of, dc, elo, len(rows))


def base_prediction(lm: LeagueModel, home_id, away_id, cfg) -> dict:
    mg = cfg["model"]["max_goals"]
    lam, mu = lm.dc.expected_goals(home_id, away_id)
    m_dc = score_matrix(lam, mu, lm.dc.rho, mg)
    p_dc = probs_1x2(m_dc)
    p_elo = lm.elo.probs(home_id, away_id)
    wdc = cfg["model"]["blend_weight_dc"]
    p_blend = wdc * p_dc + (1 - wdc) * p_elo
    m = rescale_to_1x2(m_dc, p_blend)
    return {"lam": lam, "mu": mu, "p_dc": p_dc, "p_elo": p_elo, "p_blend": p_blend,
            "matrix": m, "m_dc": m_dc}


def adjusted_prediction(lm: LeagueModel, base: dict, mult_h: float, mult_a: float, cfg) -> dict:
    """Apply bounded expected-goals multipliers, then enforce the max 1X2 shift."""
    cc, mg, rho = cfg["claude"], cfg["model"]["max_goals"], lm.dc.rho
    lo, hi = 1 - cc["max_lambda_change"], 1 + cc["max_lambda_change"]
    mult_h, mult_a = float(np.clip(mult_h, lo, hi)), float(np.clip(mult_a, lo, hi))
    cap = cc["max_prob_shift_pp"] / 100

    def build(t):
        lam = base["lam"] * mult_h ** t
        mu = base["mu"] * mult_a ** t
        m_adj = score_matrix(lam, mu, rho, mg)
        ratio = probs_1x2(m_adj) / base["p_dc"]
        target = base["p_blend"] * ratio
        target /= target.sum()
        return lam, mu, rescale_to_1x2(m_adj, target), target

    t = 1.0
    lam, mu, m, p = build(t)
    if np.abs(p - base["p_blend"]).max() > cap:
        lo_t, hi_t = 0.0, 1.0
        for _ in range(30):
            t = (lo_t + hi_t) / 2
            if np.abs(build(t)[3] - base["p_blend"]).max() > cap:
                hi_t = t
            else:
                lo_t = t
        t = lo_t
        lam, mu, m, p = build(t)
    return {"lam": lam, "mu": mu, "matrix": m, "applied_scale": round(t, 3),
            "mult_h": round(mult_h ** t, 3), "mult_a": round(mult_a ** t, 3)}


def summarize(m) -> dict:
    d = derived(m)
    return {"p_home": float(d["1x2"][0]), "p_draw": float(d["1x2"][1]),
            "p_away": float(d["1x2"][2]), "p_over25": d["over25"], "p_btts": d["btts"],
            "top_scores": top_scores(m, 5), "_derived": d}
