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
    pooled: bool = False        # cross-league model (European cups)
    blend_weight_dc: float | None = None


def league_cfg(cfg, league_id) -> dict:
    return next((l for l in cfg["leagues"] if l["id"] == league_id), {"id": league_id})


def europe_ids(cfg) -> set:
    return {l["id"] for l in cfg["leagues"] + cfg.get("data_leagues", []) if l.get("europe")}


def _football_year(ts: str) -> int:
    """Season key that turns over on 1 July, so leagues with different calendars align."""
    d = _parse(ts)
    return d.year if d.month >= 7 else d.year - 1


def _parse(ts):
    d = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def train(con, cfg, league_id: int, as_of: str | None = None) -> LeagueModel:
    as_of = as_of or datetime.now(timezone.utc).isoformat()
    lc = league_cfg(cfg, league_id)
    if lc.get("pooled"):
        return _train_pooled(con, cfg, league_id, as_of, lc)
    rows = con.execute(
        f"""SELECT season, kickoff, home_id, away_id, home_goals, away_goals FROM fixtures
            WHERE league_id=? AND status IN {FINISHED} AND kickoff < ? ORDER BY kickoff""",
        (league_id, as_of)).fetchall()
    if len(rows) < 50:
        raise ValueError(f"league {league_id}: only {len(rows)} finished matches before {as_of}")
    mc = cfg["model"]
    ref = _parse(as_of)
    days = [(ref - _parse(r["kickoff"])).days for r in rows]
    # promoted = teams that only appear in the latest season of the training window
    latest = max(r["season"] for r in rows)
    seasons_by_team: dict = {}
    for r in rows:
        for t in (r["home_id"], r["away_id"]):
            seasons_by_team.setdefault(t, set()).add(r["season"])
    has_history = any(r["season"] < latest for r in rows)
    promoted = {t for t, s in seasons_by_team.items() if s == {latest}} if has_history else set()
    dc = DixonColes(half_life_days=mc["dc_half_life_days"], l2=mc["dc_l2"]).fit(
        [r["home_id"] for r in rows], [r["away_id"] for r in rows],
        [r["home_goals"] for r in rows], [r["away_goals"] for r in rows], days,
        promoted=promoted, prior_strength=mc.get("promoted_prior_strength", 6.0))

    elo = Elo(k=mc["elo_k"], home_adv=mc["elo_home_adv"],
              season_regression=mc["elo_season_regression"])
    gaps = elo.run((r["season"], r["home_id"], r["away_id"], r["home_goals"], r["away_goals"])
                   for r in rows)
    outcomes = [2 if r["home_goals"] > r["away_goals"] else 1 if r["home_goals"] == r["away_goals"]
                else 0 for r in rows]
    w = 0.5 ** (np.array(days) / mc["dc_half_life_days"])
    burn = min(100, len(rows) // 4)  # early Elo gaps are noisy
    elo.fit_outcome_map(gaps[burn:], outcomes[burn:], w[burn:])
    return LeagueModel(league_id, as_of, dc, elo, len(rows),
                       blend_weight_dc=lc.get("blend_weight_dc"))


def _train_pooled(con, cfg, league_id, as_of, lc) -> LeagueModel:
    """One model over every stored domestic league plus all European cup matches.
    Domestic games rate teams within each league; European games link the leagues,
    which gives the cross-league strength a Champions League tie needs."""
    rows = con.execute(
        f"""SELECT league_id, kickoff, home_id, away_id, home_goals, away_goals FROM fixtures
            WHERE status IN {FINISHED} AND kickoff < ? AND home_goals IS NOT NULL
            ORDER BY kickoff""", (as_of,)).fetchall()
    eu = europe_ids(cfg)
    n_eu = sum(r["league_id"] in eu for r in rows)
    if n_eu < 50:
        raise ValueError(f"pooled model: only {n_eu} European matches before {as_of}")
    mc = cfg["model"]
    ref = _parse(as_of)
    days = [(ref - _parse(r["kickoff"])).days for r in rows]
    groups = [1 if r["league_id"] in eu else 0 for r in rows]
    # Teams seen only in European games come from leagues we don't load (usually weaker):
    # start them below average, fading as their European matches accumulate.
    domestic = {t for r, g in zip(rows, groups) if g == 0 for t in (r["home_id"], r["away_id"])}
    euro_only = {t for r in rows for t in (r["home_id"], r["away_id"])} - domestic
    dc = DixonColes(half_life_days=mc["dc_half_life_days"], l2=mc["dc_l2"]).fit(
        [r["home_id"] for r in rows], [r["away_id"] for r in rows],
        [r["home_goals"] for r in rows], [r["away_goals"] for r in rows], days, groups=groups,
        promoted=euro_only, prior_strength=mc.get("promoted_prior_strength", 6.0),
        promoted_prior=tuple(mc.get("euro_only_prior", (-0.35, 0.35))))

    elo = Elo(k=mc["elo_k"], home_adv=mc["elo_home_adv"],
              season_regression=mc["elo_season_regression"])
    gaps = elo.run((_football_year(r["kickoff"]), r["home_id"], r["away_id"],
                    r["home_goals"], r["away_goals"]) for r in rows)
    sel = np.array(groups) == 1          # fit the outcome map on European games only
    outcomes = np.array([2 if r["home_goals"] > r["away_goals"] else
                         1 if r["home_goals"] == r["away_goals"] else 0 for r in rows])
    w = 0.5 ** (np.array(days) / mc["dc_half_life_days"])
    elo.fit_outcome_map(gaps[sel], outcomes[sel], w[sel])
    return LeagueModel(league_id, as_of, dc, elo, len(rows), pooled=True,
                       blend_weight_dc=lc.get("blend_weight_dc"))


def base_prediction(lm: LeagueModel, home_id, away_id, cfg, neutral=False) -> dict:
    mg = cfg["model"]["max_goals"]
    lam, mu = lm.dc.expected_goals(home_id, away_id, group=1 if lm.pooled else 0, neutral=neutral)
    m_dc = score_matrix(lam, mu, lm.dc.rho, mg)
    p_dc = probs_1x2(m_dc)
    p_elo = lm.elo.probs(home_id, away_id, neutral=neutral)
    wdc = lm.blend_weight_dc if lm.blend_weight_dc is not None else cfg["model"]["blend_weight_dc"]
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


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def drivers(lm: LeagueModel, home_id, away_id, base: dict, form_home: str, form_away: str,
            odds_fair: dict | None, home: str, away: str) -> list[dict]:
    """Factual model inputs behind a prediction (no LLM involved)."""
    dc = lm.dc
    att_rank = {t: r + 1 for r, t in enumerate(sorted(dc.teams, key=lambda t: -dc.attack[dc._idx[t]]))}
    def_rank = {t: r + 1 for r, t in enumerate(sorted(dc.teams, key=lambda t: dc.defence[dc._idx[t]]))}

    def att(t):
        i = dc._idx.get(t)
        return "n/a (new)" if i is None else f"{dc.attack[i]:+.2f} ({_ordinal(att_rank[t])})"

    def de(t):
        i = dc._idx.get(t)
        return "n/a (new)" if i is None else f"{-dc.defence[i]:+.2f} ({_ordinal(def_rank[t])})"

    scope = "Europe" if lm.pooled else "league"
    rows = [
        {"label": f"Attack rating ({scope} rank)", "home": att(home_id), "away": att(away_id)},
        {"label": f"Defence rating ({scope} rank)", "home": de(home_id), "away": de(away_id)},
        {"label": "Elo rating", "home": f"{lm.elo.rating(home_id):.0f}", "away": f"{lm.elo.rating(away_id):.0f}"},
        {"label": "Expected goals", "home": f"{base['lam']:.2f}", "away": f"{base['mu']:.2f}"},
        {"label": "Form (last 6, newest first)", "home": form_home or "–", "away": form_away or "–"},
        {"label": "Top-flight matches in model", "home": str(dc.n_matches.get(home_id, 0)),
         "away": str(dc.n_matches.get(away_id, 0))},
    ]
    if odds_fair and len(odds_fair) == 3:
        p = base["p_blend"]
        fav = max(range(3), key=lambda k: p[k])
        lab = [f"{home} win", "Draw", f"{away} win"][fav]
        mk = [odds_fair.get("H"), odds_fair.get("D"), odds_fair.get("A")][fav]
        if mk:
            rows.append({"label": "Market view",
                         "note": f"Market has {lab} at {mk*100:.0f}% vs model {p[fav]*100:.0f}%"})
    return rows
