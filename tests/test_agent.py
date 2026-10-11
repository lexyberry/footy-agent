"""Offline tests: simulated league, fake API-Football, fake Claude."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from agent import backtest, engine, evaluate, ingest, pipeline, report
from agent.config import load_config
from agent.db import connect
from agent.markets import devig, find_value

N_TEAMS = 20
RNG = np.random.default_rng(7)
ATT = RNG.normal(0, 0.3, N_TEAMS); ATT -= ATT.mean()
DEF = RNG.normal(0, 0.25, N_TEAMS)
HOME, RHO = 0.25, -0.06
NOW = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)


def sim_fixtures():
    """3 seasons of double round-robin; the last season is partly unplayed."""
    out, fid = [], 1
    for s, season in enumerate([2024, 2025, 2026]):
        start = datetime(season, 8, 10, 15, tzinfo=timezone.utc)
        pairs = [(i, j) for i in range(N_TEAMS) for j in range(N_TEAMS) if i != j]
        RNG.shuffle(pairs)
        for k, (i, j) in enumerate(pairs):
            ko = start + timedelta(days=7 * (k // 10))
            played = ko < NOW
            lam, mu = np.exp(HOME + ATT[i] + DEF[j]), np.exp(ATT[j] + DEF[i])
            out.append({
                "fixture": {"id": fid, "date": ko.isoformat(),
                            "status": {"short": "FT" if played else "NS"}},
                "league": {"id": 39, "season": season, "round": f"R{k // 10 + 1}"},
                "teams": {"home": {"id": 100 + i, "name": f"Team{i}"},
                          "away": {"id": 100 + j, "name": f"Team{j}"}},
                "goals": {"home": int(RNG.poisson(lam)) if played else None,
                          "away": int(RNG.poisson(mu)) if played else None}})
            fid += 1
    return out


FIX = sim_fixtures()


class FakeApi:
    calls = 0

    def fixtures(self, league, season, ttl=0):
        return [f for f in FIX if f["league"]["season"] == season and league == 39]

    def fixture(self, fid):
        return next(f for f in FIX if f["fixture"]["id"] == fid)

    def odds(self, fid):
        return [{"bookmakers": [{"id": 1, "name": "Superbet", "bets": [
            {"name": "Match Winner", "values": [{"value": "Home", "odd": "2.40"},
                                                {"value": "Draw", "odd": "3.30"},
                                                {"value": "Away", "odd": "3.10"}]},
            {"name": "Goals Over/Under", "values": [{"value": "Over 2.5", "odd": "1.95"},
                                                     {"value": "Under 2.5", "odd": "1.85"}]},
            {"name": "Both Teams Score", "values": [{"value": "Yes", "odd": "1.80"},
                                                     {"value": "No", "odd": "2.00"}]}]}]}]

    def injuries(self, fid):
        return [{"team": {"name": "Team0"}, "player": {"name": "Star Striker", "type": "Missing Fixture",
                                                       "reason": "Hamstring"}}]

    def lineups(self, fid):
        return []

    def standings(self, league, season):
        return []


class FakeClaude:
    def __init__(self, h=0.90, a=1.30):
        self.h, self.a = h, a
        self.messages = self
        self.last = None

    def create(self, **kw):
        self.last = kw
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input={
            "home_xg_multiplier": self.h, "away_xg_multiplier": self.a,
            "key_factors": ["Star Striker out (hamstring)"], "reasoning": "test",
            "confidence": "medium", "summary": "Team0 are favourites because of stronger ratings.",
            "reasons": ["Attack rank 1st vs 12th", "Elo gap +150"]})])


@pytest.fixture
def setup(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["leagues"] = [{"id": 39, "name": "Sim League"}]
    cfg["db_path"] = str(tmp_path / "t.db")
    cfg["reports_dir"] = str(tmp_path / "reports")
    con = connect(cfg["db_path"])
    api = FakeApi()
    ingest.ingest_fixtures(api, con, cfg)
    pipeline._models.clear()
    return api, con, cfg


def test_ingest_counts(setup):
    _, con, _ = setup
    assert con.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0] == 3 * 380
    assert con.execute("SELECT COUNT(*) FROM fixtures WHERE status='NS'").fetchone()[0] > 0


def test_model_recovers_strength(setup):
    _, con, cfg = setup
    lm = engine.train(con, cfg, 39, NOW.isoformat())
    idx = [lm.dc.teams.index(100 + i) for i in range(N_TEAMS)]
    assert np.corrcoef(ATT, lm.dc.attack[idx])[0, 1] > 0.8
    assert np.corrcoef(DEF, lm.dc.defence[idx])[0, 1] > 0.7
    assert 0.1 < lm.dc.home_adv < 0.4
    # Elo agrees with true strength ordering
    strength = ATT - DEF
    elo = [lm.elo.rating(100 + i) for i in range(N_TEAMS)]
    assert np.corrcoef(strength, elo)[0, 1] > 0.75


def test_probabilities_coherent(setup):
    _, con, cfg = setup
    lm = engine.train(con, cfg, 39, NOW.isoformat())
    b = engine.base_prediction(lm, 100, 101, cfg)
    assert abs(b["matrix"].sum() - 1) < 1e-9
    np.testing.assert_allclose(engine.summarize(b["matrix"])["p_home"], b["p_blend"][0], atol=1e-9)


def test_claude_adjustment_is_bounded(setup):
    _, con, cfg = setup
    lm = engine.train(con, cfg, 39, NOW.isoformat())
    b = engine.base_prediction(lm, 100, 101, cfg)
    adj = engine.adjusted_prediction(lm, b, 0.5, 2.0, cfg)  # wild inputs
    assert adj["mult_h"] >= 0.85 - 1e-9 and adj["mult_a"] <= 1.15 + 1e-9
    shift = np.abs(np.array([engine.summarize(adj["matrix"])[k] for k in ("p_home", "p_draw", "p_away")])
                   - b["p_blend"]).max()
    assert shift <= cfg["claude"]["max_prob_shift_pp"] / 100 + 1e-6
    neutral = engine.adjusted_prediction(lm, b, 1.0, 1.0, cfg)
    np.testing.assert_allclose(neutral["matrix"], b["matrix"], atol=1e-12)


def test_end_to_end_predict_and_report(setup):
    api, con, cfg = setup
    fx = con.execute("SELECT id FROM fixtures WHERE status='NS' AND (home_id=100 OR away_id=100) "
                     "ORDER BY kickoff LIMIT 1").fetchone()["id"]
    fc = FakeClaude()
    r = pipeline.predict_fixture(api, con, cfg, fx, claude_client=fc)
    assert abs(sum(r["probs"][k] for k in ("home", "draw", "away")) - 100) < 0.5
    assert r["bookmaker"] == "Superbet" and "1x2" in r["odds"]
    assert r["context"]["key_factors"] == ["Star Striker out (hamstring)"]
    assert "Star Striker" in str(fc.last["messages"])          # injuries reached Claude
    assert fc.last["tools"][0]["name"] == "submit_adjustment"
    assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
    html = report.render([r], cfg, "Test")
    assert "Team" in html and "viewport" in html


def test_claude_failure_falls_back_to_neutral(setup):
    api, con, cfg = setup

    class Broken:
        messages = SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(RuntimeError("down")))

    fx = con.execute("SELECT id FROM fixtures WHERE status='NS' LIMIT 1").fetchone()["id"]
    r = pipeline.predict_fixture(api, con, cfg, fx, claude_client=Broken(), save=False)
    assert r["adjustment"]["home"] == 1.0 and r["adjustment"]["away"] == 1.0


def test_value_and_devig():
    fair = devig({"H": 2.0, "D": 3.5, "A": 4.0})
    assert abs(sum(fair.values()) - 1) < 1e-12
    cfg = load_config()
    d = {"1x2": np.array([0.60, 0.22, 0.18]), "over25": 0.5, "btts": 0.5}
    vb = find_value(d, {"1x2": {"H": 2.0, "D": 3.5, "A": 4.0}}, cfg)
    assert vb and vb[0]["selection"] == "H"
    assert vb[0]["stake_pct"] <= cfg["market"]["max_stake_pct"]
    # no value when model == market
    assert find_value({"1x2": np.array(list(fair.values())), "over25": .5, "btts": .5},
                      {"1x2": {"H": 2.0, "D": 3.5, "A": 4.0}}, cfg) == []


def test_settle_and_scoreboard(setup):
    api, con, cfg = setup
    fx = con.execute("SELECT * FROM fixtures WHERE status='NS' ORDER BY kickoff LIMIT 1").fetchone()
    cfg["market"]["min_edge"] = -1  # force a bet to be logged
    cfg["market"]["min_prob"] = 0
    pipeline.predict_fixture(api, con, cfg, fx["id"], claude_client=FakeClaude(1, 1))
    con.execute("UPDATE predictions SET created_at='2000-01-01'")
    con.execute("UPDATE odds_snapshots SET taken_at='2000-01-01'")
    con.execute("UPDATE fixtures SET status='FT', home_goals=2, away_goals=1 WHERE id=?", (fx["id"],))
    con.commit()
    assert evaluate.settle(con) > 0
    sb = evaluate.scoreboard(con)
    assert sb["n_predictions"] == 1 and "market_closing" in sb and sb["bets"]["avg_clv"] == 0.0


def test_backtest_beats_baseline(setup):
    _, con, cfg = setup
    res = backtest.run(con, cfg, 39, "2025-09-01", "2026-06-01", step_days=28)
    assert res["n"] > 200
    assert res["blend_at_best"]["log_loss"] < res["baseline_base_rates"]["log_loss"]


def test_promoted_team_shrinkage():
    """A promoted side with a lucky 6-game start shouldn't be rated as a title contender."""
    from agent.models.dixon_coles import DixonColes
    rng = np.random.default_rng(3)
    H, A, hg, ag, days = [], [], [], [], []
    for k in range(600):  # 10 established teams, ~2 seasons
        i, j = rng.choice(10, 2, replace=False)
        H.append(i); A.append(j); hg.append(rng.poisson(1.4)); ag.append(rng.poisson(1.1))
        days.append(400 + k)
    for k in range(6):  # promoted team 99 wins 3-0 six times
        H.append(99); A.append(k); hg.append(3); ag.append(0); days.append(10 + k)
    plain = DixonColes(l2=0.05).fit(H, A, hg, ag, days)
    shrunk = DixonColes(l2=0.05).fit(H, A, hg, ag, days, promoted={99})
    i = plain.teams.index(99)
    assert shrunk.attack[i] < plain.attack[i] - 0.2
    lam_p, _ = plain.expected_goals(99, 0)
    lam_s, _ = shrunk.expected_goals(99, 0)
    assert lam_s < lam_p


def test_promoted_gradient():
    import scipy.optimize as so
    from agent.models import dixon_coles as dc
    cap = {}
    orig = so.minimize
    dc.minimize = lambda f, x0, jac=None, **k: (cap.update(f=f, g=jac), orig(f, x0, jac=jac, **k))[1]
    try:
        rng = np.random.default_rng(1)
        H = rng.integers(0, 8, 300); A = (H + rng.integers(1, 8, 300)) % 8
        dc.DixonColes().fit(H, A, rng.poisson(1.4, 300), rng.poisson(1.1, 300),
                            rng.integers(0, 700, 300), promoted={3})
        x = rng.normal(0, .2, 18); x[-1] = -0.1
        assert so.check_grad(cap["f"], cap["g"], x) < 1e-3
    finally:
        dc.minimize = orig


def test_explanations(setup):
    api, con, cfg = setup
    fx = con.execute("SELECT id FROM fixtures WHERE status='NS' AND home_id=100 LIMIT 1").fetchone()["id"]
    fc = FakeClaude()
    r = pipeline.predict_fixture(api, con, cfg, fx, claude_client=fc, save=False)
    assert r["pick"].endswith("win") or r["pick"] == "Draw"
    assert r["summary"].startswith("Team0 are favourites")
    assert r["reasons"] == ["Attack rank 1st vs 12th", "Elo gap +150"]
    labels = [d["label"] for d in r["drivers"]]
    assert "Attack rating (league rank)" in labels and "Form (last 6, newest first)" in labels
    assert any(d["label"] == "Market view" for d in r["drivers"])
    assert '"drivers"' in str(fc.last["messages"])          # Claude sees the drivers
    form = next(d for d in r["drivers"] if d["label"].startswith("Form"))
    assert set(form["home"]) <= set("WDL") and len(form["home"]) == 6


def test_template_explanation_when_claude_fails(setup):
    api, con, cfg = setup

    class Broken:
        messages = SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(RuntimeError("down")))

    fx = con.execute("SELECT id FROM fixtures WHERE status='NS' LIMIT 1").fetchone()["id"]
    r = pipeline.predict_fixture(api, con, cfg, fx, claude_client=Broken(), save=False)
    assert r["pick"] in r["summary"] and "expected goals" in r["summary"]
    assert len(r["reasons"]) >= 3


def test_backtest_store_and_report(setup):
    _, con, cfg = setup
    res = backtest.run_all(con, cfg, "2025-09-01", "2026-06-01")
    assert res["leagues"] and res["leagues"][0]["league_name"] == "Sim League"
    assert "blend_grid_logloss" not in res["leagues"][0]
    backtest.store(con, res)
    assert "log-loss base" in backtest.one_line(res["leagues"][0])


def _pooled_world(tmp_path):
    """League 39 (strong) and league 140 (weak), each 10 teams, linked only by cup games."""
    from agent.db import upsert_fixture
    rng = np.random.default_rng(11)
    cfg = copy.deepcopy(load_config())
    cfg["leagues"] = [{"id": 39, "name": "Strong"}, {"id": 140, "name": "Weak"},
                      {"id": 2, "name": "Champions League", "europe": True, "pooled": True}]
    cfg["data_leagues"] = [{"id": 3, "name": "Europa League", "europe": True}]
    cfg["db_path"] = str(tmp_path / "p.db")
    con = connect(cfg["db_path"])
    strength = {}
    for t in range(10):
        strength[100 + t] = (0.35 + RNG.normal(0, .15), -0.35 + RNG.normal(0, .1))  # strong league
        strength[200 + t] = (-0.35 + RNG.normal(0, .15), 0.35 + RNG.normal(0, .1))  # weak league
    fid = [10_000]

    def add(lg, season, ko, h, a, home_adv):
        lam = np.exp(home_adv + strength[h][0] + strength[a][1])
        mu = np.exp(strength[a][0] + strength[h][1])
        fid[0] += 1
        upsert_fixture(con, {"fixture": {"id": fid[0], "date": ko.isoformat(), "status": {"short": "FT"}},
                             "league": {"id": lg, "season": season, "round": "R"},
                             "teams": {"home": {"id": h, "name": f"T{h}"}, "away": {"id": a, "name": f"T{a}"}},
                             "goals": {"home": int(rng.poisson(lam)), "away": int(rng.poisson(mu))}})

    for season in (2024, 2025):
        start = datetime(season, 8, 10, tzinfo=timezone.utc)
        for base, lg in ((100, 39), (200, 140)):
            k = 0
            for i in range(10):
                for j in range(10):
                    if i != j:
                        add(lg, season, start + timedelta(days=3 * (k // 5)), base + i, base + j, 0.28)
                        k += 1
        for k in range(60):  # cross-league cup games, smaller home advantage
            h, a = (100 + rng.integers(10), 200 + rng.integers(10))
            if k % 2:
                h, a = a, h
            add(2 if k % 3 else 3, season, start + timedelta(days=30 + 4 * k), int(h), int(a), 0.12)
    con.commit()
    return con, cfg


def test_pooled_model_learns_league_strength(tmp_path):
    con, cfg = _pooled_world(tmp_path)
    as_of = "2026-08-01T00:00:00+00:00"
    lm = engine.train(con, cfg, 2, as_of)
    assert lm.pooled and len(lm.dc.home_advs) == 2
    assert lm.dc.home_advs[1] < lm.dc.home_advs[0]           # cup home edge is smaller
    att = {t: lm.dc.attack[lm.dc._idx[t]] for t in lm.dc.teams}
    strong = np.mean([att[100 + t] for t in range(10)])
    weak = np.mean([att[200 + t] for t in range(10)])
    assert strong - weak > 0.4                                 # true gap is 0.7
    b = engine.base_prediction(lm, 205, 105, cfg)             # weak side at home vs strong side
    assert b["p_blend"][2] > b["p_blend"][0]                   # away (strong) still favoured
    # domestic-only model for the weak league can't see the gap
    dom = engine.train(con, cfg, 140, as_of)
    assert abs(np.mean(dom.dc.attack)) < 0.05
    # neutral venue removes home advantage
    n = engine.base_prediction(lm, 105, 205, cfg, neutral=True)
    h = engine.base_prediction(lm, 105, 205, cfg)
    assert n["lam"] < h["lam"]


def test_grouped_home_gradient():
    import scipy.optimize as so
    from agent.models import dixon_coles as dc
    cap = {}
    orig = so.minimize
    dc.minimize = lambda f, x0, jac=None, **k: (cap.update(f=f, g=jac), orig(f, x0, jac=jac, **k))[1]
    try:
        rng = np.random.default_rng(2)
        H = rng.integers(0, 8, 300); A = (H + rng.integers(1, 8, 300)) % 8
        dc.DixonColes().fit(H, A, rng.poisson(1.4, 300), rng.poisson(1.1, 300),
                            rng.integers(0, 700, 300), groups=rng.integers(0, 2, 300))
        x = rng.normal(0, .2, 19); x[-1] = -0.1
        assert so.check_grad(cap["f"], cap["g"], x) < 1e-3
    finally:
        dc.minimize = orig


def test_rate_limit_retry(monkeypatch, tmp_path):
    from agent import api_football as af
    bodies = [{"errors": {"rateLimit": "Too many requests"}, "response": []},
              {"errors": [], "response": [{"ok": 1}], "paging": {"current": 1, "total": 1}}]
    calls = []

    class R:
        status_code = 200
        def __init__(self, b): self.b = b
        def raise_for_status(self): pass
        def json(self): return self.b

    api = af.ApiFootball(key="k", cache_dir=str(tmp_path))
    monkeypatch.setattr(api.s, "get", lambda *a, **k: calls.append(1) or R(bodies[len(calls) - 1]))
    monkeypatch.setattr(af.time, "sleep", lambda s: None)
    assert api.get("fixtures", ttl=0, league=1) == [{"ok": 1}]
    assert len(calls) == 2


def test_euro_only_team_starts_below_average(tmp_path):
    """A team from an unloaded league that wins its 2 cup games shouldn't top the ratings."""
    from agent.db import upsert_fixture
    con, cfg = _pooled_world(tmp_path)
    cfg["leagues"].append({"id": 848, "name": "Conference League", "europe": True, "pooled": True})
    for k, opp in enumerate((205, 206)):
        upsert_fixture(con, {"fixture": {"id": 99_000 + k, "date": f"2026-07-0{k + 1}T18:00:00+00:00",
                                         "status": {"short": "FT"}},
                             "league": {"id": 848, "season": 2026, "round": "Q1"},
                             "teams": {"home": {"id": 900, "name": "Minnows"}, "away": {"id": opp, "name": "x"}},
                             "goals": {"home": 3, "away": 0}})
    con.commit()
    lm = engine.train(con, cfg, 848, "2026-08-01T00:00:00+00:00")
    att = lm.dc.attack[lm.dc._idx[900]]
    strong = np.mean([lm.dc.attack[lm.dc._idx[100 + t]] for t in range(10)])
    assert att < strong
