"""End-to-end: fixture -> base model -> Claude context -> value check -> store."""
import json
import logging
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import context, engine, ingest
from .markets import devig, find_value

log = logging.getLogger(__name__)
_models: dict = {}


def get_model(con, cfg, league_id):
    key = (league_id, date.today().isoformat())
    if key not in _models:
        _models[key] = engine.train(con, cfg, league_id)
    return _models[key]


def _pct(x):
    return round(100 * float(x), 1)


def predict_fixture(api, con, cfg, fixture_id: int, stage="base", manual_odds=None,
                    claude_client=None, save=True) -> dict:
    fx = con.execute("SELECT * FROM fixtures WHERE id=?", (fixture_id,)).fetchone()
    if fx is None and api is not None:
        raw = api.fixture(fixture_id)
        if raw is None:
            raise ValueError(f"fixture {fixture_id} not found")
        from .db import upsert_fixture
        upsert_fixture(con, raw)
        con.commit()
        fx = con.execute("SELECT * FROM fixtures WHERE id=?", (fixture_id,)).fetchone()
    if fx is None:
        raise ValueError(f"fixture {fixture_id} not in database")
    fx = dict(fx)

    lm = get_model(con, cfg, fx["league_id"])
    neutral = lm.pooled and (fx.get("round") or "").strip().lower() == "final"
    base = engine.base_prediction(lm, fx["home_id"], fx["away_id"], cfg, neutral=neutral)
    base_sum = engine.summarize(base["matrix"])

    ctx = context.gather(api, con, fx, cfg)

    if manual_odds:
        book, odds = "manual", manual_odds
    elif api is not None:
        book, odds = ingest.snapshot_odds(api, con, cfg, fixture_id)
    else:
        book, odds = ingest.latest_odds(con, fixture_id)

    form_h = context.form_string(con, fx["home_id"], fx["kickoff"])
    form_a = context.form_string(con, fx["away_id"], fx["kickoff"])
    drv = engine.drivers(lm, fx["home_id"], fx["away_id"], base, form_h, form_a,
                         devig(odds.get("1x2", {})) if odds else None, fx["home"], fx["away"])
    brief = {"xg_home": round(base["lam"], 2), "xg_away": round(base["mu"], 2),
             "p_home": _pct(base_sum["p_home"]), "p_draw": _pct(base_sum["p_draw"]),
             "p_away": _pct(base_sum["p_away"])}
    adj = context.ask_claude(ctx, brief, cfg, client=claude_client, drivers=drv)
    fin = engine.adjusted_prediction(lm, base, adj["home_xg_multiplier"],
                                     adj["away_xg_multiplier"], cfg)
    s = engine.summarize(fin["matrix"])
    value = find_value(s["_derived"], odds, cfg) if odds else []

    most = max(("home", "draw", "away"), key=lambda k: s[f"p_{k}"])
    pick = {"home": f"{fx['home']} win", "draw": "Draw", "away": f"{fx['away']} win"}[most]
    summary = adj.get("summary") or _template_summary(fx, s, pick, drv)
    reasons = adj.get("reasons") or _template_reasons(fx, drv)

    result = {
        "fixture_id": fixture_id, "match": f"{fx['home']} vs {fx['away']}",
        "kickoff": fx["kickoff"], "league_id": fx["league_id"], "stage": stage,
        "xg": [round(fin["lam"], 2), round(fin["mu"], 2)],
        "base_xg": [round(base["lam"], 2), round(base["mu"], 2)],
        "probs": {"home": _pct(s["p_home"]), "draw": _pct(s["p_draw"]), "away": _pct(s["p_away"]),
                  "over25": _pct(s["p_over25"]), "btts": _pct(s["p_btts"])},
        "base_probs": {"home": _pct(base_sum["p_home"]), "draw": _pct(base_sum["p_draw"]),
                       "away": _pct(base_sum["p_away"])},
        "most_likely": most, "pick": pick, "summary": summary, "reasons": reasons,
        "drivers": drv,
        "top_scores": [(sc, _pct(p)) for sc, p in s["top_scores"]],
        "context": {k: adj[k] for k in ("key_factors", "reasoning", "confidence")},
        "adjustment": {"home": fin["mult_h"], "away": fin["mult_a"],
                       "scale_applied": fin["applied_scale"]},
        "bookmaker": book, "odds": odds, "value_bets": value,
        "has_lineups": "lineups" in ctx,
    }
    if save:
        m6 = fin["matrix"][:6, :6]
        result["prediction_id"] = _save(con, result, s, m6.round(4).tolist())
    return result


def _template_summary(fx, s, pick, drv) -> str:
    """Fallback explanation built only from model numbers (used if Claude is unavailable)."""
    p = max(s["p_home"], s["p_draw"], s["p_away"])
    strength = "a clear favourite" if p >= 0.6 else "a modest favourite" if p >= 0.45 else "a narrow lean"
    xg = next(d for d in drv if d["label"] == "Expected goals")
    text = (f"The model makes {pick} {strength} at {p*100:.0f}%, with expected goals of "
            f"{xg['home']}–{xg['away']} ({fx['home']}–{fx['away']}). "
            f"Most likely score: {s['top_scores'][0][0]}.")
    mv = next((d for d in drv if d["label"] == "Market view"), None)
    if mv:
        text += f" {mv['note']}."
    return text


def _template_reasons(fx, drv) -> list[str]:
    out = []
    for d in drv:
        if "note" in d:
            out.append(d["note"])
        elif d["label"].startswith(("Attack rating", "Defence rating", "Elo rating")):
            out.append(f"{d['label']}: {fx['home']} {d['home']} vs {fx['away']} {d['away']}")
    return out[:5]


def _save(con, r, s, matrix6) -> int:
    now = ingest.now_iso()
    payload = {"base_probs": r["base_probs"], "base_xg": r["base_xg"], "odds": r["odds"],
               "bookmaker": r["bookmaker"], "score_matrix": matrix6, "pick": r["pick"],
               "summary": r["summary"], "reasons": r["reasons"], "drivers": r["drivers"]}
    cur = con.execute(
        """INSERT INTO predictions (fixture_id, created_at, stage, p_home, p_draw, p_away,
           p_over25, p_btts, lam_home, lam_away, top_scores, adjustments, reasoning, payload)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (r["fixture_id"], now, r["stage"], s["p_home"], s["p_draw"], s["p_away"],
         s["p_over25"], s["p_btts"], r["xg"][0], r["xg"][1], json.dumps(r["top_scores"]),
         json.dumps(r["adjustment"]), json.dumps(r["context"]), json.dumps(payload)))
    for v in r["value_bets"]:
        con.execute(
            """INSERT INTO bets (prediction_id, fixture_id, created_at, market, selection, price,
               bookmaker, model_prob, market_prob, edge, stake_pct, verify)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (cur.lastrowid, r["fixture_id"], now, v["market"], v["selection"], v["price"],
             r["bookmaker"], v["model_prob"], v["market_prob"], v["edge"], v["stake_pct"],
             int(v.get("verify", False))))
    con.commit()
    return cur.lastrowid


def matchday(api, con, cfg, day: date | None = None, claude_client=None) -> list[dict]:
    tz = ZoneInfo(cfg["timezone"])
    day = day or datetime.now(tz).date()
    start = datetime.combine(day, datetime.min.time(), tz).astimezone(timezone.utc)
    end = start + timedelta(days=1)
    ids = [l["id"] for l in cfg["leagues"]]
    out = []
    for fx in ingest.upcoming(con, ids, start.isoformat(), end.isoformat()):
        try:
            out.append(predict_fixture(api, con, cfg, fx["id"], "base", claude_client=claude_client))
        except Exception as e:  # noqa: BLE001
            log.error("fixture %s failed: %s", fx["id"], e)
    return out


def lineup_pass(api, con, cfg, claude_client=None) -> list[dict]:
    """Re-predict fixtures kicking off soon once lineups are out (once per fixture)."""
    now = datetime.now(timezone.utc)
    win = now + timedelta(minutes=cfg["schedule"]["lineup_window_min"])
    ids = [l["id"] for l in cfg["leagues"]]
    out = []
    for fx in ingest.upcoming(con, ids, now.isoformat(), win.isoformat()):
        done = con.execute("SELECT 1 FROM predictions WHERE fixture_id=? AND stage='lineups'",
                           (fx["id"],)).fetchone()
        if done or not api.lineups(fx["id"]):
            ingest.snapshot_odds(api, con, cfg, fx["id"])  # keep closing-odds trail fresh
            continue
        out.append(predict_fixture(api, con, cfg, fx["id"], "lineups", claude_client=claude_client))
    return out
