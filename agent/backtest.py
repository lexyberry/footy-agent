"""Walk-forward backtest: retrain weekly, predict the next week, score out-of-sample."""
from datetime import datetime, timedelta

import numpy as np

from . import engine
from .db import FINISHED


def _score(P, y):
    P = np.clip(np.asarray(P), 1e-9, 1)
    y = np.asarray(y)
    return {"brier": float(((P - np.eye(3)[y]) ** 2).sum(1).mean()),
            "log_loss": float(-np.log(P[np.arange(len(y)), y]).mean()),
            "accuracy": float((P.argmax(1) == y).mean())}


def run(con, cfg, league_id: int, test_from: str, test_to: str, step_days=7) -> dict:
    rows = con.execute(
        f"""SELECT * FROM fixtures WHERE league_id=? AND status IN {FINISHED}
            AND kickoff >= ? AND kickoff < ? ORDER BY kickoff""",
        (league_id, test_from, test_to)).fetchall()
    if not rows:
        return {"league_id": league_id, "n": 0}
    t = datetime.fromisoformat(test_from)
    end = datetime.fromisoformat(test_to)
    recs = []
    while t < end:
        nxt = t + timedelta(days=step_days)
        week = [r for r in rows if t.isoformat() <= r["kickoff"][:19] < nxt.isoformat()]
        if week:
            lm = engine.train(con, cfg, league_id, as_of=t.isoformat())
            for r in week:
                b = engine.base_prediction(lm, r["home_id"], r["away_id"], cfg)
                y = 0 if r["home_goals"] > r["away_goals"] else 1 if r["home_goals"] == r["away_goals"] else 2
                recs.append((b["p_dc"], b["p_elo"], y))
        t = nxt
    Pdc = np.array([r[0] for r in recs])
    Pel = np.array([r[1] for r in recs])
    y = np.array([r[2] for r in recs])
    base_rate = np.bincount(y, minlength=3) / len(y)
    res = {"league_id": league_id, "n": len(y),
           "baseline_base_rates": _score(np.tile(base_rate, (len(y), 1)), y),
           "dixon_coles": _score(Pdc, y), "elo": _score(Pel, y)}
    grid = {}
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        grid[float(w)] = _score(w * Pdc + (1 - w) * Pel, y)["log_loss"]
    best = min(grid, key=grid.get)
    res["blend_grid_logloss"] = grid
    res["best_blend_weight_dc"] = best
    res["blend_at_best"] = _score(best * Pdc + (1 - best) * Pel, y)
    # calibration of the home-win probability in 10% bins
    pb = best * Pdc + (1 - best) * Pel
    bins = np.clip((pb[:, 0] * 10).astype(int), 0, 9)
    res["calibration_home"] = {f"{b*10}-{b*10+10}%": {"n": int((bins == b).sum()),
                                                       "actual": round(float((y[bins == b] == 0).mean()), 3)}
                               for b in range(10) if (bins == b).sum() >= 10}
    return res


def run_all(con, cfg, test_from: str, test_to: str, league_ids=None) -> dict:
    import logging
    from datetime import datetime, timezone
    names = {l["id"]: l["name"] for l in cfg["leagues"]}
    out = []
    for lid in league_ids or list(names):
        try:
            r = run(con, cfg, lid, test_from, test_to)
        except Exception as e:  # noqa: BLE001  (e.g. too little data for a league)
            logging.getLogger(__name__).warning("backtest league %s failed: %s", lid, e)
            continue
        if not r.get("n"):
            logging.getLogger(__name__).warning("backtest league %s skipped: no finished matches "
                                                "in %s..%s", lid, test_from, test_to)
            continue
        if r.get("n"):
            r["league_name"] = names.get(lid, str(lid))
            r.pop("blend_grid_logloss", None)
            out.append(r)
    return {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_from": test_from, "test_to": test_to, "leagues": out}


def one_line(r: dict) -> str:
    return (f"{r['league_name']}: n={r['n']} | log-loss base {r['baseline_base_rates']['log_loss']:.4f}"
            f" DC {r['dixon_coles']['log_loss']:.4f} Elo {r['elo']['log_loss']:.4f}"
            f" blend {r['blend_at_best']['log_loss']:.4f} (w_dc={r['best_blend_weight_dc']})"
            f" | acc {r['blend_at_best']['accuracy']*100:.1f}%")


def store(con, res: dict):
    import json
    con.execute("INSERT INTO sync_state VALUES ('backtest', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (json.dumps(res),))
    con.commit()
