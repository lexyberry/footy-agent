"""Scoreboard: settle bets, closing-line value, Brier/log-loss vs the market."""
import numpy as np

from .db import FINISHED
from .ingest import latest_odds
from .markets import devig


def _outcome(hg, ag):
    return 0 if hg > ag else 1 if hg == ag else 2  # index into H, D, A


def _won(market, sel, hg, ag):
    if market == "1x2":
        return {"H": hg > ag, "D": hg == ag, "A": hg < ag}[sel]
    if market == "ou25":
        return (hg + ag > 2.5) == (sel == "over")
    if market == "btts":
        return (hg > 0 and ag > 0) == (sel == "yes")
    return None


def refresh_results(api, con, cfg):
    """Update scores for fixtures that should be finished."""
    from .ingest import ingest_fixtures
    ingest_fixtures(api, con, cfg, seasons=[cfg["current_season"]])


def settle(con) -> int:
    rows = con.execute(
        f"""SELECT b.*, f.home_goals hg, f.away_goals ag, f.kickoff FROM bets b
            JOIN fixtures f ON f.id=b.fixture_id
            WHERE b.result IS NULL AND f.status IN {FINISHED}""").fetchall()
    for b in rows:
        won = _won(b["market"], b["selection"], b["hg"], b["ag"])
        _, close = latest_odds(con, b["fixture_id"], before=b["kickoff"])
        cp = close.get(b["market"], {}).get(b["selection"])
        pnl = (b["price"] - 1) if won else -1.0
        con.execute("UPDATE bets SET result=?, pnl_units=?, closing_price=? WHERE id=?",
                    ("won" if won else "lost", pnl, cp, b["id"]))
    con.commit()
    return len(rows)


def scoreboard(con) -> dict:
    # last prediction made before kickoff, per fixture
    preds = con.execute(
        f"""SELECT p.*, f.home_goals hg, f.away_goals ag, f.kickoff FROM predictions p
            JOIN fixtures f ON f.id=p.fixture_id
            WHERE f.status IN {FINISHED} AND p.created_at < f.kickoff
              AND p.id = (SELECT MAX(id) FROM predictions q WHERE q.fixture_id=p.fixture_id
                          AND q.created_at < f.kickoff)""").fetchall()
    out = {"n_predictions": len(preds)}
    if preds:
        P = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in preds])
        y = np.array([_outcome(r["hg"], r["ag"]) for r in preds])
        Y = np.eye(3)[y]
        out["model"] = {"brier": float(((P - Y) ** 2).sum(1).mean()),
                        "log_loss": float(-np.log(P[np.arange(len(y)), y]).mean()),
                        "accuracy": float((P.argmax(1) == y).mean())}
        mk, my = [], []
        for r, yy in zip(preds, y):
            _, o = latest_odds(con, r["fixture_id"], before=r["kickoff"])
            fair = devig(o.get("1x2", {}))
            if len(fair) == 3:
                mk.append([fair["H"], fair["D"], fair["A"]])
                my.append(yy)
        if mk:
            M, my = np.array(mk), np.array(my)
            out["market_closing"] = {
                "n": len(my), "brier": float(((M - np.eye(3)[my]) ** 2).sum(1).mean()),
                "log_loss": float(-np.log(M[np.arange(len(my)), my]).mean()),
                "accuracy": float((M.argmax(1) == my).mean())}
    bets = con.execute("SELECT * FROM bets WHERE result IS NOT NULL").fetchall()
    if bets:
        stakes = np.array([b["stake_pct"] for b in bets])
        pnl = np.array([b["pnl_units"] for b in bets])
        clv = [b["price"] / b["closing_price"] - 1 for b in bets if b["closing_price"]]
        out["bets"] = {"n": len(bets), "hit_rate": float((pnl > 0).mean()),
                       "roi_flat": float(pnl.mean()),
                       "bankroll_change_pct": float((pnl * stakes).sum()),
                       "avg_clv": float(np.mean(clv)) if clv else None,
                       "beat_close_rate": float(np.mean([c > 0 for c in clv])) if clv else None}
    return out
