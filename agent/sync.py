"""Push agent data to the Lovable dashboard's ingest edge function.

Needs INGEST_URL (https://<your-lovable-app>.lovable.app/api/public/ingest) and INGEST_SECRET.
All writes are idempotent upserts, so re-sending is safe.
"""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import requests

from .evaluate import scoreboard

log = logging.getLogger(__name__)
BATCH = 200


def _state(con, key, default=None):
    r = con.execute("SELECT value FROM sync_state WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def _set_state(con, key, value):
    con.execute("INSERT INTO sync_state VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)))
    con.commit()


def _pct_to_dec(pairs):
    return [[s, round(p / 100, 4)] for s, p in pairs]


def build_rows(con, cfg, since_days=14) -> dict:
    """Collect everything the dashboard needs that may have changed recently."""
    leagues = {l["id"]: l["name"] for l in cfg["leagues"]}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
    last_odds = _state(con, "odds_taken_at", "1970-01-01")

    preds = con.execute("SELECT * FROM predictions WHERE created_at >= ? ORDER BY id",
                        (cutoff,)).fetchall()
    fx_ids = sorted({p["fixture_id"] for p in preds})
    fixtures = [dict(r) for r in con.execute(
        f"SELECT * FROM fixtures WHERE id IN ({','.join('?' * len(fx_ids))})", fx_ids)] if fx_ids else []

    out = {"fixtures": [], "predictions": [], "bets": [], "odds_snapshots": []}
    for f in fixtures:
        out["fixtures"].append({
            "id": f["id"], "league_id": f["league_id"], "league_name": leagues.get(f["league_id"]),
            "season": f["season"], "round": f["round"], "kickoff": f["kickoff"], "status": f["status"],
            "home": f["home"], "away": f["away"],
            "home_goals": f["home_goals"], "away_goals": f["away_goals"]})
    for p in preds:
        pay = json.loads(p["payload"] or "{}")
        ctx = json.loads(p["reasoning"] or "{}")
        adj = json.loads(p["adjustments"] or "{}")
        bp = pay.get("base_probs", {})
        bx = pay.get("base_xg", [None, None])
        out["predictions"].append({
            "id": p["id"], "fixture_id": p["fixture_id"], "created_at": p["created_at"],
            "stage": p["stage"], "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
            "p_over25": p["p_over25"], "p_btts": p["p_btts"],
            "base_p_home": bp.get("home", 0) / 100 if bp else None,
            "base_p_draw": bp.get("draw", 0) / 100 if bp else None,
            "base_p_away": bp.get("away", 0) / 100 if bp else None,
            "xg_home": p["lam_home"], "xg_away": p["lam_away"],
            "base_xg_home": bx[0], "base_xg_away": bx[1],
            "adj_home": adj.get("home"), "adj_away": adj.get("away"),
            "top_scores": _pct_to_dec(json.loads(p["top_scores"] or "[]")),
            "score_matrix": pay.get("score_matrix"),
            "key_factors": ctx.get("key_factors", []), "reasoning": ctx.get("reasoning"),
            "confidence": ctx.get("confidence"),
            "bookmaker": pay.get("bookmaker"), "odds": pay.get("odds", {}),
            "pick": pay.get("pick"), "summary": pay.get("summary"),
            "reasons": pay.get("reasons", []), "drivers": pay.get("drivers", [])})
    for b in con.execute("SELECT * FROM bets WHERE created_at >= ?", (cutoff,)):
        out["bets"].append({k: b[k] for k in (
            "id", "prediction_id", "fixture_id", "created_at", "market", "selection", "price",
            "bookmaker", "model_prob", "market_prob", "edge", "stake_pct", "closing_price",
            "result", "pnl_units")} | {"verify": bool(b["verify"])})
    for o in con.execute("SELECT * FROM odds_snapshots WHERE taken_at > ? ORDER BY taken_at",
                         (last_odds,)):
        out["odds_snapshots"].append({k: o[k] for k in (
            "fixture_id", "taken_at", "bookmaker", "market", "selection", "price")})
    return out


def push(con, cfg, post=None, include_scoreboard=False) -> dict:
    url, secret = os.environ.get("INGEST_URL"), os.environ.get("INGEST_SECRET")
    if not (url and secret):
        log.info("INGEST_URL/INGEST_SECRET not set; skipping dashboard sync")
        return {}
    post = post or (lambda body: requests.post(
        url, json=body, headers={"x-ingest-secret": secret}, timeout=30))
    rows = build_rows(con, cfg)
    if include_scoreboard:
        data = scoreboard(con)
        bt = _state(con, "backtest")
        if bt:
            data["backtest"] = json.loads(bt)
        rows["scoreboard_snapshots"] = [{"data": data}]
    sent = {}
    # order matters: fixtures before rows that reference them
    for table in ("fixtures", "predictions", "bets", "odds_snapshots", "scoreboard_snapshots"):
        data = rows.get(table, [])
        for i in range(0, len(data), BATCH):
            r = post({"table": table, "rows": data[i:i + BATCH]})
            if getattr(r, "status_code", 200) >= 300:
                log.error("sync %s failed: %s %s", table, r.status_code, getattr(r, "text", "")[:300])
                return sent
        sent[table] = len(data)
    gone = [r["id"] for r in con.execute("SELECT id FROM deleted_bets")]
    if gone:
        r = post({"table": "bets", "delete_ids": gone})
        if getattr(r, "status_code", 200) < 300:
            con.execute(f"DELETE FROM deleted_bets WHERE id IN ({','.join('?' * len(gone))})", gone)
            con.commit()
            sent["bets_deleted"] = len(gone)
        else:
            log.error("sync bet deletions failed: %s", getattr(r, "status_code", "?"))
    if rows["odds_snapshots"]:
        _set_state(con, "odds_taken_at", rows["odds_snapshots"][-1]["taken_at"])
    log.info("dashboard sync: %s", sent)
    return sent
