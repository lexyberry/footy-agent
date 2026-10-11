"""Long-running loop for Docker: morning matchday run, lineup polling, nightly settle."""
import logging
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import load_config

log = logging.getLogger("scheduler")


def run(cmd):
    log.info("running %s", cmd)
    subprocess.run([sys.executable, "-m", "agent.cli", *cmd], check=False)


def first_boot(cfg):
    """Pull any missing league-seasons, and run a backtest if none is stored yet."""
    from .db import connect
    con = connect(cfg["db_path"])
    have = {(r[0], r[1]) for r in con.execute(
        "SELECT league_id, season FROM fixtures GROUP BY league_id, season")}
    need = {(l["id"], s) for l in cfg["leagues"] + cfg.get("data_leagues", [])
            for s in cfg["history_seasons"]}
    if need - have:
        log.info("missing league-seasons %s: running ingest", sorted(need - have))
        run(["ingest"])
    import json
    row = con.execute("SELECT value FROM sync_state WHERE key='backtest'").fetchone()
    tested = {l["league_id"] for l in json.loads(row[0])["leagues"]} if row else set()
    if not row or {l["id"] for l in cfg["leagues"]} - tested:
        run_backtest()  # none yet, or a newly added competition hasn't been tested


def run_backtest():
    run(["backtest", "--from", "2025-08-01", "--store"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    first_boot(cfg)
    tz = ZoneInfo(cfg["timezone"])
    morning = cfg["schedule"]["morning_run"]
    poll = cfg["schedule"]["poll_every_min"] * 60
    from .db import connect
    con = connect(cfg["db_path"])

    def done(key, today):
        r = con.execute("SELECT value FROM sync_state WHERE key=?", (f"last_{key}",)).fetchone()
        return r is not None and r[0] == today

    def mark(key, today):
        con.execute("INSERT INTO sync_state VALUES (?,?) ON CONFLICT(key) DO UPDATE "
                    "SET value=excluded.value", (f"last_{key}", today))
        con.commit()

    while True:
        now = datetime.now(tz)
        today = now.date().isoformat()
        # last-run markers live in the database so a restart doesn't repeat today's run
        if now.strftime("%H:%M") >= morning and not done("matchday", today):
            run(["matchday"])
            mark("matchday", today)
        if now.hour == 3 and not done("settle", today):
            run(["settle"])
            mark("settle", today)
        if now.weekday() == 0 and now.hour == 4 and not done("backtest", today):
            run_backtest()  # weekly refresh, Monday early morning
            mark("backtest", today)
        run(["lineups"])
        time.sleep(poll)


if __name__ == "__main__":
    main()
