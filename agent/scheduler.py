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
    need = {(l["id"], s) for l in cfg["leagues"] for s in cfg["history_seasons"]}
    if need - have:
        log.info("missing league-seasons %s: running ingest", sorted(need - have))
        run(["ingest"])
    if not con.execute("SELECT 1 FROM sync_state WHERE key='backtest'").fetchone():
        run_backtest()


def run_backtest():
    run(["backtest", "--from", "2025-08-01", "--store"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    first_boot(cfg)
    tz = ZoneInfo(cfg["timezone"])
    morning = cfg["schedule"]["morning_run"]
    poll = cfg["schedule"]["poll_every_min"] * 60
    last = {}
    while True:
        now = datetime.now(tz)
        today = now.date().isoformat()
        if now.strftime("%H:%M") >= morning and last.get("matchday") != today:
            run(["matchday"])
            last["matchday"] = today
        if now.hour == 3 and last.get("settle") != today:
            run(["settle"])
            last["settle"] = today
        if now.weekday() == 0 and now.hour == 4 and last.get("backtest") != today:
            run_backtest()  # weekly refresh, Monday early morning
            last["backtest"] = today
        run(["lineups"])
        time.sleep(poll)


if __name__ == "__main__":
    main()
