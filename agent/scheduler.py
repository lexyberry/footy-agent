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
    """On a fresh volume, pull history before the first matchday run."""
    from .db import connect
    con = connect(cfg["db_path"])
    if con.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0] == 0:
        log.info("empty database: running initial ingest")
        run(["ingest"])


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
        run(["lineups"])
        time.sleep(poll)


if __name__ == "__main__":
    main()
