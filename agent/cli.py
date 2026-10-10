"""Command line: python -m agent.cli <command>"""
import argparse
import json
import logging
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from . import backtest, evaluate, ingest, pipeline, report, sync
from .api_football import ApiFootball
from .config import load_config
from .db import connect


def _odds_arg(s):
    if not s:
        return None
    h, d, a = (float(x) for x in s.split(","))
    return {"1x2": {"H": h, "D": d, "A": a}}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="footy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="pull all configured seasons into the database")
    p = sub.add_parser("predict", help="predict one fixture")
    p.add_argument("fixture_id", type=int)
    p.add_argument("--odds", help="manual 1X2 odds, e.g. 2.10,3.40,3.60 (e.g. from Superbet)")
    p.add_argument("--no-claude", action="store_true")
    m = sub.add_parser("matchday", help="predict all fixtures on a date and write the report")
    m.add_argument("--date", help="YYYY-MM-DD (default today, Europe/Warsaw)")
    sub.add_parser("lineups", help="re-run fixtures whose lineups are out")
    sub.add_parser("settle", help="update results, settle bets, print scoreboard")
    b = sub.add_parser("backtest", help="walk-forward backtest")
    b.add_argument("--from", dest="frm", default="2025-08-01")
    b.add_argument("--to", dest="to", default=date.today().isoformat())
    b.add_argument("--league", type=int, help="single league id (default all)")
    sub.add_parser("bookmakers", help="list API-Football bookmakers (check Superbet's id)")
    sub.add_parser("sync", help="push recent data to the Lovable dashboard")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.v else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config()
    con = connect(cfg["db_path"])
    needs_api = a.cmd not in ("backtest", "sync")
    api = ApiFootball() if needs_api else None

    if a.cmd == "ingest":
        print(f"{ingest.ingest_fixtures(api, con, cfg)} fixtures stored; {api.calls} API calls")
    elif a.cmd == "predict":
        if a.no_claude:
            cfg["claude"]["enabled"] = False
        r = pipeline.predict_fixture(api, con, cfg, a.fixture_id, manual_odds=_odds_arg(a.odds))
        print(json.dumps(r, indent=2, default=str))
    elif a.cmd == "matchday":
        tz = ZoneInfo(cfg["timezone"])
        d = date.fromisoformat(a.date) if a.date else datetime.now(tz).date()
        ingest.ingest_fixtures(api, con, cfg, seasons=[cfg["current_season"]])
        preds = pipeline.matchday(api, con, cfg, d)
        f = report.write(preds, cfg, f"Matchday {d:%a %d %b}", f"matchday-{d}")
        print(report.text_summary(preds))
        print(f"\nReport: {f}")
        report.telegram(f"⚽ Matchday {d}\n" + report.text_summary(preds), f)
        sync.push(con, cfg)
    elif a.cmd == "lineups":
        preds = pipeline.lineup_pass(api, con, cfg)
        if preds:
            name = f"lineups-{datetime.now():%Y%m%d-%H%M}"
            f = report.write(preds, cfg, "Lineup update", name)
            print(report.text_summary(preds))
            report.telegram("🔄 Lineup update\n" + report.text_summary(preds), f)
        sync.push(con, cfg)
    elif a.cmd == "settle":
        evaluate.refresh_results(api, con, cfg)
        print(f"settled {evaluate.settle(con)} bets")
        print(json.dumps(evaluate.scoreboard(con), indent=2))
        sync.push(con, cfg, include_scoreboard=True)
    elif a.cmd == "backtest":
        leagues = [a.league] if a.league else [l["id"] for l in cfg["leagues"]]
        for lid in leagues:
            print(json.dumps(backtest.run(con, cfg, lid, a.frm, a.to), indent=2))
    elif a.cmd == "sync":
        print(sync.push(con, cfg, include_scoreboard=True))
    elif a.cmd == "bookmakers":
        for bk in api.bookmakers():
            print(bk["id"], bk["name"])


if __name__ == "__main__":
    main()
