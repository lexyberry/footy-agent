"""Pull fixtures and odds from API-Football into SQLite."""
import logging
from datetime import datetime, timezone

from .db import upsert_fixture

log = logging.getLogger(__name__)

# API-Football bet names -> our market keys
MARKETS = {
    "Match Winner": ("1x2", {"Home": "H", "Draw": "D", "Away": "A"}),
    "Goals Over/Under": ("ou25", {"Over 2.5": "over", "Under 2.5": "under"}),
    "Both Teams Score": ("btts", {"Yes": "yes", "No": "no"}),
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ingest_fixtures(api, con, cfg, seasons=None):
    seasons = seasons or cfg["history_seasons"]
    n = 0
    for lg in cfg["leagues"]:
        for s in seasons:
            done = s < cfg["current_season"]
            rows = api.fixtures(lg["id"], s, ttl=-1 if done else 1800)
            for f in rows:
                upsert_fixture(con, f)
            n += len(rows)
            log.info("%s %s: %d fixtures", lg["name"], s, len(rows))
    con.commit()
    return n


def parse_odds(resp: list, prefer: list[str]) -> tuple[str | None, dict]:
    """Return (bookmaker_name, {market: {selection: price}}) using first available preferred book."""
    if not resp:
        return None, {}
    books = {b["name"].lower(): b for b in resp[0].get("bookmakers", [])}
    chosen = next((books[p.lower()] for p in prefer if p.lower() in books), None)
    if chosen is None and books:
        chosen = next(iter(books.values()))
    if chosen is None:
        return None, {}
    out = {}
    for bet in chosen.get("bets", []):
        if bet["name"] in MARKETS:
            mkey, sel_map = MARKETS[bet["name"]]
            for v in bet["values"]:
                sel = sel_map.get(str(v["value"]))
                if sel:
                    out.setdefault(mkey, {})[sel] = float(v["odd"])
    return chosen["name"], out


def snapshot_odds(api, con, cfg, fixture_id: int) -> tuple[str | None, dict]:
    prefer = [cfg["market"]["bookmaker"], *cfg["market"]["fallback_bookmakers"]]
    book, odds = parse_odds(api.odds(fixture_id), prefer)
    if book and book.lower() != cfg["market"]["bookmaker"].lower():
        log.warning("fixture %s: %s odds unavailable, using %s", fixture_id,
                    cfg["market"]["bookmaker"], book)
    ts = now_iso()
    for m, sels in odds.items():
        for sel, price in sels.items():
            con.execute("INSERT INTO odds_snapshots VALUES (?,?,?,?,?,?)",
                        (fixture_id, ts, book, m, sel, price))
    con.commit()
    return book, odds


def latest_odds(con, fixture_id: int, before: str | None = None) -> tuple[str | None, dict]:
    q = "SELECT MAX(taken_at) t FROM odds_snapshots WHERE fixture_id=?"
    args = [fixture_id]
    if before:
        q += " AND taken_at <= ?"
        args.append(before)
    t = con.execute(q, args).fetchone()["t"]
    if not t:
        return None, {}
    rows = con.execute("SELECT * FROM odds_snapshots WHERE fixture_id=? AND taken_at=?",
                       (fixture_id, t)).fetchall()
    out = {}
    for r in rows:
        out.setdefault(r["market"], {})[r["selection"]] = r["price"]
    return rows[0]["bookmaker"], out


def upcoming(con, league_ids, start_iso, end_iso):
    q = f"""SELECT * FROM fixtures WHERE status IN ('NS','TBD')
            AND league_id IN ({",".join("?" * len(league_ids))})
            AND kickoff >= ? AND kickoff < ? ORDER BY kickoff"""
    return con.execute(q, [*league_ids, start_iso, end_iso]).fetchall()
