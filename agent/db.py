import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS fixtures (
  id INTEGER PRIMARY KEY, league_id INTEGER, season INTEGER, round TEXT,
  kickoff TEXT, status TEXT,
  home_id INTEGER, home TEXT, away_id INTEGER, away TEXT,
  home_goals INTEGER, away_goals INTEGER
);
CREATE INDEX IF NOT EXISTS ix_fix_league ON fixtures(league_id, kickoff);

CREATE TABLE IF NOT EXISTS odds_snapshots (
  fixture_id INTEGER, taken_at TEXT, bookmaker TEXT,
  market TEXT, selection TEXT, price REAL
);
CREATE INDEX IF NOT EXISTS ix_odds_fix ON odds_snapshots(fixture_id, taken_at);

CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fixture_id INTEGER, created_at TEXT, stage TEXT,       -- stage: base | lineups
  p_home REAL, p_draw REAL, p_away REAL,
  p_over25 REAL, p_btts REAL,
  lam_home REAL, lam_away REAL,
  top_scores TEXT, adjustments TEXT, reasoning TEXT
);
CREATE INDEX IF NOT EXISTS ix_pred_fix ON predictions(fixture_id, created_at);

CREATE TABLE IF NOT EXISTS bets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id INTEGER, fixture_id INTEGER, created_at TEXT,
  market TEXT, selection TEXT, price REAL, bookmaker TEXT,
  model_prob REAL, market_prob REAL, edge REAL, stake_pct REAL,
  closing_price REAL, result TEXT, pnl_units REAL
);
"""


MIGRATIONS = [
    "ALTER TABLE predictions ADD COLUMN payload TEXT",   # full result JSON for the dashboard
    "ALTER TABLE bets ADD COLUMN verify INTEGER DEFAULT 0",
    "CREATE TABLE IF NOT EXISTS sync_state (key TEXT PRIMARY KEY, value TEXT)",
]


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    for m in MIGRATIONS:
        try:
            con.execute(m)
        except sqlite3.OperationalError:
            pass  # already applied
    con.commit()
    return con


def upsert_fixture(con, f: dict):
    fx, lg, t, g = f["fixture"], f["league"], f["teams"], f["goals"]
    con.execute(
        """INSERT INTO fixtures VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET kickoff=excluded.kickoff, status=excluded.status,
           home_goals=excluded.home_goals, away_goals=excluded.away_goals, round=excluded.round""",
        (fx["id"], lg["id"], lg["season"], lg.get("round"), fx["date"], fx["status"]["short"],
         t["home"]["id"], t["home"]["name"], t["away"]["id"], t["away"]["name"],
         g["home"], g["away"]),
    )


FINISHED = ("FT", "AET", "PEN")
