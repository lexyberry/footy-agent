"""MCP server so you can ask for predictions on demand from Claude Desktop.

Run:  python -m agent.mcp_server            (stdio, for Claude Desktop)
      MCP_TRANSPORT=http python -m agent.mcp_server   (http on :8765, for Docker)
"""
import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP

from . import evaluate, ingest, pipeline
from .api_football import ApiFootball
from .config import load_config
from .db import connect

mcp = FastMCP("footy-agent", host="0.0.0.0", port=int(os.environ.get("MCP_PORT", 8765)))
_state = {}


def _deps():
    if not _state:
        cfg = load_config()
        _state.update(cfg=cfg, con=connect(cfg["db_path"]), api=ApiFootball())
    return _state["api"], _state["con"], _state["cfg"]


@mcp.tool()
def predict_fixture(fixture_id: int, home_odds: float | None = None,
                    draw_odds: float | None = None, away_odds: float | None = None) -> dict:
    """Full prediction for one API-Football fixture id: 1X2/O2.5/BTTS probabilities,
    likeliest scores, Claude's context adjustments, and value flags. Pass Superbet 1X2
    odds to check value against them directly."""
    api, con, cfg = _deps()
    manual = None
    if home_odds and draw_odds and away_odds:
        manual = {"1x2": {"H": home_odds, "D": draw_odds, "A": away_odds}}
    return pipeline.predict_fixture(api, con, cfg, fixture_id, manual_odds=manual)


@mcp.tool()
def find_fixtures(day: str | None = None, team: str | None = None) -> list[dict]:
    """List upcoming fixtures in the covered leagues for a date (YYYY-MM-DD, default today),
    optionally filtered by team name. Use the returned id with predict_fixture."""
    api, con, cfg = _deps()
    tz = ZoneInfo(cfg["timezone"])
    d = date.fromisoformat(day) if day else datetime.now(tz).date()
    start = datetime.combine(d, datetime.min.time(), tz).astimezone(ZoneInfo("UTC"))
    end = start + timedelta(days=1)
    rows = ingest.upcoming(con, [l["id"] for l in cfg["leagues"]], start.isoformat(), end.isoformat())
    out = [{"id": r["id"], "match": f"{r['home']} vs {r['away']}", "kickoff": r["kickoff"],
            "league_id": r["league_id"]} for r in rows]
    if team:
        out = [o for o in out if team.lower() in o["match"].lower()]
    return out


@mcp.tool()
def matchday_report(day: str | None = None) -> list[dict]:
    """Predict every covered fixture on a date (default today). Slower: one model + Claude call per match."""
    api, con, cfg = _deps()
    tz = ZoneInfo(cfg["timezone"])
    d = date.fromisoformat(day) if day else datetime.now(tz).date()
    return pipeline.matchday(api, con, cfg, d)


@mcp.tool()
def model_scoreboard() -> dict:
    """How the agent has actually performed: Brier/log-loss vs closing market, bet ROI and CLV."""
    _, con, _ = _deps()
    evaluate.settle(con)
    return evaluate.scoreboard(con)


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if os.environ.get("MCP_TRANSPORT") == "http" else "stdio")
