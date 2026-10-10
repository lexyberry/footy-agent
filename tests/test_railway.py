"""Auth middleware and dashboard sync."""
import json
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from agent import pipeline, sync
from agent.serve import TokenAuth
from tests.test_agent import FakeClaude, setup  # noqa: F401  (fixture)


async def ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"inner"})


def test_auth_header_and_query():
    c = TestClient(TokenAuth(ok_app, "s3cret"))
    assert c.get("/health").status_code == 200
    assert c.get("/mcp").status_code == 401
    assert c.get("/mcp", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/mcp", headers={"Authorization": "Bearer s3cret"}).text == "inner"
    assert c.get("/mcp?token=s3cret").text == "inner"


def test_no_token_configured_rejects_everything():
    c = TestClient(TokenAuth(ok_app, None))
    assert c.get("/mcp?token=").status_code == 401
    assert c.get("/health").status_code == 200


def test_real_mcp_app_mounts(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "t")
    from agent.serve import build_app
    with TestClient(build_app()) as c:
        assert c.post("/mcp", json={}).status_code == 401
        r = c.post("/mcp", headers={"Authorization": "Bearer t",
                                    "Accept": "application/json, text/event-stream"},
                   json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                                    "clientInfo": {"name": "t", "version": "1"}}})
        assert r.status_code == 200 and "footy-agent" in r.text


def test_sync_payload_shape(setup, monkeypatch):  # noqa: F811
    api, con, cfg = setup
    cfg["market"]["min_edge"] = -1
    cfg["market"]["min_prob"] = 0
    fx = con.execute("SELECT id FROM fixtures WHERE status='NS' ORDER BY kickoff LIMIT 1").fetchone()["id"]
    pipeline.predict_fixture(api, con, cfg, fx, claude_client=FakeClaude())
    monkeypatch.setenv("INGEST_URL", "https://x/functions/v1/ingest")
    monkeypatch.setenv("INGEST_SECRET", "k")
    calls = []
    sent = sync.push(con, cfg, post=lambda b: calls.append(b) or SimpleNamespace(status_code=200),
                     include_scoreboard=True)
    tables = [c["table"] for c in calls]
    assert tables.index("fixtures") < tables.index("predictions") < tables.index("bets")
    p = next(c for c in calls if c["table"] == "predictions")["rows"][0]
    assert 0 < p["p_home"] < 1 and 0 < p["base_p_home"] < 1
    assert len(p["score_matrix"]) == 6 and len(p["score_matrix"][0]) == 6
    assert 0 < p["top_scores"][0][1] < 1
    assert p["key_factors"] == ["Star Striker out (hamstring)"]
    f = next(c for c in calls if c["table"] == "fixtures")["rows"][0]
    assert f["league_name"] == "Sim League"
    b = next(c for c in calls if c["table"] == "bets")["rows"][0]
    assert isinstance(b["verify"], bool)
    json.dumps(calls)  # everything serialisable
    assert sent["odds_snapshots"] > 0
    # second push sends no duplicate odds
    calls.clear()
    sync.push(con, cfg, post=lambda b: calls.append(b) or SimpleNamespace(status_code=200))
    assert all(c["table"] != "odds_snapshots" or not c["rows"] for c in calls)


def test_sync_skips_without_env(setup, monkeypatch):  # noqa: F811
    _, con, cfg = setup
    monkeypatch.delenv("INGEST_URL", raising=False)
    assert sync.push(con, cfg, post=lambda b: pytest.fail("should not post")) == {}
