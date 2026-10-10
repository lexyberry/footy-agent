"""Railway entrypoint: scheduler in a background thread + token-protected MCP over HTTP.

Env:
  PORT            set by Railway
  MCP_AUTH_TOKEN  required; clients send "Authorization: Bearer <token>" or "?token=<token>"
"""
import hmac
import json
import logging
import os
import threading
from urllib.parse import parse_qs

import uvicorn

log = logging.getLogger("serve")


class TokenAuth:
    """Pure ASGI middleware (doesn't buffer, so MCP streaming keeps working)."""

    def __init__(self, app, token: str | None):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == "/health":
            return await self._reply(send, 200, {"ok": True})
        if not self.token or not self._valid(scope):
            return await self._reply(send, 401, {"error": "unauthorized"})
        return await self.app(scope, receive, send)

    def _valid(self, scope) -> bool:
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode()
        given = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not given:
            given = parse_qs(scope.get("query_string", b"").decode()).get("token", [""])[0]
        return bool(given) and hmac.compare_digest(given, self.token)

    @staticmethod
    async def _reply(send, status, body):
        data = json.dumps(body).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": data})


def build_app():
    from .mcp_server import mcp
    token = os.environ.get("MCP_AUTH_TOKEN")
    if not token:
        log.warning("MCP_AUTH_TOKEN not set: MCP endpoint will reject every request")
    return TokenAuth(mcp.streamable_http_app(), token)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if os.environ.get("DISABLE_SCHEDULER") != "1":
        from .scheduler import main as scheduler_main
        threading.Thread(target=scheduler_main, daemon=True, name="scheduler").start()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", 8765)),
                log_level="info")


if __name__ == "__main__":
    main()
