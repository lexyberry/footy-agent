"""Claude context layer: gathers match context and returns bounded adjustments."""
import json
import logging
import os

log = logging.getLogger(__name__)

SYSTEM = """You are the context analyst inside a football prediction system.

A statistical model (time-weighted Dixon-Coles + Elo) has already produced expected goals
for each team from past results. It already captures team strength, recent form, and home
advantage. Your ONLY job is to adjust for information the results cannot know:
- confirmed absences of key players (injuries, suspensions), especially if the lineup confirms them
- heavy rotation or a weakened XI (e.g. before a European tie)
- clear motivation asymmetry (title/relegation pressure vs nothing to play for)
- a new manager, or other strong situational factors in the data provided

Rules:
- Default to 1.0 (no change). Most matches need little or no adjustment.
- Multipliers apply to each team's expected goals. A team's attackers missing lowers its own
  multiplier; its defenders/goalkeeper missing raises the OPPONENT's multiplier.
- Stay within 0.85-1.15. Typical meaningful adjustments are 0.95-1.05.
- Use only the data given. Do not invent injuries, news, or facts. If data is thin, say so
  and stay close to 1.0.
- Do not adjust for form or table position alone — the model already has them.

You also explain the prediction to a human reader:
- "summary": 2-3 plain-English sentences on why the model's most likely outcome is what it is,
  grounded in the "drivers" (ratings, Elo, expected goals, form, market view) and any
  adjustment you made. Say how confident the call is (e.g. a 45% favourite is far from certain).
- "reasons": 3-5 short bullet reasons, each tied to a specific number or fact provided.
- If the market view disagrees strongly with the model, say so plainly in the summary.
- Never invent statistics, players or news that are not in the data."""

TOOL = {
    "name": "submit_adjustment",
    "description": "Submit bounded expected-goals adjustments for this fixture.",
    "input_schema": {
        "type": "object",
        "properties": {
            "home_xg_multiplier": {"type": "number", "minimum": 0.85, "maximum": 1.15},
            "away_xg_multiplier": {"type": "number", "minimum": 0.85, "maximum": 1.15},
            "key_factors": {"type": "array", "items": {"type": "string"}, "maxItems": 5,
                            "description": "Each factor that drove an adjustment, one line each"},
            "reasoning": {"type": "string", "description": "2-4 sentences"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            "summary": {"type": "string",
                        "description": "2-3 sentences: why the most likely outcome was chosen"},
            "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5,
                        "description": "Key reasons, each tied to a number/fact provided"},
        },
        "required": ["home_xg_multiplier", "away_xg_multiplier", "key_factors",
                     "reasoning", "confidence", "summary", "reasons"],
    },
}

NEUTRAL = {"home_xg_multiplier": 1.0, "away_xg_multiplier": 1.0, "key_factors": [],
           "reasoning": "No context adjustment applied.", "confidence": "low",
           "summary": None, "reasons": []}


def recent_form(con, team_id, before, n=6):
    rows = con.execute(
        """SELECT kickoff, home_id, home, away, home_goals hg, away_goals ag FROM fixtures
           WHERE (home_id=? OR away_id=?) AND status IN ('FT','AET','PEN') AND kickoff < ?
           ORDER BY kickoff DESC LIMIT ?""", (team_id, team_id, before, n)).fetchall()
    return [f"{r['kickoff'][:10]} {r['home']} {r['hg']}-{r['ag']} {r['away']}" for r in rows]


def form_string(con, team_id, before, n=6) -> str:
    """W/D/L for the team's last n results, newest first."""
    rows = con.execute(
        """SELECT home_id, home_goals hg, away_goals ag FROM fixtures
           WHERE (home_id=? OR away_id=?) AND status IN ('FT','AET','PEN') AND kickoff < ?
           ORDER BY kickoff DESC LIMIT ?""", (team_id, team_id, before, n)).fetchall()
    out = []
    for r in rows:
        gf, ga = (r["hg"], r["ag"]) if r["home_id"] == team_id else (r["ag"], r["hg"])
        out.append("W" if gf > ga else "D" if gf == ga else "L")
    return "".join(out)


def gather(api, con, fx, cfg) -> dict:
    """Collect injuries, lineups and table context. API failures degrade gracefully."""
    ctx = {"fixture": f"{fx['home']} vs {fx['away']}", "kickoff": fx["kickoff"],
           "round": fx["round"],
           "home_recent": recent_form(con, fx["home_id"], fx["kickoff"]),
           "away_recent": recent_form(con, fx["away_id"], fx["kickoff"])}
    if api is None:
        return ctx
    try:
        inj = api.injuries(fx["id"])
        ctx["absences"] = [f"{i['team']['name']}: {i['player']['name']} ({i['player'].get('type')}"
                           f" - {i['player'].get('reason')})" for i in inj]
    except Exception as e:  # noqa: BLE001
        log.warning("injuries failed: %s", e)
    try:
        lu = api.lineups(fx["id"])
        if lu:
            ctx["lineups"] = {l["team"]["name"]: {
                "formation": l.get("formation"),
                "xi": [p["player"]["name"] for p in l.get("startXI", [])]} for l in lu}
    except Exception as e:  # noqa: BLE001
        log.warning("lineups failed: %s", e)
    try:
        st = api.standings(fx["league_id"], fx["season"])
        table = st[0]["league"]["standings"][0] if st else []
        n = len(table)
        ctx["table_size"] = n
        for row in table:
            if row["team"]["id"] in (fx["home_id"], fx["away_id"]):
                side = "home_table" if row["team"]["id"] == fx["home_id"] else "away_table"
                ctx[side] = {"rank": row["rank"], "points": row["points"],
                             "played": row["all"]["played"], "gd": row["goalsDiff"],
                             "form": row.get("form")}
    except Exception as e:  # noqa: BLE001
        log.warning("standings failed: %s", e)
    return ctx


def ask_claude(ctx: dict, base_summary: dict, cfg: dict, client=None, drivers=None) -> dict:
    if not cfg["claude"]["enabled"]:
        return NEUTRAL
    if client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            log.warning("ANTHROPIC_API_KEY not set; skipping context layer")
            return NEUTRAL
        import anthropic
        client = anthropic.Anthropic()
    payload = {"context": ctx, "model_base": base_summary, "drivers": drivers or []}
    try:
        # Forced tool_choice isn't supported by every model, so use "auto" and require
        # the tool in the instructions; anything else falls back to neutral.
        msg = client.messages.create(
            model=cfg["claude"]["model"], max_tokens=4000,
            system=SYSTEM + "\n\nAlways answer by calling submit_adjustment exactly once.",
            tools=[TOOL], tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}])
        block = next((b for b in msg.content if getattr(b, "type", "") == "tool_use"), None)
        if block is None:
            log.warning("Claude did not call submit_adjustment; using neutral")
            return NEUTRAL
        out = dict(NEUTRAL, **block.input)
        out["home_xg_multiplier"] = float(out["home_xg_multiplier"])
        out["away_xg_multiplier"] = float(out["away_xg_multiplier"])
        return out
    except Exception as e:  # noqa: BLE001
        log.error("Claude context call failed, using neutral: %s", e)
        return NEUTRAL
