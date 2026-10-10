"""Mobile-friendly HTML matchday report + optional Telegram push."""
import html
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger(__name__)
SEL = {"H": "Home", "D": "Draw", "A": "Away", "over": "Over 2.5", "under": "Under 2.5",
       "yes": "BTTS Yes", "no": "BTTS No"}

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1b1b1b;--mut:#6b6b6b;--acc:#0b7a4b;--line:#e4e4e0}
@media (prefers-color-scheme:dark){:root{--bg:#121413;--card:#1c1f1d;--fg:#ececec;--mut:#9a9a9a;--acc:#3ccf8e;--line:#2c302d}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:640px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:4px 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.top{display:flex;justify-content:space-between;gap:8px;font-weight:600}
.mut{color:var(--mut);font-size:13px}
.bar{display:flex;height:8px;border-radius:4px;overflow:hidden;margin:10px 0 4px}
.bar span:nth-child(1){background:var(--acc)}.bar span:nth-child(2){background:#9a9a9a}.bar span:nth-child(3){background:#c0563d}
.row{display:flex;justify-content:space-between;font-size:13px}
.vb{margin-top:10px;padding:8px 10px;border-radius:8px;background:color-mix(in srgb,var(--acc) 12%,transparent);font-size:14px}
ul{margin:6px 0 0;padding-left:18px;font-size:13px}
"""


def render(preds: list[dict], cfg: dict, title: str) -> str:
    tz = ZoneInfo(cfg["timezone"])
    leagues = {l["id"]: l["name"] for l in cfg["leagues"]}
    cards = []
    for p in sorted(preds, key=lambda x: x["kickoff"]):
        ko = datetime.fromisoformat(p["kickoff"].replace("Z", "+00:00")).astimezone(tz)
        pr = p["probs"]
        vbs = "".join(
            f'<div class="vb"><b>{SEL.get(v["selection"], v["selection"])}</b> @ {v["price"]:.2f}'
            f' · edge {v["edge"]*100:.1f}% · '
            + ('⚠ verify news/odds before betting' if v.get("verify") else f'stake {v["stake_pct"]}%') +
            f' <span class="mut">(model {v["model_prob"]*100:.0f}% vs mkt {v["market_prob"]*100:.0f}%)</span></div>'
            for v in p["value_bets"])
        factors = "".join(f"<li>{html.escape(f)}</li>" for f in p["context"]["key_factors"])
        book = f' · odds: {html.escape(p["bookmaker"])}' if p.get("bookmaker") else " · no odds"
        cards.append(f"""<div class="card">
<div class="top"><span>{html.escape(p['match'])}</span><span>{ko:%H:%M}</span></div>
<div class="mut">{leagues.get(p['league_id'], '')} · xG {p['xg'][0]}–{p['xg'][1]}{book}{' · lineups ✓' if p['has_lineups'] else ''}</div>
<div class="bar"><span style="width:{pr['home']}%"></span><span style="width:{pr['draw']}%"></span><span style="width:{pr['away']}%"></span></div>
<div class="row"><span>1 {pr['home']}%</span><span>X {pr['draw']}%</span><span>2 {pr['away']}%</span></div>
<div class="row mut"><span>O2.5 {pr['over25']}%</span><span>BTTS {pr['btts']}%</span><span>Likely: {p['top_scores'][0][0]}</span></div>
{f'<ul>{factors}</ul>' if factors else ''}{vbs}</div>""")
    n_vb = sum(len(p["value_bets"]) for p in preds)
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body><main><h1>{html.escape(title)}</h1>
<p class="mut">{len(preds)} matches · {n_vb} value flags · probabilities, not certainties</p>
{''.join(cards) or '<p>No fixtures.</p>'}</main></body></html>"""


def write(preds, cfg, title, name) -> Path:
    d = Path(cfg["reports_dir"])
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}.html"
    f.write_text(render(preds, cfg, title))
    return f


def text_summary(preds) -> str:
    lines = []
    for p in sorted(preds, key=lambda x: x["kickoff"]):
        pr = p["probs"]
        lines.append(f"{p['match']}: {pr['home']}/{pr['draw']}/{pr['away']} · {p['top_scores'][0][0]}")
        for v in p["value_bets"]:
            tag = "⚠ verify" if v.get("verify") else f"{v['stake_pct']}%"
            lines.append(f"  ★ {SEL.get(v['selection'], v['selection'])} @ {v['price']:.2f}"
                         f" (edge {v['edge']*100:.1f}%, {tag})")
    return "\n".join(lines) or "No fixtures."


def telegram(text: str, file: Path | None = None):
    tok, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text[:4000]}, timeout=20)
        if file:
            with open(file, "rb") as fh:
                requests.post(f"https://api.telegram.org/bot{tok}/sendDocument",
                              data={"chat_id": chat}, files={"document": fh}, timeout=30)
    except Exception as e:  # noqa: BLE001
        log.warning("telegram failed: %s", e)
