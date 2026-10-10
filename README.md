# Footy Agent

Football prediction agent for the Top 5 leagues + Ekstraklasa. It estimates probabilities, compares them with Superbet odds, and keeps an honest scoreboard of its own performance.

## How it works

1. **Statistical core.** A time-weighted Dixon-Coles model (attack/defence per team, home advantage, low-score correction) is blended with a goal-difference Elo model. The output is a full scoreline matrix, from which 1X2, Over 2.5, BTTS and the likeliest scores are derived.
2. **Claude context layer.** Claude reads injuries, lineups, table position and recent results, then adjusts each team's expected goals. Code enforces the limits: ±15% on expected goals and at most 8pp movement on any 1X2 outcome. Every adjustment comes with stated reasons. If the Claude call fails, the prediction falls back to neutral.
3. **Market check.** Odds are de-vigged into fair probabilities. The model is shrunk 25% toward the market, and a selection is flagged as value only if EV ≥ 4%. Stakes use quarter-Kelly, capped at 2% of bankroll. Any edge above 12% is marked **⚠ verify** with a stake of 0, because an edge that large usually means stale odds or missing news rather than a real opportunity.
4. **Scoreboard.** Every prediction and flagged bet is stored. Bets are settled nightly. The scoreboard tracks Brier score, log-loss and accuracy against the closing market, plus ROI and closing-line value (CLV). CLV is the metric to trust; short-run ROI is mostly noise.

## Deploy on Railway (production)

The agent runs as one service: the scheduler runs in a background thread, and a token-protected MCP endpoint is served on `/mcp`. A health check lives on `/health`.

1. Push this folder to a private GitHub repo.
2. In Railway, create a new project → **Deploy from GitHub repo**. Railway reads `railway.toml` and builds from the Dockerfile.
3. Add a **volume** mounted at `/app/data`. This holds the database; without it, history is wiped on every deploy.
4. Under **Variables**, add:
   - `API_FOOTBALL_KEY`
   - `ANTHROPIC_API_KEY`
   - `MCP_AUTH_TOKEN`: a long random string. Generate one with `python -c "import secrets;print(secrets.token_urlsafe(32))"`.
   - `INGEST_URL` and `INGEST_SECRET`: from the Lovable dashboard (below).
   - Optional: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
5. Under **Settings → Networking**, generate a domain.

On first boot, an empty volume triggers the full history ingest automatically (~18 API calls). After that the daily schedule runs:

- **08:30:** matchday predictions.
- **Every 15 min:** lineup re-runs and odds snapshots.
- **03:00:** results are settled.

Each run pushes its data to the dashboard.

**Claude connector:** add a custom connector with the URL `https://<your-domain>/mcp?token=<MCP_AUTH_TOKEN>`. Treat that URL like a password, since anyone with it can spend your API-Football and Anthropic credits.

**One-off commands:** open a Railway shell, then run `python -m agent.cli backtest --from 2025-08-01` or `python -m agent.cli sync`.

## Lovable dashboard

The dashboard is a separate Lovable app. It only displays data, and the agent writes to it through the app's `/api/public/ingest` route.

1. In Lovable, go to **Project Settings → Secrets** and add `INGEST_SECRET`. Use a long random string, and put the same value in Railway.
2. Publish the Lovable app, then set `INGEST_URL` in Railway to `https://<your-app>.lovable.app/api/public/ingest`.
3. Open the app and sign up, then confirm your email. The first account becomes the owner, and later sign-ups are refused.
4. Optional check: run `python -m agent.cli sync` in a Railway shell. It should report rows sent.

## Local development (Mac)

```bash
cp .env.example .env    # fill in keys
docker compose up --build     # same single service, on http://127.0.0.1:8765
docker compose exec agent python -m agent.cli bookmakers | grep -i superbet
```

## Daily use

- **Mobile:** use the Lovable dashboard, plus Telegram if configured.
- **Claude:** ask through the connector, e.g. "predict Lech vs Legia, Superbet has 2.30/3.40/3.10".
- **CLI:** run `python -m agent.cli predict <fixture_id> --odds 2.10,3.40,3.60`.

If Superbet isn't in API-Football's bookmaker list, the agent falls back to Pinnacle, Bet365 or Unibet and logs a warning. Check the Superbet price yourself before betting.

## Tuning (config.yaml)

| Setting | Default | Notes |
|---|---|---|
| `blend_weight_dc` | 0.6 | Set from backtest output |
| `dc_half_life_days` | 365 | Shorter reacts faster but is noisier |
| `max_lambda_change` / `max_prob_shift_pp` | 0.15 / 8 | Limits on Claude's influence |
| `market_shrink` | 0.25 | Raise this if the scoreboard shows the market beating the model |
| `min_edge` / `suspicious_edge` | 4% / 12% | Value window |
| `kelly_fraction` / `max_stake_pct` | 0.25 / 2% | Stake sizing |

## Honest limits

- Good models land around 50–55% accuracy on 1X2. The real target is to match or beat the closing market on log-loss and to beat the closing line on average. If that isn't happening after 200+ bets, the edge isn't real.
- API-Football only keeps odds for about a week, so the market comparison builds up from the day you start running the agent. The backtest compares models against each other and a base-rate baseline, not against bookmakers.
- Claude only sees the data passed to it (injuries, lineups, table, results). It does not read news, so late team news can still be missed.
- v2 ideas: xG-based ratings, news feed into the context layer, Shin de-vig, per-league tuning.

## Tests

`pip install -r requirements.txt && pytest -q` runs 14 offline tests on a simulated league with a fake API and a fake Claude. They cover parameter recovery, probability coherence, adjustment limits, fallback behaviour, value detection, settlement and the backtest, plus the MCP auth layer and the dashboard sync payload.
