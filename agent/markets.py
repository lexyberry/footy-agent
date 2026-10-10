"""Turn a score matrix into market probabilities; compare with bookmaker odds."""
import numpy as np


def outcome_masks(n):
    i, j = np.indices((n, n))
    return i > j, i == j, i < j


def probs_1x2(m):
    h, d, a = outcome_masks(m.shape[0])
    return np.array([m[h].sum(), m[d].sum(), m[a].sum()])


def derived(m):
    n = m.shape[0]
    i, j = np.indices((n, n))
    return {
        "1x2": probs_1x2(m),
        "over25": float(m[(i + j) > 2.5].sum()),
        "btts": float(m[(i > 0) & (j > 0)].sum()),
    }


def rescale_to_1x2(m, target):
    """Rescale H/D/A regions of the matrix so they sum to target, keeping shape inside each."""
    m = m.copy()
    cur = probs_1x2(m)
    for mask, c, t in zip(outcome_masks(m.shape[0]), cur, target):
        if c > 0:
            m[mask] *= t / c
    return m / m.sum()


def top_scores(m, k=5):
    flat = np.argsort(m, axis=None)[::-1][:k]
    return [(f"{i}-{j}", float(m[i, j])) for i, j in zip(*np.unravel_index(flat, m.shape))]


def devig(prices: dict) -> dict:
    """Multiplicative de-vig: implied probability / overround."""
    inv = {k: 1 / v for k, v in prices.items() if v and v > 1}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()} if s else {}


def model_probs_for(market: str, d: dict) -> dict:
    if market == "1x2":
        return dict(zip("HDA", d["1x2"]))
    if market == "ou25":
        return {"over": d["over25"], "under": 1 - d["over25"]}
    if market == "btts":
        return {"yes": d["btts"], "no": 1 - d["btts"]}
    return {}


def find_value(d: dict, odds: dict, cfg: dict) -> list[dict]:
    """Shrink model toward market, then flag selections with EV above threshold."""
    mc = cfg["market"]
    out = []
    for market, prices in odds.items():
        mp = model_probs_for(market, d)
        fair = devig(prices)
        for sel, price in prices.items():
            if sel not in mp or sel not in fair:
                continue
            p = (1 - mc["market_shrink"]) * mp[sel] + mc["market_shrink"] * fair[sel]
            ev = p * price - 1
            if ev < mc["min_edge"] or p < mc["min_prob"] or price > mc["max_odds"]:
                continue
            kelly = (p * price - 1) / (price - 1)
            stake = min(mc["kelly_fraction"] * kelly * 100, mc["max_stake_pct"])
            verify = ev > mc.get("suspicious_edge", 1.0)
            out.append({"market": market, "selection": sel, "price": price,
                        "model_prob": round(p, 4), "market_prob": round(fair[sel], 4),
                        "edge": round(ev, 4), "stake_pct": 0.0 if verify else round(stake, 2),
                        "verify": verify})
    return sorted(out, key=lambda x: -x["edge"])
