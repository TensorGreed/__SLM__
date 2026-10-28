"""List-price cost estimates for cloud LLM token usage.

Prices are USD per million tokens (input, output) by model family, as listed
by the providers (checked 2026-10; edit ``LLM_PRICES_JSON`` — e.g.
``{"sonnet": [3, 15]}`` — when they change or for a negotiated rate).
Anthropic prompt caching bills a cache write at 1.25x and a cache read at
0.1x the input price. Anthropic Message Batches bill every token (cache
factors included) at ``BATCH_DISCOUNT`` of the list price. Always shown as an *estimate*: the provider's invoice
is the truth.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "haiku": (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus": (5.0, 25.0),
    # Model generations priced off the family rate (longest match wins).
    "haiku-5-5": (0.1, 0.5),
    "sonnet-5": (2.0, 10.0),
    "opus-5-5": (4.0, 20.0),
    "fable-5": (10.0, 50.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
    "deepseek": (0.27, 1.1),
}
# Usage recorded by the command (CLI) backend is keyed ``subscription:<binary>``.
SUBSCRIPTION_PREFIX = "subscription:"
CACHE_WRITE_FACTOR = 1.25
CACHE_READ_FACTOR = 0.1
# Message Batches: half the list price on input, output and cache tokens.
BATCH_DISCOUNT = 0.5


def _prices() -> dict[str, tuple[float, float]]:
    prices = dict(DEFAULT_PRICES)
    raw = os.environ.get("LLM_PRICES_JSON")
    if raw:
        try:
            for k, v in json.loads(raw).items():
                prices[str(k).lower()] = (float(v[0]), float(v[1]))
        except (ValueError, TypeError, IndexError):
            pass
    return prices


def price_for(model: str | None) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens for the longest matching family."""
    name = str(model or "").lower()
    matches = [(k, v) for k, v in _prices().items() if k in name]
    return max(matches, key=lambda kv: len(kv[0]))[1] if matches else None


def fallback_cost(usage: dict[str, Any] | None, *, batch: bool = False) -> float:
    """A deliberately conservative USD figure for a metered call whose model
    has no known price: priced at the HIGHEST input / output rate in the
    price table (cache factors included). Used only by the budget ledger —
    the spend views still say "price unknown"."""
    prices = _prices().values()
    pin = max(p[0] for p in prices)
    pout = max(p[1] for p in prices)
    if batch:
        pin, pout = pin * BATCH_DISCOUNT, pout * BATCH_DISCOUNT
    u = usage or {}
    inp, out = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    cw, cr = int(u.get("cache_write_tokens") or 0), int(u.get("cache_read_tokens") or 0)
    return round((inp * pin + cw * pin * CACHE_WRITE_FACTOR + cr * pin * CACHE_READ_FACTOR + out * pout) / 1e6, 6)


def estimate_cost(model: str | None, usage: dict[str, Any] | None, *, batch: bool = False) -> dict[str, Any]:
    """``{tokens, usd, basis}`` for one model's usage counters (``usd`` None
    when the model's price is unknown). ``batch``: the calls went through
    Message Batches — every price times ``BATCH_DISCOUNT``."""
    u = usage or {}
    inp, out = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    cw, cr = int(u.get("cache_write_tokens") or 0), int(u.get("cache_read_tokens") or 0)
    if str(model or "").startswith(SUBSCRIPTION_PREFIX):
        # A subscription CLI is not billed per call: always $0, and this runs
        # BEFORE price matching ("subscription:sonnet" must not hit the list price).
        return {
            "model": model, "calls": int(u.get("calls") or 0),
            "input_tokens": inp, "output_tokens": out, "cache_write_tokens": cw, "cache_read_tokens": cr,
            "total_tokens": inp + out + cw + cr, "usd": 0.0, "usd_uncached": 0.0, "batch": bool(batch),
            "basis": "subscription (not billed per call)",
        }
    price = price_for(model)
    if price is not None and batch:
        price = (price[0] * BATCH_DISCOUNT, price[1] * BATCH_DISCOUNT)
    usd = None
    if price is not None:
        pin, pout = price
        usd = round((inp * pin + cw * pin * CACHE_WRITE_FACTOR + cr * pin * CACHE_READ_FACTOR + out * pout) / 1e6, 4)
    return {
        "model": model, "calls": int(u.get("calls") or 0),
        "input_tokens": inp, "output_tokens": out, "cache_write_tokens": cw, "cache_read_tokens": cr,
        "total_tokens": inp + out + cw + cr, "usd": usd,
        # What the same calls cost without the prompt cache (every cached
        # token billed as plain input) — the saving is usd_uncached - usd.
        "usd_uncached": round(((inp + cw + cr) * price[0] + out * price[1]) / 1e6, 4) if price is not None else None,
        "batch": bool(batch),
        "basis": ("list-price estimate" + (" (batch, half price)" if batch else "")) if usd is not None else "price unknown",
    }
