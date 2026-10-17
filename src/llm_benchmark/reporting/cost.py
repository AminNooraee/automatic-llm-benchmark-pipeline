"""Static reference-model pricing for an actual judge workload.

Costs are derived only from token usage reported by the actual judge and a
committed pricing snapshot. They are estimates because provider tokenization
may differ. Incomplete usage or an invalid reference entry produces an
unavailable estimate for that entry rather than a misleading number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from llm_benchmark.reporting.token_usage import RoleUsage

_PRICING_KEYS = ("input_per_million", "output_per_million")


@dataclass(frozen=True)
class JudgeCost:
    """A priced judge cost (flat) or a cost range (tiered pricing)."""

    priced: bool
    model_display: str | None
    currency: str
    snapshot_date: str
    kind: str  # "flat" | "range" | "none"
    cost: float | None = None
    cost_low: float | None = None
    cost_high: float | None = None
    range_label: str | None = None


@dataclass(frozen=True)
class JudgeCostComparison:
    """Reference-model estimates for one actual judge workload."""

    currency: str
    snapshot_date: str
    estimates: tuple[JudgeCost, ...]


def load_pricing(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Pricing configuration root must be a mapping")
    return raw


def _match_model(pricing: dict, judge_name: str) -> dict | None:
    target = (judge_name or "").strip().lower()
    for entry in pricing.get("models", []) or []:
        if not isinstance(entry, dict):
            continue
        names = [entry.get("name", "")]
        aliases = entry.get("aliases") or []
        if isinstance(aliases, list):
            names.extend(aliases)
        if target in [str(name).strip().lower() for name in names]:
            return entry
    return None


def _cost_for_tier(tier: dict, prompt: int, completion: int) -> float:
    if not isinstance(tier, dict):
        raise ValueError("Pricing tier must be a mapping")
    input_rate = _rate(tier.get("input_per_million"))
    output_rate = _rate(tier.get("output_per_million"))
    return (
        (prompt / 1_000_000) * input_rate
        + (completion / 1_000_000) * output_rate
    )


def _rate(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Pricing rates must be finite non-negative numbers")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Pricing rates must be finite non-negative numbers")
    return number


def _unavailable(
    metadata: dict[str, str], model_display: str | None = None
) -> JudgeCost:
    return JudgeCost(
        priced=False,
        model_display=model_display,
        currency=metadata["currency"],
        snapshot_date=metadata["snapshot_date"],
        kind="none",
    )


def estimate_judge_cost(
    *,
    judge_name: str,
    judge_usage: RoleUsage,
    pricing: dict,
) -> JudgeCost:
    """Estimate judge API cost from reported usage and the pricing snapshot."""

    metadata = _metadata(pricing)
    entry = _match_model(pricing, judge_name)
    display = (
        str(entry.get("display_name") or judge_name)
        if entry is not None
        else None
    )
    if not judge_usage.complete:
        return _unavailable(metadata, display)
    prompt = judge_usage.prompt_tokens
    completion = judge_usage.completion_tokens
    if prompt is None or completion is None:
        return _unavailable(metadata, display)

    if entry is None:
        return _unavailable(metadata)
    pricing_block = entry.get("pricing") or {}
    if not isinstance(pricing_block, dict):
        return _unavailable(metadata, display)

    tiers = pricing_block.get("tiers")
    if tiers:
        if not isinstance(tiers, list):
            return _unavailable(metadata, display)
        try:
            values = [_cost_for_tier(t, prompt, completion) for t in tiers]
        except (KeyError, TypeError, ValueError):
            return _unavailable(metadata, display)
        return JudgeCost(
            priced=True,
            model_display=display,
            currency=metadata["currency"],
            snapshot_date=metadata["snapshot_date"],
            kind="range",
            cost_low=min(values),
            cost_high=max(values),
            range_label=" → ".join(str(t.get("label", "")) for t in tiers) or None,
        )

    if all(key in pricing_block for key in _PRICING_KEYS):
        try:
            cost = _cost_for_tier(pricing_block, prompt, completion)
        except (KeyError, TypeError, ValueError):
            return _unavailable(metadata, display)
        return JudgeCost(
            priced=True,
            model_display=display,
            currency=metadata["currency"],
            snapshot_date=metadata["snapshot_date"],
            kind="flat",
            cost=cost,
        )
    return _unavailable(metadata, display)


def _metadata(pricing: dict) -> dict:
    return {
        "currency": str(pricing.get("currency", "USD")),
        "snapshot_date": str(pricing.get("snapshot_date", "")),
    }


def estimate_reference_costs(
    *, judge_usage: RoleUsage, pricing: dict
) -> JudgeCostComparison:
    """Price one judge workload against every configured reference model."""

    metadata = _metadata(pricing)
    models = pricing.get("models", [])
    if not isinstance(models, list):
        models = []
    estimates: list[JudgeCost] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        display = str(
            entry.get("display_name") or entry.get("name") or "Unavailable"
        )
        estimates.append(
            _estimate_reference_entry(
                entry=entry,
                display=display,
                judge_usage=judge_usage,
                metadata=metadata,
            )
        )
    return JudgeCostComparison(
        currency=metadata["currency"],
        snapshot_date=metadata["snapshot_date"],
        estimates=tuple(estimates),
    )


def _estimate_reference_entry(
    *,
    entry: dict,
    display: str,
    judge_usage: RoleUsage,
    metadata: dict[str, str],
) -> JudgeCost:
    if not judge_usage.complete:
        return _unavailable(metadata, display)
    prompt = judge_usage.prompt_tokens
    completion = judge_usage.completion_tokens
    if prompt is None or completion is None:
        return _unavailable(metadata, display)

    pricing_block = entry.get("pricing")
    if not isinstance(pricing_block, dict):
        return _unavailable(metadata, display)
    tiers = pricing_block.get("tiers")
    if tiers is not None:
        if not isinstance(tiers, list) or not tiers:
            return _unavailable(metadata, display)
        try:
            values = [_cost_for_tier(tier, prompt, completion) for tier in tiers]
        except (KeyError, TypeError, ValueError):
            return _unavailable(metadata, display)
        return JudgeCost(
            priced=True,
            model_display=display,
            currency=metadata["currency"],
            snapshot_date=metadata["snapshot_date"],
            kind="range",
            cost_low=min(values),
            cost_high=max(values),
            range_label=" → ".join(str(tier.get("label", "")) for tier in tiers),
        )
    if not all(key in pricing_block for key in _PRICING_KEYS):
        return _unavailable(metadata, display)
    try:
        value = _cost_for_tier(pricing_block, prompt, completion)
    except (KeyError, TypeError, ValueError):
        return _unavailable(metadata, display)
    return JudgeCost(
        priced=True,
        model_display=display,
        currency=metadata["currency"],
        snapshot_date=metadata["snapshot_date"],
        kind="flat",
        cost=value,
    )
