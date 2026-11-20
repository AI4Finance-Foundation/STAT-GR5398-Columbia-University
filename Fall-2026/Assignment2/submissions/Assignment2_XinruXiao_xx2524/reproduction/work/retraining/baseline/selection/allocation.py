"""WeeklyML's numerical allocation rules, independent of runtime and accounts.

These preserve the existing algorithm, including its final normalization. The
floor/cap are redistribution inputs, not guaranteed final position limits;
the framework's execution policy remains responsible for hard risk limits.
"""
from dataclasses import dataclass
from typing import Mapping, Sequence

ScoredPicks = Sequence[tuple[str, float]]


@dataclass(frozen=True)
class ScoreAllocation:
    min_weight_pct: float
    max_weight_pct: float

    def weights(self, picks: ScoredPicks, budget: float, *, clamp_budget: float | None = None) -> dict[str, float]:
        if not picks:
            return {}
        clamp_budget = budget if clamp_budget is None else clamp_budget
        floor = self.min_weight_pct * clamp_budget
        cap = self.max_weight_pct * clamp_budget
        scores = dict(picks)
        min_score = min(scores.values())
        shifted = {sym: score - min_score + 0.01 for sym, score in scores.items()}
        total = sum(shifted.values())
        weights = {sym: value / total * budget for sym, value in shifted.items()}
        for _ in range(10):
            excess = 0.0
            clamped = set()
            for sym in weights:
                if weights[sym] < floor:
                    excess += weights[sym] - floor
                    weights[sym] = floor
                    clamped.add(sym)
                elif weights[sym] > cap:
                    excess += weights[sym] - cap
                    weights[sym] = cap
                    clamped.add(sym)
            if abs(excess) < 1e-8:
                break
            free = [sym for sym in weights if sym not in clamped]
            if not free:
                break
            free_total = sum(weights[sym] for sym in free)
            if free_total > 0:
                for sym in free:
                    weights[sym] += excess * (weights[sym] / free_total)
        total_weight = sum(weights.values())
        if total_weight > 0:
            factor = budget / total_weight
            weights = {sym: weight * factor for sym, weight in weights.items()}
        return weights

    def buckets(self, picks: Mapping[str, ScoredPicks], quotas: Mapping[str, int], budget: float) -> dict[str, float]:
        """Allocate bucket quotas through the same numerical rules."""
        configured_total = sum(int(quotas.get(bucket, 0) or 0) for bucket in picks if bucket in quotas)
        if configured_total <= 0:
            return self.weights([item for items in picks.values() for item in items], budget)
        weights = {}
        for bucket, items in picks.items():
            if not items:
                continue
            configured_n = int(quotas.get(bucket, len(items)) or len(items))
            bucket_budget = budget * configured_n / configured_total
            allocated = self.weights(items, bucket_budget, clamp_budget=budget)
            for sym, weight in allocated.items():
                weights[sym] = weights.get(sym, 0.0) + weight
        total_weight = sum(weights.values())
        if total_weight > 0 and abs(total_weight - budget) > 1e-8:
            weights = {sym: weight * budget / total_weight for sym, weight in weights.items()}
        return weights
