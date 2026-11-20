"""Pure selection transformations; input acquisition belongs to the caller."""
from dataclasses import dataclass
from typing import Mapping, Sequence

from .allocation import ScoredPicks


def rank_with_incumbents(picks: Sequence[Mapping], available: set[str], positions: Mapping[str, float],
                         count: int, incumbent_multiplier: int) -> list[tuple[str, float]]:
    ranked = []
    for pick in picks:
        symbol = pick["symbol"]
        if symbol in available:
            score = pick.get("pred_ensemble_avg", pick.get("predicted_return", 0.0))
            ranked.append((symbol, score, pick.get("pred_raw_score", score)))
    ranked.sort(key=lambda row: row[1], reverse=True)
    top = {symbol for symbol, *_ in ranked[:count * incumbent_multiplier]}
    incumbents = [row for row in ranked if positions.get(row[0], 0) > 0 and row[0] in top]
    incumbent_symbols = {row[0] for row in incumbents}
    result = [(symbol, weight_score) for symbol, _, weight_score in incumbents]
    for symbol, _, weight_score in ranked:
        if len(result) >= count:
            break
        if symbol not in incumbent_symbols:
            result.append((symbol, weight_score))
    return result[:count]


def rank_momentum(symbols: Sequence[str], momentum: Mapping[str, float | None], count: int) -> list[str]:
    scored = [(symbol, momentum.get(symbol)) for symbol in symbols if momentum.get(symbol) is not None]
    scored.sort(key=lambda row: row[1], reverse=True)
    return [symbol for symbol, _ in scored[:count]]


@dataclass(frozen=True)
class CandidateSelection:
    picks: list[tuple[str, float]]
    buckets: dict[str, list[tuple[str, float]]]
    original_count: int


def select_candidates(buckets: Mapping[str, ScoredPicks], *, global_count: int,
                      fallbacks: Sequence[str], available: set[str]) -> CandidateSelection:
    """Pool/truncate candidates or use the existing two-name fallback rule."""
    groups = {bucket: list(items) for bucket, items in buckets.items()}
    picks = [item for items in groups.values() for item in items]
    original_count = len(picks)
    if global_count > 0 and len(picks) > global_count:
        picks.sort(key=lambda item: item[1], reverse=True)
        picks = picks[:global_count]
    if not picks:
        picks = [(symbol, 0.0) for symbol in [s for s in fallbacks if s in available][:2]]
        groups = {"defensive_fallback": picks}
    return CandidateSelection(picks, groups, original_count)
