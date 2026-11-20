"""Point-in-time tradeable universe: who can be selected at date T.

Training may use any history before T, including departed symbols, but
selection at T is restricted to index members on T. Membership lookups are
fail-closed: asking about a date before the stored snapshot history begins
raises ``LookupError`` instead of silently answering with today's members.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, Iterable, Set


class TradeableUniverse:
    """Union of point-in-time members across the configured universes."""

    def __init__(self, store, universes: Iterable[str]):
        """``store`` needs one method: get_universe_members_at(universe, date_str)."""
        self._store = store
        self._universes = [str(u) for u in universes if str(u).strip()]
        self._cache: Dict[str, Set[str]] = {}

    @classmethod
    def from_db_path(cls, db_path: str, universes: Iterable[str]) -> "TradeableUniverse":
        from .store import FundamentalStore

        return cls(FundamentalStore(str(db_path)), universes)

    def members_at(self, as_of: "date | str") -> Set[str]:
        as_of_str = as_of.isoformat() if isinstance(as_of, date) else str(as_of)
        cached = self._cache.get(as_of_str)
        if cached is not None:
            return cached
        combined: Set[str] = set()
        for universe in self._universes:
            combined |= self._store.get_universe_members_at(universe, as_of_str)
        self._cache[as_of_str] = combined
        return combined
