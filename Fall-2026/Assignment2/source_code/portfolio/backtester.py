"""A compact daily backtester for the course portfolio (selection + valuation + timing).

Rules (deliberately simple and explicit):
  * One model per month, trained at the last session before the month starts.
  * Rebalance on the first session of each week (normally Monday), filled at
    that session's open. Decisions use only information up to the previous
    close (model cutoff, membership on the decision day).
  * Targets are scaled by (1 - cash_buffer) and, optionally, by the exposure
    engine's multiplier for that day.
  * A trade is skipped only if it is both under ``min_trade_value`` dollars
    and under ``min_weight_delta`` of equity. Positions bought fewer than
    ``min_hold_sessions`` sessions ago are not partially trimmed (full exits
    are allowed).
  * Optional two-layer stop on daily closes (10-day peak and entry price,
    both 15% by default), executed at the next open, with a re-entry cooldown.
  * Costs: commission + slippage as a fraction of traded notional.

Fractional shares are not modelled: quantities are whole shares.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from shared.prices import PriceStore
from selection.strategy import MonthlyPredictions, Parameters, target_weights
from shared.universe import TradeableUniverse

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    start: date
    end: date
    initial_cash: float = 100_000.0
    commission: float = 0.0001
    slippage: float = 0.0005
    cash_buffer: float = 0.05
    min_trade_value: float = 1_000.0
    min_weight_delta: float = 0.03
    min_hold_sessions: int = 5
    stops: bool = False
    stop_peak_window: int = 10
    stop_pct: float = 0.15
    stop_cooldown_sessions: int = 5
    use_exposure_engine: bool = False


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    weekly_targets: List[dict] = field(default_factory=list)

    def stats(self) -> dict:
        s = self.equity
        r = s.pct_change().dropna()
        vol = r.std() * np.sqrt(252)
        dd = s / s.cummax() - 1
        years = len(r) / 252
        return {
            "total_return_pct": round((s.iloc[-1] / s.iloc[0] - 1) * 100, 2),
            "cagr_pct": round(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100, 2) if years > 0 else None,
            "sharpe": round(r.mean() * 252 / vol, 2) if vol > 0 else None,
            "volatility_pct": round(vol * 100, 1),
            "max_drawdown_pct": round(dd.min() * 100, 2),
            "trades": int(len(self.trades)),
            "final_equity": round(float(s.iloc[-1]), 2),
        }


def run_backtest(prices: PriceStore, predictions: MonthlyPredictions, params: Parameters,
                 universe: TradeableUniverse, cfg: BacktestConfig, *, global_unified: bool = False,
                 exposure_fn=None) -> BacktestResult:
    sessions = [d for d in prices.sessions if cfg.start <= d.date() <= cfg.end]
    if len(sessions) < 2:
        raise ValueError("Backtest window has fewer than two sessions")
    cash = cfg.initial_cash
    pos: Dict[str, int] = {}
    entry: Dict[str, float] = {}
    bought_at: Dict[str, int] = {}
    cooldown: Dict[str, int] = {}
    pending_stops: set = set()
    trades, curve, targets_log = [], {}, []
    cost = cfg.commission + cfg.slippage
    last_week = None

    close_ff = prices.close.ffill()          # a missing bar marks at the last close

    def mark(day, which="close"):
        px = close_ff.loc[day] if which == "close" else prices.open.loc[day].fillna(close_ff.loc[day])
        return sum(q * float(px.get(s, 0.0)) for s, q in pos.items() if q)

    def trade(day, i, sym, qty, px, reason):
        nonlocal cash
        notional = qty * px
        cash -= notional + abs(notional) * cost
        pos[sym] = pos.get(sym, 0) + qty
        if qty > 0:
            entry[sym] = px if pos[sym] == qty else (entry.get(sym, px) * (pos[sym] - qty) + notional) / pos[sym]
            bought_at.setdefault(sym, i)
        if pos[sym] == 0:
            pos.pop(sym); entry.pop(sym, None); bought_at.pop(sym, None)
        trades.append({"date": day.date(), "symbol": sym, "qty": qty, "price": round(px, 4),
                       "notional": round(notional, 2), "reason": reason})

    for i, day in enumerate(sessions):
        opens = prices.open.loc[day]
        # 1) stops decided at the previous close, filled at this open
        for sym in sorted(pending_stops):
            q = pos.get(sym, 0)
            px = float(opens.get(sym, np.nan))
            if q > 0 and np.isfinite(px):
                trade(day, i, sym, -q, px, "stop")
                cooldown[sym] = i + cfg.stop_cooldown_sessions
        pending_stops.clear()

        # 2) weekly rebalance on the first session of an ISO week
        week = day.isocalendar()[:2]
        if week != last_week:
            last_week = week
            prev_day = sessions[i - 1].date() if i else (day - pd.Timedelta(days=1)).date()
            preds = predictions.get(day.date())
            try:
                members = universe.members_at(prev_day)
            except LookupError as exc:
                logger.warning("No membership at %s: %s", prev_day, exc)
                members = set()
            tradeable = {s for s in members if np.isfinite(float(opens.get(s, np.nan)))}
            tradeable -= {s for s, until in cooldown.items() if until > i}
            weights = target_weights(preds, params, tradeable, pos, global_unified=global_unified)
            budget = 1.0 - cfg.cash_buffer
            if cfg.use_exposure_engine and exposure_fn is not None:
                res = exposure_fn(prev_day)
                if res is not None:
                    budget *= float(res.exposureMultiplier)
            equity = cash + mark(day, "open")
            targets_log.append({"date": day.date(), "model_cutoff": predictions.cutoff,
                                "budget": round(budget, 4), "weights": {k: round(v, 4) for k, v in weights.items()}})
            desired = {s: int((w * budget * equity) // float(opens[s])) for s, w in weights.items()}
            orders = []
            for sym in set(pos) | set(desired):
                px = float(opens.get(sym, np.nan))
                if not np.isfinite(px):
                    continue
                cur, tgt = pos.get(sym, 0), desired.get(sym, 0)
                delta = tgt - cur
                if delta == 0:
                    continue
                if abs(delta * px) < cfg.min_trade_value and abs(delta * px) / equity < cfg.min_weight_delta and tgt > 0:
                    continue
                if delta < 0 and tgt > 0 and i - bought_at.get(sym, -10**9) < cfg.min_hold_sessions:
                    continue
                orders.append((sym, delta, px))
            for sym, delta, px in sorted(orders, key=lambda o: o[1]):     # sells first
                if delta > 0:
                    affordable = int(max(0.0, cash) // (px * (1 + cost)))
                    delta = min(delta, affordable)
                    if delta <= 0:
                        continue
                trade(day, i, sym, delta, px, "rebalance")

        # 3) end-of-day mark and stop checks
        closes = prices.close.loc[day]
        curve[day] = cash + mark(day, "close")
        if cfg.stops:
            hist = prices.close.loc[:day].tail(cfg.stop_peak_window)
            for sym, q in pos.items():
                px = float(closes.get(sym, np.nan))
                if q <= 0 or not np.isfinite(px):
                    continue
                peak = float(hist[sym].max()) if sym in hist else px
                if px <= peak * (1 - cfg.stop_pct) or px <= entry.get(sym, px) * (1 - cfg.stop_pct):
                    pending_stops.add(sym)

    return BacktestResult(pd.Series(curve, name="equity"), pd.DataFrame(trades), targets_log)
