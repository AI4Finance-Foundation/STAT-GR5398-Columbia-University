from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from strategy_framework.interfaces import StrategySignals
from strategy_framework.naming import normalize_config_name


@dataclass
class AlphaRPRegimeConfig:
    # Keep defaults aligned with the notebook alpha_rp implementation.
    top_n: int = 25
    vol_lookback_days: int = 63
    market_lookback_days: int = 21
    rebalance_every_n_days: int = 25
    rebalance_tolerance: float = 0.08

    score_power_risk_on: float = 1.35
    score_power_transition: float = 1.00
    score_power_risk_off: float = 0.75

    single_name_cap: float = 0.12
    sector_cap: float = 0.45

    risk_on_vol_th: float = 0.18
    risk_off_vol_th: float = 0.30
    risk_on_mom_th: float = -0.01
    risk_off_mom_th: float = -0.06

    # Signal enhancement controls
    score_clip_quantile: float = 0.05
    min_alpha_candidates: int = 8
    min_alpha_strength: float = 0.10
    dynamic_top_n: bool = True
    dynamic_top_n_min: int = 10
    dynamic_top_n_max: int = 22

    # Adaptive signal direction inferred from historical (realized) cross-sectional IC.
    ic_forward_days: int = 5
    ic_lookback_obs: int = 126
    ic_pos_th: float = 0.01
    ic_neg_th: float = -0.01
    default_alpha_direction: int = -1

    # Turnover and trade-gating controls for cost robustness.
    min_ic_abs_to_rebalance: float = 0.025
    force_rebalance_drift: float = 0.35
    max_turnover_per_rebalance: float = 0.45
    max_turnover_per_rebalance_weak: float = 0.3825
    ic_turnover_boost_th: float = 0.025
    min_effective_rebalance_turnover: float = 0.02

    force_config_name: str | None = None


class AlphaRPRegimeStrategy:
    """Alpha-tilted inverse-vol weighting with regime-aware alpha power."""

    name = "alpha_rp_regime"

    def __init__(self, config: AlphaRPRegimeConfig | None = None) -> None:
        self.config = config or AlphaRPRegimeConfig()
        self._regime_labels = {0: "RiskOn", 1: "Transition", 2: "RiskOff"}
        self._alpha_power_by_regime = {
            0: self.config.score_power_risk_on,
            1: self.config.score_power_transition,
            2: self.config.score_power_risk_off,
        }

    def _robust_zscore(self, s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 2:
            return pd.Series(0.0, index=s.index, dtype=float)

        q = float(self.config.score_clip_quantile)
        q = min(max(q, 0.0), 0.49)
        if q > 0:
            lo = float(x.quantile(q))
            hi = float(x.quantile(1.0 - q))
            x = x.clip(lower=lo, upper=hi)

        std = float(x.std(ddof=1))
        if not np.isfinite(std) or std <= 1e-12:
            return pd.Series(0.0, index=s.index, dtype=float)
        z = (x - float(x.mean())) / std
        return z.fillna(0.0)

    def _spearman_corr(self, x: pd.Series, y: pd.Series) -> float:
        xv = pd.to_numeric(x, errors="coerce")
        yv = pd.to_numeric(y, errors="coerce")
        m = xv.notna() & yv.notna()
        if int(m.sum()) < max(5, int(self.config.min_alpha_candidates)):
            return float("nan")
        xr = xv[m].rank(method="average")
        yr = yv[m].rank(method="average")
        xv2 = xr - float(xr.mean())
        yv2 = yr - float(yr.mean())
        denom = float(np.sqrt((xv2.pow(2).sum()) * (yv2.pow(2).sum())))
        if denom <= 1e-12 or not np.isfinite(denom):
            return float("nan")
        return float((xv2 * yv2).sum() / denom)

    def _build_ic_observations(
        self,
        selected_stocks_df: pd.DataFrame,
        cc_ret_df: pd.DataFrame,
        ticker_sector: Mapping[str, float],
    ) -> tuple[list[tuple[int, float]], dict[pd.Timestamp, int]]:
        cfg = self.config
        forward_days = max(1, int(cfg.ic_forward_days))

        ret = cc_ret_df.sort_index()
        idx = ret.index
        pos_map = {pd.Timestamp(dt).normalize(): i for i, dt in enumerate(idx)}
        obs: list[tuple[int, float]] = []

        grouped = selected_stocks_df.groupby("trade_date", sort=True)
        for dt, g in grouped:
            date = pd.Timestamp(dt).normalize()
            start_pos = pos_map.get(date, None)
            if start_pos is None:
                continue
            end_pos = start_pos + forward_days
            if end_pos >= len(idx):
                continue

            g2 = (
                g[["tic", "predicted_return"]]
                .dropna()
                .drop_duplicates(subset=["tic"])
                .copy()
            )
            if g2.empty:
                continue

            fwd_ret = (1.0 + ret.iloc[start_pos + 1 : end_pos + 1]).prod(skipna=True) - 1.0
            fwd_ret = pd.to_numeric(fwd_ret, errors="coerce")

            sec = g2["tic"].map(ticker_sector).fillna(-1)
            pred = pd.to_numeric(g2["predicted_return"], errors="coerce")
            pred_res = pred - pred.groupby(sec).transform("mean")
            ret_res = fwd_ret.reindex(g2["tic"]).to_numpy(dtype=float)
            ret_res = pd.Series(ret_res, index=g2.index, dtype=float)
            ret_res = ret_res - ret_res.groupby(sec).transform("mean")

            ic = self._spearman_corr(pred_res, ret_res)
            if np.isfinite(ic):
                obs.append((end_pos, float(ic)))

        return obs, pos_map

    def _resolve_alpha_direction(
        self,
        asof_pos: int | None,
        ic_observations: list[tuple[int, float]],
    ) -> tuple[int, float]:
        cfg = self.config
        default_dir = -1 if int(cfg.default_alpha_direction) < 0 else 1
        if asof_pos is None or asof_pos < 0 or not ic_observations:
            return default_dir, float("nan")

        hist = [ic for end_pos, ic in ic_observations if end_pos < asof_pos and np.isfinite(ic)]
        if not hist:
            return default_dir, float("nan")

        window = hist[-max(5, int(cfg.ic_lookback_obs)) :]
        ic_med = float(np.median(window))
        if ic_med >= float(cfg.ic_pos_th):
            return 1, ic_med
        if ic_med <= float(cfg.ic_neg_th):
            return -1, ic_med
        return default_dir, ic_med

    def _apply_turnover_cap(
        self,
        current_weights: dict[str, float],
        target_weights: dict[str, float],
        ic_median: float,
    ) -> tuple[dict[str, float], float]:
        cfg = self.config
        tics = list(current_weights.keys())
        drift_l1 = float(sum(abs(current_weights[t] - target_weights[t]) for t in tics))
        if drift_l1 <= 1e-12:
            return current_weights.copy(), 0.0

        ic_abs = abs(float(ic_median)) if np.isfinite(ic_median) else 0.0
        cap = float(cfg.max_turnover_per_rebalance if ic_abs >= float(cfg.ic_turnover_boost_th) else cfg.max_turnover_per_rebalance_weak)
        cap = max(0.0, cap)
        if cap <= 1e-12 or drift_l1 <= cap:
            proposed = target_weights.copy()
        else:
            ratio = cap / drift_l1
            proposed = {
                t: float(current_weights[t] + (target_weights[t] - current_weights[t]) * ratio)
                for t in tics
            }

        realized_turn = float(sum(abs(current_weights[t] - proposed[t]) for t in tics))
        return proposed, realized_turn

    def _build_alpha_signal(
        self,
        day_df: pd.DataFrame,
        regime: int,
        alpha_direction: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        cfg = self.config
        rank_power = self._alpha_power_by_regime.get(regime, cfg.score_power_transition)

        d = day_df.copy()
        d["sector"] = d["sector"].fillna(-1)
        raw = pd.to_numeric(d["predicted_return"], errors="coerce").fillna(0.0)

        # Sector-neutralized signal (cross-sectional) to reduce single-theme crowding.
        sec_mean = raw.groupby(d["sector"]).transform("mean")
        residual = raw - sec_mean
        z = self._robust_zscore(residual)
        oriented = z * (-1.0 if int(alpha_direction) < 0 else 1.0)

        z_strength = float(oriented.std(ddof=1)) if oriented.size > 1 else 0.0
        use_dynamic = bool(cfg.dynamic_top_n)
        n_min = max(1, int(cfg.dynamic_top_n_min))
        n_max = max(n_min, int(cfg.dynamic_top_n_max))
        n_cap = int(min(n_max, len(d)))

        if use_dynamic:
            if z_strength >= 1.15:
                n_target = n_min
            elif z_strength <= 0.70:
                n_target = n_cap
            else:
                frac = (1.15 - z_strength) / (1.15 - 0.70)
                n_target = int(round(n_min + frac * (n_cap - n_min)))
        else:
            n_target = int(min(len(d), cfg.top_n))

        n_target = int(max(1, min(n_target, n_cap)))
        d["_score"] = oriented
        d = d.sort_values("_score", ascending=False).head(n_target).copy()
        score = d["_score"]

        # Long-only alpha transform with floor to suppress weak/noisy names.
        alpha = np.power(np.clip(score.to_numpy(dtype=float), a_min=0.0, a_max=None), rank_power)
        alpha = pd.Series(alpha, index=d.index, dtype=float)
        if alpha.gt(0).sum() < int(cfg.min_alpha_candidates) or float(alpha.sum()) <= float(cfg.min_alpha_strength):
            rank_pct = d["_score"].rank(method="average", pct=True).clip(lower=1e-6)
            alpha = np.power(rank_pct.to_numpy(dtype=float), rank_power)
            alpha = pd.Series(alpha, index=d.index, dtype=float)

        return d, alpha

    def _infer_market_regime(self, asof_date: pd.Timestamp, cc_ret_df: pd.DataFrame) -> tuple[int, float, float]:
        # Use only information strictly before asof_date to avoid look-ahead.
        hist = cc_ret_df.loc[cc_ret_df.index < asof_date].tail(self.config.market_lookback_days)
        market_ret = hist.mean(axis=1, skipna=True).dropna()
        if market_ret.empty:
            return 1, float("nan"), float("nan")

        market_vol = float(market_ret.std() * np.sqrt(252))
        market_mom = float((1.0 + market_ret).prod() - 1.0)

        if market_vol <= self.config.risk_on_vol_th and market_mom >= self.config.risk_on_mom_th:
            regime = 0
        elif market_vol <= self.config.risk_off_vol_th and market_mom >= self.config.risk_off_mom_th:
            regime = 1
        else:
            regime = 2
        return regime, market_vol, market_mom

    def _allocate_day(
        self,
        day_df: pd.DataFrame,
        asof_date: pd.Timestamp,
        cc_ret_df: pd.DataFrame,
        ticker_sector: Mapping[str, float],
        alpha_direction: int,
        ic_median: float,
    ) -> tuple[dict[str, float], dict[str, float]]:
        cfg = self.config
        day_df = (
            day_df[["tic", "predicted_return"]]
            .dropna()
            .drop_duplicates(subset=["tic"])
            .sort_values("predicted_return", ascending=False)
            .head(max(int(cfg.top_n), int(cfg.dynamic_top_n_max)))
            .copy()
        )
        if day_df.empty:
            return {}, {"regime": 1, "market_vol": np.nan, "market_mom": np.nan, "alpha_power": cfg.score_power_transition}

        regime, market_vol, market_mom = self._infer_market_regime(asof_date, cc_ret_df)
        alpha_power = self._alpha_power_by_regime.get(regime, cfg.score_power_transition)

        day_df["sector"] = day_df["tic"].map(ticker_sector).fillna(-1)
        day_df, alpha = self._build_alpha_signal(day_df=day_df, regime=regime, alpha_direction=alpha_direction)
        tics = day_df["tic"].tolist()
        if not tics:
            return {}, {
                "regime": regime,
                "market_vol": market_vol,
                "market_mom": market_mom,
                "alpha_power": alpha_power,
                "alpha_direction": int(-1 if int(alpha_direction) < 0 else 1),
                "ic_median": float(ic_median),
            }

        # Volatility estimate must also be based on history before rebalance date.
        hist_ret = cc_ret_df.loc[cc_ret_df.index < asof_date, tics].tail(cfg.vol_lookback_days)
        vol = hist_ret.std(skipna=True).replace(0, np.nan)
        median_vol = vol.dropna().median()
        if pd.isna(median_vol):
            median_vol = 0.02
        vol = vol.fillna(median_vol).clip(lower=0.005)

        inv_vol = 1.0 / vol
        raw = inv_vol.to_numpy() * alpha.to_numpy(dtype=float)
        if raw.sum() <= 0:
            raw = np.ones_like(raw)
        base = raw / raw.sum()

        sectors = day_df["sector"].tolist()
        weights = {tic: 0.0 for tic in tics}
        sector_alloc: dict[float, float] = {}
        active = set(range(len(tics)))
        remaining = 1.0

        for _ in range(40):
            if remaining <= 1e-8 or not active:
                break

            base_sum = base[list(active)].sum()
            if base_sum <= 1e-12:
                break

            allocated = 0.0
            for i in list(active):
                tic = tics[i]
                sec = sectors[i]

                name_room = max(0.0, cfg.single_name_cap - weights[tic])
                sec_room = max(0.0, cfg.sector_cap - sector_alloc.get(sec, 0.0))
                room = min(name_room, sec_room)

                if room <= 1e-10:
                    active.discard(i)
                    continue

                target_add = remaining * (base[i] / base_sum)
                add = min(room, target_add)

                if add > 0:
                    weights[tic] += add
                    sector_alloc[sec] = sector_alloc.get(sec, 0.0) + add
                    allocated += add

                if name_room - add <= 1e-10:
                    active.discard(i)

            if allocated <= 1e-10:
                break
            remaining -= allocated

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        meta = {
            "regime": regime,
            "market_vol": market_vol,
            "market_mom": market_mom,
            "alpha_power": alpha_power,
            "alpha_direction": int(-1 if int(alpha_direction) < 0 else 1),
            "ic_median": float(ic_median),
        }
        return weights, meta

    def generate_signals(
        self,
        selected_stocks_df: pd.DataFrame,
        ticker_sector: Mapping[str, float],
        cc_ret_df: pd.DataFrame,
    ) -> StrategySignals:
        cfg = self.config
        selected = selected_stocks_df.copy()
        selected["trade_date"] = pd.to_datetime(selected["trade_date"]).dt.normalize()
        selected = selected.dropna(subset=["tic", "predicted_return", "trade_date"])

        portfolio_dates = pd.to_datetime(sorted(selected["trade_date"].unique()))
        all_stocks = sorted(selected["tic"].unique())
        rebalance_dates = set(portfolio_dates[:: cfg.rebalance_every_n_days])
        ic_obs, pos_map = self._build_ic_observations(
            selected_stocks_df=selected,
            cc_ret_df=cc_ret_df,
            ticker_sector=ticker_sector,
        )

        weights_dict: dict[pd.Timestamp, dict[str, float]] = {}
        regime_rows: list[dict[str, float]] = []
        current_weights = {tic: 0.0 for tic in all_stocks}

        for date in portfolio_dates:
            regime_today, market_vol_today, market_mom_today = self._infer_market_regime(date, cc_ret_df)
            alpha_power_today = self._alpha_power_by_regime.get(regime_today, cfg.score_power_transition)
            alpha_direction_today, ic_median_today = self._resolve_alpha_direction(
                asof_pos=pos_map.get(pd.Timestamp(date).normalize(), None),
                ic_observations=ic_obs,
            )
            trade_executed = 0

            if date in rebalance_dates or sum(current_weights.values()) == 0.0:
                day_df = selected[selected["trade_date"] == date]
                target_weights, meta = self._allocate_day(
                    day_df=day_df,
                    asof_date=date,
                    cc_ret_df=cc_ret_df,
                    ticker_sector=ticker_sector,
                    alpha_direction=alpha_direction_today,
                    ic_median=ic_median_today,
                )

                target_full = {tic: target_weights.get(tic, 0.0) for tic in all_stocks}
                drift_l1 = sum(abs(current_weights[tic] - target_full[tic]) for tic in all_stocks)

                if sum(current_weights.values()) == 0.0:
                    current_weights = target_full
                    trade_executed = 1
                elif drift_l1 >= cfg.rebalance_tolerance:
                    ic_abs = abs(float(ic_median_today)) if np.isfinite(ic_median_today) else 0.0
                    allow_trade = (
                        ic_abs >= float(cfg.min_ic_abs_to_rebalance)
                        or drift_l1 >= float(cfg.force_rebalance_drift)
                    )
                    if allow_trade:
                        proposed, realized_turn = self._apply_turnover_cap(
                            current_weights=current_weights,
                            target_weights=target_full,
                            ic_median=ic_median_today,
                        )
                        if realized_turn >= float(cfg.min_effective_rebalance_turnover):
                            current_weights = proposed
                            trade_executed = 1

                regime_today = int(meta["regime"])
                market_vol_today = float(meta["market_vol"])
                market_mom_today = float(meta["market_mom"])
                alpha_power_today = float(meta["alpha_power"])
                alpha_direction_today = int(meta["alpha_direction"])
                ic_median_today = float(meta["ic_median"])

            weights_dict[date] = current_weights.copy()
            regime_rows.append(
                {
                    "date": date,
                    "regime": regime_today,
                    "regime_label": self._regime_labels.get(regime_today, "Transition"),
                    "market_vol": market_vol_today,
                    "market_mom": market_mom_today,
                    "alpha_power": alpha_power_today,
                    "alpha_direction": alpha_direction_today,
                    "ic_median": ic_median_today,
                    "trade_executed": trade_executed,
                }
            )

        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index").sort_index()
        weights_df.index.name = "date"
        regime_df = pd.DataFrame(regime_rows).set_index("date").sort_index()

        auto_name = (
            f"alpha_rp_regime_topk{cfg.top_n}_vol{cfg.vol_lookback_days}"
            f"_cap{int(cfg.single_name_cap * 100)}_sec{int(cfg.sector_cap * 100)}"
            f"_reb{cfg.rebalance_every_n_days}_tol{int(cfg.rebalance_tolerance * 100)}"
        )
        config_name = normalize_config_name(cfg.force_config_name) or auto_name

        return StrategySignals(
            weights_signal_df=weights_df,
            regime_df=regime_df,
            config_name=config_name,
            metadata={
                "strategy": self.name,
                "top_n": cfg.top_n,
                "rebalance_every_n_days": cfg.rebalance_every_n_days,
            },
        )
