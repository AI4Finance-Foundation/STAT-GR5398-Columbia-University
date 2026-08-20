from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from strategy_framework.interfaces import StrategySignals
from strategy_framework.naming import normalize_config_name


@dataclass
class EqualWeightTopKConfig:
    top_n: int = 20
    rebalance_every_n_days: int = 15
    rebalance_tolerance: float = 0.05

    score_clip_quantile: float = 0.05
    dynamic_top_n: bool = True
    dynamic_top_n_min: int = 10
    dynamic_top_n_max: int = 22
    hold_buffer: int = 5

    ic_forward_days: int = 5
    ic_lookback_obs: int = 126
    ic_pos_th: float = 0.01
    ic_neg_th: float = -0.01
    default_alpha_direction: int = -1

    force_config_name: str | None = None


class EqualWeightTopKStrategy:
    name = "equal_weight_topk"

    def __init__(self, config: EqualWeightTopKConfig | None = None) -> None:
        self.config = config or EqualWeightTopKConfig()

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
        mask = xv.notna() & yv.notna()
        if int(mask.sum()) < 5:
            return float("nan")
        xr = xv[mask].rank(method="average")
        yr = yv[mask].rank(method="average")
        xr = xr - float(xr.mean())
        yr = yr - float(yr.mean())
        denom = float(np.sqrt((xr.pow(2).sum()) * (yr.pow(2).sum())))
        if denom <= 1e-12 or not np.isfinite(denom):
            return float("nan")
        return float((xr * yr).sum() / denom)

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

        for dt, g in selected_stocks_df.groupby("trade_date", sort=True):
            date = pd.Timestamp(dt).normalize()
            start_pos = pos_map.get(date, None)
            if start_pos is None:
                continue
            end_pos = start_pos + forward_days
            if end_pos >= len(idx):
                continue

            d = (
                g[["tic", "predicted_return"]]
                .dropna()
                .drop_duplicates(subset=["tic"])
                .copy()
            )
            if d.empty:
                continue

            sec = d["tic"].map(ticker_sector).fillna(-1)
            pred = pd.to_numeric(d["predicted_return"], errors="coerce")
            pred_res = pred - pred.groupby(sec).transform("mean")

            fwd = (1.0 + ret.iloc[start_pos + 1 : end_pos + 1]).prod(skipna=True) - 1.0
            fwd = pd.to_numeric(fwd.reindex(d["tic"]), errors="coerce")
            fwd = pd.Series(fwd.to_numpy(dtype=float), index=d.index)
            fwd_res = fwd - fwd.groupby(sec).transform("mean")

            ic = self._spearman_corr(pred_res, fwd_res)
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

    def _rank_candidates(
        self,
        day_df: pd.DataFrame,
        alpha_direction: int,
    ) -> tuple[list[str], int]:
        cfg = self.config
        d = (
            day_df[["tic", "predicted_return", "sector"]]
            .dropna(subset=["tic", "predicted_return"])
            .drop_duplicates(subset=["tic"])
            .copy()
        )
        if d.empty:
            return [], 0

        raw = pd.to_numeric(d["predicted_return"], errors="coerce").fillna(0.0)
        sec_mean = raw.groupby(d["sector"]).transform("mean")
        residual = raw - sec_mean
        z = self._robust_zscore(residual)
        score = z * (-1.0 if int(alpha_direction) < 0 else 1.0)

        z_strength = float(score.std(ddof=1)) if score.size > 1 else 0.0
        n_min = max(1, int(cfg.dynamic_top_n_min))
        n_max = max(n_min, int(cfg.dynamic_top_n_max))
        n_cap = int(min(n_max, len(d)))
        if bool(cfg.dynamic_top_n):
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
        d["_score"] = score
        ranked = d.sort_values("_score", ascending=False)["tic"].tolist()
        return ranked, n_target

    def _hysteresis_select(
        self,
        ranked: list[str],
        prev_selected: set[str],
        n_target: int,
    ) -> list[str]:
        if not ranked or n_target <= 0:
            return []

        cfg = self.config
        rank_pos = {tic: i for i, tic in enumerate(ranked)}
        keep_cut = int(min(len(ranked), n_target + max(0, int(cfg.hold_buffer))))

        kept = [tic for tic in ranked[:keep_cut] if tic in prev_selected]
        selected = kept[:n_target]
        if len(selected) < n_target:
            for tic in ranked:
                if tic not in selected:
                    selected.append(tic)
                    if len(selected) >= n_target:
                        break
        return selected[:n_target]

    def generate_signals(
        self,
        selected_stocks_df: pd.DataFrame,
        ticker_sector,
        cc_ret_df,
    ) -> StrategySignals:
        cfg = self.config
        selected = selected_stocks_df.copy()
        selected["trade_date"] = pd.to_datetime(selected["trade_date"]).dt.normalize()
        selected = selected.dropna(subset=["tic", "predicted_return", "trade_date"])

        dates = pd.to_datetime(sorted(selected["trade_date"].unique()))
        all_tics = sorted(selected["tic"].unique())
        rebalance_dates = set(dates[:: cfg.rebalance_every_n_days])

        ic_obs, pos_map = self._build_ic_observations(
            selected_stocks_df=selected,
            cc_ret_df=cc_ret_df,
            ticker_sector=ticker_sector,
        )

        weights_dict: dict[pd.Timestamp, dict[str, float]] = {}
        regime_rows = []
        current = {t: 0.0 for t in all_tics}
        prev_selected: set[str] = set()

        for d in dates:
            alpha_direction, ic_median = self._resolve_alpha_direction(
                asof_pos=pos_map.get(pd.Timestamp(d).normalize(), None),
                ic_observations=ic_obs,
            )
            trade_executed = 0

            if d in rebalance_dates or sum(current.values()) == 0.0:
                day = selected[selected["trade_date"] == d].copy()
                day["sector"] = day["tic"].map(ticker_sector).fillna(-1)
                ranked, n_target = self._rank_candidates(day_df=day, alpha_direction=alpha_direction)
                chosen = self._hysteresis_select(
                    ranked=ranked,
                    prev_selected=prev_selected,
                    n_target=n_target,
                )

                next_w = {t: 0.0 for t in all_tics}
                if chosen:
                    w = 1.0 / len(chosen)
                    for t in chosen:
                        next_w[t] = w

                drift = float(sum(abs(next_w[t] - current[t]) for t in all_tics))
                if sum(current.values()) == 0.0 or drift >= float(cfg.rebalance_tolerance):
                    current = next_w
                    prev_selected = {t for t in chosen}
                    trade_executed = 1

            weights_dict[d] = current.copy()
            regime_rows.append(
                {
                    "date": d,
                    "regime": 1,
                    "regime_label": "Neutral",
                    "market_vol": np.nan,
                    "market_mom": np.nan,
                    "alpha_power": np.nan,
                    "alpha_direction": int(-1 if int(alpha_direction) < 0 else 1),
                    "ic_median": float(ic_median),
                    "trade_executed": trade_executed,
                }
            )

        weights_df = pd.DataFrame.from_dict(weights_dict, orient="index").sort_index()
        weights_df.index.name = "date"
        regime_df = pd.DataFrame(regime_rows).set_index("date").sort_index()

        auto_name = f"eqw_topk{cfg.top_n}_reb{cfg.rebalance_every_n_days}"
        config_name = normalize_config_name(cfg.force_config_name) or auto_name

        return StrategySignals(weights_signal_df=weights_df, regime_df=regime_df, config_name=config_name)
