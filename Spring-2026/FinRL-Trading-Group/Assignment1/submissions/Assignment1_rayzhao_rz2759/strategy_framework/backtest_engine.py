from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    return_mode: str = "close_to_close"

    # Allow zero lag when signals are already timestamp-safe.
    weight_lag_days: int = 0
    exposure_signal_lag_days: int = 0
    vol_scale_lag_days: int = 0

    target_annual_vol: float = 0.15
    vol_target_lookback: int = 20
    min_exposure: float = 0.20
    max_vol_scale: float = 1.00
    min_vol_scale: float = 0.35
    fee_rate: float = 0.001
    default_base_exposure: float = 0.70
    # Deprecated: retained only for backward compatibility.
    drop_dates_with_missing_held_returns: bool = False

    regime_base_exposure: dict[int, float] | None = None
    # Strategy already embeds regime adaptation; keep engine regime overlay off by default
    # to avoid double-penalty unless explicitly enabled.
    apply_regime_exposure_overlay: bool = False
    apply_vol_target_overlay: bool = True

    # Missing held-return handling:
    # - "zero": fill missing with 0 return
    # - "raise": fail immediately when held assets have missing returns
    # - "liquidate": apply missing_ret_penalty (default -100%) then force 0 on later missing days
    missing_ret_policy: str = "liquidate"
    missing_ret_penalty: float = -1.0
    missing_ret_penalty_once: bool = True

    # Capacity and impact controls. Require volume column + portfolio_aum.
    max_participation_of_adv: float | None = None
    adv_lookback_days: int = 20
    portfolio_aum: float | None = None
    impact_bps_at_100pct_adv: float = 0.0
    impact_exponent: float = 1.0
    volume_col: str = "cshtrd"

    def resolved_regime_base_exposure(self) -> dict[int, float]:
        if self.regime_base_exposure is not None:
            return self.regime_base_exposure
        return {0: 1.00, 1: 0.70, 2: 0.35}


def _normalize_lag_days(value: int, field_name: str) -> int:
    lag = int(value)
    if lag < 0:
        raise ValueError(f"{field_name} must be >= 0, got {lag}.")
    return lag


def resolve_effective_lags(cfg: BacktestConfig) -> tuple[int, int, int]:
    """Resolve lag settings with anti-lookahead guardrails.

    For close-to-close returns, same-day signals are not tradable without
    timestamp-level validation. Enforce at least 1 business-day lag.
    """
    weight_lag_days = _normalize_lag_days(cfg.weight_lag_days, "weight_lag_days")
    exposure_signal_lag_days = _normalize_lag_days(cfg.exposure_signal_lag_days, "exposure_signal_lag_days")
    vol_scale_lag_days = _normalize_lag_days(cfg.vol_scale_lag_days, "vol_scale_lag_days")

    mode = str(cfg.return_mode).strip().lower()
    if mode == "close_to_close":
        weight_lag_days = max(1, weight_lag_days)
        exposure_signal_lag_days = max(1, exposure_signal_lag_days)
        vol_scale_lag_days = max(1, vol_scale_lag_days)

    return weight_lag_days, exposure_signal_lag_days, vol_scale_lag_days


def _normalize_missing_ret_policy(policy: str) -> str:
    canonical = str(policy).strip().lower()
    aliases = {
        "fill_zero": "zero",
        "flat": "zero",
        "error": "raise",
        "strict": "raise",
        "liquidate_with_penalty": "liquidate",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in {"zero", "raise", "liquidate"}:
        raise ValueError(
            "Unsupported missing_ret_policy. Use one of {'zero','raise','liquidate'}."
        )
    return canonical


def _prepare_return_matrix(daily_price_df: pd.DataFrame, return_mode: str) -> pd.DataFrame:
    daily = daily_price_df.copy()
    daily = daily.rename(columns={"datadate": "date", "prcod": "open", "prccd": "close"})
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["open_adj"] = pd.to_numeric(daily["open"], errors="coerce") / pd.to_numeric(daily["ajexdi"], errors="coerce")
    daily["close_adj"] = pd.to_numeric(daily["close"], errors="coerce") / pd.to_numeric(
        daily["ajexdi"], errors="coerce"
    )

    mode = return_mode.strip().lower()
    if mode == "close_to_close":
        daily = daily.sort_values(["tic", "date"])
        daily["ret"] = daily.groupby("tic")["close_adj"].pct_change()
    elif mode == "intraday":
        daily["ret"] = daily["close_adj"] / daily["open_adj"] - 1
    else:
        raise ValueError(f"Unsupported return_mode: {return_mode}")

    ret_df = daily.pivot(index="date", columns="tic", values="ret").sort_index()
    return ret_df.replace([float("inf"), float("-inf")], float("nan"))


def _prepare_adv_dollar_matrix(
    daily_price_df: pd.DataFrame,
    volume_col: str,
    lookback_days: int,
) -> pd.DataFrame | None:
    required_cols = {"datadate", "tic", "ajexdi", "prccd", volume_col}
    if not required_cols.issubset(set(daily_price_df.columns)):
        return None

    daily = daily_price_df[list(required_cols)].copy()
    daily = daily.rename(columns={"datadate": "date", "prccd": "close", volume_col: "volume"})
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["close_adj"] = pd.to_numeric(daily["close"], errors="coerce") / pd.to_numeric(
        daily["ajexdi"], errors="coerce"
    )
    daily["volume"] = pd.to_numeric(daily["volume"], errors="coerce").abs()
    daily["dollar_volume"] = (daily["close_adj"].abs() * daily["volume"]).replace(
        [float("inf"), float("-inf")], float("nan")
    )

    dollar_vol_df = daily.pivot(index="date", columns="tic", values="dollar_volume").sort_index()
    window = max(1, int(lookback_days))
    return dollar_vol_df.rolling(window=window, min_periods=1).mean()


def _sanitize_day_returns(
    day_ret_raw: pd.Series,
    live_equity: pd.Series,
    has_seen_valid_return: pd.Series,
    missing_ret_policy: str,
    missing_ret_penalty: float,
    missing_ret_penalty_once: bool,
    penalized_assets: set[str],
    date: pd.Timestamp,
) -> tuple[pd.Series, float]:
    day_ret = day_ret_raw.copy()
    held_mask = live_equity.abs() > 1e-12
    missing_held_mask = held_mask & day_ret.isna()
    penalty_component = 0.0

    if missing_held_mask.any():
        missing_tics = day_ret.index[missing_held_mask].tolist()
        if missing_ret_policy == "raise":
            preview = ", ".join(missing_tics[:8])
            raise ValueError(
                f"Held assets have missing returns on {date.date()}: {preview}. "
                "Use a different missing_ret_policy or provide complete prices."
            )

        if missing_ret_policy == "liquidate":
            for tic in missing_tics:
                if not bool(has_seen_valid_return.get(tic, False)):
                    # Do not penalize assets before the first valid return observation.
                    day_ret.at[tic] = 0.0
                    continue
                apply_penalty = (not missing_ret_penalty_once) or (tic not in penalized_assets)
                if apply_penalty:
                    day_ret.at[tic] = missing_ret_penalty
                    penalty_component += float(live_equity.at[tic] * missing_ret_penalty)
                    penalized_assets.add(tic)
                else:
                    day_ret.at[tic] = 0.0
        else:
            day_ret.loc[missing_held_mask] = 0.0

    day_ret = day_ret.fillna(0.0).astype(float)
    return day_ret, penalty_component


def _simulate_drift_aware_execution(
    target_weights_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    fee_rate: float,
    rebalance_mask: pd.Series | None = None,
    trade_capacity_df: pd.DataFrame | None = None,
    weight_at_100pct_adv_df: pd.DataFrame | None = None,
    impact_bps_at_100pct_adv: float = 0.0,
    impact_exponent: float = 1.0,
    missing_ret_policy: str = "zero",
    missing_ret_penalty: float = -1.0,
    missing_ret_penalty_once: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """Trade only on rebalance days, drift holdings otherwise."""
    if target_weights_df.empty:
        empty = pd.Series(dtype=float)
        return empty, empty, empty, empty, empty, empty, empty, empty, pd.DataFrame()

    target_weights_df = target_weights_df.fillna(0.0).astype(float)
    ret_df = ret_df.reindex(index=target_weights_df.index, columns=target_weights_df.columns)

    if trade_capacity_df is not None:
        trade_capacity_df = (
            trade_capacity_df.reindex(index=target_weights_df.index, columns=target_weights_df.columns)
            .fillna(0.0)
            .clip(lower=0.0)
            .astype(float)
        )
    if weight_at_100pct_adv_df is not None:
        weight_at_100pct_adv_df = (
            weight_at_100pct_adv_df.reindex(index=target_weights_df.index, columns=target_weights_df.columns)
            .fillna(0.0)
            .clip(lower=0.0)
            .astype(float)
        )

    index = target_weights_df.index
    columns = target_weights_df.columns

    gross_ret = pd.Series(0.0, index=index, dtype=float)
    equity_turnover = pd.Series(0.0, index=index, dtype=float)
    cash_turnover = pd.Series(0.0, index=index, dtype=float)
    turnover = pd.Series(0.0, index=index, dtype=float)
    impact_cost = pd.Series(0.0, index=index, dtype=float)
    missing_penalty_ret = pd.Series(0.0, index=index, dtype=float)
    capacity_shortfall = pd.Series(0.0, index=index, dtype=float)
    executed_equity_weights = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)

    if rebalance_mask is None:
        rebalance_mask = pd.Series(True, index=index, dtype=bool)
    else:
        rebalance_mask = rebalance_mask.reindex(index).fillna(False).astype(bool)

    prev_post_equity = pd.Series(0.0, index=columns, dtype=float)
    prev_post_cash = 1.0
    penalized_assets: set[str] = set()
    has_seen_valid_return = pd.Series(False, index=columns, dtype=bool)

    for i, date in enumerate(index):
        do_rebalance = bool(rebalance_mask.loc[date]) or i == 0
        exec_delta = pd.Series(0.0, index=columns, dtype=float)
        cap_short = 0.0

        if do_rebalance:
            target_equity = target_weights_df.loc[date].copy()
            sum_w = float(target_equity.sum())
            if sum_w <= 0:
                target_equity[:] = 0.0
                target_cash = 1.0
            elif sum_w > 1.0:
                # Disallow free leverage in engine execution.
                target_equity = target_equity / sum_w
                target_cash = 0.0
            else:
                target_cash = 1.0 - sum_w

            desired_delta = target_equity - prev_post_equity
            exec_delta = desired_delta.copy()

            if trade_capacity_df is not None:
                cap_row = trade_capacity_df.loc[date]
                exec_delta = exec_delta.clip(lower=-cap_row, upper=cap_row)

            sells = (-exec_delta.clip(upper=0.0)).clip(lower=0.0)
            sells = pd.concat([sells, prev_post_equity], axis=1).min(axis=1)
            buys = exec_delta.clip(lower=0.0)

            cash_available = float(prev_post_cash + sells.sum())
            buy_sum = float(buys.sum())
            if buy_sum > cash_available + 1e-12 and buy_sum > 0:
                buys = buys * (cash_available / buy_sum)

            exec_delta = buys - sells
            live_equity = (prev_post_equity + exec_delta).clip(lower=0.0)
            live_cash = float(prev_post_cash + sells.sum() - buys.sum())
            if live_cash < 0 and abs(live_cash) <= 1e-12:
                live_cash = 0.0

            if live_cash < -1e-9:
                raise RuntimeError(
                    "Execution produced negative cash. Check liquidity constraints and trade sizing."
                )

            eq_turn = float(exec_delta.abs().sum())
            ca_turn = float(abs(live_cash - prev_post_cash))
            cap_short = float((desired_delta - exec_delta).abs().sum())
        else:
            live_equity = prev_post_equity
            live_cash = prev_post_cash
            eq_turn = 0.0
            ca_turn = 0.0

        turn = eq_turn
        day_impact_cost = 0.0

        if i == 0:
            # Do not charge initialization costs for the first modeled date.
            eq_turn = 0.0
            ca_turn = 0.0
            turn = 0.0
            cap_short = 0.0

        day_ret_raw = ret_df.loc[date]
        day_ret, penalty_component = _sanitize_day_returns(
            day_ret_raw=day_ret_raw,
            live_equity=live_equity,
            has_seen_valid_return=has_seen_valid_return,
            missing_ret_policy=missing_ret_policy,
            missing_ret_penalty=missing_ret_penalty,
            missing_ret_penalty_once=missing_ret_penalty_once,
            penalized_assets=penalized_assets,
            date=date,
        )
        has_seen_valid_return = has_seen_valid_return | day_ret_raw.notna()
        missing_penalty_ret.iloc[i] = penalty_component
        executed_equity_weights.loc[date] = live_equity

        if (
            do_rebalance
            and i > 0
            and impact_bps_at_100pct_adv > 0
            and weight_at_100pct_adv_df is not None
        ):
            adv_weight = weight_at_100pct_adv_df.loc[date].replace(0.0, np.nan)
            participation = (exec_delta.abs() / adv_weight).replace([float("inf"), float("-inf")], np.nan).fillna(0.0)
            impact_rate = (impact_bps_at_100pct_adv / 10000.0) * participation.pow(impact_exponent)
            day_impact_cost = float((impact_rate * exec_delta.abs()).sum())

        equity_turnover.iloc[i] = eq_turn
        cash_turnover.iloc[i] = ca_turn
        turnover.iloc[i] = turn
        impact_cost.iloc[i] = day_impact_cost
        capacity_shortfall.iloc[i] = cap_short

        gross = float((live_equity * day_ret).sum())
        gross_ret.iloc[i] = gross

        denom = 1.0 + gross
        if denom <= 1e-12:
            prev_post_equity = pd.Series(0.0, index=columns, dtype=float)
            prev_post_cash = 1.0
            continue

        prev_post_equity = (
            live_equity * (1.0 + day_ret) / denom
        ).replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
        prev_post_cash = float(live_cash / denom)

    fee_cost = turnover * float(fee_rate)
    return (
        gross_ret,
        turnover,
        fee_cost,
        impact_cost,
        equity_turnover,
        cash_turnover,
        missing_penalty_ret,
        capacity_shortfall,
        executed_equity_weights,
    )


def run_backtest(
    weights_signal_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    daily_price_df: pd.DataFrame,
    config: BacktestConfig | None = None,
    return_exec_weights: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or BacktestConfig()

    ret_df = _prepare_return_matrix(daily_price_df, cfg.return_mode)

    weights_signal_df = weights_signal_df.sort_index()
    regime_df = regime_df.sort_index()

    weight_lag_days, exposure_signal_lag_days, vol_scale_lag_days = resolve_effective_lags(cfg)
    missing_ret_policy = _normalize_missing_ret_policy(cfg.missing_ret_policy)
    impact_exponent = float(cfg.impact_exponent)
    if impact_exponent < 0:
        raise ValueError(f"impact_exponent must be >= 0, got {impact_exponent}.")

    weights_exec_df = weights_signal_df.shift(weight_lag_days).ffill().fillna(0.0)

    common_dates = weights_exec_df.index.intersection(ret_df.index)
    weights_signal_df = weights_signal_df.loc[common_dates]
    weights_exec_df = weights_exec_df.loc[common_dates]
    missing_ret_assets = weights_exec_df.columns.difference(ret_df.columns)
    if not missing_ret_assets.empty:
        held_on_missing = (weights_exec_df[missing_ret_assets].abs().sum(axis=1) > 0).any()
        if held_on_missing:
            preview = ", ".join(list(missing_ret_assets[:8]))
            raise ValueError(
                f"Return matrix missing held assets: {preview}. "
                "Please align signal universe and price universe."
            )

    ret_df = ret_df.reindex(index=common_dates, columns=weights_exec_df.columns)
    regime_aligned = regime_df.reindex(common_dates).copy()
    if "regime" not in regime_aligned.columns:
        regime_aligned["regime"] = 1
    regime_aligned["regime"] = (
        pd.to_numeric(regime_aligned["regime"], errors="coerce").ffill().bfill().fillna(1).astype(int)
    )
    if "regime_label" not in regime_aligned.columns:
        regime_aligned["regime_label"] = pd.NA
    regime_aligned["regime_label"] = regime_aligned["regime_label"].ffill().bfill()
    regime_label_map = {0: "RiskOn", 1: "Transition", 2: "RiskOff"}
    regime_aligned["regime_label"] = (
        regime_aligned["regime_label"].fillna(regime_aligned["regime"].map(regime_label_map)).fillna("Unknown")
    )

    weight_at_100pct_adv_df = None
    trade_capacity_df = None
    use_liquidity_controls = (
        cfg.max_participation_of_adv is not None
        or float(cfg.impact_bps_at_100pct_adv) > 0
    )
    if use_liquidity_controls:
        if cfg.portfolio_aum is None or float(cfg.portfolio_aum) <= 0:
            raise ValueError("Liquidity controls require portfolio_aum > 0.")

        adv_dollar_df = _prepare_adv_dollar_matrix(
            daily_price_df=daily_price_df,
            volume_col=cfg.volume_col,
            lookback_days=cfg.adv_lookback_days,
        )
        if adv_dollar_df is None:
            raise ValueError(
                "Liquidity controls require volume data. "
                f"Column '{cfg.volume_col}' is missing from daily_price_df."
            )

        weight_at_100pct_adv_df = (
            adv_dollar_df.reindex(index=common_dates, columns=weights_exec_df.columns)
            / float(cfg.portfolio_aum)
        ).fillna(0.0).clip(lower=0.0)

        if cfg.max_participation_of_adv is not None:
            max_participation = float(cfg.max_participation_of_adv)
            if max_participation <= 0:
                raise ValueError(
                    f"max_participation_of_adv must be > 0, got {max_participation}."
                )
            trade_capacity_df = weight_at_100pct_adv_df * max_participation

    signal_rebalance_mask = (weights_exec_df.diff().abs().sum(axis=1).fillna(0.0) > 1e-12)
    if not signal_rebalance_mask.empty:
        signal_rebalance_mask.iloc[0] = True

    if bool(cfg.apply_regime_exposure_overlay):
        regime_base = cfg.resolved_regime_base_exposure()
        base_exposure_signal = regime_aligned["regime"].map(regime_base)
        base_exposure_signal = pd.to_numeric(base_exposure_signal, errors="coerce")
        base_exposure_signal = base_exposure_signal.fillna(float(cfg.default_base_exposure)).clip(lower=0.0, upper=1.0)
    else:
        base_exposure_signal = pd.Series(1.0, index=weights_exec_df.index, dtype=float)
    base_exposure_default = float(cfg.default_base_exposure) if bool(cfg.apply_regime_exposure_overlay) else 1.0
    base_exposure = (
        base_exposure_signal.shift(exposure_signal_lag_days).ffill().fillna(base_exposure_default)
    ).clip(lower=0.0, upper=1.0)

    if bool(cfg.apply_vol_target_overlay):
        # Strictly history-only volatility signal:
        # use lagged holdings proxy and shift by 1 day before rolling stats.
        lagged_weights_for_vol = weights_exec_df.shift(1).fillna(0.0)
        signal_portfolio_ret = (lagged_weights_for_vol * ret_df).sum(axis=1, min_count=1)
        vol_window = max(2, int(cfg.vol_target_lookback))
        vol_min_periods = max(2, vol_window // 2)
        rolling_ann_vol = signal_portfolio_ret.shift(1).rolling(
            window=vol_window,
            min_periods=vol_min_periods,
        ).std() * np.sqrt(252)
        target_ann_vol = float(cfg.target_annual_vol)
        vol_scale_signal = (target_ann_vol / rolling_ann_vol).replace([float("inf"), float("-inf")], float("nan"))
        vol_scale_signal = vol_scale_signal.fillna(1.0).clip(
            lower=float(cfg.min_vol_scale),
            upper=float(cfg.max_vol_scale),
        )
    else:
        vol_scale_signal = pd.Series(1.0, index=weights_exec_df.index, dtype=float)
    vol_scale = (
        vol_scale_signal.shift(vol_scale_lag_days).ffill().fillna(1.0)
    ).clip(lower=float(cfg.min_vol_scale), upper=float(cfg.max_vol_scale))

    exposure_raw = base_exposure * vol_scale
    has_equity_target = weights_exec_df.abs().sum(axis=1) > 1e-12
    exposure_signal = pd.Series(0.0, index=weights_exec_df.index, dtype=float)
    exposure_signal.loc[has_equity_target] = exposure_raw.loc[has_equity_target].clip(
        lower=float(cfg.min_exposure),
        upper=1.0,
    )
    # Do not let daily overlay jitter force high-frequency trading.
    # Overlay is applied only when the underlying strategy weights rebalance.
    exposure = exposure_signal.where(signal_rebalance_mask).ffill()
    if not exposure.empty:
        exposure = exposure.fillna(exposure_signal.iloc[0] if not pd.isna(exposure_signal.iloc[0]) else 1.0)
    else:
        exposure = exposure.fillna(1.0)
    exposure = exposure.where(has_equity_target, 0.0)
    effective_weights = weights_exec_df.mul(exposure, axis=0)
    rebalance_mask = signal_rebalance_mask

    (
        gross_ret,
        turnover,
        fee_cost,
        impact_cost,
        equity_turnover,
        cash_turnover,
        missing_penalty_ret,
        capacity_shortfall,
        executed_equity_weights,
    ) = _simulate_drift_aware_execution(
        target_weights_df=effective_weights,
        ret_df=ret_df,
        fee_rate=cfg.fee_rate,
        rebalance_mask=rebalance_mask,
        trade_capacity_df=trade_capacity_df,
        weight_at_100pct_adv_df=weight_at_100pct_adv_df,
        impact_bps_at_100pct_adv=float(cfg.impact_bps_at_100pct_adv),
        impact_exponent=impact_exponent,
        missing_ret_policy=missing_ret_policy,
        missing_ret_penalty=float(cfg.missing_ret_penalty),
        missing_ret_penalty_once=bool(cfg.missing_ret_penalty_once),
    )
    cost = fee_cost + impact_cost
    net_ret = gross_ret - cost
    nav = (1.0 + net_ret).cumprod()

    result = pd.DataFrame(
        {
            "gross_ret": gross_ret,
            "equity_turnover": equity_turnover,
            "cash_turnover": cash_turnover,
            "turnover": turnover,
            "fee_cost": fee_cost,
            "impact_cost": impact_cost,
            "cost": cost,
            "missing_ret_penalty_ret": missing_penalty_ret,
            "capacity_shortfall": capacity_shortfall,
            "base_exposure_signal": base_exposure_signal,
            "base_exposure": base_exposure,
            "vol_scale_signal": vol_scale_signal,
            "vol_scale": vol_scale,
            "exposure": exposure,
            "regime": regime_aligned["regime"],
            "regime_label": regime_aligned["regime_label"],
            "net_ret": net_ret,
            "nav": nav,
            "return_mode": cfg.return_mode,
            "weight_lag_days": weight_lag_days,
            "exposure_signal_lag_days": exposure_signal_lag_days,
            "vol_scale_lag_days": vol_scale_lag_days,
            "missing_ret_policy": missing_ret_policy,
            "max_participation_of_adv": cfg.max_participation_of_adv,
            "impact_bps_at_100pct_adv": float(cfg.impact_bps_at_100pct_adv),
        }
    )

    if result.empty:
        if return_exec_weights:
            return result, executed_equity_weights
        return result

    initial_date = result.index[0] - pd.Timedelta(days=1)
    initial_row = pd.DataFrame(
        {
            "gross_ret": [0.0],
            "equity_turnover": [0.0],
            "cash_turnover": [0.0],
            "turnover": [0.0],
            "fee_cost": [0.0],
            "impact_cost": [0.0],
            "cost": [0.0],
            "missing_ret_penalty_ret": [0.0],
            "capacity_shortfall": [0.0],
            "base_exposure_signal": [base_exposure_signal.iloc[0]],
            "base_exposure": [base_exposure.iloc[0]],
            "vol_scale_signal": [1.0],
            "vol_scale": [1.0],
            "exposure": [exposure.iloc[0]],
            "regime": [regime_aligned["regime"].iloc[0]],
            "regime_label": [regime_aligned["regime_label"].iloc[0]],
            "net_ret": [0.0],
            "nav": [1.0],
            "return_mode": [cfg.return_mode],
            "weight_lag_days": [weight_lag_days],
            "exposure_signal_lag_days": [exposure_signal_lag_days],
            "vol_scale_lag_days": [vol_scale_lag_days],
            "missing_ret_policy": [missing_ret_policy],
            "max_participation_of_adv": [cfg.max_participation_of_adv],
            "impact_bps_at_100pct_adv": [float(cfg.impact_bps_at_100pct_adv)],
        },
        index=[initial_date],
    )

    final_result = pd.concat([initial_row, result], axis=0)
    if return_exec_weights:
        return final_result, executed_equity_weights
    return final_result
