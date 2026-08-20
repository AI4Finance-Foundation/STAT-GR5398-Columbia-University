from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from strategy_framework.runner import load_cc_return_matrix


@dataclass
class DiagnosticsResult:
    backtest_dir: Path
    aligned_rows: int
    aligned_cols: int
    summary: pd.DataFrame
    lag_sensitivity: pd.DataFrame
    placebo_summary: pd.DataFrame
    random_topn_summary: pd.DataFrame
    cost_decay: pd.DataFrame
    oos_summary: pd.DataFrame
    flags: pd.DataFrame
    section_status: pd.DataFrame


def _perf_stats(daily_ret: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(daily_ret, errors="coerce").dropna()
    if x.empty:
        return {
            "ann_ret": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_dd": float("nan"),
            "total_ret": float("nan"),
        }
    ann_ret = float(x.mean() * 252)
    ann_vol = float(x.std(ddof=1) * np.sqrt(252))
    sharpe = float(ann_ret / (ann_vol + 1e-12))
    nav = (1.0 + x).cumprod()
    max_dd = float((nav / nav.cummax() - 1.0).min())
    total_ret = float(nav.iloc[-1] - 1.0)
    return {
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_ret": total_ret,
    }


def _resolve_existing(candidates: Iterable[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    checked = "\n".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Cannot find {label}. Checked paths:\n{checked}")


def _discover_assignment_roots() -> list[Path]:
    start = Path.cwd().resolve()
    raw_candidates: list[Path] = []
    for base in [start, *start.parents]:
        raw_candidates.extend(
            [
                base,
                base / "Assignment1_rayzhao_rz2759",
                base / "Assignment1" / "submissions" / "Assignment1_rayzhao_rz2759",
                base / "FinRL-Trading-Group" / "Assignment1_rayzhao_rz2759",
                base / "FinRL-Trading-Group" / "Assignment1" / "submissions" / "Assignment1_rayzhao_rz2759",
            ]
        )

    roots: list[Path] = []
    seen: set[str] = set()
    for p in raw_candidates:
        rp = p.resolve()
        key = str(rp).lower()
        if key in seen or not rp.exists():
            continue
        seen.add(key)
        if (rp / "strategy_framework").exists() or (rp / "nasdaq_stock.csv").exists():
            roots.append(rp)
    return roots


def _resolve_backtest_dir(strategy_output_dir: str | Path | None) -> Path:
    candidates: list[Path] = []
    if strategy_output_dir:
        candidates.append(Path(str(strategy_output_dir)))
    for root in _discover_assignment_roots():
        candidates.append(root / "backtest_result" / "eqw_topk20_reb20")
    candidates.append(Path("backtest_result/eqw_topk20_reb20"))
    bt = _resolve_existing(candidates, "backtest output directory")
    needed = [bt / "strategy_result.csv", bt / "portfolio_weights_exec.csv", bt / "portfolio_weights.csv"]
    for fp in needed:
        if not fp.exists():
            raise FileNotFoundError(f"Missing required file: {fp}")
    return bt


def _resolve_stock_price_csv(stock_price_csv: str | Path | None) -> Path:
    candidates: list[Path] = []
    if stock_price_csv:
        candidates.append(Path(str(stock_price_csv)))
    for root in _discover_assignment_roots():
        candidates.append(root / "nasdaq_stock.csv")
    candidates.append(Path("nasdaq_stock.csv"))
    return _resolve_existing(candidates, "nasdaq_stock.csv")


def _infer_linear_rate(numer: pd.Series, denom: pd.Series) -> float:
    if numer.empty or denom.empty:
        return 0.0
    num = pd.to_numeric(numer, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denom, errors="coerce").fillna(0.0)
    mask = den > 1e-12
    if not mask.any():
        return 0.0
    ratio = (num.loc[mask] / den.loc[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 0.0
    return float(ratio.clip(lower=0.0).median())


def run_diagnostics(
    strategy_output_dir: str | Path | None = None,
    stock_price_csv: str | Path | None = None,
    stock_selected_csv: str | Path | None = None,
    output_path_step2: str | Path | None = None,
    mc_runs: int = 100,
    lag_grid: tuple[int, ...] = (0, 1, 2, 5, 10),
    fee_grid: tuple[float, ...] = (0.001, 0.003, 0.005),
    oos_split_date: str = "2023-01-01",
    save_artifacts: bool = True,
) -> DiagnosticsResult:
    # Backward-compatible API:
    # stock_selected_csv/output_path_step2 are intentionally kept in the signature.
    # Current MC baseline uses tradable universe from return matrix by date.
    _ = stock_selected_csv, output_path_step2

    status_rows: list[dict[str, str]] = []

    def ok(section: str) -> None:
        status_rows.append({"section": section, "status": "OK"})

    def fail(section: str, err: Exception) -> None:
        status_rows.append({"section": section, "status": f"ERROR: {type(err).__name__}: {err}"})

    # 0) Load and align
    try:
        backtest_dir = _resolve_backtest_dir(strategy_output_dir)
        result_df = pd.read_csv(backtest_dir / "strategy_result.csv", index_col=0, parse_dates=True)
        weights_exec_df = pd.read_csv(backtest_dir / "portfolio_weights_exec.csv", index_col=0, parse_dates=True)
        weights_signal_df = pd.read_csv(backtest_dir / "portfolio_weights.csv", index_col=0, parse_dates=True)
        px_path = _resolve_stock_price_csv(stock_price_csv)
        ret_df = load_cc_return_matrix(str(px_path))
        common_idx = result_df.index.intersection(weights_exec_df.index).intersection(ret_df.index)
        cols = weights_exec_df.columns.intersection(ret_df.columns)
        res = result_df.loc[common_idx].copy()
        w_exec = weights_exec_df.loc[common_idx, cols].fillna(0.0)
        w_sig = weights_signal_df.loc[common_idx, cols].fillna(0.0)
        ret_raw = ret_df.loc[common_idx, cols]
        ret = ret_raw.fillna(0.0)
        actual_gross = (w_exec * ret).sum(axis=1)
        turn_series = res["turnover"].fillna(0.0) if "turnover" in res.columns else pd.Series(0.0, index=res.index)
        fee_series = res["fee_cost"].fillna(0.0) if "fee_cost" in res.columns else pd.Series(0.0, index=res.index)
        impact_series = res["impact_cost"].fillna(0.0) if "impact_cost" in res.columns else pd.Series(0.0, index=res.index)
        inferred_fee_rate = _infer_linear_rate(fee_series, turn_series)
        inferred_impact_rate = _infer_linear_rate(impact_series, turn_series)
        inferred_total_cost_rate = inferred_fee_rate + inferred_impact_rate
        ok("0. Data Load")
    except Exception as e:
        fail("0. Data Load", e)
        empty = pd.DataFrame()
        return DiagnosticsResult(
            backtest_dir=Path("."),
            aligned_rows=0,
            aligned_cols=0,
            summary=empty,
            lag_sensitivity=empty,
            placebo_summary=empty,
            random_topn_summary=empty,
            cost_decay=empty,
            oos_summary=empty,
            flags=empty,
            section_status=pd.DataFrame(status_rows),
        )

    # 1) Sanity metrics
    summary_items: dict[str, float | str] = {}
    try:
        nav_err = (res["nav"].pct_change().fillna(0.0) - res["net_ret"].fillna(0.0)).abs()
        gross_err = (actual_gross - res["gross_ret"]).abs()
        w_sum = w_exec.sum(axis=1)
        summary_items.update(
            {
                "date_start": str(common_idx.min().date()),
                "date_end": str(common_idx.max().date()),
                "rows": int(len(common_idx)),
                "cols": int(len(cols)),
                "nav_consistency_max_err": float(nav_err.max()),
                "gross_consistency_max_err": float(gross_err.max()),
                "any_negative_weights": bool((w_exec < -1e-10).any().any()),
                "days_weight_sum_gt_1": int((w_sum > 1.0 + 1e-10).sum()),
                "weight_sum_mean": float(w_sum.mean()),
                "turnover_mean": float(res["turnover"].mean()) if "turnover" in res.columns else float("nan"),
                "exposure_mean": float(res["exposure"].mean()) if "exposure" in res.columns else float("nan"),
                "inferred_fee_rate": float(inferred_fee_rate),
                "inferred_impact_rate": float(inferred_impact_rate),
                "inferred_total_cost_rate": float(inferred_total_cost_rate),
                "return_mode": str(res["return_mode"].iloc[-1]) if "return_mode" in res.columns else "unknown",
                "weight_lag_days": int(pd.to_numeric(res["weight_lag_days"], errors="coerce").dropna().iloc[-1])
                if "weight_lag_days" in res.columns and pd.to_numeric(res["weight_lag_days"], errors="coerce").dropna().size
                else -1,
                "exposure_signal_lag_days": int(
                    pd.to_numeric(res["exposure_signal_lag_days"], errors="coerce").dropna().iloc[-1]
                )
                if "exposure_signal_lag_days" in res.columns
                and pd.to_numeric(res["exposure_signal_lag_days"], errors="coerce").dropna().size
                else -1,
                "vol_scale_lag_days": int(pd.to_numeric(res["vol_scale_lag_days"], errors="coerce").dropna().iloc[-1])
                if "vol_scale_lag_days" in res.columns
                and pd.to_numeric(res["vol_scale_lag_days"], errors="coerce").dropna().size
                else -1,
            }
        )
        ok("1. Sanity")
    except Exception as e:
        fail("1. Sanity", e)

    # 2) Permutation placebo
    placebo_summary = pd.DataFrame()
    try:
        rng = np.random.default_rng(0)
        w_placebo = w_exec.copy()
        col_list = list(w_placebo.columns)
        for dt in w_placebo.index:
            perm = rng.permutation(col_list)
            w_placebo.loc[dt, :] = w_placebo.loc[dt, perm].values
        gross_placebo = (w_placebo * ret).sum(axis=1)
        actual_stats = _perf_stats(actual_gross)
        placebo_stats = _perf_stats(gross_placebo)
        placebo_summary = pd.DataFrame(
            [
                {"scenario": "actual", **actual_stats},
                {"scenario": "permutation_placebo", **placebo_stats},
            ]
        )
        ok("2. Placebo Permutation")
    except Exception as e:
        fail("2. Placebo Permutation", e)

    # 3) Lag sensitivity
    lag_df = pd.DataFrame()
    try:
        lag_rows = []
        for extra_lag in lag_grid:
            w_lag = w_exec.shift(extra_lag).ffill().fillna(0.0)
            gross = (w_lag * ret).sum(axis=1)
            turn_lag = w_lag.diff().abs().sum(axis=1).fillna(0.0)
            net = gross - inferred_total_cost_rate * turn_lag
            net_stats = _perf_stats(net)
            gross_stats = _perf_stats(gross)
            lag_rows.append(
                {
                    "extra_lag_days": int(extra_lag),
                    **net_stats,
                    "ann_ret_gross": gross_stats["ann_ret"],
                    "sharpe_gross": gross_stats["sharpe"],
                    "turnover_mean": float(turn_lag.mean()),
                    "cost_rate_used": float(inferred_total_cost_rate),
                }
            )
        lag_df = pd.DataFrame(lag_rows)
        ok("3. Lag Sensitivity")
    except Exception as e:
        fail("3. Lag Sensitivity", e)

    # 4) Random Top-N MC placebo
    mc_summary = pd.DataFrame()
    try:
        rebalance_mask = (w_sig.diff().abs().sum(axis=1).fillna(0.0) > 1e-12)
        if not rebalance_mask.empty:
            rebalance_mask.iloc[0] = True
        reb_idx = np.flatnonzero(rebalance_mask.values)
        n_hold_by_reb = (w_sig.iloc[reb_idx] > 0).sum(axis=1).astype(int).values

        ret_np = ret.to_numpy(dtype=float)
        ret_raw_np = ret_raw.to_numpy(dtype=float)
        exp_vec = (
            res["exposure"].ffill().fillna(0.0).to_numpy(dtype=float)
            if "exposure" in res.columns
            else np.ones(len(common_idx))
        )
        actual_net = res["net_ret"].fillna(0.0) if "net_ret" in res.columns else actual_gross
        actual_stats = _perf_stats(actual_net)

        T, N = ret_np.shape
        runs = max(20, int(mc_runs))
        rng_mc = np.random.default_rng(42)
        mc_ann = np.empty(runs, dtype=float)
        mc_sh = np.empty(runs, dtype=float)
        pool_sizes = []

        tradable_pool_by_t0: dict[int, np.ndarray] = {}
        for t0 in reb_idx:
            avail = np.flatnonzero(np.isfinite(ret_raw_np[t0, :]))
            if avail.size == 0:
                avail = np.arange(N, dtype=int)
            tradable_pool_by_t0[int(t0)] = avail
            pool_sizes.append(int(avail.size))

        for m in range(runs):
            W = np.zeros((T, N), dtype=float)
            prev_w = np.zeros(N, dtype=float)
            for j, t0 in enumerate(reb_idx):
                t1 = reb_idx[j + 1] if (j + 1) < len(reb_idx) else T
                pool = tradable_pool_by_t0.get(int(t0), np.arange(N, dtype=int))
                k = int(n_hold_by_reb[j]) if j < len(n_hold_by_reb) else 20
                if k <= 0:
                    k = 20
                k_eff = min(k, int(pool.size))
                picks = rng_mc.choice(pool, size=k_eff, replace=False)
                w = np.zeros(N, dtype=float)
                w[picks] = 1.0 / float(k_eff)
                prev_w = w
                W[t0:t1, :] = prev_w

            W_exec = np.vstack([np.zeros((1, N), dtype=float), W[:-1, :]]) * exp_vec[:, None]
            gross_mc = pd.Series((W_exec * ret_np).sum(axis=1), index=common_idx)
            turn_mc = pd.Series(np.abs(np.diff(W_exec, axis=0)).sum(axis=1), index=common_idx[1:]).reindex(common_idx, fill_value=0.0)
            net_mc = gross_mc - inferred_total_cost_rate * turn_mc
            st = _perf_stats(net_mc)
            mc_ann[m] = st["ann_ret"]
            mc_sh[m] = st["sharpe"]

        p_ann = float((np.sum(mc_ann >= actual_stats["ann_ret"]) + 1) / (runs + 1))
        p_sh = float((np.sum(mc_sh >= actual_stats["sharpe"]) + 1) / (runs + 1))
        mc_summary = pd.DataFrame(
            [
                {
                    "pool_mode": "tradable_universe_by_date",
                    "mc_runs": runs,
                    "actual_ann_ret": actual_stats["ann_ret"],
                    "actual_sharpe": actual_stats["sharpe"],
                    "random_ann_ret_mean": float(mc_ann.mean()),
                    "random_sharpe_mean": float(mc_sh.mean()),
                    "p_ann_ge_actual": p_ann,
                    "p_sh_ge_actual": p_sh,
                    "ann_percentile": float((mc_ann <= actual_stats["ann_ret"]).mean()),
                    "sh_percentile": float((mc_sh <= actual_stats["sharpe"]).mean()),
                    "avg_pool_size": float(np.mean(pool_sizes)) if pool_sizes else float("nan"),
                    "cost_rate_used": float(inferred_total_cost_rate),
                }
            ]
        )
        ok("4. Random Top-N MC")
    except Exception as e:
        fail("4. Random Top-N MC", e)

    # 5) Cost decay
    cost_df = pd.DataFrame()
    try:
        turn = res["turnover"].fillna(0.0) if "turnover" in res.columns else pd.Series(0.0, index=res.index)
        gross = res["gross_ret"].fillna(0.0)
        trade_mask = turn > 1e-12
        rows = []
        for fee in fee_grid:
            total_cost = (fee + inferred_impact_rate) * turn
            net = gross - total_cost
            st = _perf_stats(net)
            avg_trade_gross = float(gross.loc[trade_mask].mean()) if trade_mask.any() else float("nan")
            avg_trade_cost = float(total_cost.loc[trade_mask].mean()) if trade_mask.any() else float("nan")
            rows.append(
                {
                    "fee_rate": float(fee),
                    **st,
                    "avg_trade_gross": avg_trade_gross,
                    "avg_trade_cost": avg_trade_cost,
                    "edge_per_trade": float(avg_trade_gross - avg_trade_cost) if trade_mask.any() else float("nan"),
                    "impact_rate_used": float(inferred_impact_rate),
                    "total_cost_rate_used": float(fee + inferred_impact_rate),
                }
            )
        cost_df = pd.DataFrame(rows)
        ok("5. Cost Decay")
    except Exception as e:
        fail("5. Cost Decay", e)

    # 6) OOS decay proxy
    oos_df = pd.DataFrame()
    try:
        net = res["net_ret"].fillna(0.0)
        split_dt = pd.to_datetime(oos_split_date).normalize()
        if (split_dt <= common_idx.min()) or (split_dt >= common_idx.max()):
            split_dt = pd.to_datetime(common_idx[int(len(common_idx) * 0.6)]).normalize()
        is_stats = _perf_stats(net.loc[net.index < split_dt])
        oos_stats = _perf_stats(net.loc[net.index >= split_dt])
        sharpe_decay = (
            1.0 - (oos_stats["sharpe"] / is_stats["sharpe"])
            if np.isfinite(is_stats["sharpe"]) and is_stats["sharpe"] != 0
            else float("nan")
        )
        sharpe_ratio = (
            (oos_stats["sharpe"] / is_stats["sharpe"])
            if np.isfinite(is_stats["sharpe"]) and is_stats["sharpe"] != 0
            else float("nan")
        )
        oos_df = pd.DataFrame(
            [
                {"segment": "IS", "split_date": str(split_dt.date()), **is_stats},
                {"segment": "OOS", "split_date": str(split_dt.date()), **oos_stats},
                {
                    "segment": "DECAY",
                    "split_date": str(split_dt.date()),
                    "ann_ret": np.nan,
                    "ann_vol": np.nan,
                    "sharpe": sharpe_decay,
                    "max_dd": np.nan,
                    "total_ret": np.nan,
                },
                {
                    "segment": "RATIO",
                    "split_date": str(split_dt.date()),
                    "ann_ret": np.nan,
                    "ann_vol": np.nan,
                    "sharpe": sharpe_ratio,
                    "max_dd": np.nan,
                    "total_ret": np.nan,
                },
            ]
        )
        ok("6. OOS Decay")
    except Exception as e:
        fail("6. OOS Decay", e)

    # 7) Risk flags
    flag_rows: list[dict[str, str]] = []
    try:
        nav_ok = summary_items.get("nav_consistency_max_err", 1.0) <= 1e-10
        gross_ok = summary_items.get("gross_consistency_max_err", 1.0) <= 1e-10
        constraints_ok = (not bool(summary_items.get("any_negative_weights", True))) and int(summary_items.get("days_weight_sum_gt_1", 1)) == 0

        flag_rows.append({"check": "accounting_identity", "status": "PASS" if (nav_ok and gross_ok) else "WARN"})
        flag_rows.append({"check": "weight_constraints", "status": "PASS" if constraints_ok else "WARN"})
        lag_cfg_ok = (
            int(summary_items.get("weight_lag_days", -1)) >= 1
            and int(summary_items.get("exposure_signal_lag_days", -1)) >= 1
            and int(summary_items.get("vol_scale_lag_days", -1)) >= 1
        )
        flag_rows.append({"check": "lag_config_min1", "status": "PASS" if lag_cfg_ok else "WARN"})

        if not lag_df.empty:
            base_sh = float(lag_df.loc[lag_df["extra_lag_days"] == 0, "sharpe"].iloc[0])
            lag1_sh = float(lag_df.loc[lag_df["extra_lag_days"] == 1, "sharpe"].iloc[0])
            lag_flag = "PASS"
            if np.isfinite(base_sh) and base_sh > 0 and (lag1_sh / base_sh) < 0.5:
                lag_flag = "WARN"
            flag_rows.append({"check": "lag_sensitivity", "status": lag_flag})

        if not mc_summary.empty and pd.notna(mc_summary["p_sh_ge_actual"].iloc[0]):
            mc_flag = "PASS" if float(mc_summary["p_sh_ge_actual"].iloc[0]) <= 0.05 else "WARN"
            flag_rows.append({"check": "ranking_alpha_significance", "status": mc_flag})

        if not cost_df.empty:
            row30 = cost_df.loc[np.isclose(cost_df["fee_rate"], 0.003)]
            if not row30.empty:
                robust_30 = (
                    float(row30["ann_ret"].iloc[0]) > 0
                    and float(row30["sharpe"].iloc[0]) > 0
                    and float(row30["edge_per_trade"].iloc[0]) > 0
                )
                flag_rows.append({"check": "cost_robust_30bps", "status": "PASS" if robust_30 else "WARN"})

        if not oos_df.empty:
            decay_row = oos_df.loc[oos_df["segment"] == "DECAY"]
            if not decay_row.empty and pd.notna(decay_row["sharpe"].iloc[0]):
                decay_val = float(decay_row["sharpe"].iloc[0])
                oos_flag = "WARN" if (decay_val > 0.5 or decay_val < -0.5) else "PASS"
                flag_rows.append({"check": "oos_sharpe_decay", "status": oos_flag})
        ok("7. Flags")
    except Exception as e:
        fail("7. Flags", e)

    summary_df = pd.DataFrame([summary_items]) if summary_items else pd.DataFrame()
    flags_df = pd.DataFrame(flag_rows)
    section_status_df = pd.DataFrame(status_rows)

    result = DiagnosticsResult(
        backtest_dir=backtest_dir,
        aligned_rows=int(len(common_idx)),
        aligned_cols=int(len(cols)),
        summary=summary_df,
        lag_sensitivity=lag_df,
        placebo_summary=placebo_summary,
        random_topn_summary=mc_summary,
        cost_decay=cost_df,
        oos_summary=oos_df,
        flags=flags_df,
        section_status=section_status_df,
    )

    if save_artifacts:
        diag_dir = backtest_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        if not result.summary.empty:
            result.summary.to_csv(diag_dir / "summary.csv", index=False)
        if not result.lag_sensitivity.empty:
            result.lag_sensitivity.to_csv(diag_dir / "lag_sensitivity.csv", index=False)
        if not result.placebo_summary.empty:
            result.placebo_summary.to_csv(diag_dir / "placebo_summary.csv", index=False)
        if not result.random_topn_summary.empty:
            result.random_topn_summary.to_csv(diag_dir / "random_topn_monte_carlo.csv", index=False)
        if not result.cost_decay.empty:
            result.cost_decay.to_csv(diag_dir / "cost_decay.csv", index=False)
        if not result.oos_summary.empty:
            result.oos_summary.to_csv(diag_dir / "oos_summary.csv", index=False)
        if not result.flags.empty:
            result.flags.to_csv(diag_dir / "flags.csv", index=False)
        result.section_status.to_csv(diag_dir / "section_status.csv", index=False)

    return result


def print_diagnostics_report(result: DiagnosticsResult) -> None:
    print("=" * 80)
    print("Backtest Diagnostics Report")
    print("=" * 80)
    print(f"Backtest dir: {result.backtest_dir}")
    print(f"Aligned shape: rows={result.aligned_rows}, cols={result.aligned_cols}")

    if not result.summary.empty:
        print("\n[Summary]")
        print(result.summary.to_string(index=False))
    if not result.flags.empty:
        print("\n[Flags]")
        print(result.flags.to_string(index=False))
    if not result.placebo_summary.empty:
        print("\n[Permutation Placebo]")
        print(result.placebo_summary.to_string(index=False))
    if not result.random_topn_summary.empty:
        print("\n[Random Top-N Monte Carlo]")
        print(result.random_topn_summary.to_string(index=False))
    if not result.lag_sensitivity.empty:
        print("\n[Lag Sensitivity]")
        print(result.lag_sensitivity.to_string(index=False))
    if not result.cost_decay.empty:
        print("\n[Cost Decay]")
        print(result.cost_decay.to_string(index=False))
    if not result.oos_summary.empty:
        print("\n[OOS]")
        print(result.oos_summary.to_string(index=False))

    print("\n[Section Status]")
    print(result.section_status.to_string(index=False))
