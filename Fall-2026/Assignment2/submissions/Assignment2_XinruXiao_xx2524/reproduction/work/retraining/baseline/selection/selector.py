"""
ML Stock Selector — per-bucket ML pipeline for stock ranking.

Migrated from FinRL-Trading's ml_bucket_selection.py.
Trains per-bucket models (6 base + stacking ensemble), predicts quarterly
returns, and provides ranked stock picks for each asset bucket.

Usage:
    selector = MLStockSelector(db_path="data/fundamentals.db")
    preds = selector.run_monthly(date(2026, 4, 1))
    picks = selector.get_top_picks("growth_tech", n=5)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector -> Bucket mapping (synced with FinRL group_selection_by_gics.py v1.2.2)
# ---------------------------------------------------------------------------
SECTOR_TO_BUCKET = {
    "information technology": "growth_tech",
    "technology": "growth_tech",
    "communication services": "growth_tech",
    "consumer discretionary": "cyclical",
    "consumer cyclical": "cyclical",
    "financials": "cyclical",
    "financial services": "cyclical",
    "industrials": "cyclical",
    "energy": "real_assets",
    "materials": "real_assets",
    "basic materials": "real_assets",
    "real estate": "real_assets",
    "health care": "defensive",
    "healthcare": "defensive",
    "consumer staples": "defensive",
    "consumer defensive": "defensive",
    "utilities": "defensive",
}

FEATURE_COLS = [
    # Valuation (5)
    "pe", "ps", "pb", "peg", "ev_multiple",
    # Profitability (4)
    "EPS", "roe", "gross_margin", "operating_margin",
    # Cash Flow (5)
    "fcf_per_share", "cash_per_share", "capex_per_share", "fcf_to_ocf", "ocf_ratio",
    # Leverage (3)
    "debt_ratio", "debt_to_equity", "debt_to_mktcap",
    # Liquidity (1)
    "cur_ratio",
    # Efficiency (3)
    "acc_rec_turnover", "asset_turnover", "payables_turnover",
    # Coverage (2)
    "interest_coverage", "debt_service_coverage",
    # Dividend (1)
    "dividend_yield",
    # Solvency (1)
    "solvency_ratio",
    # Per-Share (1)
    "BPS",
]

MOMENTUM_COLS = [
    "ret_1q", "ret_4q", "ret_accel",
    "eps_chg", "roe_chg", "gm_chg", "om_chg",
]

# datadate -> tradedate mapping (2-month lag: first day of next-next month)
DATADATE_TO_TRADEDATE_MAP = {
    "03-31": ("06-01", 0),
    "06-30": ("09-01", 0),
    "09-30": ("12-01", 0),
    "12-31": ("03-01", 1),
}


def datadate_to_tradedate(datadate_str: str) -> Optional[str]:
    """Map datadate to tradedate: 03-31->06-01, 06-30->09-01, etc."""
    if not isinstance(datadate_str, str) or len(datadate_str) < 10:
        return None
    mm_dd = datadate_str[5:]
    try:
        year = int(datadate_str[:4])
    except ValueError:
        return None
    mapping = DATADATE_TO_TRADEDATE_MAP.get(mm_dd)
    if mapping is None:
        return None
    target_mmdd, year_add = mapping
    return f"{year + year_add}-{target_mmdd}"


def _build_models() -> Dict[str, Any]:
    """Create the 6 base models (XGB/LGBM optional)."""
    models: Dict[str, Any] = {
        "RF": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "HistGBM": HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0),
    }
    try:
        from xgboost import XGBRegressor
        models["XGB"] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LGBM"] = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    except ImportError:
        pass
    return models


def _run_bucket(
    bucket: str,
    bdf: pd.DataFrame,
    feature_cols: List[str],
    val_cutoff: str = "2025-12-31",
    val_quarters: int = 3,
) -> tuple:
    """Train models for one bucket, return (predictions_df, model_results, feature_importances).

    Mirrors FinRL ml_bucket_selection.py run_bucket() logic:
    1. Split train/val/infer by date
    2. Fit 6 models + stacking (top 3)
    3. Retrain on train+val
    4. Predict on inference set
    5. Compute inverse-MSE weighted ensemble
    """
    # --- Split ---
    all_dates = sorted(bdf[bdf["y_return"].notna()]["datadate"].unique())
    val_end_idx = None
    for i, d in enumerate(all_dates):
        if str(d) <= val_cutoff:
            val_end_idx = i
    if val_end_idx is not None:
        val_start_idx = max(0, val_end_idx - val_quarters + 1)
        val_dates = set(all_dates[val_start_idx: val_end_idx + 1])
    else:
        val_dates = set()

    train_b = bdf[
        (~bdf["datadate"].isin(val_dates))
        & (bdf["datadate"] <= val_cutoff)
        & (bdf["y_return"].notna())
    ]
    val_b = bdf[
        (bdf["datadate"].isin(val_dates))
        & (bdf["y_return"].notna())
    ]
    infer_dates = sorted(bdf[bdf["datadate"] > val_cutoff]["datadate"].unique())
    infer_b = bdf[bdf["datadate"].isin(infer_dates)] if infer_dates else pd.DataFrame()

    logger.info(
        f"Bucket {bucket}: train={len(train_b)} val={len(val_b)} "
        f"({len(val_dates)}Q) infer={len(infer_b)}"
    )

    if len(train_b) < 20 or len(infer_b) == 0:
        logger.warning(f"Bucket {bucket}: insufficient data, skipping")
        return pd.DataFrame(), [], []

    X_train, y_train = train_b[feature_cols].values, train_b["y_return"].values
    X_val, y_val = val_b[feature_cols].values, val_b["y_return"].values
    X_infer = infer_b[feature_cols].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val) if len(val_b) > 0 else None
    X_infer_s = scaler.transform(X_infer)

    # --- Train base models ---
    models = _build_models()
    fitted: Dict[str, Any] = {}
    model_results: List[Dict[str, Any]] = []
    best_name, best_mse, best_model = None, float("inf"), None

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model
        if X_val_s is not None and len(X_val_s) > 0:
            mse = mean_squared_error(y_val, model.predict(X_val_s))
        else:
            mse = float("inf")
        model_results.append({
            "bucket": bucket, "model": name, "val_mse": round(mse, 6),
            "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
        })
        if mse < best_mse:
            best_name, best_mse, best_model = name, mse, model

    # --- Stacking (top 3) ---
    if X_val_s is not None and len(X_val_s) > 0:
        sorted_m = sorted(
            [(n, mean_squared_error(y_val, fitted[n].predict(X_val_s))) for n in fitted],
            key=lambda x: x[1],
        )
    else:
        sorted_m = [(n, 0) for n in fitted]
    top3 = [n for n, _ in sorted_m[:3]]
    stacking = StackingRegressor(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
    )
    stacking.fit(X_train_s, y_train)
    fitted["Stacking"] = stacking
    if X_val_s is not None and len(X_val_s) > 0:
        stack_mse = mean_squared_error(y_val, stacking.predict(X_val_s))
    else:
        stack_mse = float("inf")
    model_results.append({
        "bucket": bucket, "model": "Stacking", "val_mse": round(stack_mse, 6),
        "train_size": len(train_b), "val_size": len(val_b), "infer_size": len(infer_b),
    })
    if stack_mse < best_mse:
        best_name, best_mse, best_model = "Stacking", stack_mse, stacking

    logger.info(f"Bucket {bucket}: best={best_name} MSE={best_mse:.6f}")

    # --- Retrain on train+val ---
    full_train = pd.concat([train_b, val_b], ignore_index=True)
    X_full, y_full = full_train[feature_cols].values, full_train["y_return"].values
    scaler_full = StandardScaler()
    X_full_s = scaler_full.fit_transform(X_full)
    X_infer_s = scaler_full.transform(X_infer)

    for name, model in fitted.items():
        if name == "Stacking":
            continue
        model.fit(X_full_s, y_full)

    # Rebuild stacking with retrained base
    stacking = StackingRegressor(
        estimators=[(n, fitted[n]) for n in top3],
        final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1,
    )
    stacking.fit(X_full_s, y_full)
    fitted["Stacking"] = stacking
    if best_name == "Stacking":
        best_model = stacking

    # --- Feature importances (tree-based models only) ---
    feature_importances: List[Dict[str, Any]] = []
    for name, model in fitted.items():
        if name == "Stacking":
            continue
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            is_best = (name == best_name)
            for feat, importance in zip(feature_cols, imp):
                feature_importances.append({
                    "bucket": bucket,
                    "model": name,
                    "feature": feat,
                    "importance": round(float(importance), 6),
                    "is_best": is_best,
                })

    # --- Predict ---
    infer_b = infer_b.copy()
    infer_b["predicted_return"] = best_model.predict(X_infer_s)
    infer_b["best_model"] = best_name
    for n, m in fitted.items():
        infer_b[f"pred_{n}"] = m.predict(X_infer_s)

    # --- Inverse-MSE weighted ensemble ---
    mse_map = {r["model"]: r["val_mse"] for r in model_results}
    pred_cols = [c for c in infer_b.columns if c.startswith("pred_") and c != "pred_ensemble_avg"]
    weights: Dict[str, float] = {}
    for col in pred_cols:
        name = col.replace("pred_", "")
        mse = mse_map.get(name)
        weights[col] = (1.0 / mse) if mse and mse > 0 else 0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    infer_b["pred_ensemble_avg"] = sum(infer_b[col] * w for col, w in weights.items())

    # --- Rank within bucket ---
    infer_b["rank_best"] = infer_b.groupby("datadate")["predicted_return"].rank(ascending=False).astype(int)
    infer_b["rank_ensemble"] = infer_b.groupby("datadate")["pred_ensemble_avg"].rank(ascending=False).astype(int)

    infer_b = infer_b.sort_values(["datadate", "predicted_return"], ascending=[True, False])
    return infer_b, model_results, feature_importances


class MLStockSelector:
    """Monthly ML stock selection pipeline.

    Loads fundamental data from SQLite, engineers features, trains per-bucket
    ML models, and provides ranked stock picks.

    Supports three inference modes:
    - ``infer-date``: align all tickers to a single ``datadate`` quarter
      (e.g. 2025-12-31). Eliminates cross-quarter duplication in backtests.
    - ``mixed-vintage`` (default): each ticker uses its latest available
      filing.  Best for live trading where freshness matters.
    - ``latest-snapshot``: similar to mixed-vintage but truncated at
      ``as_of_date`` and tagged ``datadate="latest"``.

    Supports caching: each monthly run is saved to disk and can be reloaded
    without re-training.
    """

    VALID_INFERENCE_MODES = {"infer-date", "mixed-vintage", "latest-snapshot", "mixed-alpha", "global-unified"}

    DEFAULT_BUCKET_ALPHA = {
        "growth_tech": 0.0,
        "cyclical": 0.0,
        "real_assets": 1.0,
        "defensive": 0.0,
    }

    def __init__(
        self,
        db_path: str,
        cache_dir: str = "./data/ml_cache",
        val_cutoff: str = "2025-12-31",
        val_quarters: int = 3,
        universes: Optional[List[str]] = None,
        multi_universe_buckets: Optional[List[str]] = None,
        inference_mode: str = "mixed-vintage",
        bucket_alpha: Optional[Dict[str, float]] = None,
        api_key: str = "",
        auto_sync_missing: bool = False,
        read_only: bool = True,
        price_as_of: Optional[Callable[[Sequence[str], date], Dict[str, float]]] = None,
    ):
        from shared.store import FundamentalStore

        if inference_mode not in self.VALID_INFERENCE_MODES:
            raise ValueError(
                f"Invalid inference_mode '{inference_mode}'. "
                f"Must be one of {sorted(self.VALID_INFERENCE_MODES)}"
            )

        if read_only and auto_sync_missing:
            raise ValueError("Read-only selection cannot auto-sync data")
        self._store = FundamentalStore(db_path, read_only=True) if read_only else FundamentalStore(db_path)
        self._read_only = read_only
        self._price_as_of = price_as_of
        self._tradeable_cache = None
        self._cache_dir = cache_dir
        self._val_cutoff = val_cutoff
        self._val_quarters = val_quarters
        self._inference_mode = inference_mode
        self._bucket_alpha = bucket_alpha if bucket_alpha is not None else dict(self.DEFAULT_BUCKET_ALPHA)
        self._api_key = str(api_key or "").strip()
        self._auto_sync_missing = bool(auto_sync_missing)
        self._universes = universes or ["sp500"]
        self._multi_universe_buckets = multi_universe_buckets if multi_universe_buckets is not None else ["growth_tech"]

        # Current predictions (populated by run_monthly or load_cache)
        self._predictions: Optional[pd.DataFrame] = None
        self._model_results: List[Dict[str, Any]] = []
        self._feature_importances: List[Dict[str, Any]] = []
        self._run_date: Optional[str] = None
        # Per-universe membership (populated by run_monthly)
        self._universe_members: Dict[str, Set[str]] = {}
        self._reported_non_members: Set[str] = set()
        self.blacklist: Set[str] = set()
        self._selection_fingerprint = self.selection_fingerprint(
            inference_mode=self._inference_mode,
            universes=self._universes,
            multi_universe_buckets=self._multi_universe_buckets,
            val_cutoff=self._val_cutoff,
            val_quarters=self._val_quarters,
            bucket_alpha=self._bucket_alpha,
        )

    # ------------------------------------------------------------------
    # Cache identity
    # ------------------------------------------------------------------

    @staticmethod
    def selection_fingerprint(
        *,
        inference_mode: str,
        universes: Optional[List[str]] = None,
        multi_universe_buckets: Optional[List[str]] = None,
        val_cutoff: str = "2025-12-31",
        val_quarters: int = 3,
        bucket_alpha: Optional[Dict[str, float]] = None,
    ) -> str:
        """Digest of every input that changes what this selector predicts.

        ``cache_dir`` is derived from the fundamentals DB, which several
        sub-accounts on one host deliberately share. Without this digest in
        the file name the cache key was just mode+month, so the second
        sub-account to rebalance loaded the first one's rankings — built
        from a different universe or a different ``bucket_alpha`` — and
        traded them for the rest of the month.
        """
        alphas = bucket_alpha if bucket_alpha is not None else MLStockSelector.DEFAULT_BUCKET_ALPHA
        payload = json.dumps(
            {
                "inference_mode": str(inference_mode),
                "universes": sorted(str(u) for u in (universes or ["sp500"])),
                "multi_universe_buckets": sorted(
                    str(b) for b in (
                        multi_universe_buckets
                        if multi_universe_buckets is not None
                        else ["growth_tech"]
                    )
                ),
                "val_cutoff": str(val_cutoff),
                "val_quarters": int(val_quarters),
                "bucket_alpha": {str(k): float(v) for k, v in sorted(alphas.items())},
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def fingerprint_from_config(ml_cfg: Dict[str, Any]) -> str:
        """Fingerprint straight from an ``ml_selection`` config block.

        For callers that never build a selector (report writers). Defaults
        must stay in step with WeeklyMLStrategy's own parsing.
        """
        ml_cfg = dict(ml_cfg or {})
        return MLStockSelector.selection_fingerprint(
            inference_mode=str(ml_cfg.get("inference_mode", "mixed-vintage")),
            universes=list(ml_cfg.get("universes", ["sp500"])),
            multi_universe_buckets=list(
                ml_cfg.get("multi_universe_buckets", ["growth_tech"])
            ),
            val_cutoff=str(ml_cfg.get("val_cutoff", "2025-12-31")),
            val_quarters=int(ml_cfg.get("val_quarters", 3)),
            bucket_alpha=ml_cfg.get("bucket_alpha", None),
        )


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def run_monthly(
        self,
        as_of_date: date,
        infer_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run the ML pipeline.

        Loads fundamental data, engineers features, applies PIT filter,
        assigns buckets, assembles an inference set (mode-dependent),
        trains per-bucket models, and predicts forward returns.

        Args:
            as_of_date: The reference date for the monthly run.
            infer_date: (infer-date mode only) The single datadate to use
                for inference.  Auto-derived if *None*.

        Returns:
            DataFrame with predictions for all buckets.
        """
        # Auto-roll val_cutoff to one quarter before current infer quarter
        # to prevent information leakage in backtests
        effective_infer_date = None
        if self._inference_mode == "infer-date":
            effective_infer_date = infer_date or self._auto_infer_date(as_of_date)
        rolling_cutoff = self._auto_val_cutoff(
            effective_infer_date or self._auto_infer_date(as_of_date)
        )
        effective_val_cutoff = min(self._val_cutoff, rolling_cutoff)
        logger.info(
            f"ML run_monthly [{self._inference_mode}]: as_of={as_of_date} "
            f"val_cutoff={effective_val_cutoff} (config={self._val_cutoff}, rolling={rolling_cutoff})"
        )

        # Get members from each configured universe (keep per-universe for picking)
        as_of_str = as_of_date.isoformat()
        self._universe_members = {}
        members: Set[str] = set()
        for universe in self._universes:
            u_members = self._store.get_universe_members_at(universe, as_of_str, api_key=self._api_key)
            if u_members:
                logger.info(f"{universe} members: {len(u_members)}")
                self._universe_members[universe] = u_members
                members |= u_members
        if not members:
            if getattr(self, "_read_only", False):
                raise ValueError(f"No point-in-time model universe at {as_of_str}")
            # Fallback to the latest stored SP500 list
            members = self._store.get_latest_sp500_members()
            if members:
                self._universe_members["sp500"] = members
        logger.info(f"Combined universe members: {len(members)}")

        # Load all fundamental data
        # Point-in-time visibility is mandatory, not a mode: rows are cut at
        # what was filed and knowable by as-of.
        # Restated values are future information — training on them makes
        # backtests unreproducible against what live inference will see.
        df = self._store.get_fundamental_data(as_of_date=as_of_str)
        if df.empty:
            logger.error("No fundamental data found in cache")
            return pd.DataFrame()

        # Price momentum must not see past the decision date: a row whose
        # tradedate is still ahead of as-of carries a later close in history.
        if self._price_as_of is not None:
            df = self._clip_trade_prices_to_as_of(df, as_of_date, self._price_as_of)

        # --- Feature engineering ---
        df = self._engineer_features(df)

        # --- Point-in-time filter (for training data) ---
        df = self._apply_pit_filter(df)

        # --- Assign buckets ---
        df["bucket"] = df["gsector"].str.lower().map(SECTOR_TO_BUCKET)
        unmapped = df[df["bucket"].isna()]["gsector"].unique()
        if len(unmapped) > 0:
            logger.warning(f"Unmapped sectors: {unmapped}")
        df = df[df["bucket"].notna()].copy()

        # --- Assemble inference set (mode-dependent) ---
        train_val_part = df[df["datadate"] <= effective_val_cutoff].copy()
        infer_df = self._assemble_inference(df, as_of_date, members, effective_infer_date)

        if infer_df is None or infer_df.empty:
            logger.error(f"No inference rows after {self._inference_mode} assembly")
            return pd.DataFrame()

        df = pd.concat([train_val_part, infer_df], ignore_index=True)

        # --- One-hot encode sectors ---
        sector_dummies = pd.get_dummies(df["gsector"], prefix="sector")
        df = pd.concat([df, sector_dummies], axis=1)

        # --- Global-unified: single model on all stocks, skip per-bucket ---
        if self._inference_mode == "global-unified":
            all_sector_cols = sorted(
                c for c in df.columns
                if c.startswith("sector_") and df[c].sum() > 0
            )
            unified_features = FEATURE_COLS + MOMENTUM_COLS + all_sector_cols
            logger.info(
                "Global-unified: training single model on all stocks "
                "(%d fund + %d mom + %d sector = %d features)",
                len(FEATURE_COLS), len(MOMENTUM_COLS),
                len(all_sector_cols), len(unified_features),
            )

            preds, results, feat_imp = _run_bucket(
                "global_unified", df, unified_features,
                val_cutoff=effective_val_cutoff,
                val_quarters=self._val_quarters,
            )
            if preds.empty:
                logger.error("Global-unified model produced no predictions")
                return pd.DataFrame()

            # Map original bucket back for display/filtering
            bucket_map = df.drop_duplicates("tic").set_index("tic")["bucket"].to_dict()
            preds["bucket"] = preds["tic"].map(bucket_map).fillna("unknown")

            # Z-score normalize within each bucket so cross-bucket ranking is fair
            preds["pred_raw"] = preds["pred_ensemble_avg"]
            grp = preds.groupby("bucket")["pred_raw"]
            bucket_mean = grp.transform("mean")
            bucket_std = grp.transform("std")
            bucket_std = bucket_std.replace(0, 1)  # avoid div-by-zero
            preds["pred_ensemble_avg"] = (preds["pred_raw"] - bucket_mean) / bucket_std
            logger.info(
                "Global-unified z-score normalization applied — "
                "top raw vs top z-score: %s",
                list(zip(
                    preds.nlargest(5, "pred_raw")["tic"],
                    preds.nlargest(5, "pred_ensemble_avg")["tic"],
                )),
            )

            self._predictions = preds
            self._model_results = results
            self._feature_importances = feat_imp
            self._run_date = as_of_date.isoformat()

            logger.info(
                f"Global-unified pipeline complete: {len(self._predictions)} predictions"
            )
            return self._predictions

        # --- Run per-bucket ---
        all_preds = []
        all_results = []
        all_feat_imp = []
        buckets = ["growth_tech", "cyclical", "real_assets", "defensive"]

        for bucket in buckets:
            bdf = df[df["bucket"] == bucket].copy()
            if len(bdf) == 0:
                logger.warning(f"No data for bucket {bucket}")
                continue

            # Per-bucket features: fundamental + momentum + sector dummies within bucket
            bucket_sector_cols = sorted(
                c for c in bdf.columns
                if c.startswith("sector_") and bdf[c].sum() > 0
            )
            feature_cols = FEATURE_COLS + MOMENTUM_COLS + bucket_sector_cols

            preds, results, feat_imp = _run_bucket(
                bucket, bdf, feature_cols,
                val_cutoff=effective_val_cutoff,
                val_quarters=self._val_quarters,
            )
            if not preds.empty:
                preds["bucket"] = bucket
                all_preds.append(preds)
            all_results.extend(results)
            all_feat_imp.extend(feat_imp)

        if not all_preds:
            logger.error("No predictions generated for any bucket")
            return pd.DataFrame()

        self._predictions = pd.concat(all_preds, ignore_index=True)
        if self._inference_mode == "mixed-alpha":
            self._predictions = self._apply_mixed_alpha(
                df, self._predictions, effective_val_cutoff, all_results, all_feat_imp,
            )
        self._model_results = all_results
        self._feature_importances = all_feat_imp
        self._run_date = as_of_date.isoformat()

        logger.info(
            f"ML pipeline complete: {len(self._predictions)} predictions "
            f"across {self._predictions['bucket'].nunique()} buckets"
        )
        return self._predictions

    # ------------------------------------------------------------------
    # Mixed-alpha blending
    # ------------------------------------------------------------------

    def _apply_mixed_alpha(
        self,
        full_df: pd.DataFrame,
        bucket_preds: pd.DataFrame,
        val_cutoff: str,
        all_results: list,
        all_feat_imp: list,
    ) -> pd.DataFrame:
        """Blend unified and bucket-split ranks with per-bucket alpha."""
        logger.info("Mixed-alpha: training unified model on all stocks")
        for bucket, alpha in self._bucket_alpha.items():
            logger.info("  %s: alpha=%s", bucket, alpha)

        all_sector_cols = sorted(
            c for c in full_df.columns
            if c.startswith("sector_") and full_df[c].sum() > 0
        )
        unified_features = FEATURE_COLS + MOMENTUM_COLS + all_sector_cols
        logger.info(
            "  Unified features: %d fund + %d mom + %d sector = %d",
            len(FEATURE_COLS),
            len(MOMENTUM_COLS),
            len(all_sector_cols),
            len(unified_features),
        )

        unified_preds, unified_results, unified_imps = _run_bucket(
            "unified_all", full_df, unified_features,
            val_cutoff=val_cutoff,
            val_quarters=self._val_quarters,
        )
        if unified_preds.empty:
            logger.warning(
                "Mixed-alpha: unified model produced no predictions, falling back to bucket-split only"
            )
            return bucket_preds

        all_results.extend(unified_results)
        all_feat_imp.extend(unified_imps)

        unified_map = unified_preds.set_index("tic")["pred_ensemble_avg"].to_dict()
        bucket_preds = bucket_preds.copy()
        bucket_preds["unified_pred"] = bucket_preds["tic"].map(unified_map)

        bucket_preds["bucket_pctile"] = bucket_preds.groupby(
            ["bucket", "datadate"]
        )["pred_ensemble_avg"].rank(pct=True)
        bucket_preds["unified_pctile"] = bucket_preds.groupby(
            ["bucket", "datadate"]
        )["unified_pred"].rank(pct=True)
        bucket_preds["unified_pctile"] = bucket_preds["unified_pctile"].fillna(
            bucket_preds["bucket_pctile"]
        )

        bucket_preds["mixed_score"] = 0.0
        bucket_preds["applied_alpha"] = 0.0
        for bucket, alpha in self._bucket_alpha.items():
            mask = bucket_preds["bucket"] == bucket
            bucket_preds.loc[mask, "mixed_score"] = (
                alpha * bucket_preds.loc[mask, "unified_pctile"]
                + (1 - alpha) * bucket_preds.loc[mask, "bucket_pctile"]
            )
            bucket_preds.loc[mask, "applied_alpha"] = alpha

        unconfigured = bucket_preds["applied_alpha"].eq(0) & ~bucket_preds["bucket"].isin(self._bucket_alpha)
        if unconfigured.any():
            bucket_preds.loc[unconfigured, "mixed_score"] = bucket_preds.loc[unconfigured, "bucket_pctile"]

        bucket_preds["rank_mixed"] = bucket_preds.groupby(
            ["bucket", "datadate"]
        )["mixed_score"].rank(ascending=False).astype(int)
        bucket_preds["pred_ensemble_avg_bucket"] = bucket_preds["pred_ensemble_avg"]
        bucket_preds["pred_ensemble_avg"] = bucket_preds["mixed_score"]
        bucket_preds["rank_ensemble"] = bucket_preds["rank_mixed"]

        logger.info("Mixed-alpha blending complete: %d predictions", len(bucket_preds))
        return bucket_preds

    # ------------------------------------------------------------------
    # Inference set assembly (mode-dependent)
    # ------------------------------------------------------------------

    def _assemble_inference(
        self,
        df: pd.DataFrame,
        as_of_date: date,
        members: Set[str],
        infer_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Dispatch to the appropriate inference assembly method."""
        if self._inference_mode == "infer-date":
            return self._assemble_infer_date(df, infer_date or self._auto_infer_date(as_of_date))
        elif self._inference_mode == "latest-snapshot":
            return self._assemble_latest_snapshot(df, as_of_date, members)
        else:  # mixed-vintage, mixed-alpha, or global-unified
            return self._assemble_mixed_vintage(df, as_of_date, members)

    def _assemble_mixed_vintage(
        self,
        df: pd.DataFrame,
        as_of_date: date,
        members: Set[str],
    ) -> pd.DataFrame:
        """Mixed-vintage: each ticker uses its latest point-in-time filing."""
        infer_rows = []
        as_of_str = as_of_date.isoformat()
        available_mask = self._is_row_available_as_of(df, as_of_str)
        for tic in sorted(members):
            tic_df = df[(df["tic"] == tic) & available_mask]
            if len(tic_df) == 0:
                continue
            latest = tic_df.sort_values("datadate").iloc[-1].copy()
            latest["original_datadate"] = latest["datadate"]
            latest["datadate"] = "mixed"
            infer_rows.append(latest)

        if not infer_rows:
            return pd.DataFrame()

        infer_df = pd.DataFrame(infer_rows)
        logger.info(f"Mixed-vintage: {len(infer_df)} tickers for inference")
        return infer_df

    @staticmethod
    def _is_row_available_as_of(df: pd.DataFrame, as_of_str: str) -> pd.Series:
        """Return rows whose financials were visible by the as-of date."""
        available = pd.Series(False, index=df.index)
        for col in ("accepted_date", "filing_date"):
            if col not in df.columns:
                continue
            dates = pd.to_datetime(df[col], errors="coerce")
            available = available | dates.notna() & (dates <= pd.Timestamp(as_of_str))

        # Fallback for legacy rows without filing metadata: use only the
        # modeled tradable date. Never fall back to raw datadate because that
        # would treat quarter-end financials as immediately knowable.
        if "tradedate" in df.columns:
            trade_dates = pd.to_datetime(df["tradedate"], errors="coerce")
        else:
            trade_dates = df["datadate"].apply(datadate_to_tradedate)
            trade_dates = pd.to_datetime(trade_dates, errors="coerce")
        no_filing_metadata = ~available
        for col in ("accepted_date", "filing_date"):
            if col in df.columns:
                no_filing_metadata = no_filing_metadata & pd.to_datetime(df[col], errors="coerce").isna()
        return available | (no_filing_metadata & trade_dates.notna() & (trade_dates <= pd.Timestamp(as_of_str)))

    def _assemble_infer_date(
        self,
        df: pd.DataFrame,
        infer_date: str,
    ) -> pd.DataFrame:
        """Infer-date: align all tickers to a single datadate quarter.

        Only rows with ``datadate == infer_date`` are kept for inference.
        This ensures every ticker is evaluated on the same quarter,
        eliminating cross-quarter duplication in backtests.
        """
        infer_df = df[df["datadate"] == infer_date].copy()
        logger.info(
            f"Infer-date: {len(infer_df)} rows with datadate={infer_date}"
        )
        return infer_df

    def _assemble_latest_snapshot(
        self,
        df: pd.DataFrame,
        as_of_date: date,
        members: Set[str],
    ) -> pd.DataFrame:
        """Latest-snapshot: like mixed-vintage but truncated at as_of_date.

        Each universe member gets its latest filing up to as_of_date,
        tagged with ``datadate="latest"``.  Does *not* compute actual
        returns — that belongs in the evaluation stage.
        """
        infer_rows = []
        as_of_str = as_of_date.isoformat()
        for tic in sorted(members):
            tic_df = df[(df["tic"] == tic) & (df["datadate"] <= as_of_str)]
            if len(tic_df) == 0:
                continue
            latest = tic_df.sort_values("datadate").iloc[-1].copy()
            latest["original_datadate"] = latest["datadate"]
            latest["datadate"] = "latest"
            infer_rows.append(latest)

        if not infer_rows:
            return pd.DataFrame()

        infer_df = pd.DataFrame(infer_rows)
        logger.info(f"Latest-snapshot: {len(infer_df)} tickers for inference")
        return infer_df

    # ------------------------------------------------------------------
    # Date helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_infer_date(as_of_date: date) -> str:
        """Derive the single inference quarter from as_of_date.

        Aligned to tradedate boundaries (when filings become available):
        - Dec-Feb  → 09-30 (Q3 data, tradedate 12/1)
        - Mar-May  → 12-31 (Q4 data, tradedate 3/1)
        - Jun-Aug  → 03-31 (Q1 data, tradedate 6/1)
        - Sep-Nov  → 06-30 (Q2 data, tradedate 9/1)
        """
        m = as_of_date.month
        y = as_of_date.year
        if m in (12, 1, 2):
            ref_year = y if m == 12 else y - 1
            return f"{ref_year}-09-30"
        elif m <= 5:
            return f"{y - 1}-12-31"
        elif m <= 8:
            return f"{y}-03-31"
        else:
            return f"{y}-06-30"

    @staticmethod
    def _auto_val_cutoff(infer_date: str) -> str:
        """Return the quarter-end immediately before *infer_date*.

        This ensures the validation set ends one quarter before the
        inference quarter, so training data and model change with each
        new infer_date.

        2025-12-31 → 2025-09-30
        2025-09-30 → 2025-06-30
        2025-06-30 → 2025-03-31
        2025-03-31 → 2024-12-31
        """
        mm_dd = infer_date[5:]
        year = int(infer_date[:4])
        prev_map = {
            "03-31": (f"{year - 1}-12-31"),
            "06-30": (f"{year}-03-31"),
            "09-30": (f"{year}-06-30"),
            "12-31": (f"{year}-09-30"),
        }
        result = prev_map.get(mm_dd)
        if result is None:
            # Fallback: subtract ~90 days
            logger.warning(f"_auto_val_cutoff: unrecognized quarter-end '{infer_date}', using fallback")
            dt = date.fromisoformat(infer_date)
            from datetime import timedelta
            result = (dt - timedelta(days=92)).isoformat()
        return result

    def get_top_picks(
        self, bucket: str, n: int = 5, as_of: "Optional[date]" = None
    ) -> List[Dict[str, Any]]:
        """Get top N stock picks for a bucket.

        When multiple universes are configured, selects top N from each
        universe independently, then merges (union, deduplicated).
        The merged list is sorted by ensemble prediction descending.

        Args:
            bucket: One of 'growth_tech', 'cyclical', 'real_assets', 'defensive'.
            n: Number of picks per universe.

        Returns:
            List of dicts with 'symbol', 'predicted_return', 'rank_best',
            'pred_ensemble_avg', 'best_model'.
        """
        if self._predictions is None or self._predictions.empty:
            return []

        bdf = self._predictions[self._predictions["bucket"] == bucket].copy()
        if bdf.empty:
            return []

        bdf = self._restrict_to_universe(bdf, as_of=as_of)
        if bdf.empty:
            return []

        bdf = bdf.sort_values("pred_ensemble_avg", ascending=False)

        # Per-universe top-N then merge — only for configured buckets
        use_multi = (
            bucket in self._multi_universe_buckets
            and len(self._universe_members) > 1
        )
        if use_multi:
            seen: Set[str] = set()
            merged_rows = []
            for universe, u_members in self._universe_members.items():
                u_bdf = bdf[bdf["tic"].isin(u_members)]
                for _, row in u_bdf.head(n).iterrows():
                    tic = str(row["tic"])
                    if tic not in seen:
                        seen.add(tic)
                        merged_rows.append(row)
            # Re-sort merged by ensemble prediction
            if merged_rows:
                merged = pd.DataFrame(merged_rows).sort_values(
                    "pred_ensemble_avg", ascending=False
                )
            else:
                merged = bdf.head(n)
        else:
            # Single universe: global top N
            merged = bdf.head(n)

        picks = []
        for _, row in merged.iterrows():
            raw_score = float(row.get("pred_ensemble_avg_bucket", row.get("pred_ensemble_avg", 0)))
            picks.append({
                "symbol": str(row["tic"]),
                "predicted_return": float(row.get("predicted_return", 0)),
                "rank_best": int(row.get("rank_best", 999)),
                "pred_ensemble_avg": float(row.get("pred_ensemble_avg", 0)),
                "pred_raw_score": raw_score,
                "best_model": str(row.get("best_model", "")),
            })
        return picks

    def get_global_top_picks(
        self, n: int = 10, as_of: "Optional[date]" = None
    ) -> List[Dict[str, Any]]:
        """Get top N stock picks globally across all buckets.

        Sorts all predictions by ``pred_ensemble_avg`` descending and returns
        top N.  Used by ``global-unified`` inference mode.

        Args:
            n: Number of global picks.

        Returns:
            List of dicts with 'symbol', 'predicted_return', 'rank_best',
            'pred_ensemble_avg', 'pred_raw_score', 'best_model', 'bucket'.
        """
        if self._predictions is None or self._predictions.empty:
            return []

        sdf = self._restrict_to_universe(self._predictions, as_of=as_of)
        if sdf.empty:
            return []

        sdf = sdf.sort_values("pred_ensemble_avg", ascending=False)
        picks = []
        for _, row in sdf.head(n).iterrows():
            raw_score = float(row.get("pred_ensemble_avg_bucket", row.get("pred_ensemble_avg", 0)))
            picks.append({
                "symbol": str(row["tic"]),
                "predicted_return": float(row.get("predicted_return", 0)),
                "rank_best": int(row.get("rank_best", 999)),
                "pred_ensemble_avg": float(row.get("pred_ensemble_avg", 0)),
                "pred_raw_score": raw_score,
                "best_model": str(row.get("best_model", "")),
                "bucket": str(row.get("bucket", "unknown")),
            })
        return picks

    @property
    def _tradeable(self):
        """Shared point-in-time membership resolver (lazy: _universes is
        assigned later in __init__ than the store)."""
        if self._tradeable_cache is None:
            from shared.universe import TradeableUniverse

            self._tradeable_cache = TradeableUniverse(self._store, self._universes)
        return self._tradeable_cache

    def _restrict_to_universe(
        self, df: pd.DataFrame, as_of: "Optional[date]" = None
    ) -> pd.DataFrame:
        """Constrain picks to universe members as of the run's as-of date.

        Training and the predictions frame deliberately include departed
        symbols — their fundamentals and past membership are real history
        (survivorship-bias-free). But a pick is a trade instruction,
        and the tradeable set is membership AS OF the inference date:
        ``_universe_members`` is built point-in-time by run_monthly and
        load_cache, so a live run excludes a symbol delisted last quarter
        while a backtest replaying that quarter still ranks it. Without
        this, a dead symbol occupied a pick slot and shrank the portfolio
        (incident: DAY, delisted 2026-02-03, picked on 2026-08-04).

        Fail-open when membership is unavailable: on a store with no
        snapshots the previous unfiltered behaviour is kept rather than
        silently returning no picks at all.
        """
        if as_of is not None:
            # Daily granularity: the caller's decision date (sim time in a
            # backtest, today in live) resolves membership point-in-time via
            # the shared component, so a mid-month departure drops out of
            # picks the day after the next snapshot — not at month end.
            allowed = self._tradeable.members_at(as_of)
        elif self._universe_members:
            allowed = set().union(*self._universe_members.values())
        else:
            return df
        if not allowed:
            return df
        filtered = df[df["tic"].isin(allowed)]
        dropped = set(df["tic"]) - set(filtered["tic"])
        new_drops = dropped - self._reported_non_members
        if new_drops:
            self._reported_non_members |= new_drops
            logger.info(
                "ML picks: %d symbol(s) excluded as non-members at the "
                "inference date (departed/delisted): %s",
                len(new_drops),
                ", ".join(sorted(new_drops)[:10]),
            )
        return filtered

    def get_all_predictions(self) -> Optional[pd.DataFrame]:
        """Return the full predictions DataFrame."""
        return self._predictions

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _rebuild_universe_members(self, as_of_month: str) -> None:
        """Rebuild per-universe membership from configured universes."""
        # Use first day of month as lookup date
        as_of_date = f"{as_of_month}-01"
        self._universe_members = {}
        for universe in self._universes:
            members = self._store.get_universe_members_at(universe, as_of_date)
            if members:
                self._universe_members[universe] = members

    @staticmethod
    def cache_prefix(inference_mode: str, fingerprint: str) -> str:
        """Everything in a cache file name before the cache key.

        Shared with report builders that glob the cache dir, so they and
        the selector can never disagree about the naming.
        """
        pit_tag = "_pitfund"  # PIT is mandatory; non-PIT caches are orphaned on purpose
        stem = {
            "infer-date": "infer",
            "latest-snapshot": "snapshot",
            "mixed-alpha": "mixed_alpha",
            "global-unified": "global_unified",
        }.get(str(inference_mode), "pit")
        return f"ml_predictions_{stem}{pit_tag}_{fingerprint}_"

    def _cache_filename(self, cache_key: str) -> str:
        """Build the cache CSV filename based on inference mode.

        The selection fingerprint sits between the mode tag and the cache
        key so that configs which would predict differently never share a
        file — see ``selection_fingerprint``.

        Cache key semantics per mode:
        - infer-date:       cache_key = infer_date (e.g. '2025-12-31')
                            → ml_predictions_infer_pitfund_<fp>_2025-12-31.csv
        - mixed-vintage:    cache_key = YYYY-MM
                            → ml_predictions_pit_pitfund_<fp>_2026-01.csv
        - latest-snapshot:  cache_key = YYYY-MM
                            → ml_predictions_snapshot_pitfund_<fp>_2026-01.csv
        """
        return (
            self.cache_prefix(self._inference_mode, self._selection_fingerprint)
            + f"{cache_key}.csv"
        )

    def load_cache(self, cache_key: str) -> bool:
        """Load cached predictions.

        Args:
            cache_key: For mixed-vintage/latest-snapshot this is YYYY-MM.
                For infer-date this is the infer_date string (e.g. '2025-12-31').

        Returns True if cache was loaded successfully.
        """
        cache_path = Path(self._cache_dir) / self._cache_filename(cache_key)
        if not cache_path.exists():
            return False
        try:
            self._predictions = pd.read_csv(cache_path)
            self._run_date = cache_key
            # Rebuild universe membership for per-universe picking
            # For infer-date mode, derive a month string for membership lookup
            if self._inference_mode == "infer-date":
                # cache_key is a date like '2025-12-31'; use its YYYY-MM
                month_str = cache_key[:7]
            else:
                month_str = cache_key
            self._rebuild_universe_members(month_str)
            logger.info(f"Loaded ML cache: {cache_path} ({len(self._predictions)} rows)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load ML cache: {e}")
            return False

    @staticmethod
    def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
        """Write via a temp file in the same dir, then rename.

        Another trading process on this host may be reading the cache dir
        at any moment; a direct ``to_csv`` lets it read a half-written file.
        """
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            df.to_csv(tmp_path, index=False)
            os.replace(tmp_path, path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temp cache file %s", tmp_path, exc_info=True)

    def save_cache(self, cache_key: str) -> None:
        """Save current predictions to cache (CSV + optional Excel export)."""
        if self._predictions is None or self._predictions.empty:
            return
        cache_dir = Path(self._cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / self._cache_filename(cache_key)
        self._atomic_to_csv(self._predictions, cache_path)
        logger.info(f"Saved ML cache: {cache_path}")

        # Also save model results
        if self._model_results:
            results_path = cache_dir / cache_path.name.replace("predictions", "model_results")
            self._atomic_to_csv(pd.DataFrame(self._model_results), results_path)

        # Optional analyst-friendly export
        self.save_excel(cache_key)

        # Optional standalone HTML dashboard
        self.save_dashboard(cache_key)

    def save_excel(self, cache_key: str) -> Optional[str]:
        """Save predictions, model results, and feature importances to Excel.

        Produces a 3-sheet workbook (Rankings / Models / Features).

        Returns the path to the saved file, or None if nothing to save.
        """
        if self._predictions is None or self._predictions.empty:
            return None

        cache_dir = Path(self._cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        xlsx_name = self._cache_filename(cache_key).replace(".csv", ".xlsx")
        xlsx_path = cache_dir / xlsx_name

        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                # Rankings sheet
                self._predictions.to_excel(writer, sheet_name="Rankings", index=False)

                # Models sheet
                if self._model_results:
                    pd.DataFrame(self._model_results).to_excel(
                        writer, sheet_name="Models", index=False
                    )

                # Features sheet
                if self._feature_importances:
                    pd.DataFrame(self._feature_importances).to_excel(
                        writer, sheet_name="Features", index=False
                    )

            logger.info(f"Saved ML Excel: {xlsx_path}")
            return str(xlsx_path)
        except ImportError:
            logger.warning("openpyxl not installed — skipping Excel export")
            return None
        except Exception as e:
            logger.warning(f"Failed to save Excel: {e}")
            return None


    @staticmethod
    def _clip_trade_prices_to_as_of(
        df: pd.DataFrame,
        as_of_date: date,
        price_as_of: Callable[[Sequence[str], date], Dict[str, float]],
    ) -> pd.DataFrame:
        """Replace trade_price with the as-of close where its tradedate is later.

        ``trade_price`` is the close at a quarter's tradedate (quarter end + ~2
        months). Replaying history, that close is already stored even when the
        tradedate falls after the decision date, so ret_1q/ret_4q/ret_accel of
        the inference row would include returns realised after as-of. Live
        never sees this (the tradedate has not happened), which is the
        behaviour reproduced here. Rows with a known tradedate are untouched;
        a missing as-of close becomes NaN rather than keeping the future price.
        """
        if df.empty or "trade_price" not in df.columns:
            return df
        as_of_str = as_of_date.isoformat()
        tradedates = df["datadate"].astype(str).map(datadate_to_tradedate)
        future = tradedates.notna() & (tradedates > as_of_str)
        if not future.any():
            return df
        df = df.copy()
        symbols = sorted(df.loc[future, "tic"].astype(str).unique())
        closes = price_as_of(symbols, as_of_date) or {}
        replaced = df.loc[future, "tic"].astype(str).map(closes)
        df.loc[future, "trade_price"] = pd.to_numeric(replaced, errors="coerce").values
        missing = int(replaced.isna().sum())
        logger.info(
            "As-of trade_price: %d rows had a tradedate after %s; %d replaced with the as-of close, %d set missing",
            int(future.sum()), as_of_str, int(future.sum()) - missing, missing,
        )
        return df

    # ------------------------------------------------------------------
    # Feature Engineering (from FinRL ml_bucket_selection.py)
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum features, fill missing, winsorize."""
        df = df.sort_values(["tic", "datadate"]).copy()
        df["adj_close_q"] = pd.to_numeric(df["adj_close_q"], errors="coerce")
        df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")

        # Price momentum
        df["ret_1q"] = df.groupby("tic")["trade_price"].pct_change(1)
        df["ret_4q"] = df.groupby("tic")["trade_price"].pct_change(4)

        # Fundamental QoQ changes
        for src, dst in [("EPS", "eps_chg"), ("roe", "roe_chg"),
                         ("gross_margin", "gm_chg"), ("operating_margin", "om_chg")]:
            df[src] = pd.to_numeric(df[src], errors="coerce")
            df[dst] = df.groupby("tic")[src].diff()

        # Fill and clean
        pre_accel = [c for c in FEATURE_COLS + MOMENTUM_COLS if c != "ret_accel"]
        for c in pre_accel:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        global_medians = df[pre_accel].median()
        df[pre_accel] = df[pre_accel].fillna(global_medians).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Winsorize at 1st/99th percentile
        for c in pre_accel:
            p01, p99 = df[c].quantile(0.01), df[c].quantile(0.99)
            df[c] = df[c].clip(lower=p01, upper=p99)

        # Compute ret_accel from winsorized data
        df["ret_accel"] = df["ret_1q"] - df["ret_4q"] / 4
        df["ret_accel"] = df["ret_accel"].fillna(0)
        p01, p99 = df["ret_accel"].quantile(0.01), df["ret_accel"].quantile(0.99)
        df["ret_accel"] = df["ret_accel"].clip(lower=p01, upper=p99)

        return df

    def _apply_pit_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply point-in-time universe membership filter to training data.

        Uses all configured universes (sp500, nasdaq100, etc.) so that
        stocks belonging to ANY universe are retained.
        """
        if not self._universes:
            logger.warning("No configured universes, skipping PIT filter")
            return df

        def get_members_at(quarter_str: str) -> Set[str]:
            combined: Set[str] = set()
            for universe in self._universes:
                combined |= self._store.get_universe_members_at(
                    universe,
                    quarter_str,
                    api_key=self._api_key,
                )
            return combined

        synthetic_dates = {"latest", "mixed", "snapshot"}
        real_mask = ~df["datadate"].isin(synthetic_dates)

        # Map real datadates to tradedates
        df["_tradedate_pit"] = None
        df.loc[real_mask, "_tradedate_pit"] = df.loc[real_mask, "datadate"].apply(datadate_to_tradedate)

        pit_keep = ~real_mask  # always keep synthetic rows
        unique_tradedates = sorted(df.loc[real_mask, "_tradedate_pit"].dropna().unique())

        for td in unique_tradedates:
            universe = get_members_at(td)
            td_mask = df["_tradedate_pit"] == td
            pit_keep = pit_keep | (td_mask & df["tic"].isin(universe))

        # Drop unmappable
        unmappable = real_mask & df["_tradedate_pit"].isna()
        before = len(df)
        df = df[pit_keep].copy()
        df = df.drop(columns=["_tradedate_pit"], errors="ignore")

        logger.info(f"PIT filter: {before} -> {len(df)} records ({df['tic'].nunique()} tickers)")
        return df
