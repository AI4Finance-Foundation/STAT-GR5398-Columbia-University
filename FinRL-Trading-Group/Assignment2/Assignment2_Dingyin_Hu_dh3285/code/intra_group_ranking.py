"""
Intra-Group Asset Ranking
==========================

Ranks assets within each active group based on residual momentum.

Key Concepts:
- Residual Momentum = Asset Return - Group Return
- Robust Z-score normalization using MAD
- Top-N selection per group

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .utils.robust_stats import robust_zscore
from .config_loader import AdaptiveRotationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
A1_GROWTH_POOL_PATH = PROJECT_ROOT / "data" / "a1_growth_pool.csv"

STATIC_GROWTH_TECH = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "TSLA"]

def _load_a1_growth_pool():
    if not A1_GROWTH_POOL_PATH.exists():
        print(f"[WARN] A1 growth pool not found: {A1_GROWTH_POOL_PATH}")
        return pd.DataFrame(columns=["tic", "predicted_return", "trade_date"])

    df = pd.read_csv(A1_GROWTH_POOL_PATH)
    if "trade_date" not in df.columns or "tic" not in df.columns:
        print("[WARN] A1 growth pool missing required columns.")
        return pd.DataFrame(columns=["tic", "predicted_return", "trade_date"])

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
    df = df.sort_values(["trade_date", "predicted_return"], ascending=[True, False])
    return df

A1_GROWTH_DF = _load_a1_growth_pool()

def get_dynamic_growth_members(decision_date, fallback=None):
    if fallback is None:
        fallback = STATIC_GROWTH_TECH

    if A1_GROWTH_DF.empty:
        return fallback

    dt = pd.to_datetime(decision_date).normalize()
    eligible = A1_GROWTH_DF[A1_GROWTH_DF["trade_date"] <= dt]

    if eligible.empty:
        return fallback

    latest_dt = eligible["trade_date"].max()
    members = (
        eligible.loc[eligible["trade_date"] == latest_dt, "tic"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(members) < 3:
        return fallback

    return members

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class AssetScore:
    """Asset ranking result within a group"""
    symbol: str
    zscore: float              # Robust Z-score of residual momentum
    residual_momentum: float   # Recent residual return
    rank: int                  # Rank within group (1=best)
    is_valid: bool = True      # Whether sufficient data exists
    
    # Raw metrics for debugging
    asset_return: float = 0.0
    group_return: float = 0.0
    lookback_periods: int = 0


@dataclass
class GroupRankingResult:
    """Asset ranking result for a single group"""
    group_name: str
    asset_scores: Dict[str, AssetScore]  # symbol → score
    ranked_assets: List[str]             # Assets sorted by Z-score (best first)
    top_n_assets: List[str]              # Top N assets selected
    as_of_date: pd.Timestamp
    
    def get_asset_score(self, symbol: str) -> Optional[AssetScore]:
        """Get score for a specific asset"""
        return self.asset_scores.get(symbol)
    
    def get_top_n(self, n: int) -> List[str]:
        """Get top N assets"""
        return self.ranked_assets[:n]


# ============================================================================
# Residual Momentum Computation
# ============================================================================

def compute_residual_returns(
    asset_returns: pd.Series,
    group_returns: pd.Series,
) -> pd.Series:
    """
    Compute residual returns (asset - group)
    
    Args:
        asset_returns: Asset return series
        group_returns: Group return series
    
    Returns:
        Series of residual returns
    
    Note:
        Returns are aligned by index. Missing dates are handled by
        using only common dates.
    """
    # Align indices
    common_dates = asset_returns.index.intersection(group_returns.index)
    
    if len(common_dates) == 0:
        return pd.Series(dtype=float)
    
    # Compute residual
    residual = (
        asset_returns.loc[common_dates] - 
        group_returns.loc[common_dates]
    )
    
    return residual


def compute_residual_momentum(
    asset_returns: pd.Series,
    group_returns: pd.Series,
    lookback_periods: Optional[int] = None,
) -> float:
    """
    Compute recent residual momentum (cumulative residual return)
    
    Args:
        asset_returns: Asset return series
        group_returns: Group return series
        lookback_periods: Lookback window (if None, use all data)
    
    Returns:
        Cumulative residual return over lookback period
    
    Examples:
        >>> asset_ret = pd.Series([0.01, 0.02, 0.015])
        >>> group_ret = pd.Series([0.008, 0.015, 0.012])
        >>> momentum = compute_residual_momentum(asset_ret, group_ret)
        >>> print(f"Residual momentum: {momentum:.4f}")
    """
    # Compute residual returns
    residual = compute_residual_returns(asset_returns, group_returns)
    
    if len(residual) == 0:
        return 0.0
    
    # Apply lookback
    if lookback_periods is not None:
        residual = residual.tail(lookback_periods)
    
    # Cumulative return
    # (1 + r1) * (1 + r2) * ... - 1
    cum_residual = (1 + residual).prod() - 1
    
    return float(cum_residual)


# ============================================================================
# Asset Ranking
# ============================================================================

def compute_asset_score(
    asset_returns: pd.Series,
    group_returns: pd.Series,
    symbol: str,
    lookback_periods: int,
    robust: bool = True,
) -> AssetScore:
    """
    Compute ranking score for a single asset
    
    Args:
        asset_returns: Asset return series
        group_returns: Group return series
        symbol: Asset symbol
        lookback_periods: Lookback window
        robust: Whether to use robust Z-score
    
    Returns:
        AssetScore object
    
    Examples:
        >>> score = compute_asset_score(
        ...     asset_returns=aapl_returns,
        ...     group_returns=tech_group_returns,
        ...     symbol="AAPL",
        ...     lookback_periods=12
        ... )
        >>> print(f"AAPL Z-score: {score.zscore:.2f}")
    """
    # Apply lookback to both series
    asset_ret_window = asset_returns.tail(lookback_periods)
    group_ret_window = group_returns.tail(lookback_periods)
    
    # Align
    common_dates = asset_ret_window.index.intersection(group_ret_window.index)
    
    # Check validity
    if len(common_dates) < lookback_periods * 0.8:  # Require 80% data
        return AssetScore(
            symbol=symbol,
            zscore=0.0,
            residual_momentum=0.0,
            rank=999,  # Invalid rank
            is_valid=False,
            lookback_periods=len(common_dates),
        )
    
    # Get aligned data
    asset_ret_aligned = asset_ret_window.loc[common_dates]
    group_ret_aligned = group_ret_window.loc[common_dates]
    
    # Compute residual returns
    residual_returns = asset_ret_aligned - group_ret_aligned
    
    # Compute Z-score of residual returns
    # Use fixed rolling window instead of full history to:
    # 1. Prevent extreme Z-scores from near-constant MAD
    # 2. Focus on recent relative performance
    # 3. Align with Exception framework lookback (12 weeks)
    ZSCORE_WINDOW = 12  # Fixed 12-week window for Z-score calculation
    
    if robust:
        z_scores = robust_zscore(
            residual_returns,
            window=min(ZSCORE_WINDOW, len(residual_returns)),
            center_metric="median"
        )
    else:
        # Standard Z-score
        mean = residual_returns.mean()
        std = residual_returns.std()
        if std == 0:
            z_scores = pd.Series(0.0, index=residual_returns.index)
        else:
            z_scores = (residual_returns - mean) / std
    
    # Use most recent Z-score
    current_zscore = z_scores.iloc[-1] if len(z_scores) > 0 else 0.0
    
    # Cap Z-score to reasonable range as additional safety
    MAX_ZSCORE = 20.0
    current_zscore = np.clip(current_zscore, -MAX_ZSCORE, MAX_ZSCORE)
    
    # Compute cumulative residual momentum
    residual_momentum = compute_residual_momentum(
        asset_ret_aligned,
        group_ret_aligned,
        lookback_periods=len(common_dates)
    )
    
    # Compute cumulative returns for debugging
    asset_cum_ret = (1 + asset_ret_aligned).prod() - 1
    group_cum_ret = (1 + group_ret_aligned).prod() - 1
    
    return AssetScore(
        symbol=symbol,
        zscore=float(current_zscore) if not pd.isna(current_zscore) else 0.0,
        residual_momentum=float(residual_momentum),
        rank=0,  # Will be assigned later
        is_valid=True,
        asset_return=float(asset_cum_ret),
        group_return=float(group_cum_ret),
        lookback_periods=len(common_dates),
    )


def rank_assets_in_group(
    asset_returns_dict: Dict[str, pd.Series],
    group_returns: pd.Series,
    group_members: List[str],
    lookback_periods: int,
    top_n: int = 3,
    robust: bool = True,
) -> Tuple[Dict[str, AssetScore], List[str]]:
    """
    Rank all assets in a group
    
    Args:
        asset_returns_dict: Dict mapping symbol → return series
        group_returns: Group return series
        group_members: List of symbols in this group
        lookback_periods: Lookback window
        top_n: Number of assets to select
        robust: Whether to use robust Z-score
    
    Returns:
        Tuple of (scores_dict, ranked_symbols)
    
    Examples:
        >>> scores, ranked = rank_assets_in_group(
        ...     returns_dict,
        ...     tech_returns,
        ...     ["AAPL", "MSFT", "NVDA"],
        ...     lookback_periods=12,
        ...     top_n=2
        ... )
        >>> print(f"Top 2: {ranked[:2]}")
    """
    # Compute scores for all members
    scores = {}
    
    for symbol in group_members:
        if symbol not in asset_returns_dict:
            # Create invalid score for missing data
            scores[symbol] = AssetScore(
                symbol=symbol,
                zscore=0.0,
                residual_momentum=0.0,
                rank=999,
                is_valid=False,
            )
            continue
        
        asset_returns = asset_returns_dict[symbol]
        
        score = compute_asset_score(
            asset_returns,
            group_returns,
            symbol,
            lookback_periods,
            robust,
        )
        
        scores[symbol] = score
    
    # Filter to valid scores
    valid_scores = {
        sym: score for sym, score in scores.items()
        if score.is_valid
    }
    
    if not valid_scores:
        return scores, []
    
    # Sort by Z-score (descending)
    sorted_items = sorted(
        valid_scores.items(),
        key=lambda x: x[1].zscore,
        reverse=True
    )
    
    # Assign ranks
    for rank, (symbol, score) in enumerate(sorted_items, start=1):
        scores[symbol].rank = rank
    
    # Get ranked list
    ranked_symbols = [sym for sym, _ in sorted_items]
    
    return scores, ranked_symbols


# ============================================================================
# High-Level API
# ============================================================================

class IntraGroupRanker:
    """
    Ranks assets within groups based on residual momentum
    
    Examples:
        >>> ranker = IntraGroupRanker(lookback_weeks=12)
        >>> 
        >>> result = ranker.rank_group(
        ...     asset_returns_dict=returns_dict,
        ...     group_returns=tech_returns,
        ...     group_name="group_a_growth_tech",
        ...     group_members=["AAPL", "MSFT", "NVDA"],
        ...     as_of_date=pd.Timestamp("2024-02-01"),
        ...     top_n=2
        ... )
        >>> 
        >>> print(f"Top 2 assets: {result.top_n_assets}")
    """
    
    def __init__(
        self,
        lookback_weeks: int = 12,
        robust: bool = True,
    ):
        """
        Initialize ranker
        
        Args:
            lookback_weeks: Lookback window for Z-score
            robust: Whether to use robust Z-score (MAD-based)
        """
        self.lookback_weeks = lookback_weeks
        self.robust = robust
    
    def rank_group(
        self,
        asset_returns_dict: Dict[str, pd.Series],
        group_returns: pd.Series,
        group_name: str,
        group_members: List[str],
        as_of_date: pd.Timestamp,
        top_n: int = 3,
    ) -> GroupRankingResult:
        """
        Rank assets within a single group
        
        Args:
            asset_returns_dict: Dict mapping symbol → return series
            group_returns: Group return series
            group_name: Name of the group
            group_members: List of symbols in this group
            as_of_date: Current decision date
            top_n: Number of assets to select
        
        Returns:
            GroupRankingResult object
        """
        # Replace fixed Growth Tech pool with dynamic A1-selected pool
        if group_name == "group_a_growth_tech":
            group_members = get_dynamic_growth_members(
                as_of_date,
                fallback=STATIC_GROWTH_TECH
            )

        # Rank assets
        scores, ranked = rank_assets_in_group(
            asset_returns_dict,
            group_returns,
            group_members,
            self.lookback_weeks,
            top_n,
            self.robust,
        )
        
        # Select top N
        top_n_assets = ranked[:top_n]
        
        return GroupRankingResult(
            group_name=group_name,
            asset_scores=scores,
            ranked_assets=ranked,
            top_n_assets=top_n_assets,
            as_of_date=as_of_date,
        )
    
    def rank_multiple_groups(
        self,
        asset_returns_dict: Dict[str, pd.Series],
        group_returns_dict: Dict[str, pd.Series],
        group_members_dict: Dict[str, List[str]],
        active_groups: List[str],
        as_of_date: pd.Timestamp,
        top_n: int = 3,
    ) -> Dict[str, GroupRankingResult]:
        """
        Rank assets in multiple groups
        
        Args:
            asset_returns_dict: Dict mapping symbol → return series
            group_returns_dict: Dict mapping group_name → return series
            group_members_dict: Dict mapping group_name → list of symbols
            active_groups: List of groups to rank
            as_of_date: Current decision date
            top_n: Number of assets to select per group
        
        Returns:
            Dict mapping group_name → GroupRankingResult
        
        Examples:
            >>> results = ranker.rank_multiple_groups(
            ...     asset_returns_dict=returns,
            ...     group_returns_dict=group_rets,
            ...     group_members_dict=config.get_group_symbols(),
            ...     active_groups=["group_a", "group_b"],
            ...     as_of_date=date,
            ...     top_n=2
            ... )
        """
        results = {}
        
        for group_name in active_groups:
            if group_name not in group_returns_dict:
                continue
            
            if group_name not in group_members_dict:
                continue
            
            result = self.rank_group(
                asset_returns_dict,
                group_returns_dict[group_name],
                group_name,
                group_members_dict[group_name],
                as_of_date,
                top_n,
            )
            
            results[group_name] = result
        
        return results


if __name__ == "__main__":
    """Quick test of intra-group ranking"""
    
    print("Testing Intra-Group Ranking")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="W-FRI")
    
    np.random.seed(42)
    
    # Strong performer
    aapl_returns = pd.Series(
        np.random.randn(len(dates)) * 0.02 + 0.003,  # 0.3% mean
        index=dates
    )
    
    # Moderate performer
    msft_returns = pd.Series(
        np.random.randn(len(dates)) * 0.02 + 0.002,  # 0.2% mean
        index=dates
    )
    
    # Weak performer
    nvda_returns = pd.Series(
        np.random.randn(len(dates)) * 0.02 + 0.001,  # 0.1% mean
        index=dates
    )
    
    # Group average
    group_returns = pd.Series(
        (aapl_returns + msft_returns + nvda_returns) / 3,
        index=dates
    )
    
    # Create returns dict
    returns_dict = {
        "AAPL": aapl_returns,
        "MSFT": msft_returns,
        "NVDA": nvda_returns,
    }
    
    # Test ranking
    ranker = IntraGroupRanker(lookback_weeks=12)
    
    test_date = pd.Timestamp("2024-06-30")
    
    # Get data as of date
    returns_as_of = {
        sym: series[series.index <= test_date]
        for sym, series in returns_dict.items()
    }
    group_ret_as_of = group_returns[group_returns.index <= test_date]
    
    result = ranker.rank_group(
        asset_returns_dict=returns_as_of,
        group_returns=group_ret_as_of,
        group_name="test_tech",
        group_members=["AAPL", "MSFT", "NVDA"],
        as_of_date=test_date,
        top_n=2,
    )
    
    print(f"\n[Test Date: {test_date.date()}]")
    print(f"\nAsset Rankings:")
    for symbol in result.ranked_assets:
        score = result.asset_scores[symbol]
        print(f"  {score.rank}. {symbol}")
        print(f"     Z-score: {score.zscore:.3f}")
        print(f"     Residual Momentum: {score.residual_momentum:.2%}")
        print(f"     Asset Return: {score.asset_return:.2%}")
        print(f"     Group Return: {score.group_return:.2%}")
    
    print(f"\nTop {len(result.top_n_assets)} Selected:")
    for sym in result.top_n_assets:
        print(f"  - {sym}")
    
    print(f"\n{'='*60}")
    print("[PASS] Intra-group ranking test complete!")
