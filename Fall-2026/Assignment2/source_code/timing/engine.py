"""
Deterministic Risk Parameters Calculator
timing/engine.py

Computes two execution-ready risk parameters from market data.
Core factors are pure math; Factor 6 (News Sentiment) optionally
calls OpenAI GPT for LLM-based headline analysis.

Outputs:
  exposureMultiplier     : Maximum allowed gross exposure [0.10, 1.0]
  maxDrawdownTolerance   : Trailing stop loss threshold  [0.05, 0.15]

Formula:
  Layer 1 — Volatility Scaling:
    exposure_vol = target_vol / realized_vol_annualized

  Layer 2 — Composite Risk Score (6 factors):
    risk_score = 0.22 × vix_score
               + 0.17 × yield_curve_score
               + 0.17 × hy_spread_score
               + 0.17 × momentum_score
               + 0.15 × breadth_score
               + 0.12 × news_sentiment_score
    risk_multiplier = 0.3 + 0.7 × risk_score  →  [0.3, 1.0]

  Final:
    exposureMultiplier   = clamp(exposure_vol × risk_multiplier, 0.10, 1.0)
    maxDrawdownTolerance = clamp(2 × daily_vol × √5,            0.05, 0.15)

Usage:
    from timing.engine import compute_risk_parameters

    result = compute_risk_parameters(technicals, fred_data)
    print(result.exposureMultiplier)       # e.g. 0.65
    print(result.maxDrawdownTolerance)     # e.g. 0.10
"""
from __future__ import annotations

import json
import math
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 11 GICS sector ETFs used for breadth calculation
SECTOR_ETFS = (
    "XLK", "XLE", "XLV", "XLF", "XLY",
    "XLI", "XLB", "XLU", "XLRE", "XLC", "XLP",
)

# Composite risk score factor weights (must sum to 1.0)
WEIGHT_VIX            = 0.22
WEIGHT_YIELD_CURVE    = 0.17
WEIGHT_HY_SPREAD      = 0.17
WEIGHT_MOMENTUM       = 0.17
WEIGHT_BREADTH        = 0.15
WEIGHT_NEWS_SENTIMENT = 0.12

# Default target annualized volatility
DEFAULT_TARGET_VOL = 0.16  # 16%


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RiskParametersResult:
    """
    Output of the deterministic risk parameter calculation.

    Fields consumed by the strategy:
        exposureMultiplier     → cap on total exposure (0.10–1.0)
        maxDrawdownTolerance   → informational (not used for stops)

    Diagnostic fields (for logging and audit):
        exposure_vol, risk_score, risk_multiplier, daily_volatility,
        components, raw_inputs
    """
    # ── Execution parameters ──────────────────────────────────────────────
    exposureMultiplier: float       # Final exposure 0.10–1.0
    maxDrawdownTolerance: float     # Trailing stop loss 0.05–0.15

    # ── Diagnostic fields ─────────────────────────────────────────────────
    exposure_vol: float             # target_vol / realized_vol
    risk_score: float               # Composite score 0–1
    risk_multiplier: float          # 0.3 + 0.7 × risk_score → [0.3, 1.0]
    daily_volatility: float         # Realized daily volatility (annualized / √252)
    components: dict[str, float] = field(default_factory=dict)   # Per-factor scores
    raw_inputs: dict[str, float] = field(default_factory=dict)   # Raw market values


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_risk_parameters(
    technicals: list[dict[str, Any]],
    fred_data: list[dict[str, Any]],
    target_vol: float = DEFAULT_TARGET_VOL,
    vix_percentile_override: float | None = None,
    news_sentiment_override: float | None = None,
) -> RiskParametersResult:
    """
    Compute deterministic risk parameters from market data.

    Args:
        technicals : List of technical indicator dicts.
                     Must contain '^GSPC 20D realized volatility', '^GSPC RSI(14)',
                     '^GSPC price vs SMA200', and '{ETF} price vs SMA200' for sector ETFs.
        fred_data  : List of FRED observation dicts.
                     Must contain 'T10Y2Y' and 'BAMLH0A0HYM2'.
        target_vol : Annualized target volatility (default 16%).

    Returns:
        RiskParametersResult with exposureMultiplier and maxDrawdownTolerance.
    """
    raw_inputs: dict[str, float] = {}

    # ── Layer 1: Volatility Scaling ───────────────────────────────────────
    realized_vol_pct = _extract_metric_value(technicals, "^GSPC 20D realized volatility")

    if realized_vol_pct is not None and realized_vol_pct > 0:
        realized_vol = realized_vol_pct / 100.0          # Convert % → decimal
        daily_vol    = realized_vol / math.sqrt(252)     # Annualized → daily
        exposure_vol = target_vol / realized_vol
        raw_inputs["realized_vol_annualized"] = realized_vol
        raw_inputs["realized_vol_daily"]      = daily_vol
    else:
        # Fallback: assume target vol is met → no scaling needed
        realized_vol = target_vol
        daily_vol    = target_vol / math.sqrt(252)
        exposure_vol = 1.0
        logger.warning("[risk_params] realized vol unavailable — defaulting exposure_vol=1.0")

    # maxDrawdownTolerance = 2 × daily_vol × √5
    # Rationale: 2-sigma move over 5 trading days (≈ 1 week expected drawdown)
    raw_stop = 2.0 * daily_vol * math.sqrt(5)
    stop_loss = max(0.05, min(0.15, raw_stop))
    raw_inputs["raw_stop_loss"] = raw_stop

    # ── Layer 2: Composite Risk Score ─────────────────────────────────────

    # Factor 1: VIX Percentile (25%)
    if vix_percentile_override is not None:
        vix_percentile = vix_percentile_override
    else:
        vix_percentile = _fetch_vix_percentile()
    vix_score = _score_vix(vix_percentile) if vix_percentile is not None else 0.5
    raw_inputs["vix_percentile"] = vix_percentile if vix_percentile is not None else -1.0

    # Factor 2: Yield Curve — T10Y2Y (20%)
    t10y2y = _extract_fred_value(fred_data, "T10Y2Y")
    yield_curve_score = _score_yield_curve(t10y2y) if t10y2y is not None else 0.5
    raw_inputs["t10y2y"] = t10y2y if t10y2y is not None else -999.0

    # Factor 3: High-Yield OAS — BAMLH0A0HYM2 (20%)
    hy_oas = _extract_fred_value(fred_data, "BAMLH0A0HYM2")
    hy_spread_score = _score_hy_spread(hy_oas) if hy_oas is not None else 0.5
    raw_inputs["hy_oas"] = hy_oas if hy_oas is not None else -1.0

    # Factor 4: S&P 500 Momentum — RSI + SMA200 (20%)
    rsi       = _extract_metric_value(technicals, "^GSPC RSI(14)")
    sma200_dist = _extract_metric_value(technicals, "^GSPC price vs SMA200")
    momentum_score = _score_momentum(rsi, sma200_dist)
    raw_inputs["sp500_rsi"]        = rsi        if rsi        is not None else -1.0
    raw_inputs["sp500_sma200_dist"] = sma200_dist if sma200_dist is not None else -999.0

    # Factor 5: Market Breadth — sector ETFs above SMA200 (15%)
    breadth_score = _score_breadth(technicals)
    raw_inputs["breadth_above_sma200"] = breadth_score

    # Factor 6: News Sentiment — LLM analysis of market headlines (12%)
    if news_sentiment_override is not None:
        news_sentiment_score = max(0.0, min(1.0, news_sentiment_override))
    else:
        news_sentiment_score = _fetch_news_sentiment()
    raw_inputs["news_sentiment"] = news_sentiment_score

    # ── Composite ─────────────────────────────────────────────────────────
    risk_score = (
        WEIGHT_VIX            * vix_score
        + WEIGHT_YIELD_CURVE    * yield_curve_score
        + WEIGHT_HY_SPREAD      * hy_spread_score
        + WEIGHT_MOMENTUM       * momentum_score
        + WEIGHT_BREADTH        * breadth_score
        + WEIGHT_NEWS_SENTIMENT * news_sentiment_score
    )
    risk_multiplier = 0.3 + 0.7 * risk_score   # Maps [0, 1] → [0.3, 1.0]

    # ── Final exposure ────────────────────────────────────────────────────
    raw_exposure = exposure_vol * risk_multiplier
    exposure = max(0.10, min(1.0, raw_exposure))

    components = {
        "vix_score":            round(vix_score, 4),
        "yield_curve_score":    round(yield_curve_score, 4),
        "hy_spread_score":      round(hy_spread_score, 4),
        "momentum_score":       round(momentum_score, 4),
        "breadth_score":        round(breadth_score, 4),
        "news_sentiment_score": round(news_sentiment_score, 4),
    }

    logger.info(
        "[risk_params] vol=%.2f risk=%.3f mult=%.3f → "
        "exposure=%.3f stop=%.3f  "
        "[VIX=%.2f YC=%.2f HY=%.2f Mom=%.2f Br=%.2f News=%.2f]",
        exposure_vol, risk_score, risk_multiplier,
        exposure, stop_loss,
        vix_score, yield_curve_score, hy_spread_score, momentum_score,
        breadth_score, news_sentiment_score,
    )

    return RiskParametersResult(
        exposureMultiplier   = round(exposure,  4),
        maxDrawdownTolerance = round(stop_loss, 4),
        exposure_vol         = round(exposure_vol,    4),
        risk_score           = round(risk_score,      4),
        risk_multiplier      = round(risk_multiplier, 4),
        daily_volatility     = round(daily_vol,       4),
        components           = components,
        raw_inputs           = {k: round(v, 4) for k, v in raw_inputs.items()},
    )


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_metric_value(technicals: list[dict[str, Any]], pattern: str) -> float | None:
    """Extract a numeric value from technicals by exact 'name' match."""
    for t in technicals:
        if t.get("name", "") == pattern:
            return _parse_numeric(t.get("value", ""))
    return None


def _extract_fred_value(fred_data: list[dict[str, Any]], series_id: str) -> float | None:
    """Extract the latest value for a FRED series."""
    for entry in fred_data:
        if entry.get("series_id") == series_id:
            return _parse_numeric(entry.get("value", ""))
    return None


def _parse_numeric(value: Any) -> float | None:
    """Parse strings like '+14.23%', '-2.1%', '56.3' to float."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.upper() == "N/A":
        return None
    s = s.replace("%", "").replace("x", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# VIX percentile fetch
# ---------------------------------------------------------------------------

def _fetch_vix_percentile() -> float | None:
    """
    Fetch VIX 1-year daily close history via FMP and return
    the percentile rank of the latest close.
    """
    try:
        import requests
        import pandas as pd

        fmp_api_key = os.environ.get("FMP_API_KEY", "")
        if not fmp_api_key:
            logger.warning("[risk_params] FMP_API_KEY not set — VIX percentile unavailable")
            return None

        url = "https://financialmodelingprep.com/stable/historical-price-eod/full"
        resp = requests.get(url, params={"symbol": "^VIX", "apikey": fmp_api_key}, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        records = data if isinstance(data, list) else (data or {}).get("historical", [])
        if not records:
            logger.warning("[risk_params] VIX history unavailable from FMP")
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        # Keep only the last ~252 trading days (1 year)
        close = df["close"].astype(float).tail(252)
        if close.empty:
            return None

        latest = float(close.iloc[-1])
        percentile = float((close < latest).sum()) / len(close)
        logger.debug("[risk_params] VIX=%.2f percentile=%.2f (n=%d)", latest, percentile, len(close))
        return percentile

    except Exception as exc:
        logger.warning("[risk_params] VIX percentile fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Factor scoring functions
# ---------------------------------------------------------------------------

def _score_vix(percentile: float) -> float:
    """
    High VIX percentile = high fear = low score.
    VIX at 90th percentile → score = 0.10
    VIX at 20th percentile → score = 0.80
    """
    return max(0.0, min(1.0, 1.0 - percentile))


def _score_yield_curve(t10y2y: float) -> float:
    """
    Continuous linear mapping: score = clamp(0.5 + T10Y2Y × 0.5, 0, 1)
    +1.0 → 1.0  (steep positive, expansion)
    +0.5 → 0.75 (moderate positive)
     0.0 → 0.50 (flat, caution)
    -0.5 → 0.25 (inverted, recession risk)
    -1.0 → 0.0  (deeply inverted)
    """
    return max(0.0, min(1.0, 0.5 + t10y2y * 0.5))


def _score_hy_spread(oas: float) -> float:
    """
    OAS < 3  → 1.0  (tight spreads, low credit stress)
    OAS 3~5  → linear interpolation
    OAS > 5  → 0.0  (wide spreads, credit stress)
    """
    if oas < 3.0:
        return 1.0
    if oas > 5.0:
        return 0.0
    # Linear: 3 → 1.0, 5 → 0.0  ⟹  score = 1.0 - (oas - 3) / 2
    return 1.0 - (oas - 3.0) / 2.0


def _score_momentum(rsi: float | None, sma200_dist: float | None) -> float:
    """
    Blend RSI and SMA200 distance for S&P 500 momentum.
    RSI component  : rsi / 100
    SMA200 component: maps [-5%, +5%] distance to [0, 1]
    """
    rsi_score = max(0.0, min(1.0, rsi / 100.0)) if rsi is not None else 0.5
    sma200_score = (
        max(0.0, min(1.0, 0.5 + sma200_dist / 10.0))
        if sma200_dist is not None else 0.5
    )
    return 0.5 * rsi_score + 0.5 * sma200_score


def _score_breadth(technicals: list[dict[str, Any]]) -> float:
    """
    Fraction of 11 GICS sector ETFs trading above their SMA200.
    0/11 → 0.0  (all sectors in downtrend)
    11/11 → 1.0 (all sectors in uptrend)
    """
    above_count = 0
    total = 0
    for etf in SECTOR_ETFS:
        dist = _extract_metric_value(technicals, f"{etf} price vs SMA200")
        if dist is not None:
            total += 1
            if dist > 0:
                above_count += 1
    if total == 0:
        return 0.5   # No data → neutral
    return above_count / total


# ---------------------------------------------------------------------------
# Factor 6: News Sentiment (LLM-based)
# ---------------------------------------------------------------------------

_NEWS_SENTIMENT_PROMPT = """\
You are a financial market sentiment analyst.
Analyze the following recent market news headlines and return a single
sentiment score between 0.0 and 1.0:
  0.0 = extremely bearish (crisis, crash, severe downturn)
  0.3 = moderately bearish
  0.5 = neutral / mixed signals
  0.7 = moderately bullish
  1.0 = extremely bullish (strong rally, broad optimism)

Consider overall market tone, not individual stocks.
Respond with ONLY a JSON object: {"score": <float>, "rationale": "<one sentence>"}

Headlines:
"""


def _fetch_news_sentiment() -> float:
    """
    Fetch recent market news via FMP, analyze with OpenAI GPT, return score [0, 1].
    Falls back to 0.5 (neutral) on any failure.
    """
    try:
        import requests

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        fmp_api_key = os.environ.get("FMP_API_KEY", "")

        if not openai_api_key:
            logger.warning("[risk_params] OPENAI_API_KEY not set — news sentiment unavailable")
            return 0.5
        if not fmp_api_key:
            logger.warning("[risk_params] FMP_API_KEY not set — news headlines unavailable")
            return 0.5

        # ── Step 1: Fetch recent general market news from FMP ──────────
        headlines = _fetch_fmp_headlines(fmp_api_key)
        if not headlines:
            logger.warning("[risk_params] no news headlines retrieved — defaulting to neutral")
            return 0.5

        # ── Step 2: Send to OpenAI for sentiment scoring ───────────────
        return _analyze_sentiment_openai(openai_api_key, headlines)

    except Exception as exc:
        logger.warning("[risk_params] news sentiment fetch failed: %s", exc)
        return 0.5


def _fetch_fmp_headlines(fmp_api_key: str, limit: int = 20) -> list[str]:
    """Fetch recent general market news headlines from FMP."""
    import requests

    try:
        url = "https://financialmodelingprep.com/stable/news/stock"
        resp = requests.get(
            url,
            params={"apikey": fmp_api_key, "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        articles = resp.json()

        if not isinstance(articles, list):
            return []

        headlines = []
        for article in articles:
            title = article.get("title", "").strip()
            if title:
                headlines.append(title)
        return headlines[:limit]

    except Exception as exc:
        logger.warning("[risk_params] FMP news fetch failed: %s", exc)
        return []


def _analyze_sentiment_openai(openai_api_key: str, headlines: list[str]) -> float:
    """Call OpenAI GPT to score market sentiment from headlines. Returns [0, 1]."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)

        headline_text = "\n".join(f"- {h}" for h in headlines)
        prompt = _NEWS_SENTIMENT_PROMPT + headline_text

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )

        content = response.choices[0].message.content.strip()
        # Parse JSON response
        parsed = json.loads(content)
        score = float(parsed["score"])
        rationale = parsed.get("rationale", "")
        score = max(0.0, min(1.0, score))

        logger.info(
            "[risk_params] news sentiment=%.2f (%d headlines) — %s",
            score, len(headlines), rationale,
        )
        return score

    except Exception as exc:
        logger.warning("[risk_params] OpenAI sentiment analysis failed: %s", exc)
        return 0.5


# ---------------------------------------------------------------------------
# Point-in-time computation from the bundled history
# ---------------------------------------------------------------------------

def exposure_from_history(market, macro, as_of, *, index_symbol: str = "SPY",
                          news_sentiment: float | None = 0.5) -> "RiskParametersResult | None":
    """Exposure multiplier at ``as_of`` from data/prices/market_series.parquet
    and data/macro/*.parquet, using only observations on or before ``as_of``.

    ``market``: DataFrame with columns symbol, date, close (index proxy, ^VIX,
    11 sector ETFs). ``macro``: {"T10Y2Y": Series, "BAMLH0A0HYM2": Series}
    indexed by date. The original pipeline used ^GSPC; SPY is the bundled
    proxy. Historical dates take a neutral news score (0.5), as the original
    did; pass ``news_sentiment=None`` to call the live FMP + OpenAI path.
    """
    import pandas as pd

    as_of = pd.Timestamp(as_of)
    m = market[market["date"] <= as_of]
    def series(sym):
        s = m[m["symbol"] == sym].set_index("date")["close"].sort_index()
        return s if len(s) else None

    idx = series(index_symbol)
    if idx is None or len(idx) < 200:
        return None
    rets = idx.pct_change().dropna()
    vol_20d = float(rets.tail(20).std() * (252 ** 0.5) * 100)
    delta = idx.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = 100.0 if loss == 0 else float(100 - 100 / (1 + gain / loss))
    sma200 = float(idx.rolling(200).mean().iloc[-1])
    technicals = [
        {"name": f"^GSPC 20D realized volatility", "value": f"{vol_20d:.2f}%"},
        {"name": f"^GSPC RSI(14)", "value": f"{rsi:.2f}"},
        {"name": f"^GSPC price vs SMA200", "value": f"{(float(idx.iloc[-1]) / sma200 - 1) * 100:+.2f}%"},
    ]
    for etf in SECTOR_ETFS:
        s = series(etf)
        if s is not None and len(s) >= 200:
            technicals.append({"name": f"{etf} price vs SMA200",
                               "value": f"{(float(s.iloc[-1]) / float(s.rolling(200).mean().iloc[-1]) - 1) * 100:+.2f}%"})
    vix = series("^VIX")
    vix_pct = None
    if vix is not None and len(vix) >= 50:
        last = vix.tail(252)
        vix_pct = float((last < float(last.iloc[-1])).sum()) / len(last)
    fred = []
    for sid in ("T10Y2Y", "BAMLH0A0HYM2"):
        s = macro.get(sid)
        if s is not None:
            s = s[s.index <= as_of].dropna()
            if len(s):
                fred.append({"series_id": sid, "value": str(float(s.iloc[-1]))})
    return compute_risk_parameters(technicals, fred, vix_percentile_override=vix_pct,
                                   news_sentiment_override=news_sentiment)
