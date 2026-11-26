"""E8: Regime-aware dual-market timing strategy."""

def _market_signal(prices):
    prices = prices.dropna()

    if len(prices) < 200:
        return 1.0, False

    price = prices.iloc[-1]
    sma20 = prices.iloc[-20:].mean()
    sma50 = prices.iloc[-50:].mean()
    sma200 = prices.mean()

    # Original E2 trend exposure
    exposure_value = (
        0.4
        + 0.3 * (price > sma50)
        + 0.3 * (price > sma200)
    )

    # Short-term recovery confirmation
    short_recovery = (
        price > sma20
        and price > prices.iloc[-21]
    )

    # Original E5 recovery adjustment
    if price < sma200 and short_recovery:
        exposure_value += 0.15

    exposure_value = float(
        min(1.0, max(0.10, exposure_value))
    )

    return exposure_value, bool(short_recovery)


def exposure(as_of, data) -> float:
    prices = data.close(
        ["SPY", "QQQ"], lookback=200
    )

    spy_exposure, spy_recovery = _market_signal(
        prices["SPY"]
    )

    qqq_exposure, qqq_recovery = _market_signal(
        prices["QQQ"]
    )

    # Relax conservative exposure only when both
    # markets confirm positive short-term momentum
    if spy_recovery and qqq_recovery:
        exposure_value = (
            spy_exposure + qqq_exposure
        ) / 2.0
    else:
        exposure_value = min(
            spy_exposure, qqq_exposure
        )

    return float(
        min(1.0, max(0.10, exposure_value))
    )
