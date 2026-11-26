"""E3: Trend-aware volatility targeting strategy."""

def exposure(as_of, data) -> float:
    spy = data.close(["SPY"], lookback=200)["SPY"].dropna()

    if len(spy) < 200:
        return 1.0

    # Volatility targeting
    returns = spy.iloc[-21:].pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5)

    vol_multiplier = (
        min(1.0, 0.16 / volatility)
        if volatility > 0 else 1.0
    )

    # Trend control
    price = spy.iloc[-1]
    sma50 = spy.iloc[-50:].mean()
    sma200 = spy.mean()

    trend_multiplier = (
        0.4
        + 0.3 * (price > sma50)
        + 0.3 * (price > sma200)
    )

    exposure_value = vol_multiplier * trend_multiplier

    return float(min(1.0, max(0.10, exposure_value)))
