"""E2: Dual-moving-average trend timing strategy."""

def exposure(as_of, data) -> float:
    spy = data.close(["SPY"], lookback=200)["SPY"].dropna()

    if len(spy) < 200:
        return 1.0

    price = spy.iloc[-1]
    sma50 = spy.iloc[-50:].mean()
    sma200 = spy.mean()

    exposure_value = (
        0.4
        + 0.3 * (price > sma50)
        + 0.3 * (price > sma200)
    )

    return float(exposure_value)
