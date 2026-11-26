"""E5: Recovery-aware trend timing strategy."""

def exposure(as_of, data) -> float:
    spy = data.close(["SPY"], lookback=200)["SPY"].dropna()

    if len(spy) < 200:
        return 1.0

    price = spy.iloc[-1]
    sma20 = spy.iloc[-20:].mean()
    sma50 = spy.iloc[-50:].mean()
    sma200 = spy.mean()

    # Original E2 trend exposure
    exposure_value = (
        0.4
        + 0.3 * (price > sma50)
        + 0.3 * (price > sma200)
    )

    # Early recovery confirmation
    recovery = (
        price < sma200
        and price > sma20
        and price > spy.iloc[-21]
    )

    if recovery:
        exposure_value += 0.10

    return float(min(1.0, max(0.10, exposure_value)))
