"""E9: VIX-aware bear-market exposure control."""

def exposure(as_of, data) -> float:
    spy = data.close(["SPY"], lookback=200)["SPY"].dropna()

    if len(spy) < 200:
        return 1.0

    price = spy.iloc[-1]
    sma20 = spy.iloc[-20:].mean()
    sma50 = spy.iloc[-50:].mean()
    sma200 = spy.mean()

    # Original E5 trend rule
    exposure_value = (
        0.4
        + 0.3 * (price > sma50)
        + 0.3 * (price > sma200)
    )

    recovery = (
        price < sma200
        and price > sma20
        and price > spy.iloc[-21]
    )

    if recovery:
        exposure_value += 0.15

    # Additional market-stress protection
    vix = data.market("^VIX").dropna().tail(5)

    if (
        len(vix) == 5
        and price < sma200
        and vix.mean() >= 25.0
    ):
        exposure_value = min(exposure_value, 0.55)

    return float(min(1.0, max(0.10, exposure_value)))
