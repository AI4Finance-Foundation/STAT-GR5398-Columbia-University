"""E10: Early VIX stress protection added to E5."""

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
        exposure_value += 0.15


    # Reduce exposure during accelerating market stress
    vix = data.market("^VIX").dropna().tail(20)

    if len(vix) == 20:
        vix_fast = float(vix.iloc[-5:].mean())
        vix_slow = float(vix.mean())

        early_stress = (
            price < sma50
            and vix_fast >= 25.0
            and vix_fast >= 1.15 * vix_slow
        )

        if early_stress:
            exposure_value -= 0.25

    return float(min(1.0, max(0.10, exposure_value)))
