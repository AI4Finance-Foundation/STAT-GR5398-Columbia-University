"""E6: QQQ-based recovery-aware trend timing strategy."""

def exposure(as_of, data) -> float:
    qqq = data.close(["QQQ"], lookback=200)["QQQ"].dropna()

    if len(qqq) < 200:
        return 1.0

    price = qqq.iloc[-1]
    sma20 = qqq.iloc[-20:].mean()
    sma50 = qqq.iloc[-50:].mean()
    sma200 = qqq.mean()

    # Original E2 trend exposure, using QQQ
    exposure_value = (
        0.4
        + 0.3 * (price > sma50)
        + 0.3 * (price > sma200)
    )

    # E5 recovery confirmation, using QQQ
    recovery = (
        price < sma200
        and price > sma20
        and price > qqq.iloc[-21]
    )

    if recovery:
        exposure_value += 0.15

    return float(min(1.0, max(0.10, exposure_value)))
