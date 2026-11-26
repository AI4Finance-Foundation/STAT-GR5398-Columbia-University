"""E7: Conservative dual-market consensus timing."""

def _market_exposure(prices):
    prices = prices.dropna()

    if len(prices) < 200:
        return 1.0

    price = prices.iloc[-1]
    sma20 = prices.iloc[-20:].mean()
    sma50 = prices.iloc[-50:].mean()
    sma200 = prices.mean()

    exposure_value = (
        0.4
        + 0.3 * (price > sma50)
        + 0.3 * (price > sma200)
    )

    recovery = (
        price < sma200
        and price > sma20
        and price > prices.iloc[-21]
    )

    if recovery:
        exposure_value += 0.15

    return float(min(1.0, max(0.10, exposure_value)))


def exposure(as_of, data) -> float:
    prices = data.close(["SPY", "QQQ"], lookback=200)

    spy_exposure = _market_exposure(prices["SPY"])
    qqq_exposure = _market_exposure(prices["QQQ"])

    return float(min(spy_exposure, qqq_exposure))
