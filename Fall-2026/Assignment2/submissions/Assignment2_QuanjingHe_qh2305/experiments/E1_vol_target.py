"""E1: Volatility-targeting timing strategy."""

def exposure(as_of, data) -> float:
    spy = data.close(["SPY"], lookback=21)["SPY"].dropna()

    if len(spy) < 21:
        return 1.0

    returns = spy.pct_change().dropna()
    volatility = returns.std() * (252 ** 0.5)

    if not volatility > 0:
        return 1.0

    target_vol = 0.16
    exposure_value = target_vol / volatility

    return float(min(1.0, max(0.10, exposure_value)))
