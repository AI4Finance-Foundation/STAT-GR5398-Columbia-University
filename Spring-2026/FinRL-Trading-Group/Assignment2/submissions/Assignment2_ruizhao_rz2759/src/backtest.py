import pandas as pd
def weights_to_returns(ret,w,tcost):
    weights=w.reindex(ret.index,method="ffill").fillna(0); exe=weights.shift(1).fillna(0)
    gross=(exe*ret.reindex(columns=exe.columns).fillna(0)).sum(axis=1)
    turn=weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1)); r=gross-turn*float(tcost)/10000
    return r.rename("return"),weights,float(turn[turn>0].mean()) if (turn>0).any() else 0.0
def benchmark_returns(ret,ticker): return ret[ticker].fillna(0).rename(ticker.lower())
def equal_weight_returns(ret,tickers): return ret.reindex(columns=[t for t in tickers if t in ret]).mean(axis=1).fillna(0).rename("equal_weight_optional")

