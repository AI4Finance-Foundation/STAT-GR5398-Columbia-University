"""Build strategy objects from a JSON config (see configs/)."""
from __future__ import annotations

import json
from pathlib import Path

from shared.prices import PriceStore
from .strategy import ModelOptions, MonthlyPredictions, Parameters
from shared.universe import TradeableUniverse

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load(config_path: str | Path, *, cap_prices_at_decision: bool = True):
    cfg = json.loads(Path(config_path).read_text())
    m, p = cfg["model"], cfg["parameters"]
    options = ModelOptions(db_path=str(DATA / "fundamentals.db"), universes=tuple(m["universes"]),
                           inference_mode=m["inference_mode"], bucket_alpha=dict(m["bucket_alpha"]),
                           val_cutoff=m.get("val_cutoff", "2099-12-31"), val_quarters=int(m.get("val_quarters", 3)))
    params = Parameters(picks_per_bucket={k: int(v) for k, v in p["picks_per_bucket"].items()},
                        global_picks=int(p.get("global_picks", 0)), exposure_rate=float(p.get("exposure_rate", 1.0)),
                        weight_mode=p.get("weight_mode", "score-global"), min_weight_pct=float(p.get("min_weight_pct", .05)),
                        max_weight_pct=float(p.get("max_weight_pct", .15)),
                        incumbent_multiplier=int(p.get("incumbent_multiplier", 2)),
                        blacklist=tuple(p.get("blacklist", ())))
    prices = PriceStore(DATA / "prices" / "daily_adjusted.parquet")
    predictions = MonthlyPredictions(options, cutoff_for=prices.last_session_before_month,
                                     price_as_of=prices.price_as_of_lookup() if cap_prices_at_decision else None)
    universe = TradeableUniverse.from_db_path(options.db_path, options.universes)
    return cfg, options, params, prices, predictions, universe


def load_exposure_inputs():
    """(market_series, macro) for timing.engine.exposure_from_history."""
    import pandas as pd
    market = pd.read_parquet(DATA / "prices" / "market_series.parquet")
    market["date"] = pd.to_datetime(market["date"])
    macro = {}
    for path in (DATA / "macro").glob("*.parquet"):
        f = pd.read_parquet(path)
        macro[path.stem] = pd.Series(f["value"].values, index=pd.to_datetime(f["date"])).sort_index()
    return market, macro
