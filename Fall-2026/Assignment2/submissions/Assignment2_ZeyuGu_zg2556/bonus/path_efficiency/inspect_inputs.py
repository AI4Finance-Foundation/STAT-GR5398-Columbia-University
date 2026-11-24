"""Inspect the extra feature on the exact set of scored growth stocks, with no new fit."""
from pathlib import Path
import pandas as pd


def inspect_inputs(config: Path, scores: Path, output: Path):
    from selection.config import load
    from selection.selector import MLStockSelector
    from shared.store import FundamentalStore

    _, opts, _, _, predictions, _ = load(config)
    store = FundamentalStore(opts.db_path, read_only=True)
    selector = MLStockSelector(opts.db_path, universes=list(opts.universes), read_only=True,
                               price_as_of=predictions.price_as_of, auto_sync_missing=False)
    scored = pd.read_parquet(scores)
    rows = []
    for (month, cutoff), group in scored[scored.bucket == 'growth_tech'].groupby(['month', 'cutoff']):
        assert cutoff == pd.Timestamp(predictions.cutoff_for(month.date())) and cutoff < month
        raw = store.get_fundamental_data(as_of_date=cutoff.date().isoformat())
        capped = selector._clip_trade_prices_to_as_of(raw, cutoff.date(), predictions.price_as_of)
        features = selector._apply_pit_filter(selector._engineer_features(capped))
        inference = selector._assemble_mixed_vintage(features, cutoff.date(), set(group.symbol))
        assert len(inference) == len(group) and set(inference.tic) == set(group.symbol)
        factor = inference.trend_efficiency_4q
        assert factor.between(-1, 1).all()
        rows.append({'month': month.date(), 'cutoff': cutoff.date(), 'growth_names': len(factor),
                     'negative': int(factor.lt(0).sum()), 'zero': int(factor.eq(0).sum()),
                     'positive': int(factor.gt(0).sum()), 'p10': factor.quantile(0.1),
                     'median': factor.median(), 'p90': factor.quantile(0.9)})
    pd.DataFrame(rows).to_csv(output, index=False)
