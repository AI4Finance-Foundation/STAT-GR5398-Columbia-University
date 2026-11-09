"""Print the target portfolio for a date.

    python selection/run_picks.py selection/configs/growth.json 2026-11-06
    python selection/run_picks.py selection/configs/growth.json 2026-11-06 --exposure
"""
import argparse, logging, sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from selection.config import load
from selection.strategy import target_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("date")
    ap.add_argument("--exposure", action="store_true", help="scale by the exposure engine's multiplier")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)
    cfg, options, params, prices, predictions, universe = load(a.config)
    day = date.fromisoformat(a.date)
    rows = predictions.get(day)
    members = universe.members_at(day)
    eligible = set(members) & set(prices.close.columns)
    weights = target_weights(rows, params, eligible, {}, global_unified=options.inference_mode == "global-unified")
    budget = 1.0
    if a.exposure:
        from selection.config import load_exposure_inputs
        from timing.engine import exposure_from_history
        market, macro = load_exposure_inputs()
        res = exposure_from_history(market, macro, day)
        if res is not None:
            budget = res.exposureMultiplier
            print(f"exposure multiplier {budget:.3f} (risk score {res.risk_score:.3f})")
    by = {r.symbol: r for r in rows}
    print(f"{cfg['name']} · as of {day} · model cutoff {predictions.cutoff} · members {len(members)} · eligible {len(eligible)}")
    for sym, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        print(f"  {sym:6} {by[sym].bucket:12} score {by[sym].rank_score:8.4f}  weight {w*100:6.2f}%  invested {w*budget*100:6.2f}%")
    print(f"  total {sum(weights.values())*100:.2f}%")


if __name__ == "__main__":
    main()
