"""Retrain the selection model for every month and save each month's scores.

The course data bundle already contains the result
(data/selection/monthly_scores.parquet); run this only after changing the model
or the data. Each month takes 1-4 minutes to train.

Students: always pass --out, so the course scores stay untouched (the grader
uses them), then backtest your scores with run_portfolio.py --scores:

    python selection/build_scores.py --config my_growth.json --out output/retrained/scores.parquet --workers 4
    python run_portfolio.py --period dev --scores output/retrained/scores.parquet

Each month is written to a parts/ folder next to the output as it finishes,
so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from selection.config import DATA, load

PARTS = DATA / "selection" / "parts"
OUT = DATA / "selection" / "monthly_scores.parquet"
_state = {}


def _init(config: str, parts: str):
    _state["parts"] = Path(parts)
    logging.basicConfig(level=logging.ERROR)
    _, _, _, prices, predictions, _ = load(config)
    _state.update(prices=prices, predictions=predictions)


def _run_month(month: str) -> str:
    prices, predictions = _state["prices"], _state["predictions"]
    first = pd.Timestamp(month + "-01")
    sessions = prices.sessions[prices.sessions >= first]
    day = sessions[0].date() if len(sessions) and sessions[0].month == first.month else first.date()
    rows = predictions.get(day)
    df = pd.DataFrame([{"month": first, "cutoff": pd.Timestamp(predictions.cutoff), "symbol": p.symbol,
                        "bucket": p.bucket, "rank_score": p.rank_score, "weight_score": p.weight_score}
                       for p in rows])
    parts = _state["parts"]
    tmp = parts / f".{month}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(parts / f"{month}.parquet")
    return f"{month}: {len(df)} rows, cutoff {predictions.cutoff}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="selection/configs/growth.json")
    ap.add_argument("--start", default="2017-01")
    ap.add_argument("--end", default="2026-11")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where to write the scores (default: the course file, data/selection/monthly_scores.parquet)")
    a = ap.parse_args()
    out = a.out.resolve()
    parts = PARTS if out == OUT.resolve() else out.parent / f"{out.stem}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    months = [m.strftime("%Y-%m") for m in pd.period_range(a.start, a.end, freq="M")]
    todo = [m for m in months if not (parts / f"{m}.parquet").is_file()]
    print(f"{len(months)} months, {len(months) - len(todo)} already done, {len(todo)} to train", flush=True)
    if todo:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.config, str(parts))) as pool:
            for fut in as_completed([pool.submit(_run_month, m) for m in todo]):
                print(fut.result(), flush=True)
    df = pd.concat([pd.read_parquet(parts / f"{m}.parquet") for m in months], ignore_index=True)
    df.to_parquet(out, index=False)
    meta = {"config": a.config, "months": [months[0], months[-1]], "rows": len(df),
            "fundamentals_sha256": _sha256(DATA / "fundamentals.db"),
            "prices_sha256": _sha256(DATA / "prices" / "daily_adjusted.parquet")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
