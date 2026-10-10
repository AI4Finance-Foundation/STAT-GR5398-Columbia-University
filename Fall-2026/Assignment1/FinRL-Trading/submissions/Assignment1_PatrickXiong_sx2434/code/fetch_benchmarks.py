"""Fetch benchmark inputs for a new run; preserve existing frozen inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import yfinance as yf

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
records = []
for ticker in ["SPY", "QQQ"]:
    path = args.output / f"{ticker}_adjusted_close.csv"
    if not path.exists():
        data = yf.Ticker(ticker).history(start="2015-06-15", end="2025-10-01",
                                       auto_adjust=True, actions=False)
        if data.empty:
            raise RuntimeError(f"No benchmark history returned: {ticker}")
        series = data["Close"].dropna()
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        if series.index.has_duplicates or (series <= 0).any():
            raise RuntimeError(f"Invalid benchmark history: {ticker}")
        series.to_csv(path)
    records.append({"ticker": ticker, "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(args.output / "benchmark_input_manifest.json").write_text(json.dumps({
    "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
    "provider": "Yahoo Finance via yfinance; auto_adjust=True",
    "mutable_input": True, "files": records}, indent=2))
