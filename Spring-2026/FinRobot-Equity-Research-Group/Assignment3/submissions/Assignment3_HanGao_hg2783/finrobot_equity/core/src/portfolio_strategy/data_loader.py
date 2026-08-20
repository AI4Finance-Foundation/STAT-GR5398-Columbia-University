# portfolio_strategy/data_loader.py

import os
import pandas as pd
import yfinance as yf


def download_adjusted_prices(tickers, start_date, end_date, output_path=None):
    """
    Download adjusted close prices from Yahoo Finance.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(axis=1, how="all")
    prices = prices.ffill().dropna(how="all")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        prices.to_csv(output_path)

    return prices


def load_or_download_prices(tickers, start_date, end_date, cache_path):
    """
    Load prices from cache if available. Otherwise download and save.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print("Downloading prices...")
        prices = download_adjusted_prices(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_path=cache_path
        )

    return prices


def to_monthly_returns(prices):
    """
    Convert daily adjusted prices to monthly returns.
    """
    monthly_prices = prices.resample("M").last()
    monthly_returns = monthly_prices.pct_change().dropna(how="all")
    return monthly_returns