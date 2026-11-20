"""Shared by all three modules.

    shared/store.py       fundamentals database, point-in-time reads and index membership snapshots
    shared/prices.py      daily prices and the trading calendar
    shared/universe.py    S&P 500 + Nasdaq-100 members on a date
    shared/data_view.py   DataView: what valuation and timing functions are allowed to see
    shared/guard.py       blocks file and network access inside student functions
    shared/metrics.py     Sharpe, drawdown, Calmar, crisis table
    shared/periods.py     dev / val / test / forward periods and the test-run log
"""
