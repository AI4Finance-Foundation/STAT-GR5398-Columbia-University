"""
a3_data_pipeline.py
GR5398 Assignment 3 — ContrarianFusion Data Pipeline
Beibei Xian (bx2233) · Columbia University · Spring 2026

Fetches all data needed before FinGPT inference:
  1. Weekly prices       → data/prices_2026q1.csv
  2. News articles       → data/news_2026q1.csv
  3. FinGPT prompt inputs→ data/fingpt_inputs.csv   ← feed to FinGPT_A3_Inference.ipynb
  4. Technical features  → data/features_2026q1.csv
  5. Macro regime        → data/macro_regime.csv

Run:
    python code/a3_data_pipeline.py
"""

import os, time, requests
import pandas as pd
import numpy as np
from pathlib import Path
from fredapi import Fred

# ── Config ────────────────────────────────────────────────────────────────────
POLYGON_KEY = os.getenv("POLYGON_API_KEY", "8S0vgBEz3jtUMsOQlXTQNzij1gQnB7fH")
FRED_KEY    = os.getenv("FRED_API_KEY",    "db1e550a25dbac2b3817204c4b39717c")
POLY_BASE   = "https://api.polygon.io"

START       = "2026-01-01"
END         = "2026-04-30"
HIST_START  = "2024-01-01"

DOW30 = [
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","INTC","JNJ","JPM","KO","MCD","MMM",
    "MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WMT",
]

OUT = Path("data")
OUT.mkdir(exist_ok=True)


# ── 1. Weekly prices ──────────────────────────────────────────────────────────
def fetch_prices():
    print("\n[1/5] Fetching weekly prices from Polygon.io ...")
    price_rows = {}
    for tk in DOW30 + ["SPY"]:
        url = f"{POLY_BASE}/v2/aggs/ticker/{tk}/range/1/week/{HIST_START}/{END}"
        r   = requests.get(url, params={"adjusted":"true","sort":"asc",
                                        "limit":500,"apiKey":POLYGON_KEY}, timeout=15)
        res = r.json().get("results", [])
        if res:
            df_tk = pd.DataFrame(res)
            df_tk["date"] = pd.to_datetime(df_tk["t"], unit="ms").dt.normalize()
            price_rows[tk] = df_tk.set_index("date")["c"]
            print(f"  {tk:5s}: {len(res)} weeks")
        else:
            print(f"  {tk:5s}: ✗ no data — trying yfinance fallback")
        time.sleep(0.13)

    prices = pd.DataFrame(price_rows)

    # yfinance fallback for any missing / 2026 Q1 gaps
    missing = [t for t in DOW30 + ["SPY"] if t not in prices.columns]
    if missing or prices.empty:
        import yfinance as yf
        yf_raw = yf.download(DOW30 + ["SPY"], start=HIST_START, end=END,
                             interval="1wk", auto_adjust=True, progress=False)["Close"]
        yf_raw.index = pd.to_datetime(yf_raw.index).tz_localize(None)
        prices = prices.combine_first(yf_raw) if not prices.empty else yf_raw

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.to_csv(OUT / "prices_2026q1.csv")
    print(f"  ✓ prices_2026q1.csv — {prices.shape}")
    return prices


# ── 2. News articles ──────────────────────────────────────────────────────────
def fetch_news():
    print("\n[2/5] Fetching news from Polygon.io ...")
    all_articles = []
    for tk in DOW30:
        r = requests.get(
            f"{POLY_BASE}/v2/reference/news",
            params={"ticker": tk, "published_utc.gte": START,
                    "published_utc.lte": END, "limit": 50, "apiKey": POLYGON_KEY},
            timeout=15,
        )
        articles = r.json().get("results", [])
        for a in articles:
            sent = 0.0
            for ins in a.get("insights", []):
                if ins.get("ticker") == tk:
                    s    = ins.get("sentiment", "neutral")
                    sent = 1.0 if s == "positive" else (-1.0 if s == "negative" else 0.0)
                    break
            all_articles.append({
                "ticker":    tk,
                "date":      a.get("published_utc", "")[:10],
                "title":     a.get("title", ""),
                "summary":   str(a.get("description", ""))[:300],
                "sentiment": sent,
            })
        print(f"  {tk:5s}: {len(articles)} articles")
        time.sleep(0.25)

    news_df = pd.DataFrame(all_articles)
    news_df["date"] = pd.to_datetime(news_df["date"])
    news_df.to_csv(OUT / "news_2026q1.csv", index=False)
    print(f"  ✓ news_2026q1.csv — {len(news_df)} articles")
    return news_df


# ── 3. FinGPT prompt inputs ───────────────────────────────────────────────────
def build_fingpt_inputs(news_df):
    print("\n[3/5] Building FinGPT prompt inputs ...")

    def make_prompt(ticker, date_str, articles):
        lines = [f"- {a['title']}. {str(a.get('summary',''))[:150]}"
                 for _, a in articles.iterrows()][:5]
        headlines = "\n".join(lines) if lines else "No news today."
        return (f"Analyze the following news for {ticker} "
                f"and predict the 5-day price direction.\n"
                f"Date: {date_str}\nNews:\n{headlines}\n\n"
                f'{{"prediction": "')

    bdays = pd.bdate_range(START, END)
    rows  = []
    for day in bdays:
        day_str      = day.strftime("%Y-%m-%d")
        week_start   = day.to_period("W").start_time.strftime("%Y-%m-%d")
        window_start = day - pd.Timedelta(days=3)
        for tk in DOW30:
            arts      = news_df[(news_df["ticker"] == tk) &
                                (news_df["date"] >= window_start) &
                                (news_df["date"] <  day)]
            sent_mean = arts["sentiment"].mean() if len(arts) > 0 else 0.0
            rows.append({
                "date":                   day_str,
                "week_start":             week_start,
                "ticker":                 tk,
                "fingpt_prompt":          make_prompt(tk, day_str, arts),
                "massive_sentiment_mean": 0.0 if np.isnan(sent_mean) else sent_mean,
                "n_articles":             len(arts),
            })

    inputs_df = pd.DataFrame(rows)
    inputs_df.to_csv(OUT / "fingpt_inputs.csv", index=False)
    print(f"  ✓ fingpt_inputs.csv — {len(inputs_df)} stock-day rows")
    print(f"    Rows with news: {(inputs_df['n_articles']>0).sum()} / {len(inputs_df)}")
    return inputs_df


# ── 4. Technical features (FinRL-style) ───────────────────────────────────────
def compute_features(prices):
    print("\n[4/5] Computing technical features ...")
    dow_p  = prices[[t for t in DOW30 if t in prices.columns]]

    def zscore_cs(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-9, axis=0)

    ret4w  = dow_p.pct_change(4,  fill_method=None)
    ret52w = dow_p.pct_change(52, fill_method=None)
    vol4w  = dow_p.pct_change(fill_method=None).rolling(4).std()
    ma20   = dow_p.rolling(20).mean()
    std20  = dow_p.rolling(20).std()
    bb_pos = (dow_p - (ma20 - 2 * std20)) / (4 * std20 + 1e-9)

    tech_sig = np.tanh(
        (0.40 * zscore_cs(ret4w)
       + 0.25 * zscore_cs(ret52w)
       - 0.20 * zscore_cs(vol4w)
       - 0.15 * zscore_cs(bb_pos)) / 2
    )

    rows = []
    for dt in tech_sig.index:
        if dt < pd.Timestamp(START) or dt > pd.Timestamp(END):
            continue
        for tk in tech_sig.columns:
            v = tech_sig.loc[dt, tk]
            if not np.isnan(v):
                rows.append({"date": dt, "ticker": tk, "tech_score": float(v)})

    # also store raw feature columns for cross-sectional z-scoring in backtest
    raw_rows = []
    for dt in dow_p.index:
        if dt < pd.Timestamp(START) or dt > pd.Timestamp(END):
            continue
        for tk in dow_p.columns:
            raw_rows.append({
                "date":   dt,
                "ticker": tk,
                "ret4w":  float(ret4w.loc[dt, tk])  if not np.isnan(ret4w.loc[dt, tk])  else 0.0,
                "ret52w": float(ret52w.loc[dt, tk]) if not np.isnan(ret52w.loc[dt, tk]) else 0.0,
                "vol4w":  float(vol4w.loc[dt, tk])  if not np.isnan(vol4w.loc[dt, tk])  else 0.0,
                "bb_pos": float(bb_pos.loc[dt, tk]) if not np.isnan(bb_pos.loc[dt, tk]) else 0.5,
            })

    feat_df = pd.DataFrame(raw_rows)
    # add z-scored columns
    def zscore(s):
        return (s - s.mean()) / (s.std() + 1e-9)
    for col in ["ret4w", "ret52w", "vol4w", "bb_pos"]:
        feat_df[col + "_z"] = feat_df.groupby("date")[col].transform(zscore)
    feat_df["tech_score"] = (0.40 * feat_df["ret4w_z"]
                           + 0.25 * feat_df["ret52w_z"]
                           - 0.20 * feat_df["vol4w_z"]
                           - 0.15 * feat_df["bb_pos_z"])

    feat_df.to_csv(OUT / "features_2026q1.csv", index=False)
    print(f"  ✓ features_2026q1.csv — {len(feat_df)} rows")
    return feat_df


# ── 5. Macro regime (FRED) ────────────────────────────────────────────────────
def fetch_macro():
    print("\n[5/5] Fetching macro data from FRED ...")
    fred = Fred(api_key=FRED_KEY)
    series = {
        "VIXCLS":       "vix",
        "T10Y2Y":       "yield_curve",
        "FEDFUNDS":     "fed_funds",
        "BAMLH0A0HYM2": "credit_spread",
    }
    macro_daily = pd.DataFrame()
    for sid, name in series.items():
        try:
            s = fred.get_series(sid, observation_start=START, observation_end=END)
            macro_daily[name] = s
            print(f"  ✓ {name}: {len(s)} obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    macro_weekly = macro_daily.ffill().resample("W-FRI").last().ffill()

    def classify(row):
        vix = row.get("vix",          22) or 22
        yc  = row.get("yield_curve",   0) or 0
        cs  = row.get("credit_spread", 5) or 5
        if vix < 20 and yc > -0.2 and cs < 4.5:
            return pd.Series({"regime":"RISK_ON",  "long_mult":1.0, "short_mult":1.0})
        if vix > 25 or  yc < -0.5 or  cs > 6.0:
            return pd.Series({"regime":"RISK_OFF", "long_mult":0.4, "short_mult":0.0})
        return     pd.Series({"regime":"NEUTRAL",  "long_mult":0.7, "short_mult":0.7})

    macro_out = pd.concat([macro_weekly, macro_weekly.apply(classify, axis=1)], axis=1)
    macro_out.to_csv(OUT / "macro_regime.csv")
    print(f"  ✓ macro_regime.csv — {len(macro_out)} weeks")
    print(f"    Regimes: {macro_out['regime'].value_counts().to_dict()}")
    return macro_out


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("A3 Data Pipeline — ContrarianFusion")
    print("=" * 60)
    prices   = fetch_prices()
    news_df  = fetch_news()
    _        = build_fingpt_inputs(news_df)
    _        = compute_features(prices)
    _        = fetch_macro()
    print("\n" + "=" * 60)
    print("✓ All 5 data files written to data/")
    print("  Next step: open notebooks/FinGPT_A3_Inference.ipynb in Colab (A100)")
    print("=" * 60)
