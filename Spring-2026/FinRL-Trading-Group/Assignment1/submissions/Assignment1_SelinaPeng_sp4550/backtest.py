import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")
import os

OUTPUT_DIR = '/home/claude/outputs'
DATA_FILE  = os.path.join(OUTPUT_DIR, 'final_ratios.csv')

print("Loading data...")
df = pd.read_csv(DATA_FILE, parse_dates=['date'])

BACKTEST_START = '2018-01-01'
BACKTEST_END   = '2025-12-31'
df = df[(df['date'] >= BACKTEST_START) & (df['date'] <= BACKTEST_END)].copy()
df.sort_values(['date','tic'], inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Backtest data: {df.shape}, dates: {df['date'].min().date()} → {df['date'].max().date()}")




QUALITY_FACTORS = ['ROE', 'ROA', 'OPM', 'NPM']   # higher = better
VALUE_FACTORS   = ['pe', 'pb', 'ps']               # lower = better (invert)
SAFETY_FACTORS  = ['debt_ratio']                   # lower = better (invert)

FEATURE_COLS = QUALITY_FACTORS + VALUE_FACTORS + SAFETY_FACTORS

def winsorize(series, low=0.01, high=0.99):
    lo, hi = series.quantile(low), series.quantile(high)
    return series.clip(lo, hi)

for col in FEATURE_COLS:
    if col in df.columns:
        df[col] = df.groupby('date')[col].transform(winsorize)

def zscore(series):
    mu, sd = series.mean(), series.std()
    return (series - mu) / sd if sd > 1e-9 else series * 0

for col in QUALITY_FACTORS:
    if col in df.columns:
        df[f'z_{col}'] = df.groupby('date')[col].transform(zscore)

for col in VALUE_FACTORS + SAFETY_FACTORS:
    if col in df.columns:
        # Invert: lower PE/PB/PS/debt is better
        df[f'z_{col}'] = df.groupby('date')[col].transform(lambda x: -zscore(x))

z_cols = [f'z_{c}' for c in FEATURE_COLS if f'z_{c}' in df.columns]
df['composite_score'] = df[z_cols].mean(axis=1)

STOCKS_PER_SECTOR = 2

def select_portfolio(date_df):
    """Select top N stocks per sector by composite score."""
    selected = (
        date_df.dropna(subset=['composite_score','y_return'])
        .sort_values('composite_score', ascending=False)
        .groupby('gsector')
        .head(STOCKS_PER_SECTOR)
    )
    return selected['tic'].tolist()



print("\nRunning backtest...")

dates = sorted(df['date'].unique())
portfolio_log = []
quarterly_returns = []

for i, date in enumerate(dates[:-1]):  # exclude last (no forward return)
    date_df = df[df['date'] == date].copy()
    if date_df.empty:
        continue

    selected_tickers = select_portfolio(date_df)
    if not selected_tickers:
        continue

    # Get returns
    selected_rows = date_df[date_df['tic'].isin(selected_tickers)]
    valid_rows = selected_rows.dropna(subset=['y_return'])

    if valid_rows.empty:
        continue


    
    port_return = valid_rows['y_return'].mean()

    
    # Benchmark
    bench_return = date_df.dropna(subset=['y_return'])['y_return'].mean()

    quarterly_returns.append({
        'date': date,
        'portfolio_return': port_return,
        'benchmark_return': bench_return,
        'n_stocks': len(valid_rows),
        'tickers': ', '.join(sorted(valid_rows['tic'].tolist()))
    })

    portfolio_log.append({
        'date': date,
        'tickers': selected_tickers,
        'sector_counts': date_df[date_df['tic'].isin(selected_tickers)].groupby('gsector').size().to_dict()
    })

results = pd.DataFrame(quarterly_returns)
results['date'] = pd.to_datetime(results['date'])
results.sort_values('date', inplace=True)

results['port_cum']  = results['portfolio_return'].cumsum()
results['bench_cum'] = results['benchmark_return'].cumsum()

results['port_value']  = 100 * np.exp(results['portfolio_return'].cumsum())
results['bench_value'] = 100 * np.exp(results['benchmark_return'].cumsum())



def sharpe(returns, freq=4):
    """Annualized Sharpe (assume 0% risk-free)."""
    if returns.std() == 0: return np.nan
    return (returns.mean() * freq) / (returns.std() * np.sqrt(freq))

def max_drawdown(cum_log_returns):
    wealth = np.exp(cum_log_returns.values)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    return dd.min()

def annualized_return(cum_log_ret, n_quarters):
    return np.exp(cum_log_ret / n_quarters * 4) - 1

n_q = len(results)
port_total_log  = results['portfolio_return'].sum()
bench_total_log = results['benchmark_return'].sum()

metrics = {
    'Portfolio': {
        'Annualized Return':    f"{annualized_return(port_total_log, n_q)*100:.2f}%",
        'Cumulative Return':    f"{(np.exp(port_total_log)-1)*100:.2f}%",
        'Annualized Sharpe':    f"{sharpe(results['portfolio_return']):.3f}",
        'Max Drawdown':         f"{max_drawdown(results['port_cum'])*100:.2f}%",
        'Win Rate vs Bench':    f"{(results['portfolio_return']>results['benchmark_return']).mean()*100:.1f}%",
        'Avg Stocks/Quarter':   f"{results['n_stocks'].mean():.1f}",
    },
    'Benchmark': {
        'Annualized Return':    f"{annualized_return(bench_total_log, n_q)*100:.2f}%",
        'Cumulative Return':    f"{(np.exp(bench_total_log)-1)*100:.2f}%",
        'Annualized Sharpe':    f"{sharpe(results['benchmark_return']):.3f}",
        'Max Drawdown':         f"{max_drawdown(results['bench_cum'])*100:.2f}%",
        'Win Rate vs Bench':    '-',
        'Avg Stocks/Quarter':   f"{df.groupby('date')['tic'].nunique().mean():.1f}",
    }
}

print("\n" + "=" * 60)
print("BACKTEST RESULTS (2018–2025)")
print("=" * 60)
header = f"{'Metric':<28} {'Portfolio':>14} {'Benchmark':>14}"
print(header)
print("-" * 60)
for metric in list(metrics['Portfolio'].keys()):
    p = metrics['Portfolio'][metric]
    b = metrics['Benchmark'][metric]
    print(f"{metric:<28} {p:>14} {b:>14}")
print("=" * 60)

# ── Save results CSV ───────────────────────────────────────────────────────────
results_out = results[['date','portfolio_return','benchmark_return',
                        'port_value','bench_value','n_stocks','tickers']].copy()
results_out['date'] = results_out['date'].dt.strftime('%Y-%m-%d')
results_out.to_csv(os.path.join(OUTPUT_DIR, 'backtest_results.csv'), index=False)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(13, 14))
fig.suptitle('Assignment 1 – Sector-Neutral Multi-Factor Strategy\nBacktest: 2018–2025',
             fontsize=15, fontweight='bold', y=0.98)

# Panel 1: Cumulative performance
ax1 = axes[0]
ax1.plot(results['date'], results['port_value'],  color='#1f77b4', lw=2, label='Strategy (Sector-Neutral Multi-Factor)')
ax1.plot(results['date'], results['bench_value'], color='#ff7f0e', lw=2, linestyle='--', label='Benchmark (Equal-Weight Universe)')
ax1.axhline(100, color='gray', lw=0.8, linestyle=':')
ax1.fill_between(results['date'], results['port_value'], results['bench_value'],
                  where=results['port_value'] >= results['bench_value'],
                  alpha=0.15, color='green', label='Outperformance')
ax1.fill_between(results['date'], results['port_value'], results['bench_value'],
                  where=results['port_value'] < results['bench_value'],
                  alpha=0.15, color='red', label='Underperformance')
ax1.set_ylabel('Portfolio Value ($, starting $100)', fontsize=11)
ax1.set_title('Cumulative Performance', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel 2: Quarterly returns bar chart
ax2 = axes[1]
colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in results['portfolio_return']]
ax2.bar(results['date'], results['portfolio_return'] * 100, width=60,
        color=colors, alpha=0.8, label='Strategy Quarterly Return')
ax2.plot(results['date'], results['benchmark_return'] * 100,
         color='#ff7f0e', lw=1.5, linestyle='--', label='Benchmark', marker='o', markersize=3)
ax2.axhline(0, color='black', lw=0.8)
ax2.set_ylabel('Quarterly Log-Return (%)', fontsize=11)
ax2.set_title('Quarterly Returns', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Panel 3: Active return (alpha per quarter)
ax3 = axes[2]
active = (results['portfolio_return'] - results['benchmark_return']) * 100
cum_active = active.cumsum()
ax3.bar(results['date'], active, width=60,
        color=['#27ae60' if a > 0 else '#c0392b' for a in active], alpha=0.7, label='Active Return per Quarter')
ax3.plot(results['date'], cum_active, color='navy', lw=2, label='Cumulative Active Return')
ax3.axhline(0, color='black', lw=0.8)
ax3.set_ylabel('Active Return vs Benchmark (%)', fontsize=11)
ax3.set_xlabel('Date', fontsize=11)
ax3.set_title('Active Return (Strategy – Benchmark)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, 'backtest_chart.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart saved: {OUTPUT_DIR}/backtest_chart.png")




GICS = {10:'Energy',15:'Materials',20:'Industrials',25:'Cons. Disc.',
        30:'Cons. Staples',35:'Health Care',40:'Financials',
        45:'Info Tech',50:'Comm. Svcs',55:'Utilities',60:'Real Estate'}

sector_exp = df[df['tic'].isin(
    [t for pl in portfolio_log for t in pl['tickers']]
)].groupby('gsector')['tic'].nunique()

fig2, ax = plt.subplots(figsize=(10, 5))
sector_labels = [GICS.get(s, str(s)) for s in sector_exp.index]
bars = ax.bar(sector_labels, sector_exp.values,
              color=plt.cm.tab20.colors[:len(sector_labels)], alpha=0.85, edgecolor='white')
ax.set_title('Unique Stocks Selected per Sector (Backtest Period)', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Unique Tickers Selected')
ax.set_xlabel('GICS Sector')
for bar, val in zip(bars, sector_exp.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, str(val),
            ha='center', fontsize=10, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sector_exposure.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Sector chart saved: {OUTPUT_DIR}/sector_exposure.png")

print("\n✓ Backtest complete. All outputs saved to:", OUTPUT_DIR)
print("  backtest_results.csv")
print("  backtest_chart.png")
print("  sector_exposure.png")

# Print a quarterly holdings snapshot
print("\n── Sample Holdings (most recent quarters) ──")
print(results[['date','n_stocks','tickers']].tail(6).to_string(index=False))
