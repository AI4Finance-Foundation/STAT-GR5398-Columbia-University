
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = '/home/claude/outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FUND_FILE = '/home/claude/sp500_tickers_fundamental_quarterly_synthetic.csv'
PRICE_FILE = '/home/claude/sp500_tickers_daily_price_synthetic.csv'

print("=" * 70)
print("STEP 1: Loading data")
print("=" * 70)

fund_df = pd.read_csv(FUND_FILE)
price_df = pd.read_csv(PRICE_FILE, usecols=['gvkey','tic','datadate','prccd','ajexdi'])

fund_df['datadate'] = pd.to_datetime(fund_df['datadate'])
price_df['datadate'] = pd.to_datetime(price_df['datadate'])

print(fund_df.shape)
print(price_df.shape)
print(fund_df.tic.nunique())
print(price_df.tic.nunique())


def adjust_trade_date(d):
    m = d.month
    y = d.year
    if m <= 3:
        return pd.Timestamp(y, 6, 1)     # Q1 report → trade after Jun 1
    elif m <= 6:
        return pd.Timestamp(y, 9, 1)     # Q2 report → trade after Sep 1
    elif m <= 9:
        return pd.Timestamp(y, 12, 1)    # Q3 report → trade after Dec 1
    else:
        return pd.Timestamp(y + 1, 3, 1) # Q4 report → trade after Mar 1 next yr

fund_df['tradedate'] = fund_df['datadate'].apply(adjust_trade_date)

fund_df['adj_close_q'] = fund_df['prccq'] / fund_df['adjex']
common_tickers = set(fund_df.tic.unique()) & set(price_df.tic.unique())
fund_df = fund_df[fund_df.tic.isin(common_tickers)].copy()
print(f"Common tickers: {len(common_tickers)}")

print("\nSTEP 2.2: Calculating next-quarter log-returns...")

fund_df['date'] = fund_df['tradedate']
fund_df.drop_duplicates(['date','gvkey'], keep='last', inplace=True)
fund_df.sort_values(['gvkey','date'], inplace=True)
fund_df.reset_index(drop=True, inplace=True)

fund_df['y_return'] = (
    fund_df.groupby('gvkey')['adj_close_q']
    .transform(lambda x: np.log(x.shift(-1) / x))
)

print("\nSTEP 2.3: Calculating valuation ratios (PE, PS, PB)...")

fund_df['pe'] = fund_df['prccq'] / fund_df['epspiy'].replace(0, np.nan)
fund_df['ps'] = fund_df['prccq'] / (fund_df['revtq'] / fund_df['cshoq']).replace(0, np.nan)
fund_df['pb'] = fund_df['prccq'] / ((fund_df['atq'] - fund_df['ltq']) / fund_df['cshoq']).replace(0, np.nan)



items = [
    'date','gvkey','tic','gsector',
    'oiadpq','revtq','niq','atq','teqq','epspiy','ceqq','cshoq','dvpspq',
    'actq','lctq','cheq','rectq','cogsq','invtq','apq','dlttq','dlcq','ltq',
    'pe','ps','pb','adj_close_q','y_return'
]
fund_data = fund_df[items].rename(columns={
    'oiadpq':'op_inc_q','revtq':'rev_q','niq':'net_inc_q','atq':'tot_assets',
    'teqq':'sh_equity','epspiy':'eps_incl_ex','ceqq':'com_eq','cshoq':'sh_outstanding',
    'dvpspq':'div_per_sh','actq':'cur_assets','lctq':'cur_liabilities',
    'cheq':'cash_eq','rectq':'receivables','cogsq':'cogs_q','invtq':'inventories',
    'apq':'payables','dlttq':'long_debt','dlcq':'short_debt','ltq':'tot_liabilities'
}).reset_index(drop=True)

print("\nSTEP 2.3: Calculating comprehensive financial ratios (vectorized)...")

def trailing_4q_ratio(df, num_col, denom_col=None, denom_point=False):
    """
    For each row, sum the trailing 4 quarters of num_col,
    divide by either sum of denom_col (flow) or point-in-time denom_col.
    Uses groupby + rolling for speed.
    """
    result = pd.Series(np.nan, index=df.index)
    for gvk, grp in df.groupby('gvkey'):
        idx = grp.index
        num_roll = grp[num_col].rolling(4, min_periods=4).sum()
        if denom_col is None:
            result.loc[idx] = num_roll
        elif denom_point:
            result.loc[idx] = num_roll / grp[denom_col].replace(0, np.nan)
        else:
            denom_roll = grp[denom_col].rolling(4, min_periods=4).sum()
            result.loc[idx] = num_roll / denom_roll.replace(0, np.nan)
    return result

fund_data['OPM'] = trailing_4q_ratio(fund_data, 'op_inc_q', 'rev_q')          # Operating Margin
fund_data['NPM'] = trailing_4q_ratio(fund_data, 'net_inc_q', 'rev_q')         # Net Profit Margin
fund_data['ROA'] = trailing_4q_ratio(fund_data, 'net_inc_q', 'tot_assets', denom_point=True)
fund_data['ROE'] = trailing_4q_ratio(fund_data, 'net_inc_q', 'sh_equity', denom_point=True)

# Per share
fund_data['EPS'] = fund_data['eps_incl_ex']
fund_data['BPS'] = fund_data['com_eq'] / fund_data['sh_outstanding'].replace(0, np.nan)
fund_data['DPS'] = fund_data['div_per_sh']

# Liquidity
fund_data['cur_ratio']   = fund_data['cur_assets']  / fund_data['cur_liabilities'].replace(0, np.nan)
fund_data['quick_ratio'] = (fund_data['cash_eq'] + fund_data['receivables']) / fund_data['cur_liabilities'].replace(0, np.nan)
fund_data['cash_ratio']  = fund_data['cash_eq'] / fund_data['cur_liabilities'].replace(0, np.nan)

# Efficiency 
fund_data['inv_turnover']      = trailing_4q_ratio(fund_data, 'cogs_q', 'inventories', denom_point=True)
fund_data['acc_rec_turnover']  = trailing_4q_ratio(fund_data, 'rev_q', 'receivables', denom_point=True)
fund_data['acc_pay_turnover']  = trailing_4q_ratio(fund_data, 'cogs_q', 'payables', denom_point=True)

# Leverage
fund_data['debt_ratio']      = fund_data['tot_liabilities'] / fund_data['tot_assets'].replace(0, np.nan)
fund_data['debt_to_equity']  = fund_data['tot_liabilities'] / fund_data['sh_equity'].replace(0, np.nan)


# Final Sele
ratio_cols = [
    'date','gvkey','tic','gsector','adj_close_q','y_return',
    'OPM','NPM','ROA','ROE','EPS','BPS','DPS',
    'cur_ratio','quick_ratio','cash_ratio',
    'inv_turnover','acc_rec_turnover','acc_pay_turnover',
    'debt_ratio','debt_to_equity','pe','ps','pb'
]
ratios = fund_data[ratio_cols].copy()

print(f"Ratios shape before cleaning: {ratios.shape}")






GICS_NAMES = {
    10: 'Energy', 15: 'Materials', 20: 'Industrials',
    25: 'Consumer Discretionary', 30: 'Consumer Staples',
    35: 'Health Care', 40: 'Financials',
    45: 'Information Technology', 50: 'Communication Services',
    55: 'Utilities', 60: 'Real Estate'
}

FEATURE_COLS = [
    'OPM','NPM','ROA','ROE','EPS','BPS','DPS',
    'cur_ratio','quick_ratio','cash_ratio',
    'inv_turnover','acc_rec_turnover','acc_pay_turnover',
    'debt_ratio','debt_to_equity','pe','ps','pb'
]

# Replace inf with NaN globally first
ratios.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with no adjusted close or no return
ratios = ratios[ratios['adj_close_q'] > 0].copy()

all_sector_dfs = []
sector_summary = []

for sector, sname in GICS_NAMES.items():
    df_sec = ratios[ratios['gsector'] == sector].copy()
    if df_sec.empty:
        continue

    feat_cols_available = [c for c in FEATURE_COLS if c in df_sec.columns]

    # Drop factors with >5% missing 
    missing_pct = df_sec[feat_cols_available].isnull().mean()
    drop_factors = missing_pct[missing_pct > 0.05].index.tolist()
    if drop_factors:
        df_sec.drop(columns=drop_factors, inplace=True)
        feat_cols_available = [c for c in feat_cols_available if c not in drop_factors]


    miss_by_stock = df_sec.groupby('tic')[feat_cols_available].apply(
        lambda x: x.isnull().sum().sum()
    )
    if miss_by_stock.max() > 0:
        worst_tic = miss_by_stock.idxmax()
        df_sec = df_sec[df_sec['tic'] != worst_tic].copy()

    keep_cols = feat_cols_available + ['y_return']
    df_sec.dropna(subset=keep_cols, inplace=True)
    df_sec.reset_index(drop=True, inplace=True)

    sector_summary.append({
        'sector': sector, 'name': sname,
        'records': len(df_sec),
        'tickers': df_sec['tic'].nunique(),
        'dropped_factors': drop_factors
    })

    # Save sector Excel file
    df_sec['date'] = df_sec['date'].apply(
        lambda x: x.strftime('%Y-%m-%d') if hasattr(x,'strftime') else str(x))
    xls_path = os.path.join(OUTPUT_DIR, f'sector{sector}.xlsx')
    df_sec.to_excel(xls_path, index=False)
    print(f"  Sector {sector:2d} ({sname:30s}): {len(df_sec):5d} rows, "
          f"{df_sec['tic'].nunique():2d} tickers | dropped factors: {drop_factors}")

    all_sector_dfs.append(df_sec)

final_ratios = pd.concat(all_sector_dfs, ignore_index=True)
final_ratios.to_csv(os.path.join(OUTPUT_DIR, 'final_ratios.csv'), index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Final combined dataset: {final_ratios.shape}")
print(f"Date range: {final_ratios['date'].min()} → {final_ratios['date'].max()}")
print(f"Unique tickers: {final_ratios['tic'].nunique()}")
print(f"Sectors saved: {len(all_sector_dfs)}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("  final_ratios.csv")
for sector, sname in GICS_NAMES.items():
    if any(s['sector']==sector for s in sector_summary):
        print(f"  sector{sector}.xlsx  ({sname})")

print("\nDone! ✓")
