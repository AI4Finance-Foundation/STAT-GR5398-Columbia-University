"""
scripts/run_preprocessing.py

Command-line wrapper to run the preprocessing pipeline from the notebook
FinRL-Trading-Full-Workload (2).ipynb.

This script replicates the notebook's functions:
 - load_data
 - adjust_trade_dates
 - calculate_adjusted_close
 - match_tickers_and_gvkey
 - calculate_next_quarter_returns
 - calculate_basic_ratios
 - select_columns
 - calculate_financial_ratios
 - handle_missing_values
 - save_results

Usage:
  python3 scripts/run_preprocessing.py --fundamental path/to/fundamental.csv --price data/daily.csv --output outputs

Notes:
 - The fundamental CSV must contain the quarterly Compustat fields used in the notebook
   (e.g., prccq, adjex, rdq, epspxq, revtq, cshoq, atq, ltq, etc.).
 - The price CSV should be `data/daily.csv` generated earlier and contain columns
   ['gvkey','tic','datadate','prccd','ajexdi'].

Only this file is added by the assistant. Do not modify other files unless instructed.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def load_data(fundamental_file, price_file):
    print("Loading data files...")
    if not os.path.isfile(fundamental_file):
        raise FileNotFoundError(f"Fundamental file {fundamental_file} not found.")
    if not os.path.isfile(price_file):
        raise FileNotFoundError(f"Price file {price_file} not found.")

    fund_df = pd.read_csv(fundamental_file)
    print(f"Fundamental shape: {fund_df.shape}")

    price_columns = ['gvkey', 'tic', 'datadate', 'prccd', 'ajexdi']
    df_daily_price = pd.read_csv(price_file, usecols=price_columns)
    print(f"Price shape: {df_daily_price.shape}")

    return fund_df, df_daily_price


def adjust_trade_dates(fund_df):
    print("Adjusting trade dates...")
    datadate_dt = pd.to_datetime(fund_df['datadate'])
    fund_df['tradedate'] = ((datadate_dt.dt.to_period('Q')).dt.end_time.dt.normalize())
    fund_df['reportdate'] = fund_df.get('rdq', '')
    return fund_df


def calculate_adjusted_close(fund_df):
    print("Calculating adjusted close price...")
    # note: notebook used prccq / adjex -> adj_close_q
    if 'prccq' in fund_df.columns and 'adjex' in fund_df.columns:
        fund_df['adj_close_q'] = fund_df['prccq'] / fund_df['adjex']
    else:
        fund_df['adj_close_q'] = np.nan
    return fund_df


def match_tickers_and_gvkey(fund_df, df_daily_price):
    print("Matching tickers and gvkey...")
    tic_to_gvkey = {}
    df_daily_groups = list(df_daily_price.groupby('tic'))
    for tic, df_ in df_daily_groups:
        tic_to_gvkey[tic] = df_.gvkey.iloc[0]

    # Filter fundamental data to only include tickers present in price data
    fund_df = fund_df[np.isin(fund_df.tic, list(tic_to_gvkey.keys()))]
    fund_df['gvkey'] = [tic_to_gvkey[x] for x in fund_df['tic']]
    return fund_df


def calculate_next_quarter_returns(fund_df):
    print("Calculating next quarter returns...")
    fund_df['date'] = pd.to_datetime(fund_df['tradedate'])
    fund_df.drop_duplicates(['date', 'gvkey'], keep='last', inplace=True)

    l_df = list(fund_df.groupby('gvkey'))
    out_frames = []
    for tic, df in l_df:
        df = df.sort_values('date').reset_index(drop=True)
        if 'adj_close_q' in df.columns:
            df['y_return'] = np.log(df['adj_close_q'].shift(-1) / df['adj_close_q'])
        else:
            df['y_return'] = np.nan
        out_frames.append(df)

    fund_df = pd.concat(out_frames, ignore_index=True)
    print(f"Data shape after calculating returns: {fund_df.shape}")
    return fund_df


def calculate_basic_ratios(fund_df):
    print("Calculating basic financial ratios...")
    # Guard columns
    if 'prccq' in fund_df.columns and 'epspxq' in fund_df.columns:
        fund_df['pe'] = fund_df['prccq'] / fund_df['epspxq']
    else:
        fund_df['pe'] = np.nan

    if all(c in fund_df.columns for c in ['prccq', 'revtq', 'cshoq']):
        fund_df['ps'] = fund_df['prccq'] / (fund_df['revtq'] / fund_df['cshoq'])
    else:
        fund_df['ps'] = np.nan

    if all(c in fund_df.columns for c in ['prccq', 'atq', 'ltq', 'cshoq']):
        fund_df['pb'] = fund_df['prccq'] / ((fund_df['atq'] - fund_df['ltq']) / fund_df['cshoq'])
    else:
        fund_df['pb'] = np.nan

    return fund_df


def select_columns(fund_df):
    print("Selecting relevant columns...")
    items = [
        'date', 'gvkey', 'tic', 'gsector',
        'oiadpq', 'revtq', 'niq', 'atq', 'teqq', 'epspiy', 'ceqq', 'cshoq', 'dvpspq',
        'actq', 'lctq', 'cheq', 'rectq', 'cogsq', 'invtq', 'apq', 'dlttq', 'dlcq', 'ltq',
        'pe', 'ps', 'pb', 'adj_close_q', 'y_return', 'reportdate'
    ]
    available = [c for c in items if c in fund_df.columns]
    fund_data = fund_df[available].copy()

    # Rename for readability if columns exist
    rename_map = {}
    if 'oiadpq' in fund_data.columns:
        rename_map['oiadpq'] = 'op_inc_q'
    if 'revtq' in fund_data.columns:
        rename_map['revtq'] = 'rev_q'
    if 'niq' in fund_data.columns:
        rename_map['niq'] = 'net_inc_q'
    if 'atq' in fund_data.columns:
        rename_map['atq'] = 'tot_assets'
    if 'teqq' in fund_data.columns:
        rename_map['teqq'] = 'sh_equity'
    if 'epspiy' in fund_data.columns:
        rename_map['epspiy'] = 'eps_incl_ex'
    if 'ceqq' in fund_data.columns:
        rename_map['ceqq'] = 'com_eq'
    if 'cshoq' in fund_data.columns:
        rename_map['cshoq'] = 'sh_outstanding'
    if 'dvpspq' in fund_data.columns:
        rename_map['dvpspq'] = 'div_per_sh'
    if 'actq' in fund_data.columns:
        rename_map['actq'] = 'cur_assets'
    if 'lctq' in fund_data.columns:
        rename_map['lctq'] = 'cur_liabilities'
    if 'cheq' in fund_data.columns:
        rename_map['cheq'] = 'cash_eq'
    if 'rectq' in fund_data.columns:
        rename_map['rectq'] = 'receivables'
    if 'cogsq' in fund_data.columns:
        rename_map['cogsq'] = 'cogs_q'
    if 'invtq' in fund_data.columns:
        rename_map['invtq'] = 'inventories'
    if 'apq' in fund_data.columns:
        rename_map['apq'] = 'payables'
    if 'dlttq' in fund_data.columns:
        rename_map['dlttq'] = 'long_debt'
    if 'dlcq' in fund_data.columns:
        rename_map['dlcq'] = 'short_debt'
    if 'ltq' in fund_data.columns:
        rename_map['ltq'] = 'tot_liabilities'

    fund_data = fund_data.rename(columns=rename_map)
    return fund_data


def calculate_financial_ratios(fund_data):
    print("Calculating comprehensive financial ratios...")
    # Ensure index resets
    fund_data = fund_data.reset_index(drop=True)

    # Extract series if exist
    date = fund_data['date'].to_frame('date') if 'date' in fund_data.columns else pd.DataFrame({'date': [pd.NaT]*len(fund_data)})
    reportdate = fund_data['reportdate'].to_frame('reportdate') if 'reportdate' in fund_data.columns else pd.DataFrame({'reportdate': [0]*len(fund_data)})
    tic = fund_data['tic'].to_frame('tic') if 'tic' in fund_data.columns else pd.DataFrame({'tic': [None]*len(fund_data)})
    gvkey = fund_data['gvkey'].to_frame('gvkey') if 'gvkey' in fund_data.columns else pd.DataFrame({'gvkey': [None]*len(fund_data)})
    adj_close_q = fund_data['adj_close_q'].to_frame('adj_close_q') if 'adj_close_q' in fund_data.columns else pd.DataFrame({'adj_close_q': [np.nan]*len(fund_data)})
    y_return = fund_data['y_return'].to_frame('y_return') if 'y_return' in fund_data.columns else pd.DataFrame({'y_return': [np.nan]*len(fund_data)})
    gsector = fund_data['gsector'].to_frame('gsector') if 'gsector' in fund_data.columns else pd.DataFrame({'gsector': [0]*len(fund_data)})
    pe = fund_data['pe'].to_frame('pe') if 'pe' in fund_data.columns else pd.DataFrame({'pe': [np.nan]*len(fund_data)})
    ps = fund_data['ps'].to_frame('ps') if 'ps' in fund_data.columns else pd.DataFrame({'ps': [np.nan]*len(fund_data)})
    pb = fund_data['pb'].to_frame('pb') if 'pb' in fund_data.columns else pd.DataFrame({'pb': [np.nan]*len(fund_data)})

    n = fund_data.shape[0]

    # Profitability ratios (simple rolling 3-quarter sums per group)
    print("  Calculating profitability ratios...")
    OPM = pd.Series(np.nan, index=range(n), name='OPM')
    NPM = pd.Series(np.nan, index=range(n), name='NPM')
    ROA = pd.Series(np.nan, index=range(n), name='ROA')
    ROE = pd.Series(np.nan, index=range(n), name='ROE')

    # Group by gvkey and compute for each group
    for gv, group in fund_data.groupby('gvkey'):
        idx = group.index.tolist()
        for pos, i in enumerate(idx):
            # compute rolling 3-quarter sums if available
            start = max(0, pos-2)
            window_idx = idx[start:pos+1]
            if len(window_idx) < 3:
                continue
            window = fund_data.loc[window_idx]
            if set(['op_inc_q','rev_q']).issubset(window.columns):
                try:
                    OPM.iloc[i] = window['op_inc_q'].sum() / window['rev_q'].sum()
                except Exception:
                    OPM.iloc[i] = np.nan
            if set(['net_inc_q','rev_q']).issubset(window.columns):
                try:
                    NPM.iloc[i] = window['net_inc_q'].sum() / window['rev_q'].sum()
                except Exception:
                    NPM.iloc[i] = np.nan
            if 'net_inc_q' in window.columns and 'tot_assets' in fund_data.columns:
                try:
                    ROA.iloc[i] = window['net_inc_q'].sum() / fund_data.loc[i,'tot_assets']
                except Exception:
                    ROA.iloc[i] = np.nan
            if 'net_inc_q' in window.columns and 'sh_equity' in fund_data.columns:
                try:
                    ROE.iloc[i] = window['net_inc_q'].sum() / fund_data.loc[i,'sh_equity']
                except Exception:
                    ROE.iloc[i] = np.nan

    EPS = fund_data['eps_incl_ex'].to_frame('EPS') if 'eps_incl_ex' in fund_data.columns else pd.DataFrame({'EPS': [np.nan]*n})
    BPS = (fund_data['com_eq'] / fund_data['sh_outstanding']).to_frame('BPS') if set(['com_eq','sh_outstanding']).issubset(fund_data.columns) else pd.DataFrame({'BPS': [np.nan]*n})
    DPS = fund_data['div_per_sh'].to_frame('DPS') if 'div_per_sh' in fund_data.columns else pd.DataFrame({'DPS': [np.nan]*n})

    # Liquidity ratios
    print("  Calculating liquidity ratios...")
    cur_ratio = (fund_data['cur_assets'] / fund_data['cur_liabilities']).to_frame('cur_ratio') if set(['cur_assets','cur_liabilities']).issubset(fund_data.columns) else pd.DataFrame({'cur_ratio':[np.nan]*n})
    quick_ratio = ((fund_data['cash_eq'] + fund_data['receivables']) / fund_data['cur_liabilities']).to_frame('quick_ratio') if set(['cash_eq','receivables','cur_liabilities']).issubset(fund_data.columns) else pd.DataFrame({'quick_ratio':[np.nan]*n})
    cash_ratio = (fund_data['cash_eq'] / fund_data['cur_liabilities']).to_frame('cash_ratio') if set(['cash_eq','cur_liabilities']).issubset(fund_data.columns) else pd.DataFrame({'cash_ratio':[np.nan]*n})

    # Efficiency ratios (inventory, receivables, payables turnover)
    print("  Calculating efficiency ratios...")
    inv_turnover = pd.Series(np.nan, index=range(n), name='inv_turnover')
    acc_rec_turnover = pd.Series(np.nan, index=range(n), name='acc_rec_turnover')
    acc_pay_turnover = pd.Series(np.nan, index=range(n), name='acc_pay_turnover')

    for gv, group in fund_data.groupby('gvkey'):
        idx = group.index.tolist()
        for pos, i in enumerate(idx):
            start = max(0, pos-2)
            window_idx = idx[start:pos+1]
            if len(window_idx) < 3:
                continue
            window = fund_data.loc[window_idx]
            if 'cogs_q' in window.columns and 'inventories' in fund_data.columns:
                try:
                    inv_turnover.iloc[i] = window['cogs_q'].sum() / fund_data.loc[i,'inventories']
                except Exception:
                    inv_turnover.iloc[i] = np.nan
            if 'rev_q' in window.columns and 'receivables' in fund_data.columns:
                try:
                    acc_rec_turnover.iloc[i] = window['rev_q'].sum() / fund_data.loc[i,'receivables']
                except Exception:
                    acc_rec_turnover.iloc[i] = np.nan
            if 'cogs_q' in window.columns and 'payables' in fund_data.columns:
                try:
                    acc_pay_turnover.iloc[i] = window['cogs_q'].sum() / fund_data.loc[i,'payables']
                except Exception:
                    acc_pay_turnover.iloc[i] = np.nan

    # Leverage ratios
    print("  Calculating leverage ratios...")
    debt_ratio = (fund_data['tot_liabilities'] / fund_data['tot_assets']).to_frame('debt_ratio') if set(['tot_liabilities','tot_assets']).issubset(fund_data.columns) else pd.DataFrame({'debt_ratio':[np.nan]*n})
    debt_to_equity = (fund_data['tot_liabilities'] / fund_data['sh_equity']).to_frame('debt_to_equity') if set(['tot_liabilities','sh_equity']).issubset(fund_data.columns) else pd.DataFrame({'debt_to_equity':[np.nan]*n})

    # Assemble final ratios
    frames = [date, gvkey, tic, gsector, adj_close_q, y_return,
              OPM.to_frame(), NPM.to_frame(), ROA.to_frame(), ROE.to_frame(), EPS, BPS, DPS,
              cur_ratio, quick_ratio, cash_ratio, inv_turnover.to_frame(), acc_rec_turnover.to_frame(), acc_pay_turnover.to_frame(),
              debt_ratio, debt_to_equity, pe, ps, pb, reportdate]

    ratios = pd.concat([f.reset_index(drop=True) for f in frames], axis=1)
    return ratios


def handle_missing_values(ratios):
    print("Handling missing values...")
    final_ratios = ratios.copy()
    final_ratios = final_ratios.fillna(0)
    final_ratios = final_ratios.replace(np.inf, 0)

    features_column_financial = [
        'OPM', 'NPM', 'ROA', 'ROE', 'EPS', 'BPS', 'DPS', 'cur_ratio',
        'quick_ratio', 'cash_ratio', 'inv_turnover', 'acc_rec_turnover',
        'acc_pay_turnover', 'debt_ratio', 'debt_to_equity', 'pe', 'ps', 'pb'
    ]

    # Remove rows with zero adjusted close price if column exists
    if 'adj_close_q' in final_ratios.columns:
        final_ratios = final_ratios.drop(list(final_ratios[final_ratios.adj_close_q == 0].index)).reset_index(drop=True)

    final_ratios['y_return'] = pd.to_numeric(final_ratios['y_return'], errors='coerce')
    for col in features_column_financial:
        if col in final_ratios.columns:
            final_ratios[col] = pd.to_numeric(final_ratios[col], errors='coerce')

    final_ratios['y_return'].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)
    final_ratios[features_column_financial].replace([np.nan, np.inf, -np.inf], np.nan, inplace=True)

    dropped_col = []
    for col in features_column_financial:
        if col in final_ratios.columns and np.any(~np.isfinite(final_ratios[col])):
            final_ratios.drop(columns=[col], axis=1, inplace=True)
            dropped_col.append(col)

    final_ratios.dropna(axis=0, inplace=True)
    if 'reportdate' in final_ratios.columns:
        final_ratios = final_ratios[final_ratios['reportdate'].ne(0)]
    final_ratios = final_ratios.reset_index(drop=True)

    print(f"Dropped columns: {dropped_col}")
    print(f"Final data shape: {final_ratios.shape}")
    return final_ratios


def save_results(final_ratios, output_dir="outputs", include_sector0=False):
    print("Saving results...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    main_output_file = os.path.join(output_dir, 'final_ratios.csv')
    final_ratios.to_csv(main_output_file, index=False)
    print(f"Main results saved to: {main_output_file}")

    # Save sector-specific files
    sector_count = 0
    if 'gsector' in final_ratios.columns:
        for sec, df_ in list(final_ratios.groupby('gsector')):
            if sec == 0 and not include_sector0:
                continue
            sector_file = os.path.join(output_dir, f"sector{int(sec)}.xlsx")
            df_.to_excel(sector_file, index=False)
            sector_count += 1
    print(f"Total sectors saved: {sector_count}")
    return main_output_file


def parse_args():
    parser = argparse.ArgumentParser(description='Run preprocessing pipeline')
    parser.add_argument('--fundamental', required=True, help='Path to fundamental CSV (quarterly Compustat)')
    parser.add_argument('--price', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'daily.csv'), help='Path to daily price CSV')
    parser.add_argument('--output', default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs'), help='Output directory')
    parser.add_argument('--include-sector0', action='store_true', help='Include sector 0 in sector outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    fundamental_file = args.fundamental
    price_file = args.price
    output_dir = args.output
    include_sector0 = args.include_sector0

    print('='*80)
    print('Running preprocessing pipeline')
    print(f'Fundamental file: {fundamental_file}')
    print(f'Price file: {price_file}')
    print(f'Output dir: {output_dir}')

    fund_df, df_daily_price = load_data(fundamental_file, price_file)
    fund_df = adjust_trade_dates(fund_df)
    fund_df = calculate_adjusted_close(fund_df)
    fund_df = match_tickers_and_gvkey(fund_df, df_daily_price)
    fund_df = calculate_next_quarter_returns(fund_df)
    fund_df = calculate_basic_ratios(fund_df)

    fund_data = select_columns(fund_df)
    ratios = calculate_financial_ratios(fund_data)
    final_ratios = handle_missing_values(ratios)
    output_file = save_results(final_ratios, output_dir, include_sector0)

    print('\nProcessing completed successfully!')
    print(f'Final dataset shape: {final_ratios.shape}')
    print(f'Output saved to: {output_file}')
    print('='*80)


if __name__ == '__main__':
    main()
