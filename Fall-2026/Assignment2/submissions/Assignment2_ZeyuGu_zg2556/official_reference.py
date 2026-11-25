"""Run the unchanged official valuation evaluator with horizon labels restricted to each split."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    out = parser.parse_args().destination.resolve()
    source = Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    for split, last in [('dev', '2021-12-31'), ('val', '2023-12-29')]:
        with tempfile.TemporaryDirectory() as scratch:
            src = Path(scratch)/'source_code'
            shutil.copytree(source, src, ignore=shutil.ignore_patterns('data','output','__pycache__'))
            data = src/'data'
            (data/'prices').mkdir(parents=True)
            for item in (source/'data').iterdir():
                if item.name != 'prices':
                    (data/item.name).symlink_to(item, target_is_directory=item.is_dir())
            market = 'market_series.parquet'
            (data/'prices'/market).symlink_to(source/'data/prices'/market)
            daily = 'daily_adjusted.parquet'
            prices = pd.read_parquet(source/'data/prices'/daily)
            prices = prices.loc[pd.to_datetime(prices.date) <= pd.Timestamp(last)]
            prices.to_parquet(data/'prices'/daily, index=False)
            for arm, option in [('student', []), ('trailing_pe', ['--example'])]:
                name = f'valuation_eval_{split}' if arm == 'student' else f'valuation_example_{split}'
                with (out/f'{name}.txt').open('w') as stream:
                    subprocess.run([sys.executable,'evaluate_valuation.py','--period',split,*option],
                                   cwd=src, check=True, stdout=stream)
                shutil.copy2(src/f'output/valuation_eval_{split}.csv', out/f'{name}.csv')


if __name__ == '__main__':
    main()
