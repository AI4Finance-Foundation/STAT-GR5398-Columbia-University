"""Run a paired six-month selection experiment with the unmodified course trainer."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
START, END = '2018-01', '2018-06'
FIRST, LAST = '2018-01-02', '2018-06-29'
ARMS = ('control', 'six_dates')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def run_arm(arm, source, work, results, cfg, workers):
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'LOKY_MAX_CPU_COUNT', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    # Each arm gets fresh checkpoints, staging paths, and a private backtest output/.
    stage = work / arm
    stage.mkdir()
    (stage / 'selection').symlink_to(source / 'selection', target_is_directory=True)
    output = stage / f'{arm}.parquet'
    with (stage / 'training.log').open('w') as log:
        begin = time.monotonic()
        subprocess.run([sys.executable, str(source / 'selection/build_scores.py'),
                        '--config', str(cfg), '--start', START, '--end', END,
                        '--workers', str(workers), '--out', str(output)], cwd=stage, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        train_seconds = time.monotonic() - begin
    with (stage / 'backtest.log').open('w') as log:
        begin = time.monotonic()
        subprocess.run([sys.executable, str(source / 'run_portfolio.py'), '--scores', str(output),
                        '--start', FIRST, '--end', LAST], cwd=stage, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        replay_seconds = time.monotonic() - begin
    destination = results / arm
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output, destination / 'scores.parquet')
    metadata = read_json(output.with_suffix('.json'))
    metadata['config'] = cfg.name
    (destination / 'scores.json').write_text(json.dumps(metadata, indent=2) + '\n')
    stats_dir = stage / 'output' / f'{FIRST}_{LAST}_{arm}_selection'
    for name in ('stats.json', 'trades.csv', 'equity.csv'):
        shutil.copy2(stats_dir / name, destination / name)
    print(f'{arm}: {metadata["rows"]} scores in {train_seconds:.1f}s', flush=True)
    return {'train_wall_seconds': round(train_seconds, 3), 'replay_wall_seconds': round(replay_seconds, 3),
            'config_sha256': digest(cfg), 'scores_sha256': digest(output)}


def compare(results, course_scores):
    cols = ['month', 'cutoff', 'symbol', 'bucket']
    frames = [pd.read_parquet(results / arm / 'scores.parquet') for arm in ARMS]
    for frame in frames:
        assert len(frame) and not frame.duplicated(cols).any()
        assert frame['month'].nunique() == 6 and (frame['cutoff'] < frame['month']).all()
    pairs = frames[0].merge(frames[1], on=cols, how='outer', validate='1:1',
                            indicator=True, suffixes=('_control', '_six_dates'))
    coverage = pairs.groupby(['month', 'bucket', '_merge'], observed=True).size().unstack(fill_value=0)
    coverage.to_csv(results / 'score_coverage.csv')
    assert (pairs.loc[pairs.bucket == 'growth_tech', '_merge'] == 'both').all(), 'Growth sleeve mismatch'
    released = pd.read_parquet(course_scores, columns=[*cols, 'rank_score'],
                               filters=[('month', '>=', pd.Timestamp(START + '-01')),
                                        ('month', '<=', pd.Timestamp(END + '-01'))])
    pairs = pairs.merge(released.rename(columns={'rank_score': 'rank_score_released'}),
                        how='left', on=cols, validate='1:1')
    rows = []
    for month, group in pairs[pairs.bucket == 'growth_tech'].groupby('month', sort=True):
        scores = group[['rank_score_control', 'rank_score_six_dates']]
        top = [set(group.sort_values([col, 'symbol'], ascending=[False, True]).head(10).symbol)
               for col in scores.columns]
        rows.append({'month': str(month.date()), 'cutoff': str(group['cutoff'].iloc[0].date()),
                     'growth_names': len(group), 'spearman': scores.rank().corr().iloc[0, 1],
                     'shared_top10': len(top[0] & top[1]),
                     'released_matches': group.rank_score_released.notna().sum(),
                     'control_vs_released_spearman': group[['rank_score_control', 'rank_score_released']]
                         .dropna().rank().corr().iloc[0, 1]})
    pd.DataFrame(rows).to_csv(results / 'monthly_rank_comparison.csv', index=False)
    pd.DataFrame([{'arm': arm, **read_json(results / arm / 'stats.json')} for arm in ARMS]).to_csv(
        results / 'portfolio_comparison.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Path to official source_code with course data')
    parser.add_argument('--out', type=Path, required=True, help='Fresh or existing destination for these results')
    parser.add_argument('--workers', type=int, default=2, help='Monthly workers per arm (two arms in parallel)')
    args = parser.parse_args()
    source = args.source.resolve()
    results = args.out.resolve()
    if results.is_relative_to(source / 'data') or args.workers < 1:
        parser.error('Use a results directory outside the course data, and at least one worker.')
    control = source / 'selection/configs/growth.json'
    variant = HERE / 'growth_six_reporting_dates.json'
    expected = read_json(control)
    assert expected['model']['val_quarters'] == 3
    expected['model']['val_quarters'] = 6
    assert read_json(variant) == expected, 'Experimental config must differ in exactly one value'
    course_scores = source / 'data/selection/monthly_scores.parquet'
    original_hash = digest(course_scores)
    results.mkdir(parents=True, exist_ok=True)
    packages = ('numpy', 'pandas', 'scipy', 'pyarrow', 'scikit-learn', 'xgboost', 'lightgbm', 'joblib')
    versions = {p: importlib.metadata.version(p) for p in packages}
    # Fail immediately if the optional libraries are installed but cannot actually be loaded.
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    begin = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='gr5398-selection-') as temp:
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {arm: pool.submit(run_arm, arm, source, Path(temp), results, cfg, args.workers)
                    for arm, cfg in zip(ARMS, [control, variant])}
            try:
                runs = {arm: job.result() for arm, job in jobs.items()}
            except subprocess.CalledProcessError:
                for log in Path(temp).glob('*/*.log'):
                    print(log.name, log.read_text()[-3000:], file=sys.stderr)
                raise
    compare(results, course_scores)
    assert original_hash == digest(course_scores), 'Original course scores changed'
    manifest = {'months': [START, END], 'start': FIRST, 'end': LAST,
                'control_val_quarters': 3, 'variant_val_quarters': 6, 'workers_per_arm': args.workers,
                'course_scores_sha256_before_and_after': original_hash, 'python': platform.python_version(),
                'python_platform': platform.platform(), 'libraries': versions, 'arms': runs,
                'total_wall_seconds': round(time.monotonic() - begin, 3)}
    (results / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
