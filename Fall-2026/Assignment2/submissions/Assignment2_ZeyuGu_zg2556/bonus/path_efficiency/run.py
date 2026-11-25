"""Pair an original selection model with the same model trained on one extra price-path feature."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from inspect_inputs import inspect_inputs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from retrain import digest, read_json, run_arm


def audit_formula(source):
    sys.path.insert(0, str(source))
    from selection.selector import _trend_efficiency
    rows = []
    dates = pd.period_range('2015Q1', periods=5, freq='Q').end_time.normalize()
    test_cases = {'rising': [1, 2, 4, 8, 16], 'falling': [16, 8, 4, 2, 1],
                  'round_trip': [1, 2, 1, 2, 1], 'flat': [3] * 5,
                  'bad_price': [1, 2, 0, 2, 1], 'gap': [1, 2, 4, 8, 16]}
    for label, path in test_cases.items():
        used_dates = list(dates)
        if label == 'gap':
            used_dates[-1] += pd.DateOffset(months=3)
        rows.extend((label, d, p) for d, p in zip(used_dates, path))
    df = pd.DataFrame(rows, columns=['tic', 'datadate', 'trade_price'])
    feature = _trend_efficiency(df)
    assert np.isfinite(feature.values).all() and feature.abs().max() <= 1
    expected = [1, -1, 0, 0, 0, 0]
    assert np.allclose(feature.groupby(df.tic, sort=False).last(), expected)
    assert (feature[df.groupby('tic').cumcount() < 4] == 0).all()
    # Appending one future report to every ticker leaves all earlier feature values unchanged.
    extra = df.groupby('tic', sort=False).tail(1).copy()
    extra['datadate'] += pd.DateOffset(months=3)
    extra['trade_price'] = 10000
    extended = pd.concat([df, extra], ignore_index=True).sort_values(['tic', 'datadate'])
    assert np.allclose(_trend_efficiency(extended).reindex(df.index), feature)
    return {'monotone_up': 1, 'monotone_down': -1, 'round_trip': 0, 'flat': 0, 'nonpositive': 0,
            'missing_quarter': 0, 'four_return_minimum_checked': True, 'future_append_invariant': True}


def compare(control, modified, out):
    keys = ['month', 'cutoff', 'symbol', 'bucket']
    frames = [pd.read_parquet(folder / 'scores.parquet') for folder in (control, modified)]
    pair = frames[0].merge(frames[1], how='outer', on=keys, suffixes=('_base', '_new'),
                           validate='one_to_one', indicator=True)
    assert pair._merge.eq('both').all(), 'Model inputs changed the scored universe'
    assert (pair.cutoff < pair.month).all() and pair.month.nunique() == 6
    assert ((pair.month.min(), pair.month.max()) == (pd.Timestamp('2018-01-01'), pd.Timestamp('2018-06-01')))
    comparisons = []
    for (month, cutoff), group in pair[pair.bucket == 'growth_tech'].groupby(['month', 'cutoff']):
        columns = ['rank_score_base', 'rank_score_new']
        top = [set(group.sort_values([col, 'symbol'], ascending=[False, True]).head(10).symbol) for col in columns]
        comparisons.append({'month': str(month.date()), 'cutoff': str(cutoff.date()), 'growth_names': len(group),
                            'spearman': group[columns].rank().corr().iloc[0, 1], 'shared_top10': len(top[0] & top[1])})
    pd.DataFrame(comparisons).to_csv(out / 'monthly_rank_comparison.csv', index=False)
    stats = [{'arm': name, **read_json(folder / 'stats.json')}
             for name, folder in [('control', control), ('path_efficiency', modified)]]
    pd.DataFrame(stats).to_csv(out / 'portfolio_comparison.csv', index=False)
    return {'matched_score_rows': len(pair), 'matched_growth_rows': sum(m['growth_names'] for m in comparisons),
            'distinct_months': len(comparisons)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path, help='Official populated course source_code directory')
    ap.add_argument('--out', required=True, type=Path, help='Result directory outside course data')
    ap.add_argument('--workers', type=int, choices=[1, 2], default=2)
    controls = ap.add_mutually_exclusive_group()
    controls.add_argument('--refit-control', action='store_true', help='Also rebuild the three-date control from scratch')
    controls.add_argument('--control-results', type=Path, help='A fresh original control/ from bonus/retrain.py')
    args = ap.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    for entry in (HERE / 'course_files.sha256').read_text().splitlines():
        expected, name = entry.split('  ', 1)
        if digest(source / name) != expected:
            ap.error(f'Pinned course file differs: {name}')
    control_artifacts = HERE.parent / 'results'
    if out.is_relative_to(source) or out.is_relative_to(control_artifacts):
        ap.error('Choose an output outside course source and the original bonus results')
    out.mkdir(parents=True, exist_ok=True)
    old = read_json(control_artifacts / 'manifest.json')
    cfg = source / 'selection/configs/growth.json'
    public_scores = source / 'data/selection/monthly_scores.parquet'
    assert digest(cfg) == old['arms']['control']['config_sha256'] and read_json(cfg)['model']['val_quarters'] == 3
    assert digest(public_scores) == old['course_scores_sha256_before_and_after']
    control_info = read_json(control_artifacts / 'control/scores.json')
    assert digest(source / 'data/fundamentals.db') == control_info['fundamentals_sha256']
    assert digest(source / 'data/prices/daily_adjusted.parquet') == control_info['prices_sha256']
    assert digest(control_artifacts / 'control/scores.parquet') == old['arms']['control']['scores_sha256']
    versions = {name: importlib.metadata.version(name) for name in old['libraries']}
    # Reusing the control requires the exact environment of its original fits.
    if not args.refit_control and (versions != old['libraries'] or platform.python_version() != old['python']):
        ap.error('Environment differs from the archived control. Use --refit-control for a fresh paired experiment.')
    control = control_artifacts / 'control'
    control_run = old['arms']['control']
    if args.control_results is not None:
        external = args.control_results.resolve()
        if out.is_relative_to(external):
            ap.error('Keep new outputs separate from the supplied control directory')
        external_manifest = read_json(external.parent / 'manifest.json')
        for key in ('months', 'start', 'end', 'libraries', 'python', 'course_scores_sha256_before_and_after'):
            if old[key] != external_manifest[key]:
                ap.error(f'External control manifest differs on {key}')
        for name in ('scores.parquet', 'scores.json', 'equity.csv', 'stats.json', 'trades.csv'):
            if digest(external / name) != digest(control / name):
                ap.error(f'External control differs from pristine original fits on {name}')
        control = external
        control_run = external_manifest['arms']['control']
        for key in ('config_sha256', 'scores_sha256'):
            if control_run[key] != old['arms']['control'][key]:
                ap.error(f'External control manifest differs on {key}')
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    with tempfile.TemporaryDirectory(prefix='gr5398-path-efficiency-') as tmp:
        work = Path(tmp)
        isolated = work / 'patched'
        shutil.copytree(source, isolated, ignore=shutil.ignore_patterns('data', 'output', 'student_data',
                                                                         '__pycache__', '*.pyc', '.git'))
        (isolated / 'data').symlink_to(source / 'data', target_is_directory=True)
        (isolated / 'student_data').mkdir()
        selector = source / 'selection/selector.py'
        patch = HERE / 'selector.patch'
        original = digest(selector)
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=isolated, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=isolated, check=True)
        assert digest(isolated / 'selection/selector.py') != original
        formula_checks = audit_formula(isolated)
        runs = {'control': control_run}
        if args.refit_control:
            runs['control'] = run_arm('control', source, work, out, cfg, args.workers)
            control = out / 'control'
        runs['path_efficiency'] = run_arm('path_efficiency', isolated, work, out, cfg, args.workers)
        audit = compare(control, out / 'path_efficiency', out)
        inspect_inputs(cfg, out / 'path_efficiency/scores.parquet', out / 'feature_distribution.csv')
        manifest = {'original_selector_sha256': original, 'changed_selector_sha256': digest(isolated / 'selection/selector.py'),
                    'patch_sha256': digest(patch), 'course_scores_sha256_before_and_after': digest(public_scores),
                    'control_refit_this_invocation': args.refit_control,
                    'external_control_verified': args.control_results is not None, 'python': platform.python_version(),
                    'libraries': versions, 'arms': runs, 'formula_checks': formula_checks, 'score_audit': audit,
                    'workers_per_arm': args.workers, 'course_config_sha256': digest(cfg), 'months': old['months'],
                    'start': old['start'], 'end': old['end']}
        assert digest(selector) == original and digest(public_scores) == old['course_scores_sha256_before_and_after']
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
