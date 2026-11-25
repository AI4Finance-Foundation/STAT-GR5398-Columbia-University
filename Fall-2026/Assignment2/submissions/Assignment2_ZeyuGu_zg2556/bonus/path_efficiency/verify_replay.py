"""Verify a fresh paired refit with byte comparisons of scores and complete execution paths."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path, help='Verification JSON to write')
    args = parser.parse_args()
    delivered = HERE / 'results'
    archived = json.loads((delivered / 'manifest.json').read_text())
    repeat = json.loads((args.replay / 'manifest.json').read_text())
    assert repeat['control_refit_this_invocation'] and repeat['months'] == ['2018-01', '2018-06']
    for key in ('libraries', 'python', 'original_selector_sha256', 'changed_selector_sha256',
                'patch_sha256', 'course_config_sha256', 'course_scores_sha256_before_and_after',
                'formula_checks', 'score_audit', 'months', 'start', 'end'):
        assert archived[key] == repeat[key], key
    verified = {}
    for arm, expected in [('control', HERE.parent / 'results/control'),
                          ('path_efficiency', delivered / 'path_efficiency')]:
        hashes = {}
        for name in ('scores.parquet', 'scores.json', 'equity.csv', 'trades.csv', 'stats.json'):
            hashes[name] = checksum(expected / name)
            assert hashes[name] == checksum(args.replay / arm / name), f'{arm}/{name}'
        verified[arm] = {'files_sha256': hashes,
                         'score_rows': len(pd.read_parquet(expected / 'scores.parquet')),
                         'equity_sessions': len(pd.read_csv(expected / 'equity.csv')),
                         'trades': len(pd.read_csv(expected / 'trades.csv')),
                         'stats': json.loads((expected / 'stats.json').read_text()),
                         'repeat_train_wall_seconds': repeat['arms'][arm]['train_wall_seconds'],
                         'repeat_replay_wall_seconds': repeat['arms'][arm]['replay_wall_seconds']}
    tables = {}
    for name in ('portfolio_comparison.csv', 'monthly_rank_comparison.csv', 'feature_distribution.csv'):
        tables[name] = checksum(delivered / name)
        assert tables[name] == checksum(args.replay / name), name
    record = {'course_base': '43eeda7b011270814fa28cc38e9081e36128cf8a',
              'command_from_submission_folder': 'python bonus/path_efficiency/run.py --source "$SOURCE" '
                                                 '--out "$REPLAY" --refit-control --workers 2',
              'dev_dates': [repeat['start'], repeat['end']], 'simultaneous_workers': repeat['workers_per_arm'],
              'sequential_arms': ['control', 'path_efficiency'], 'months_fitted_per_arm': repeat['score_audit']['distinct_months'],
              'python': repeat['python'], 'libraries': repeat['libraries'], 'matched_arm_files': verified,
              'matched_comparison_files': tables, 'total_files_matched_byte_for_byte': 13,
              'official_score_sha256_before_and_after': repeat['course_scores_sha256_before_and_after']}
    args.record.write_text(json.dumps(record, indent=2) + '\n')
    print('Verified 10 arm files and 3 paired diagnostics byte for byte.')


if __name__ == '__main__':
    main()
