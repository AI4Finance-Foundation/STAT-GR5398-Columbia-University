"""Run a separate, offline forecast-assumption experiment on frozen NVDA/AMD.

This is a forecast audit, not an Option A treatment, backtest, or accuracy claim.
The patched upstream processor and EV/EBITDA engine execute unchanged. Decimal
arithmetic independently checks the forecasts and valuation from saved inputs.
No credentials, HTTP client, model SDK, or writer/verifier is used.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SHA = '2717499b8e30f242640af08c4ad9afd1113c2d45'
GROWTH = {'NVDA': [.30, .20, .10], 'AMD': [.15, .10, .08]}
TABLES = ('income_statement', 'balance_sheet', 'cash_flow', 'ratios', 'key_metrics')
MODULES = ('financial_data_processor', 'valuation_engine')
# Bound to the real baseline files in the saved snapshot manifests. Git can
# change CRLF to LF when applying the compatibility patch; no other source
# difference is accepted. Data, prompts and responses are never normalized.
BASELINE_SOURCE = {
    'financial_data_processor': ('be237216f95c028827c0ad745c5ae32f5743fd1658c02abca28d571a882141bd',
                                 'd38f85d0f6e47cb288fde64140e7a6f0bbb86c432ab25431fac928ada1eaf81a'),
    'valuation_engine': ('decce50ea666b7b1b2236efe737e0e3c4814b876764ebe8407dea4cf7100745c',
                         '3253ee53248935c7ae7fd8ac9ff18a2e760a58913d72e1c29e5b541ab13081cd'),
}
D = Decimal


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, records):
    if not records:
        raise ValueError('Refusing an empty audit table')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(module, raw, manifest_digest):
    baseline_raw, baseline_lf = BASELINE_SOURCE[module]
    if manifest_digest != baseline_raw:
        raise ValueError('The snapshot does not bind the expected baseline source: ' + module)
    actual = hashlib.sha256(raw).hexdigest()
    normalized = hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()
    if actual != baseline_raw and normalized != baseline_lf:
        raise ValueError('Processor/valuation code differs beyond CRLF-to-LF: ' + module)
    return {'actual_sha256': actual, 'baseline_raw_sha256': baseline_raw,
            'crlf_to_lf_sha256': normalized, 'comparison': 'raw bytes' if actual == baseline_raw else 'CRLF-to-LF only'}


def immutable_bindings(root):
    """Record unchanged data and Option A artifacts; do not include new outputs."""
    paths = set()
    for relative in ('snapshots', 'runs', 'audits/grounding', 'evidence/verifier_versions'):
        paths.update(p for p in (root / relative).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update((root / 'config').glob('*pairs*.json'))
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(paths)}


class OfflineGuard:
    """Reject network attempts, DNS and child-process escape in this process."""
    def __init__(self):
        self.probes = []
        self.unexpected_attempts = 0
        self.probing = False
        self.stack = ExitStack()

    def reject(self, *args, **kwargs):
        if not self.probing:
            self.unexpected_attempts += 1
        raise RuntimeError('Network and subprocess execution are disabled in this offline experiment')

    def __enter__(self):
        for obj, name in ((socket, 'create_connection'), (socket, 'getaddrinfo'),
                          (socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                          (socket.socket, 'sendto'), (subprocess, 'Popen')):
            self.stack.enter_context(patch.object(obj, name, self.reject))
        self.probing = True
        probes = [('socket.create_connection', lambda: socket.create_connection(('example.invalid', 443))),
                  ('socket.getaddrinfo', lambda: socket.getaddrinfo('example.invalid', 443)),
                  ('socket.connect', self._connect_probe),
                  ('subprocess.Popen', lambda: subprocess.Popen([sys.executable, '-V']))]
        try:
            for label, probe in probes:
                try:
                    probe()
                except RuntimeError as exc:
                    self.probes.append({'probe': label, 'result': 'REJECTED_BEFORE_IO', 'reason': str(exc)})
                else:
                    raise AssertionError('Offline guard probe unexpectedly succeeded: ' + label)
        finally:
            self.probing = False
        return self

    @staticmethod
    def _connect_probe():
        with socket.socket() as client:
            client.connect(('127.0.0.1', 9))

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def compare_csv(expected, observed, ticker):
    """Require every literal CSV field, including percentages, to match."""
    with expected.open(encoding='utf-8-sig', newline='') as a, observed.open(encoding='utf-8-sig', newline='') as b:
        left, right = list(csv.reader(a)), list(csv.reader(b))
    if not left or left[0] != right[0] or len(left) != len(right):
        raise ValueError(f'{ticker}: default replay CSV dimensions/headers changed')
    result = []
    for row_number, (old, new) in enumerate(zip(left[1:], right[1:]), 2):
        if len(old) != len(new):
            raise ValueError(f'{ticker}: default replay row width changed')
        for column, saved, replayed in zip(left[0], old, new):
            result.append({'ticker': ticker, 'csv_row_1based': row_number, 'metric': old[0],
                           'column': column, 'saved_value': saved, 'replay_value': replayed,
                           'verdict': 'MATCH' if saved == replayed else 'MISMATCH'})
    if any(r['verdict'] != 'MATCH' for r in result):
        raise ValueError(f'{ticker}: default replay changed one or more CSV fields')
    return result


def scenario_config(default, ticker):
    config = deepcopy(default)
    periods = sorted(config['revenue_growth_assumptions'])
    if len(periods) != 3:
        raise ValueError('Exactly three annual forecast periods are required')
    config['revenue_growth_assumptions'] = dict(zip(periods, GROWTH[ticker]))
    config['margin_improvement'] = {'Contribution Margin': 0, 'EBITDA Margin': 0}
    config['sga_margin_change'] = 0
    return config


def percent(value):
    return str((value * 100).quantize(D('.1'), rounding=ROUND_HALF_EVEN)) + '%'


def money(value):
    return value.quantize(D('.01'), rounding=ROUND_HALF_EVEN)


def independent_forecast(raw_income, config):
    """No pandas or upstream call: reconstruct the formula from raw statements.

    The original processor rounds historical ratios to 0.1 percentage point
    before using them. Preserve that definition, rather than silently replacing
    it with full-precision ratios or a new earnings model.
    """
    latest = max(raw_income, key=lambda row: row['date'])
    if config['revenue_base_year'] != latest['year'] + 'A':
        raise ValueError('Forecast origin is not the latest frozen annual observation')
    with localcontext() as ctx:
        ctx.prec = 40
        revenue = D(latest['revenue'])
        contribution = D(percent((revenue - D(latest['costOfRevenue'])) / revenue).rstrip('%')) / 100
        sga_margin = D(percent(D(latest['sellingGeneralAndAdministrativeExpenses']) / revenue).rstrip('%')) / 100
        eps = D(latest['eps'])
        values = {}
        for period, growth in sorted(config['revenue_growth_assumptions'].items()):
            revenue *= 1 + D(str(growth))
            displayed_revenue = money(revenue)
            contribution += D(str(config['margin_improvement']['Contribution Margin']))
            sga_margin += D(str(config['sga_margin_change']))
            used_contribution = D(percent(contribution).rstrip('%')) / 100
            contribution_profit = displayed_revenue * used_contribution
            sga = money(displayed_revenue * sga_margin)
            ebitda = contribution_profit - sga
            eps *= (1 + D(str(growth))) * (1 + D(str(config['margin_improvement']['EBITDA Margin'])))
            values[period] = {'Revenue': money(revenue), 'Contribution Profit': money(contribution_profit),
                              'Cost of Operations': money(displayed_revenue - contribution_profit),
                              'SG&A': sga, 'EBITDA': money(ebitda),
                              'Contribution Margin': percent(contribution), 'SG&A Margin': percent(sga_margin),
                              'EBITDA Margin': percent(ebitda / displayed_revenue),
                              'Revenue Growth': percent(D(str(growth))), 'EPS': money(eps)}
        return values


def independent_price(ebitda, multiples, shares):
    """Independent Decimal mean and EV-to-equity bridge, without engine helpers."""
    with localcontext() as ctx:
        ctx.prec = 40
        values = [D(str(v)) for v in multiples]
        if not values or any(not v.is_finite() for v in values) or D(str(shares)) <= 0:
            raise ValueError('Observed finite multiples and positive shares required')
        multiple = sum(values) / len(values)
        enterprise = D(str(ebitda)) * multiple
        assumed_debt = enterprise / 10
        return {'target_multiple': multiple, 'enterprise_value': enterprise,
                'assumed_net_debt': assumed_debt, 'equity_value': enterprise - assumed_debt,
                'target_price': max((enterprise - assumed_debt) / D(str(shares)), D(0))}


def check_forecast(ticker, scenario, dataframe, independent):
    table = dataframe.set_index('metrics')
    result = []
    for period, expected_values in independent.items():
        for metric, expected in expected_values.items():
            actual = table.at[metric, period]
            if isinstance(expected, str):
                passed = str(actual) == expected
                delta = None
            else:
                delta = D(str(actual)) - expected
                passed = abs(delta) <= D('.005')
            result.append({'ticker': ticker, 'scenario': scenario, 'period': period, 'metric': metric,
                           'processor_value': str(actual), 'independent_value': str(expected),
                           'difference': str(delta) if delta is not None else '',
                           'tolerance': 'exact formatted percent' if isinstance(expected, str) else 'absolute USD 0.005 (float versus Decimal)',
                           'verdict': 'MATCH' if passed else 'MISMATCH'})
    if any(r['verdict'] != 'MATCH' for r in result):
        raise ValueError(f'{ticker} {scenario}: independent forecast arithmetic failed')
    return result


def execute(upstream, output):
    import pandas as pd
    from modules.financial_data_processor import extract_historical_metrics_from_api_data, calculate_growth_and_forecasts
    from modules.valuation_engine import ValuationEngine

    all_checks, baseline_checks, comparison, prices, caveats, runtime_rows = [], [], [], [], [], []
    for ticker in GROWTH:
        snapshot = ROOT / 'snapshots' / ticker
        source = snapshot / 'analysis'
        raw_tables = {name: pd.read_csv(source / f'{name}_raw_data.csv', float_precision='round_trip') for name in TABLES}
        historical = extract_historical_metrics_from_api_data(raw_tables)
        if historical is None or historical.empty:
            raise ValueError('Actual upstream extraction produced no history')
        default_config = read_json(source / 'forecast_config.json')
        inputs = read_json(snapshot / 'valuation_inputs.json')
        baseline_engine = next(x for x in read_json(snapshot / 'valuation/engine_valuation.json')['results'] if x['method'] == 'EV/EBITDA')
        fixed_period = baseline_engine['assumptions']['EBITDA Period']
        raw_income = rows(source / 'income_statement_raw_data.csv')
        raw_multiple_rows = rows(source / 'key_metrics_raw_data.csv')
        multiples = [r['enterpriseValueOverEBITDA'] for r in raw_multiple_rows]
        scenario_tables, scenario_values = {}, {}
        for name, config in [('default_replay', default_config), ('analyst_deceleration', scenario_config(default_config, ticker))]:
            started = time.perf_counter()
            folder = output / ticker / name
            folder.mkdir(parents=True)
            write_json(folder / 'forecast_config.json', config)
            historical.to_csv(folder / 'historical_metrics.csv', index=False)
            forecast = calculate_growth_and_forecasts(historical.copy(deep=True), deepcopy(config))
            csv_path = folder / 'financial_metrics_and_forecasts.csv'
            forecast.to_csv(csv_path, index=False)
            if name == 'default_replay':
                baseline_checks.extend(compare_csv(source / csv_path.name, csv_path, ticker))
            independent = independent_forecast(raw_income, config)
            checks = check_forecast(ticker, name, forecast, independent)
            all_checks.extend(checks)
            write_csv(folder / 'independent_forecast_check.csv', checks)
            financial = dict(inputs, analysis=forecast, key_metrics=raw_tables['key_metrics'])
            valuation = asdict(ValuationEngine(financial).calculate_ev_ebitda_valuation(inputs.get('target_multiple')))
            if valuation['assumptions']['EBITDA Period'] != fixed_period:
                raise ValueError('Scenario changed the original selected valuation period')
            if 'target_multiple' in inputs:
                raise ValueError('This independent mean-based audit requires the observed default multiple, not an override')
            price_check = independent_price(independent[fixed_period]['EBITDA'], multiples, inputs['shares_outstanding'])
            write_json(folder / 'engine_ev_ebitda.json', valuation)
            write_json(folder / 'independent_ev_ebitda.json', {'period': fixed_period,
                'formula': 'EV = EBITDA * mean(saved historical multiples); debt = 0.1 * EV; target = max((EV-debt)/saved shares, 0)',
                'shares': inputs['shares_outstanding'], 'historical_multiples': multiples,
                'values_decimal': {key: str(value) for key, value in price_check.items()}})
            for field, engine_name in [('target_multiple', 'Target Multiple'), ('enterprise_value', 'Enterprise Value'),
                                       ('assumed_net_debt', 'Net Debt'), ('equity_value', 'Equity Value'), ('target_price', None)]:
                actual = valuation['target_price'] if engine_name is None else valuation['assumptions'][engine_name]
                expected = float(price_check[field])
                passed = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-8)
                all_checks.append({'ticker': ticker, 'scenario': name, 'period': fixed_period, 'metric': 'EV/EBITDA.' + field,
                    'processor_value': str(actual), 'independent_value': str(price_check[field]),
                    'difference': str(actual - expected), 'tolerance': 'relative 1e-12 or absolute 1e-8', 'verdict': 'MATCH' if passed else 'MISMATCH'})
                if not passed:
                    raise ValueError('Independent valuation reconciliation failed')
            if name == 'default_replay' and not math.isclose(valuation['target_price'], baseline_engine['target_price'], rel_tol=1e-12):
                raise ValueError('Default target price no longer matches the original engine result')
            scenario_tables[name] = forecast.set_index('metrics')
            scenario_values[name] = valuation
            runtime_rows.append({'ticker': ticker, 'scenario': name, 'runtime_seconds': time.perf_counter() - started,
                                 'api_calls': 0, 'tokens': 0, 'api_cost_usd': 0})
        original, adjusted = scenario_tables['default_replay'], scenario_tables['analyst_deceleration']
        for period in sorted(default_config['revenue_growth_assumptions']):
            record = {'ticker': ticker, 'period': period, 'period_type': 'forecast, annual issuer period', 'currency': 'USD',
                      'default_growth_pct': default_config['revenue_growth_assumptions'][period] * 100,
                      'adjusted_growth_pct': scenario_config(default_config, ticker)['revenue_growth_assumptions'][period] * 100}
            for metric in ('Revenue', 'Contribution Profit', 'Contribution Margin', 'SG&A', 'SG&A Margin', 'EBITDA', 'EBITDA Margin'):
                old, new = original.at[metric, period], adjusted.at[metric, period]
                is_margin = metric.endswith('Margin')
                a = float(str(old).rstrip('%')); b = float(str(new).rstrip('%'))
                slug = metric.lower().replace('&', '').replace(' ', '_')
                record[slug + '_default'] = a
                record[slug + '_adjusted'] = b
                record[slug + ('_delta_pp' if is_margin else '_delta_usd')] = b - a
                if not is_margin:
                    record[slug + '_delta_pct'] = (b / a - 1) * 100
            comparison.append(record)
        before, after = scenario_values['default_replay'], scenario_values['analyst_deceleration']
        revenue_ratio = float(adjusted.at['Revenue', fixed_period]) / float(original.at['Revenue', fixed_period])
        margin_ratio = ((after['assumptions']['EBITDA'] / float(adjusted.at['Revenue', fixed_period])) /
                        (before['assumptions']['EBITDA'] / float(original.at['Revenue', fixed_period])))
        target_ratio = after['target_price'] / before['target_price']
        if not math.isclose(revenue_ratio * margin_ratio, target_ratio, rel_tol=1e-12):
            raise ValueError('Revenue-times-margin target decomposition failed')
        prices.append({'ticker': ticker, 'method': 'EV/EBITDA', 'valuation_period': fixed_period,
                       'annual_growth_periods_feeding_selected_year': '|'.join(p for p in sorted(default_config['revenue_growth_assumptions']) if p <= fixed_period),
                       'default_ebitda_usd': before['assumptions']['EBITDA'], 'adjusted_ebitda_usd': after['assumptions']['EBITDA'],
                       'fixed_target_multiple': before['assumptions']['Target Multiple'], 'fixed_shares': inputs['shares_outstanding'],
                       'fixed_current_price_usd': inputs['current_price'], 'net_debt_rule': '10% of each scenario EV; amount is not fixed debt',
                       'default_assumed_debt_usd': before['assumptions']['Net Debt'], 'adjusted_assumed_debt_usd': after['assumptions']['Net Debt'],
                       'default_target_usd': before['target_price'], 'adjusted_target_usd': after['target_price'],
                       'target_delta_usd': after['target_price'] - before['target_price'],
                       'target_delta_pct': (after['target_price'] / before['target_price'] - 1) * 100,
                       'terminal_revenue_ratio': revenue_ratio, 'terminal_ebitda_margin_ratio': margin_ratio,
                       'target_ratio': target_ratio, 'revenue_times_margin_ratio': revenue_ratio * margin_ratio,
                       'decomposition': 'target ratio = terminal revenue ratio * terminal EBITDA margin ratio, with unchanged shares/multiple/10%-EV debt rule; algebraic identity, not extra experiments',
                       'discount_rate': 'not used: undiscounted multiple method', 'terminal_growth': 'not used: multiple method',
                       'interpretation': 'Conditional scenario only; no accuracy test; no change to Option A reports or prompts'})
        latest = max(raw_income, key=lambda r: r['date'])
        actual_revenue, actual_ebitda = float(latest['revenue']), float(latest['ebitda'])
        gross_less_sga = actual_revenue - float(latest['costOfRevenue']) - float(latest['sellingGeneralAndAdministrativeExpenses'])
        caveats.append({'ticker': ticker, 'actual_period': latest['year'] + 'A', 'actual_period_end': latest['date'],
                        'fiscal_year': latest.get('fiscalYear', ''), 'actual_provider_ebitda_usd': actual_ebitda,
                        'actual_provider_ebitda_margin_pct': actual_ebitda / actual_revenue * 100,
                        'gross_profit_less_sga_usd': gross_less_sga, 'definition_gap_usd': gross_less_sga - actual_ebitda,
                        'reported_rd_expense_usd': float(latest['researchAndDevelopmentExpenses']),
                        'reported_da_usd': float(latest['depreciationAndAmortization']),
                        'adjusted_forecast_ebitda_margin_pct': float(str(adjusted.at['EBITDA Margin', fixed_period]).rstrip('%')),
                        'ratio_precision': 'Uses historical ratios rounded to 0.1 percentage point, as upstream does',
                        'definition_warning': 'Forecast EBITDA = gross contribution profit - SG&A. R&D and other operating expenses are not deducted; D&A is not explicitly reconciled. Gap is not attributed solely to R&D. Zero margin improvement does not repair this discontinuity.',
                        'unused_setting': 'ebitda_growth_factor is present but unused; EBITDA Margin input influences EPS, then its EBITDA margin is overwritten by derived gross profit less SG&A'})
    write_csv(output / 'baseline_cell_reconciliation.csv', baseline_checks)
    write_csv(output / 'independent_arithmetic_reconciliation.csv', all_checks)
    write_csv(output / 'forecast_comparison.csv', comparison)
    write_csv(output / 'target_price_comparison.csv', prices)
    write_csv(output / 'definition_caveats.csv', caveats)
    write_csv(output / 'runtime_cost.csv', runtime_rows)
    return {'baseline_fields_compared': len(baseline_checks), 'independent_checks': len(all_checks),
            'forecast_rows': len(comparison), 'valuation_rows': len(prices), 'all_checks_passed': True,
            'target_price_comparison': prices, 'runtime_cost': runtime_rows,
            'pandas_version': pd.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are rejected')
    args = parser.parse_args()
    started = time.perf_counter()
    upstream, output = args.upstream.resolve(), args.output.resolve()
    if any(output == ROOT / folder or (ROOT / folder) in output.parents for folder in ('snapshots', 'runs', 'config', 'audits/grounding', 'evidence/verifier_versions')):
        raise ValueError('Output must not be inside preserved experiment directories')
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=upstream, check=True, capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_SHA:
        raise ValueError('Upstream checkout is not the fixed commit')
    source_hashes, source_comparison, input_hashes = {}, {}, {}
    for ticker in GROWTH:
        snapshot = ROOT / 'snapshots' / ticker
        manifest = read_json(snapshot / 'manifest.json')
        for module in MODULES:
            relative = 'finrobot_equity/core/src/modules/' + module + '.py'
            source_hashes[relative] = digest(upstream / relative)
            source_comparison[relative] = validate_source(module, (upstream / relative).read_bytes(), manifest['source_sha256'][relative])
        inputs = [snapshot / 'manifest.json', snapshot / 'valuation_inputs.json', snapshot / 'valuation/engine_valuation.json',
                  snapshot / 'analysis/forecast_config.json', snapshot / 'analysis/financial_metrics_and_forecasts.csv']
        inputs += [snapshot / 'analysis' / f'{name}_raw_data.csv' for name in TABLES]
        for path in inputs:
            sha = digest(path)
            relative = path.relative_to(snapshot).as_posix()
            if relative != 'manifest.json' and sha != manifest['file_hashes'][relative]:
                raise ValueError('Frozen snapshot input has changed: ' + relative)
            input_hashes[path.relative_to(ROOT).as_posix()] = sha
    preserved = immutable_bindings(ROOT)
    write_json(output / 'preserved_experiment_hashes.json', preserved)
    write_json(output / 'input_hashes.json', input_hashes)
    sys.path.insert(0, str(upstream / 'finrobot_equity/core/src'))
    log = io.StringIO()
    with redirect_stdout(log), redirect_stderr(log), OfflineGuard() as guard:
        result = execute(upstream, output)
    (output / 'execution.log').write_text(log.getvalue(), encoding='utf-8')
    if guard.unexpected_attempts:
        raise AssertionError('Unexpected network/subprocess attempt occurred')
    after = immutable_bindings(ROOT)
    if preserved != after:
        raise AssertionError('The preserved snapshots or Option A experiment files changed during the run')
    write_json(output / 'network_guard.json', {'probes': guard.probes, 'unexpected_attempts': guard.unexpected_attempts,
        'new_model_calls': 0, 'limitation': 'Python socket/DNS/subprocess rejection, not an operating-system sandbox. Only processor and valuation modules execute; no model SDK imported.'})
    output_hashes = {path.relative_to(output).as_posix(): digest(path) for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = {'status': 'PASS', 'completed_utc': datetime.now(timezone.utc).isoformat(), 'upstream_commit': head,
        'upstream_path': str(upstream), 'submission_path': str(ROOT), 'command_argv': [sys.executable, *sys.argv],
        'command_windows': subprocess.list2cmdline([sys.executable, *sys.argv]),
        'python': sys.version, 'platform': platform.platform(), 'processor_source_hashes': source_hashes,
        'source_comparison': source_comparison,
        'driver_sha256': digest(Path(__file__)), 'input_hashes': input_hashes,
        'preserved_experiment_files': len(preserved), 'preserved_experiment_hashes_file': 'preserved_experiment_hashes.json',
        'original_data_and_verifier_experiment_unchanged': True,
        'new_api_calls': 0, 'new_model_calls': 0, 'tokens': 0, 'api_cost_usd': 0,
        'runtime_seconds': time.perf_counter() - started, 'network_disabled': True,
        'forecast_scope': 'Separate analyst-selected deceleration scenarios on frozen annual FMP inputs; no guidance claim, fitting, backtest, earnings-definition repair, writer regeneration or Option A change.',
        'gross_and_sga_ratios': 'Held at the upstream rounded historical ratios in the adjusted scenarios.',
        'cost_limitations': 'Local CPU time only; no new API or model usage.',
        'output_hashes': output_hashes, **result}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'status': 'PASS', 'output': str(output), 'baseline_fields': result['baseline_fields_compared'],
                      'independent_checks': result['independent_checks'], 'preserved_files': len(preserved),
                      'api_cost_usd': 0, 'target_prices': result['target_price_comparison']}, indent=2))


if __name__ == '__main__':
    main()
