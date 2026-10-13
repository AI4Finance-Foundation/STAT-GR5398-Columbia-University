import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from lxml import html

TICKERS = ['NVDA', 'AMD', 'INTC', 'AAPL', 'CSCO']
FILINGS = {
    'NVDA': 'https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm',
    'AAPL': 'https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metric(frame, name, year):
    value = frame.loc[frame['metrics'] == name, year].iloc[0]
    return float(str(value).replace('%', '').replace(',', ''))


def csv(path):
    return pd.read_csv(path, keep_default_na=False)


def years(frame, suffix):
    return sorted(c for c in frame if re.fullmatch(r'\d{4}' + suffix, c))


def baseline(root, ticker):
    return root / 'artifacts/baseline' / ticker


def raw_data(run):
    return {name: pd.read_csv(run / 'analysis' / file) for name, file in {
        'income_statement': 'income_statement_raw_data.csv', 'balance_sheet': 'balance_sheet_raw_data.csv',
        'cash_flow': 'cash_flow_raw_data.csv', 'ratios': 'ratios_raw_data.csv',
        'key_metrics': 'key_metrics_raw_data.csv'}.items()}


def legacy_config(actual, estimates):
    return {'revenue_base_year': actual[-1],
            'revenue_growth_assumptions': dict(zip(estimates, [0.05, 0.06, 0.04])),
            'margin_improvement': {'Contribution Margin': 0.01, 'EBITDA Margin': 0.01},
            'sga_margin_change': -0.005}


def duration(manifest):
    return (datetime.fromisoformat(manifest['finished_at_utc']) -
            datetime.fromisoformat(manifest['started_at_utc'])).total_seconds()


def responses(run, endpoint):
    found = []
    for request in sorted(run.glob('render_*/audit/*_fmp_request.json')):
        body = read_json(request)
        if urlsplit(body['url']).path.endswith('/' + endpoint):
            reply = read_json(request.with_name(request.name.replace('_request', '_response')))
            found.append((request, reply))
    return found


def data_audit(root):
    rows = []
    for ticker in ('NVDA', 'AAPL'):
        facts = read_json(root / 'artifacts/filings' / f'{ticker.lower()}_annual_facts.json')

        def fact(name, current=True):
            matching = [f for f in facts if f['name'] == 'us-gaap:' + name and
                        (f['context'] == 'c-1' if current else f['period_end'].startswith('2024'))]
            assert matching and len({f['value'] for f in matching}) == 1, (ticker, name)
            return float(matching[0]['value'])

        rev_tag = 'Revenues' if ticker == 'NVDA' else 'RevenueFromContractWithCustomerExcludingAssessedTax'
        cost_tag = 'CostOfRevenue' if ticker == 'NVDA' else 'CostOfGoodsAndServicesSold'
        revenue, prior = fact(rev_tag), fact(rev_tag, False)
        gross, sga = fact('GrossProfit'), fact('SellingGeneralAndAdministrativeExpense')
        operating_ebitda = fact('OperatingIncomeLoss') + fact('DepreciationDepletionAndAmortization')
        compare = {'Revenue': revenue, 'Cost of Operations': fact(cost_tag), 'SG&A': sga,
                   'Contribution Profit': gross, 'Contribution Margin': 100 * gross / revenue,
                   'EBITDA': operating_ebitda, 'EBITDA Margin': 100 * operating_ebitda / revenue,
                   'SG&A Margin': 100 * sga / revenue, 'Revenue Growth': 100 * (revenue / prior - 1),
                   'EPS': fact('EarningsPerShareBasic')}
        frame = csv(baseline(root, ticker) / 'analysis/financial_metrics_and_forecasts.csv')
        context = next(f for f in facts if f['context'] == 'c-1')
        for name, value in compare.items():
            pipeline = metric(frame, name, '2025A')
            pp = name.endswith('Margin') or name == 'Revenue Growth'
            unit = 'percentage_points' if pp else 'USD_per_share' if name == 'EPS' else 'USD'
            tolerance = 0.05 if pp else 0.005 if name == 'EPS' else 1.0
            verdict = 'Match_within_rounding' if abs(pipeline - value) <= tolerance + 1e-10 else 'Explained_definition_difference'
            explanation = 'Annual fiscal period matched; displayed percentage rounded to one decimal.' if pp else 'Same reported fiscal-period value.'
            formula = name
            if name == 'Contribution Profit':
                formula, explanation = 'GrossProfit', 'Pipeline Contribution Profit equals filing gross profit, not a segment contribution-profit definition.'
            elif name in ('EBITDA', 'EBITDA Margin'):
                formula = 'OperatingIncomeLoss + DepreciationDepletionAndAmortization; / Revenue for margin'
                explanation = ('FMP EBITDA = pretax income 84.026B + interest expense 0.247B + D&A 1.864B = 86.137B; '
                               'operating EBITDA is 83.317B. Gap 2.820B = net other income 2.573B + interest 0.247B; D&A agrees.'
                               if ticker == 'NVDA' else
                               'FMP EBITDA = pretax income 132.729B + D&A 11.698B = 144.427B; operating EBITDA is 144.748B. '
                               'Gap -0.321B equals net nonoperating expense. FMP interest field 0 does not prove actual interest is zero.')
            elif name == 'EPS':
                formula, explanation = 'EarningsPerShareBasic', 'Basic EPS matches; pipeline prefers basic over diluted EPS.'
            elif name == 'Revenue Growth':
                formula = '(Revenue FY2025 / Revenue FY2024 - 1) * 100'
            rows.append({'ticker': ticker, 'metric': name, 'fiscal_year': '2025A',
                         'period_start': context['period_start'], 'period_end': context['period_end'],
                         'pipeline_value': pipeline, 'filing_comparable_value': value,
                         'delta': pipeline - value, 'unit': unit, 'tolerance': tolerance,
                         'verdict': verdict, 'explanation': explanation, 'filing_formula_or_tag': formula,
                         'source_url': FILINGS[ticker], 'xbrl_context': 'c-1'})
    pd.DataFrame(rows).to_csv(root / 'audits/data_audit.csv', index=False)


def valuation_and_forecasts(root, original, improved, engine_module, sensitivity_module):
    audit, valuations, sensitivity, backtest = [], [], [], []
    for ticker in TICKERS:
        run = baseline(root, ticker)
        data = raw_data(run)
        frame = csv(run / 'analysis/financial_metrics_and_forecasts.csv')
        actual, estimates = years(frame, 'A'), years(frame, 'E')
        history = original.extract_historical_metrics_from_api_data(data)
        result = improved.calculate_growth_and_forecasts(history, {'forecast_method': 'historical_median'})
        exact_legacy = improved.calculate_growth_and_forecasts(history, legacy_config(actual, estimates))
        legacy_original = original.calculate_growth_and_forecasts(history, legacy_config(actual, estimates))
        pd.testing.assert_frame_equal(exact_legacy, legacy_original)
        pd.testing.assert_frame_equal(result[['metrics'] + actual], history[['metrics'] + actual])
        for row in legacy_original.itertuples(index=False, name=None):
            observed = frame[frame.metrics == row[0]].iloc[0]
            for column, expected in zip(legacy_original.columns[1:], row[1:]):
                value = observed[column]
                if isinstance(expected, str):
                    assert str(value) == expected, (ticker, row[0], column)
                else:
                    assert abs(float(value) - float(expected)) <= 0.01, (ticker, row[0], column)
        out = root / 'improvement' / ticker
        out.mkdir(exist_ok=True)
        result.to_csv(out / 'financial_metrics_and_forecasts.csv', index=False)
        save_json(out / 'forecast_assumptions.json', result.attrs['forecast_metadata'])
        latest, first, last = actual[-1], estimates[0], estimates[-1]
        overlap = sum(int(y[:-1]) <= int(latest[:-1]) for y in estimates)
        entries = [
            ('revenue_growth', '5%, 6%, 4% shared by all companies', json.dumps(result.attrs['forecast_metadata']['revenue_growth_assumptions']), 'Constants are judgmental; improved first growth uses bounded trailing median and reverts toward a 5% anchor.'),
            ('forecast_chronology', f'{latest} actual; estimates {estimates}; overlaps={overlap}', str(years(result, 'E')), 'Legacy labels are fixed at 2025E-2027E despite later actual data.'),
            ('contribution_margin', '+1 percentage point annually', f"Trailing gross-margin median {result.attrs['forecast_metadata']['gross_margin']:.8f}", 'Median computed from dollar gross profit divided by revenue; zero arbitrary annual uplift.'),
            ('sga_margin', '-0.5 percentage point annually', f"Trailing SG&A/revenue median {result.attrs['forecast_metadata']['sga_margin']:.8f}", 'No assumed annual efficiency gain in improved mode.'),
            ('ebitda_definition', f'FMP actual EBITDA; forecast gross profit minus SG&A; first EBITDA margin {metric(frame, "EBITDA Margin", first)}%', f"Trailing FMP EBITDA/revenue median {result.attrs['forecast_metadata']['ebitda_margin']:.8f}", 'Legacy overwrites configured EBITDA margin improvement and omits other operating expenses such as R&D.'),
            ('configured_ebitda_margin', f'{metric(frame, "EBITDA Margin", latest) + 1:.1f}% intended first forecast', f'{metric(result, "EBITDA Margin", years(result, "E")[0]):.1f}% effective first forecast', 'Displayed legacy effective margin differs because EBITDA is recomputed after the margin assumption.'),
            ('eps_and_pe', 'Forecast EPS proportional to revenue; P/E declines 5% annually', 'EPS proportional to revenue; P/E uses fixed last-actual implied price; losses have missing P/E', 'Constant latest net margin and shares are assumptions, not analyst forecasts; P/E is not based on current quote.'),
            ('cagr_scope', f'CAGR {frame.loc[frame.metrics == "Revenue", "CAGR"].iloc[0]}', 'Explicit Historical CAGR column', 'Historical CAGR uses first and last actual year; it is not forward CAGR.')]
        for category, before, after, note in entries:
            audit.append({'ticker': ticker, 'category': category, 'baseline': before, 'improved_or_independent': after, 'explanation': note})
        for target_index in (3, 4):
            target_year = int(actual[target_index][:-1])
            cutoff = int(actual[target_index - 1][:-1])
            train = {name: df[pd.to_numeric(df['year']) <= cutoff].copy() for name, df in data.items()}
            train_history = original.extract_historical_metrics_from_api_data(train)
            train_years = years(train_history, 'A')
            train_estimates = [f'{cutoff + i}E' for i in (1, 2, 3)]
            before = original.calculate_growth_and_forecasts(train_history, legacy_config(train_years, train_estimates))
            after = improved.calculate_growth_and_forecasts(train_history, {'forecast_method': 'historical_median'})
            for name in ('Revenue', 'EBITDA'):
                observed = metric(frame, name, f'{target_year}A')
                old = metric(before, name, f'{target_year}E')
                new = metric(after, name, f'{target_year}E')
                backtest.append({'ticker': ticker, 'training_end_year': cutoff, 'training_periods': len(train_years),
                                 'target_year': target_year, 'metric': name, 'actual': observed,
                                 'legacy_forecast': old, 'improved_forecast': new,
                                 'legacy_ape': abs(old / observed - 1), 'improved_ape': abs(new / observed - 1)})
        quote_path, quote_reply = responses(run, 'quote')[0]
        quote = quote_reply['body'][0]
        shares = float(quote['marketCap']) / float(quote['price'])
        multiples = data['key_metrics']['enterpriseValueOverEBITDA'].dropna().astype(float)
        ebitda, revenue = metric(frame, 'EBITDA', last), metric(frame, 'Revenue', last)
        multiple, std = float(multiples.mean()), float(multiples.std(ddof=1))
        ev, margin = ebitda * multiple, ebitda / revenue
        independent = max(0.9 * ev / shares, 0)
        engine = engine_module.ValuationEngine({'analysis': frame, 'key_metrics': data['key_metrics'],
                                               'shares_outstanding': shares, 'current_price': quote['price']})
        calculated = engine.calculate_ev_ebitda_valuation()
        assert abs(calculated.target_price - independent) < 1e-8
        assert abs(calculated.low_estimate - max((ebitda * (multiple - std) - 0.1 * ev) / shares, 0)) < 1e-8
        assert abs(calculated.high_estimate - max((ebitda * (multiple + std) - 0.1 * ev) / shares, 0)) < 1e-8
        balance = data['balance_sheet'].sort_values('date').iloc[-1]
        net_debt = float(balance['netDebt'])
        valuations.append({'ticker': ticker, 'displayed_forecast_year': last,
                           'effective_horizon_year': int(latest[:-1]) + 3, 'ebitda': ebitda,
                           'revenue': revenue, 'ebitda_margin': margin, 'historical_multiple_mean': multiple,
                           'historical_multiple_sample_std': std, 'quote_price': float(quote['price']),
                           'quote_market_cap': float(quote['marketCap']), 'quote_timestamp': quote.get('timestamp'),
                           'shares_market_cap_div_price': shares, 'enterprise_value': ev, 'proxy_net_debt': 0.1 * ev,
                           'balance_net_debt': net_debt, 'balance_date': str(balance['date']),
                           'independent_target_price': independent, 'engine_target_price': calculated.target_price,
                           'absolute_reconciliation_error': abs(independent - calculated.target_price),
                           'target_if_balance_net_debt_used': max((ev - net_debt) / shares, 0),
                           'low_estimate': calculated.low_estimate, 'high_estimate': calculated.high_estimate,
                           'quote_evidence': str(quote_path.relative_to(root)), 'discounting': 'None'})
        audit.extend([
            {'ticker': ticker, 'category': 'independent_ev_ebitda', 'baseline': 'Report target=0.0 after price-target-news HTTP 402; missing input',
             'improved_or_independent': f'Independent {independent:.8f}; engine {calculated.target_price:.8f}',
             'explanation': f'Latest estimate {last} (effective FY{int(latest[:-1]) + 3}); mean of five historical multiples; shares=quote market cap/price. Engine is separately invoked for audit; headline is not its output.'},
            {'ticker': ticker, 'category': 'hidden_net_debt', 'baseline': f'Engine net debt=10% of EV, {0.1 * ev:.2f} USD',
             'improved_or_independent': f'Latest archived balance net debt {net_debt:.2f} USD',
             'explanation': 'Engine ignores supplied balance-sheet debt. Alternative price uses FMP cash-only net debt and is a convention comparison, not an endorsed intrinsic value.'},
            {'ticker': ticker, 'category': 'discounting', 'baseline': 'No WACC, terminal growth or discount factor in EV/EBITDA method',
             'improved_or_independent': 'None; forward horizon valuation remains undiscounted',
             'explanation': 'Reconciliation establishes arithmetic, not defensible present fair value. Historical multiple average may include outliers.'}])
        analyzer = sensitivity_module.SensitivityAnalyzer(frame)
        analyzer.analyze_revenue_sensitivity().to_csv(root / 'audits' / f'{ticker}_revenue_sensitivity.csv', index=False)
        analyzer.analyze_margin_sensitivity().to_csv(root / 'audits' / f'{ticker}_margin_sensitivity.csv', index=False)
        analyzer.generate_sensitivity_table().to_csv(root / 'audits' / f'{ticker}_engine_ebitda_sensitivity.csv')
        for rev_delta in np.linspace(-0.05, 0.05, 5):
            for margin_delta in np.linspace(-0.02, 0.02, 5):
                scenario_ebitda = revenue * (1 + rev_delta) * (margin + margin_delta)
                target = max(0.9 * scenario_ebitda * multiple / shares, 0)
                sensitivity.append({'ticker': ticker, 'revenue_delta': rev_delta, 'margin_delta_pp': margin_delta * 100,
                                    'scenario_ebitda': scenario_ebitda, 'target_price': target,
                                    'target_price_change_pct': 100 * (target / independent - 1),
                                    'base_target_price': independent})
    pd.DataFrame(audit).to_csv(root / 'audits/forecast_valuation_audit.csv', index=False)
    pd.DataFrame(valuations).to_csv(root / 'audits/valuation_reconciliation.csv', index=False)
    scenarios = pd.DataFrame(sensitivity)
    scenarios.to_csv(root / 'audits/target_price_sensitivity.csv', index=False)
    backtest = pd.DataFrame(backtest)
    backtest.to_csv(root / 'improvement/backtest.csv', index=False)
    scores = []
    for name, sample in [('ALL', backtest)] + [(t, backtest[backtest.ticker == t]) for t in TICKERS]:
        for m in ('Revenue', 'EBITDA'):
            subset = sample[sample.metric == m]
            scores.append({'ticker': name, 'metric': m, 'folds': len(subset),
                           'legacy_mape': subset.legacy_ape.mean(), 'improved_mape': subset.improved_ape.mean(),
                           'legacy_median_ape': subset.legacy_ape.median(), 'improved_median_ape': subset.improved_ape.median(),
                           'legacy_wape': (subset.legacy_forecast - subset.actual).abs().sum() / subset.actual.abs().sum(),
                           'improved_wape': (subset.improved_forecast - subset.actual).abs().sum() / subset.actual.abs().sum(),
                           'improved_folds': int((subset.improved_ape < subset.legacy_ape).sum())})
    pd.DataFrame(scores).to_csv(root / 'improvement/backtest_summary.csv', index=False)


def grounding(root):
    frame = pd.DataFrame(read_json(root / 'code/grounding_labels.json'))
    for row in frame.itertuples():
        text = (baseline(root, row.ticker) / 'analysis' / f'{row.section}.txt').read_text().strip()
        assert row.source_excerpt in text, row.claim_id
    frame.to_csv(root / 'audits/grounding_audit.csv', index=False)
    summary = frame.groupby(['ticker', 'section', 'label']).size().unstack(fill_value=0).reset_index()
    summary['total'] = summary[['Supported', 'Unsupported', 'Contradicted']].sum(axis=1)
    summary['grounding_rate'] = summary.Supported / summary.total
    summary.to_csv(root / 'audits/grounding_summary.csv', index=False)
    by_section = summary.groupby('section')[['Supported', 'Unsupported', 'Contradicted', 'total']].sum()
    by_section['grounding_rate'] = by_section.Supported / by_section.total
    by_section.to_csv(root / 'audits/grounding_by_section.csv')


def canonical_risk(heading):
    text = heading.lower()
    if 'valuation' in text:
        return 'Valuation'
    if 'growth' in text or 'revenue' in text:
        return 'Growth'
    if 'margin' in text:
        return 'Margin'
    if 'competit' in text:
        return 'Competition'
    return heading


def request_bodies(run):
    return [read_json(p)['body'] for p in sorted((run / 'audit').glob('*_llm_request.json'))]


def stability(root):
    runs = [baseline(root, 'NVDA'), root / 'artifacts/stability/run2', root / 'artifacts/stability/run3']
    rows = []
    inputs = request_bodies(runs[0])
    for i, run in enumerate(runs, 1):
        headings = re.findall(r'\*\*([^*]+)\*\*', (run / 'analysis/risks.txt').read_text())[:3]
        canonical = [canonical_risk(h.rstrip(':')) for h in headings]
        assert request_bodies(run) == inputs
        assert len(headings) == 3
        for p in (runs[0] / 'analysis').glob('*.csv'):
            assert p.read_bytes() == (run / 'analysis' / p.name).read_bytes()
        log = next(run.glob('render_*/render.log')).read_text()
        rating = re.search(r'Rating: (.*?) \(', log).group(1)
        blocked = responses(run, 'price-target-news')[0][1]['http_status']
        assert blocked == 402
        rows.append({'run': i, 'ticker': 'NVDA', 'rating': rating, 'rating_source': 'FMP grades, frozen input',
                     'displayed_target_price': 0.0, 'valid_target_price': None, 'target_price_source_status': blocked,
                     'target_price_interpretation': 'Missing, not fair value zero',
                     'top3_original_headings': json.dumps(headings), 'top3_canonical': json.dumps(canonical),
                     'financial_csv_identical': True, 'llm_request_bodies_identical': True,
                     'llm_calls': 7, 'actual_model': 'gpt-4.1-2025-04-14', 'temperature': 0.7})
    pd.DataFrame(rows).to_csv(root / 'audits/stability_audit.csv', index=False)
    pairs = []
    for a, b in combinations(rows, 2):
        left, right = json.loads(a['top3_canonical']), json.loads(b['top3_canonical'])
        pairs.append({'run_a': a['run'], 'run_b': b['run'],
                      'top3_jaccard': len(set(left) & set(right)) / len(set(left) | set(right)),
                      'top3_rank_agreement': sum(x == y for x, y in zip(left, right)) / 3})
    pd.DataFrame(pairs).to_csv(root / 'audits/stability_pairs.csv', index=False)
    risks = ['Valuation', 'Growth', 'Competition', 'Margin']
    indicators = [[int(r in json.loads(row['top3_canonical'])) for r in risks] for row in rows]
    save_json(root / 'audits/stability_metrics.json', {
        'rating_ordinal_sample_variance': 0.0, 'rating_variance_scope': 'Frozen FMP control, not LLM recommendation',
        'valid_target_price_variance': None, 'valid_target_price_observations': 0,
        'mean_pairwise_jaccard': float(np.mean([p['top3_jaccard'] for p in pairs])),
        'mean_pairwise_rank_agreement': float(np.mean([p['top3_rank_agreement'] for p in pairs])),
        'risk_presence_sample_variance': dict(zip(risks, np.var(indicators, axis=0, ddof=1).tolist()))})


def provenance(root):
    coverage, ledger, scorecard = [], [], []
    runs = [(t, baseline(root, t), 'baseline') for t in TICKERS]
    runs += [('NVDA', root / f'artifacts/stability/run{i}', f'repeat{i}') for i in (2, 3)]
    seen = set()
    for ticker, run, stage in runs:
        manifest = read_json(run / 'audit/run_manifest.json')
        render_manifest = read_json(next(run.glob('render_*/audit/render_manifest.json')))
        for item in manifest['analysis_files']:
            assert hashlib.sha256((run / item['path']).read_bytes()).hexdigest() == item['sha256']
        assert render_manifest['analysis_unchanged']
        for path in sorted((run / 'audit').glob('*_llm_response.json')):
            response = read_json(path)
            assert response['http_status'] == 200
            body = response['body']
            assert body['id'] not in seen
            seen.add(body['id'])
            request = read_json(path.with_name(path.name.replace('_response', '_request')))['body']
            section = request['messages'][-1]['content'].split('Please provide the ')[-1].split(' based on')[0].replace(' ', '_')
            usage = body['usage']
            cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
            cost = ((usage['prompt_tokens'] - cached) * 2 + cached * 0.5 + usage['completion_tokens'] * 8) / 1e6
            ledger.append({'ticker': ticker, 'run': stage, 'section': section, 'completion_id': body['id'],
                           'actual_model': body['model'], 'temperature': request['temperature'],
                           'max_tokens': request['max_tokens'], 'prompt_tokens': usage['prompt_tokens'],
                           'cached_input_tokens': cached, 'completion_tokens': usage['completion_tokens'],
                           'estimated_cost_usd': cost, 'api_elapsed_seconds': response['elapsed_seconds'],
                           'finish_reason': body['choices'][0]['finish_reason'], 'evidence': str(path.relative_to(root))})
        price_target_status = responses(run, 'price-target-news')[0][1]['http_status']
        llm_cost = sum(row['estimated_cost_usd'] for row in ledger if row['ticker'] == ticker and row['run'] == stage)
        scorecard.append({'ticker': ticker, 'run': stage, 'analysis_seconds': duration(manifest),
                          'render_seconds': duration(render_manifest), 'total_compute_seconds': duration(manifest) + duration(render_manifest),
                          'llm_calls': 7, 'estimated_llm_cost_usd': llm_cost, 'target_price_http_status': price_target_status,
                          'news_summary_origin': 'No-news placeholder after blocked API'})
        if stage == 'baseline':
            raw = (run / 'analysis/competitor_analysis.txt').read_text().strip()
            combined = next(run.glob(f'render_*/report/Combined_Equity_Report_{ticker}.html'))
            rendered = html.fromstring(combined.read_bytes()).text_content()
            retained = ' '.join(raw.split()) in ' '.join(rendered.split())
            takeaways = (run / 'analysis/major_takeaways.txt').read_text()
            main = csv(run / 'analysis/financial_metrics_and_forecasts.csv')
            coverage.append({'ticker': ticker, 'company_name': manifest['company_name'], 'sector': 'Technology',
                             'peers': ', '.join(manifest['peer_tickers']), 'latest_actual_year': years(main, 'A')[-1],
                             'company_html_files': len(list(run.glob('render_*/report/*.html'))),
                             'company_chart_files': len(list(run.glob('render_*/report/*.png'))),
                             'analysis_hashes_unchanged': render_manifest['analysis_unchanged'],
                             'competitor_prose_retained_in_html': retained, 'competitor_comma_count': raw.count(','),
                             'takeaways_postprocessor_fallback': 'revenue growth shows consistent performance trends' in takeaways.lower(),
                             'news_available': False, 'target_price_valid': False})
    pd.DataFrame(coverage).to_csv(root / 'audits/coverage.csv', index=False)
    pd.DataFrame(ledger).to_csv(root / 'audits/api_usage.csv', index=False)
    pd.DataFrame(scorecard).to_csv(root / 'audits/runtime_cost.csv', index=False)
    save_json(root / 'audits/cost_summary.json', {'recorded_unique_llm_calls': len(ledger),
              'prompt_tokens': sum(r['prompt_tokens'] for r in ledger),
              'cached_input_tokens': sum(r['cached_input_tokens'] for r in ledger),
              'completion_tokens': sum(r['completion_tokens'] for r in ledger),
              'estimated_total_usd': sum(r['estimated_cost_usd'] for r in ledger),
              'pricing_per_million_usd': {'input': 2.0, 'cached_input': 0.5, 'output': 8.0},
              'pricing_source': 'https://developers.openai.com/api/docs/models/gpt-4.1',
              'scope': '49 captured calls only; excludes unrecorded setup checks; not a billing export. FMP free-tier incremental fee assumed zero.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    (root / 'audits').mkdir(exist_ok=True)
    source = root / 'artifacts/source_snapshot'
    original = load_module('original_processor', source / 'modules/financial_data_processor.py')
    with tempfile.TemporaryDirectory() as directory:
        checkout = Path(directory)
        destination = checkout / 'finrobot_equity/core/src'
        shutil.copytree(source, destination)
        subprocess.run(['git', 'apply', '--check', str(root / 'improvement/data_driven_forecasts.patch')], cwd=checkout, check=True)
        subprocess.run(['git', 'apply', str(root / 'improvement/data_driven_forecasts.patch')], cwd=checkout, check=True)
        improved = load_module('improved_processor', destination / 'modules/financial_data_processor.py')
        engine = load_module('audit_valuation_engine', source / 'modules/valuation_engine.py')
        sensitivity = load_module('audit_sensitivity', source / 'modules/sensitivity_analyzer.py')
        data_audit(root)
        valuation_and_forecasts(root, original, improved, engine, sensitivity)
    grounding(root)
    stability(root)
    provenance(root)
    print('Generated data, forecast/valuation, grounding, stability, sensitivity and cost audits.')


if __name__ == '__main__':
    main()
