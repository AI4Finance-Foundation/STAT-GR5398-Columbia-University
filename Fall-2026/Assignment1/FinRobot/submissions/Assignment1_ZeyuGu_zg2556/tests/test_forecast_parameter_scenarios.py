"""Meaningful rejection/definition checks using the actual frozen study inputs."""
import csv
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import socket
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import forecast_parameter_scenarios as scenario


class ForecastScenarioTests(unittest.TestCase):
    def test_scenario_does_not_mutate_default_or_change_origin(self):
        for ticker in scenario.GROWTH:
            original = scenario.read_json(ROOT / 'snapshots' / ticker / 'analysis/forecast_config.json')
            saved = deepcopy(original)
            adjusted = scenario.scenario_config(original, ticker)
            self.assertEqual(original, saved)
            self.assertEqual(adjusted['revenue_base_year'], original['revenue_base_year'])
            self.assertEqual(list(adjusted['revenue_growth_assumptions']), list(original['revenue_growth_assumptions']))
            self.assertEqual(adjusted['sga_margin_change'], 0)
            self.assertEqual(adjusted['margin_improvement'], {'Contribution Margin': 0, 'EBITDA Margin': 0})

    def test_adjusted_forecast_retains_definition_break(self):
        for ticker, expected_margin in [('NVDA', '69.0%'), ('AMD', '37.5%')]:
            source = ROOT / 'snapshots' / ticker / 'analysis'
            raw = scenario.rows(source / 'income_statement_raw_data.csv')
            config = scenario.scenario_config(scenario.read_json(source / 'forecast_config.json'), ticker)
            result = scenario.independent_forecast(raw, config)
            self.assertEqual({row['EBITDA Margin'] for row in result.values()}, {expected_margin})
            latest = max(raw, key=lambda row: row['date'])
            observed = scenario.percent(Decimal(latest['ebitda']) / Decimal(latest['revenue']))
            self.assertNotEqual(observed, expected_margin, 'Zero margin improvement must not be misreported as zero definition discontinuity')

    def test_baseline_cell_audit_rejects_single_falsified_claim(self):
        original = ROOT / 'snapshots/NVDA/analysis/financial_metrics_and_forecasts.csv'
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / 'changed.csv'
            changed.write_bytes(original.read_bytes())
            checks = scenario.compare_csv(original, changed, 'NVDA')
            self.assertEqual(len(checks), 110)
            data = scenario.rows(original)
            data[0]['2029E'] = '1.0'
            scenario.write_csv(changed, data)
            with self.assertRaisesRegex(ValueError, 'changed one or more'):
                scenario.compare_csv(original, changed, 'NVDA')

    def test_rejects_older_forecast_origin_with_newer_actuals(self):
        source = ROOT / 'snapshots/AMD/analysis'
        config = scenario.read_json(source / 'forecast_config.json')
        config['revenue_base_year'] = '2024A'
        with self.assertRaisesRegex(ValueError, 'not the latest'):
            scenario.independent_forecast(scenario.rows(source / 'income_statement_raw_data.csv'), config)

    def test_network_probe_and_unexpected_attempt_are_rejected(self):
        with scenario.OfflineGuard() as guard:
            self.assertEqual(len(guard.probes), 4)
            self.assertEqual(guard.unexpected_attempts, 0)
            with self.assertRaisesRegex(RuntimeError, 'disabled'):
                socket.create_connection(('example.invalid', 443))
            self.assertEqual(guard.unexpected_attempts, 1)

    def test_target_is_conditional_on_retained_ten_percent_ev_debt(self):
        source = ROOT / 'snapshots/AMD'
        result = scenario.read_json(source / 'valuation/engine_valuation.json')['results'][0]
        shares = scenario.read_json(source / 'valuation_inputs.json')['shares_outstanding']
        multiples = [r['enterpriseValueOverEBITDA'] for r in scenario.rows(source / 'analysis/key_metrics_raw_data.csv')]
        calc = scenario.independent_price(result['assumptions']['EBITDA'], multiples, shares)
        self.assertAlmostEqual(float(calc['target_price']), result['target_price'], places=9)
        self.assertEqual(calc['assumed_net_debt'], calc['enterprise_value'] / 10)
        with self.assertRaisesRegex(ValueError, 'positive shares'):
            scenario.independent_price(result['assumptions']['EBITDA'], multiples, 0)


if __name__ == '__main__':
    unittest.main()
