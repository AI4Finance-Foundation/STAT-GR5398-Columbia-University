import tempfile
import unittest
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from evaluate import load_module


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkout = Path(cls.directory.name)
        target = cls.checkout / 'finrobot_equity/core/src'
        shutil.copytree(cls.root / 'artifacts/source_snapshot', target)
        subprocess.run(['git', 'apply', str(cls.root / 'improvement/data_driven_forecasts.patch')],
                       cwd=cls.checkout, check=True)
        cls.processor = load_module('tested_forecast', target / 'modules/financial_data_processor.py')

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def history(self, revenues=(100, 200, 400), eps=2.0):
        names = ['Revenue', 'Contribution Profit', 'SG&A', 'EBITDA', 'EPS', 'PE Ratio']
        values = {'metrics': names}
        for year, revenue in zip(range(2021, 2024), revenues):
            values[f'{year}A'] = [revenue, 0.6 * revenue, 0.1 * revenue, 0.3 * revenue, eps, 20.0]
        return pd.DataFrame(values).astype(object)

    def forecast(self, frame, **settings):
        return self.processor.calculate_growth_and_forecasts(frame, {'forecast_method': 'historical_median', **settings})

    def test_dynamic_years_and_history_preservation(self):
        frame = self.history()
        result = self.forecast(frame)
        self.assertEqual([c for c in result if c.endswith('E')], ['2024E', '2025E', '2026E'])
        pd.testing.assert_frame_equal(result[frame.columns], frame)

    def test_growth_clipping_and_reversion(self):
        assumptions = self.forecast(self.history()).attrs['forecast_metadata']
        self.assertEqual(list(assumptions['revenue_growth_assumptions'].values()), [0.5, 0.275, 0.1625])
        declining = self.forecast(self.history((400, 200, 100))).attrs['forecast_metadata']
        self.assertAlmostEqual(declining['revenue_growth_assumptions']['2024E'], -0.25)

    def test_forecast_ebitda_uses_historical_definition(self):
        result = self.forecast(self.history())
        for year in ['2024E', '2025E', '2026E']:
            revenue = result.loc[result.metrics == 'Revenue', year].iloc[0]
            ebitda = result.loc[result.metrics == 'EBITDA', year].iloc[0]
            self.assertLessEqual(abs(ebitda - 0.3 * revenue), 0.01)

    def test_zero_and_negative_eps_have_no_pe(self):
        for eps in (0, -1):
            result = self.forecast(self.history(eps=eps))
            self.assertTrue(result.loc[result.metrics == 'PE Ratio', '2024E'].isna().all())

    def test_too_few_periods_and_gaps(self):
        with self.assertRaises(ValueError):
            self.forecast(self.history().drop(columns='2021A'))
        with self.assertRaises(ValueError):
            self.forecast(self.history().rename(columns={'2022A': '2020A'}))

    def test_bad_revenue_and_missing_metrics(self):
        for value in (0, -1, np.nan):
            frame = self.history()
            frame.loc[frame.metrics == 'Revenue', '2022A'] = value
            with self.assertRaises(ValueError):
                self.forecast(frame)
        with self.assertRaises(ValueError):
            self.forecast(self.history().query("metrics != 'EBITDA'"))

    def test_duplicate_metrics(self):
        frame = self.history()
        with self.assertRaises(ValueError):
            self.forecast(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))

    def test_invalid_configuration(self):
        for config in ({'growth_floor': -1}, {'growth_anchor': 0.6}, {'growth_decay': 1.1},
                       {'projection_years': 0}, {'lookback': 0}):
            with self.assertRaises(ValueError):
                self.forecast(self.history(), **config)


if __name__ == '__main__':
    unittest.main()
