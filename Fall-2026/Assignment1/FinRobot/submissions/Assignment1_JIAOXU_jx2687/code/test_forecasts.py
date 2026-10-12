import sys,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(sys.argv.pop()).resolve()/'finrobot_equity/core/src'))
from modules.financial_data_processor import build_data_driven_forecast_config
class ForecastTests(unittest.TestCase):
 def test_known_cagr_and_future_years(self):
  d=pd.DataFrame([dict(metrics='Revenue',**{'2024A':100,'2025A':121,'2026A':144})])
  c=build_data_driven_forecast_config(d)
  self.assertAlmostEqual(c['growth_used'],.2)
  self.assertEqual(list(c['revenue_growth_assumptions']),['2027E','2028E','2029E'])
 def test_clipping_is_disclosed(self):
  d=pd.DataFrame([dict(metrics='Revenue',**{'2024A':100,'2025A':300,'2026A':900})])
  c=build_data_driven_forecast_config(d)
  self.assertAlmostEqual(c['uncapped_growth'],2)
  self.assertEqual(c['growth_used'],.5)
 def test_missing_year_not_silently_used(self):
  with self.assertRaises(ValueError):
   build_data_driven_forecast_config(pd.DataFrame([dict(metrics='Revenue',**{'2023A':100,'2025A':120,'2026A':130})]))
 def test_invalid_revenue_rejected(self):
  with self.assertRaises(ValueError):
   build_data_driven_forecast_config(pd.DataFrame([dict(metrics='Revenue',**{'2024A':100,'2025A':0,'2026A':130})]))
if __name__=='__main__':unittest.main()
