## Sensitivity Analysis Summary

### Key Assumptions:
- 2027E Revenue Growth: 5.0%
- 2027E EBITDA Margin: 70.5%
- 2028E Revenue Growth: 6.0%
- 2028E EBITDA Margin: 72.0%
- 2029E Revenue Growth: 4.0%
- 2029E EBITDA Margin: 73.5%

### Assumed Scenario Bands (not calibrated confidence intervals):
- Revenue: $176.5B - $323.4B (normal-theory 95% label; fixed 15% assumed standard deviation)
- EBITDA: $129.7B - $237.7B (normal-theory 95% label; fixed 15% assumed standard deviation)

### Sensitivity Notes:
- revenue: {'range': (-0.05, 0.05), 'steps': 5, 'unit': 'relative change in forecast revenue LEVEL, not growth-rate percentage points'}
- margin: {'range': (-0.02, 0.02), 'steps': 5, 'unit': 'absolute margin fraction change; multiply by 100 for percentage points'}
- combined: {'revenue_range': (-0.05, 0.05), 'margin_range': (-0.02, 0.02), 'steps': 5, 'output': 'EBITDA in billions; not target price'}
- Revenue shocks multiply each forecast revenue level; they do not compound an annual growth-rate shock.
- A margin input of 0.02 is +2 percentage points, not a relative 2% margin change.
- The native combined matrix is rounded EBITDA, not a target-price sensitivity or valuation confidence range.