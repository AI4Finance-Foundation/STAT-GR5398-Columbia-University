#!/usr/bin/env bash
# Run the finrobot_equity pipeline end to end for one company.
#
# Usage (from the root of your FinRobot clone):
#   ./run_pipeline.sh NVDA "NVIDIA Corporation" AMD INTC
#
# Loop it over a coverage universe:
#   ./run_pipeline.sh AMD "Advanced Micro Devices" NVDA INTC
#   ./run_pipeline.sh INTC "Intel Corporation" NVDA AMD

set -euo pipefail

TICKER="${1:?usage: run_pipeline.sh TICKER \"Company Name\" [PEER ...]}"
NAME="${2:?usage: run_pipeline.sh TICKER \"Company Name\" [PEER ...]}"
shift 2
# Optional peers; kept empty-safe for bash 3.2 (macOS default) under `set -u`.
PEER_ARGS=()
if [ "$#" -gt 0 ]; then
    PEER_ARGS=(--peer-tickers "$@")
fi

CONFIG="finrobot_equity/core/config/config.ini"
SRC="finrobot_equity/core/src"
ANALYSIS="output/${TICKER}/analysis"

echo "=== Step 1: financial analysis + AI text sections -> ${ANALYSIS}"
python "${SRC}/generate_financial_analysis.py" \
    --company-ticker "${TICKER}" \
    --company-name "${NAME}" \
    --config-file "${CONFIG}" \
    ${PEER_ARGS[@]+"${PEER_ARGS[@]}"} \
    --generate-text-sections \
    --enable-sensitivity-analysis \
    --enable-catalyst-analysis \
    --enable-enhanced-news

echo "=== Step 2: charts + HTML report -> output/${TICKER}/report"
python "${SRC}/create_equity_report.py" \
    --company-ticker "${TICKER}" \
    --company-name "${NAME}" \
    --analysis-csv "${ANALYSIS}/financial_metrics_and_forecasts.csv" \
    --ratios-csv "${ANALYSIS}/ratios_raw_data.csv" \
    --tagline-file "${ANALYSIS}/tagline.txt" \
    --company-overview-file "${ANALYSIS}/company_overview.txt" \
    --investment-overview-file "${ANALYSIS}/investment_overview.txt" \
    --valuation-overview-file "${ANALYSIS}/valuation_overview.txt" \
    --risks-file "${ANALYSIS}/risks.txt" \
    --competitor-analysis-file "${ANALYSIS}/competitor_analysis.txt" \
    --major-takeaways-file "${ANALYSIS}/major_takeaways.txt" \
    --peer-ev-ebitda-csv "${ANALYSIS}/peer_ev_ebitda_comparison.csv" \
    --enable-text-regeneration \
    --config-file "${CONFIG}"

echo "=== Done: output/${TICKER}/"
