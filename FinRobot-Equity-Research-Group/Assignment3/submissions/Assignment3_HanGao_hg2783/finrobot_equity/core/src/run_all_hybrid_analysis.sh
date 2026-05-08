#!/bin/bash

TICKERS=(
  NVDA
  MSFT
  AAPL
  AMZN
  GOOGL
  META
  AVGO
  AMD
  CRM
  ORCL
  ADBE
  NFLX
  TSLA
  COST
  NOW
  PANW
  CRWD
  SNOW
  INTU
  CSCO
  QCOM
  TXN
  MU
  AMAT
  LRCX
  KLAC
  ANET
  PLTR
  IBM
  INTC
)

NAMES=(
  "NVIDIA Corporation"
  "Microsoft Corporation"
  "Apple Inc."
  "Amazon.com Inc."
  "Alphabet Inc."
  "Meta Platforms Inc."
  "Broadcom Inc."
  "Advanced Micro Devices Inc."
  "Salesforce Inc."
  "Oracle Corporation"
  "Adobe Inc."
  "Netflix Inc."
  "Tesla Inc."
  "Costco Wholesale Corporation"
  "ServiceNow Inc."
  "Palo Alto Networks Inc."
  "CrowdStrike Holdings Inc."
  "Snowflake Inc."
  "Intuit Inc."
  "Cisco Systems Inc."
  "QUALCOMM Incorporated"
  "Texas Instruments Incorporated"
  "Micron Technology Inc."
  "Applied Materials Inc."
  "Lam Research Corporation"
  "KLA Corporation"
  "Arista Networks Inc."
  "Palantir Technologies Inc."
  "International Business Machines Corporation"
  "Intel Corporation"
)

for i in "${!TICKERS[@]}"; do
  TICKER=${TICKERS[$i]}
  NAME=${NAMES[$i]}

  echo "=========================================="
  echo "Running hybrid analysis for $TICKER - $NAME"
  echo "=========================================="

  python generate_financial_analysis.py \
    --company-ticker "$TICKER" \
    --company-name "$NAME" \
    --generate-text-sections \
    --enable-sec-filing-analysis \
    --enable-revenue-driver-analysis \
    --output-dir "./output/$TICKER/hybrid_analysis"

  echo "Finished $TICKER"
  echo ""
done