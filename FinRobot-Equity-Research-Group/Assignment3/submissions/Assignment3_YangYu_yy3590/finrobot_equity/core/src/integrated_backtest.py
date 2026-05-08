#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "..", ".."))

from modules.common_utils import DEFAULT_CONFIG_PATH, load_config
from modules.llm_gateway import load_llm_settings
from integrations.active_setup import load_active_research_setup
from integrations.backtest_runner import run_integrated_backtest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FinRL + FinRobot integrated backtest")
    parser.add_argument("--start", required=True, default="2026-01-01")
    parser.add_argument("--end", required=True, default="2026-04-30")
    parser.add_argument("--research-frequency", default="monthly", choices=["monthly", "weekly", "once"])
    parser.add_argument("--finrl-weights", required=True)
    parser.add_argument("--finrobot-run-root", default=None, help="Optional existing FinRobot research root for --research-mode existing.")
    parser.add_argument("--output-dir", default=None, help="Integration output base. Default: ./output/integration")
    parser.add_argument("--integration-run-id", default=None)
    parser.add_argument("--active-research-setup", default=None, help="Path to active research setup JSON registry.")
    parser.add_argument("--research-mode", default="generate-missing", choices=["existing", "generate-missing", "regenerate"])
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--fmp-cache-policy", default="reuse-until-forced", choices=["same-day", "reuse-until-forced"])
    parser.add_argument("--force-refresh-fmp", action="store_true")
    parser.add_argument("--fmp-cache-dir", default=None)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--force-refresh-flags", action="store_true")
    parser.add_argument("--enable-enhanced-news", action="store_true")
    parser.add_argument("--enable-catalyst-analysis", action="store_true")
    parser.add_argument("--news-days-back", type=int, default=5)
    parser.add_argument("--news-limit", type=int, default=50)
    parser.add_argument("--no-op-band-low", type=float, default=0.90)
    parser.add_argument("--no-op-band-high", type=float, default=1.10)
    parser.add_argument("--max-relative-adjustment", type=float, default=0.20)
    parser.add_argument("--defensive-stock-threshold", type=float, default=0.50)
    parser.add_argument("--defensive-cash-threshold", type=float, default=0.20)
    parser.add_argument("--defensive-max-relative-adjustment", type=float, default=0.10)
    parser.add_argument("--stock-boost-cash-funding-limit", type=float, default=0.0)
    parser.add_argument("--stock-boost-etf-funding-limit", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_file = args.config_file or DEFAULT_CONFIG_PATH
    config = load_config(config_file)
    llm_settings = load_llm_settings(config, provider=args.llm_provider, model=args.llm_model)
    active_setup = load_active_research_setup(args.active_research_setup)

    summary = run_integrated_backtest(
        start=args.start,
        end=args.end,
        research_frequency=args.research_frequency,
        finrl_weights=args.finrl_weights,
        finrobot_run_root=args.finrobot_run_root,
        output_dir=args.output_dir,
        llm_settings=llm_settings,
        active_setup=active_setup,
        integration_run_id=args.integration_run_id,
        research_mode=args.research_mode,
        config_file=config_file,
        fmp_cache_policy=args.fmp_cache_policy,
        force_refresh_fmp=args.force_refresh_fmp,
        fmp_cache_dir=args.fmp_cache_dir,
        python_executable=args.python_executable,
        force_refresh_flags=args.force_refresh_flags,
        enable_enhanced_news=args.enable_enhanced_news,
        enable_catalyst_analysis=args.enable_catalyst_analysis,
        news_days_back=args.news_days_back,
        news_limit=args.news_limit,
        no_op_band=(args.no_op_band_low, args.no_op_band_high),
        max_relative_adjustment=args.max_relative_adjustment,
        defensive_stock_threshold=args.defensive_stock_threshold,
        defensive_cash_threshold=args.defensive_cash_threshold,
        defensive_max_relative_adjustment=args.defensive_max_relative_adjustment,
        stock_boost_cash_funding_limit=args.stock_boost_cash_funding_limit,
        stock_boost_etf_funding_limit=args.stock_boost_etf_funding_limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
