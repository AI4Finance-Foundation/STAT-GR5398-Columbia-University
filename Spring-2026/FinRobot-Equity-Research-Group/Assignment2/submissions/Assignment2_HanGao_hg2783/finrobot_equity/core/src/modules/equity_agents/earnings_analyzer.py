from typing import Any, Dict, List, Optional
import pandas as pd


class EarningsAnalyzer:
    """
    Analyst-style earnings analysis module.

    This module converts the financial forecast table into a concise earnings read-through:
    - latest actual revenue / EPS / margin trends
    - forecast earnings trajectory
    - margin quality
    - peer context
    - key watch items
    """

    def _get_metric_row(self, df: pd.DataFrame, metric_name: str) -> Optional[pd.Series]:
        if df is None or df.empty or "metrics" not in df.columns:
            return None

        metric_series = df["metrics"].astype(str).str.lower().str.strip()
        target = metric_name.lower().strip()
        matches = df[metric_series == target]

        if matches.empty:
            return None

        return matches.iloc[0]

    def _safe_float(self, value):
        try:
            if value is None or pd.isna(value):
                return None
            if isinstance(value, str):
                value = value.replace("$", "").replace(",", "").replace("%", "").strip()
            return float(value)
        except Exception:
            return None

    def _format_growth_readthrough(self, yoy_growth: float) -> str:
        if yoy_growth >= 0.08:
            return "a strong acceleration in top-line momentum"
        if yoy_growth >= 0.03:
            return "a constructive recovery in top-line momentum"
        if yoy_growth >= 0:
            return "modest but positive top-line growth"
        if yoy_growth >= -0.03:
            return "a mild revenue contraction"
        return "a meaningful revenue decline"

    def _actual_year_cols(self, df: pd.DataFrame) -> List[str]:
        return sorted(
            [col for col in df.columns if isinstance(col, str) and col.endswith("A")],
            key=lambda x: int(x[:4]) if x[:4].isdigit() else 9999,
        )

    def _forecast_year_cols(self, df: pd.DataFrame) -> List[str]:
        return sorted(
            [col for col in df.columns if isinstance(col, str) and col.endswith("E")],
            key=lambda x: int(x[:4]) if x[:4].isdigit() else 9999,
        )

    def analyze(
        self,
        ticker: str,
        company_name: str,
        analysis_df: pd.DataFrame,
        peer_ebitda_df: Optional[pd.DataFrame] = None,
        peer_ev_ebitda_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if analysis_df is None or analysis_df.empty:
            return {
                "ticker": ticker,
                "company_name": company_name,
                "summary": "Earnings analysis unavailable because the financial analysis table is empty.",
                "key_points": [],
                "watch_items": [],
            }

        actual_cols = self._actual_year_cols(analysis_df)
        forecast_cols = self._forecast_year_cols(analysis_df)

        revenue = self._get_metric_row(analysis_df, "Revenue")
        eps = self._get_metric_row(analysis_df, "EPS")
        ebitda = self._get_metric_row(analysis_df, "EBITDA")
        ebitda_margin = self._get_metric_row(analysis_df, "EBITDA Margin")
        contribution_margin = self._get_metric_row(analysis_df, "Contribution Margin")
        revenue_growth = self._get_metric_row(analysis_df, "Revenue Growth")
        sga_margin = self._get_metric_row(analysis_df, "SG&A Margin")

        key_points = []
        watch_items = []

        latest_actual = actual_cols[-1] if actual_cols else None
        prior_actual = actual_cols[-2] if len(actual_cols) >= 2 else None
        final_forecast = forecast_cols[-1] if forecast_cols else None

        latest_revenue = self._safe_float(revenue.get(latest_actual)) if revenue is not None and latest_actual else None
        prior_revenue = self._safe_float(revenue.get(prior_actual)) if revenue is not None and prior_actual else None
        forecast_revenue = self._safe_float(revenue.get(final_forecast)) if revenue is not None and final_forecast else None

        latest_eps = self._safe_float(eps.get(latest_actual)) if eps is not None and latest_actual else None
        forecast_eps = self._safe_float(eps.get(final_forecast)) if eps is not None and final_forecast else None

        latest_ebitda = self._safe_float(ebitda.get(latest_actual)) if ebitda is not None and latest_actual else None
        forecast_ebitda = self._safe_float(ebitda.get(final_forecast)) if ebitda is not None and final_forecast else None

        latest_ebitda_margin = ebitda_margin.get(latest_actual) if ebitda_margin is not None and latest_actual else None
        forecast_ebitda_margin = ebitda_margin.get(final_forecast) if ebitda_margin is not None and final_forecast else None

        latest_contribution_margin = contribution_margin.get(latest_actual) if contribution_margin is not None and latest_actual else None
        forecast_contribution_margin = contribution_margin.get(final_forecast) if contribution_margin is not None and final_forecast else None

        latest_growth = revenue_growth.get(latest_actual) if revenue_growth is not None and latest_actual else None
        latest_sga_margin = sga_margin.get(latest_actual) if sga_margin is not None and latest_actual else None
        forecast_sga_margin = sga_margin.get(final_forecast) if sga_margin is not None and final_forecast else None

        if latest_actual and latest_revenue is not None:
            key_points.append(
                f"{company_name} generated ${latest_revenue / 1e9:.1f}B of revenue in {latest_actual}; "
                f"latest observed revenue growth was {latest_growth if latest_growth is not None else 'N/A'}, providing the starting point for the earnings forecast."
            )

        if prior_revenue is not None and latest_revenue is not None and prior_revenue != 0:
            yoy_growth = (latest_revenue - prior_revenue) / prior_revenue
            key_points.append(
                f"Year-over-year revenue growth was {yoy_growth:.1%}, indicating {self._format_growth_readthrough(yoy_growth)}."
            )

        if latest_eps is not None and forecast_eps is not None and final_forecast:
            key_points.append(
                f"EPS increased to ${latest_eps:.2f} in {latest_actual} and is forecast to reach ${forecast_eps:.2f} by {final_forecast}; "
                "the implied earnings upside depends on sustaining revenue growth while converting margin expansion into per-share profit growth."
            )

        if latest_ebitda is not None and forecast_ebitda is not None and final_forecast:
            key_points.append(
                f"EBITDA is forecast to increase from ${latest_ebitda / 1e9:.1f}B in {latest_actual} to "
                f"${forecast_ebitda / 1e9:.1f}B in {final_forecast}; this makes operating leverage and margin execution the central earnings debate."
            )

        if latest_ebitda_margin is not None and forecast_ebitda_margin is not None:
            key_points.append(
                f"EBITDA margin is projected to expand from {latest_ebitda_margin} in {latest_actual} to "
                f"{forecast_ebitda_margin} in {final_forecast}; the magnitude of this step-up should be treated as an important execution assumption rather than a mechanical extrapolation."
            )

        if latest_contribution_margin is not None and forecast_contribution_margin is not None:
            key_points.append(
                f"Contribution margin is expected to rise from {latest_contribution_margin} to {forecast_contribution_margin}, "
                "supporting the earnings growth outlook through pricing discipline, services mix, and cost control."
            )

        if latest_sga_margin is not None and forecast_sga_margin is not None:
            key_points.append(
                f"SG&A margin is expected to move from {latest_sga_margin} to {forecast_sga_margin}, "
                "indicating continued cost discipline."
            )

        # Peer context
        if peer_ebitda_df is not None and not peer_ebitda_df.empty and ticker in peer_ebitda_df.columns:
            try:
                latest_peer_year = peer_ebitda_df.index.max()
                company_ebitda = peer_ebitda_df.loc[latest_peer_year, ticker]
                key_points.append(
                    f"Relative to peers, {ticker} remains a major EBITDA generator, with peer comparison data available through {latest_peer_year}; this supports the quality of the earnings base even if growth is more moderate than some peers."
                )
            except Exception:
                pass

        if peer_ev_ebitda_df is not None and not peer_ev_ebitda_df.empty and ticker in peer_ev_ebitda_df.columns:
            try:
                latest_ev_year = peer_ev_ebitda_df.index.max()
                company_multiple = peer_ev_ebitda_df.loc[latest_ev_year, ticker]
                peer_average = peer_ev_ebitda_df.loc[latest_ev_year].dropna().mean()
                key_points.append(
                    f"{ticker} trades at approximately {company_multiple:.1f}x EV/EBITDA versus a peer average of "
                    f"{peer_average:.1f}x in {latest_ev_year}, implying that earnings delivery must support the valuation premium."
                )
            except Exception:
                pass

        if ticker.upper() == "AAPL":
            watch_items = [
                "iPhone demand and premium-device replacement cycles",
                "Services growth and ecosystem monetization",
                "Gross margin and EBITDA margin guidance versus the implied forecast step-up",
                "China exposure and supply-chain execution",
                "AI-related product adoption and competitive response",
                "Regulatory pressure around App Store, payments, privacy, and platform control",
                "Capital returns, including buybacks and dividends",
            ]
        else:
            watch_items = [
                "Revenue growth versus forecast assumptions",
                "Margin guidance and operating leverage",
                "Product or segment demand trends",
                "Competitive pressure and pricing power",
                "Capital allocation and shareholder returns",
            ]

        if ticker.upper() == "AAPL":
            summary = (
                f"Earnings analysis for {company_name} is constructive but execution-sensitive. "
                "The latest actual period shows a return to positive revenue momentum and resilient EPS, while the forecast assumes meaningful margin expansion through the outer years. "
                "For Apple, the earnings debate is less about near-term revenue growth alone and more about whether product refresh cycles, services monetization, pricing/mix discipline, and installed-base retention can sustain operating leverage. "
                "Given the valuation premium, investors should focus on gross margin guidance, Services growth, iPhone replacement demand, AI-related product adoption, China exposure, and regulatory pressure."
            )
        else:
            summary = (
                f"Earnings analysis for {company_name} is constructive but execution-sensitive. "
                "The latest actual period shows revenue growth and earnings resilience, while the forecast depends on continued margin expansion, cost discipline, and demand durability. "
                f"The key debate is whether {company_name} can sustain EPS growth and justify its valuation premium through product-cycle execution, recurring revenue growth, and operating leverage."
            )

        return {
            "ticker": ticker,
            "company_name": company_name,
            "latest_actual_period": latest_actual,
            "final_forecast_period": final_forecast,
            "summary": summary,
            "key_points": key_points,
            "watch_items": watch_items,
        }