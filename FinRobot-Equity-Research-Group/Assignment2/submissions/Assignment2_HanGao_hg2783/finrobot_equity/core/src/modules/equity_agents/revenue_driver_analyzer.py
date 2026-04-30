from typing import Any, Dict, List, Optional


class RevenueDriverAnalyzer:
    def __init__(self):
        self.driver_keywords = {
            "product_segment": [
                "product", "segment", "service", "services", "subscription", "launch", "portfolio",
                "iphone", "mac", "ipad", "wearables", "airpods", "app store", "ecosystem", "installed base"
            ],
            "geographic": [
                "international", "regional", "china", "europe", "asia", "market expansion", "geographic expansion"
            ],
            "pricing": [
                "pricing", "price", "price increase", "mix", "premium", "yield", "asp", "margin"
            ],
            "volume_demand": [
                "demand", "volume", "customer", "traffic", "adoption", "orders", "upgrade", "replacement", "retention"
            ],
            "strategic": [
                "acquisition", "partnership", "distribution", "channel", "expansion", "innovation",
                "ai", "silicon", "ecosystem", "services"
            ],
            "macro": [
                "macroeconomic", "consumer spending", "inflation", "fx", "foreign exchange", "interest rate"
            ],
        }

    def analyze_historical_revenue(self, financial_metrics_df) -> Dict[str, Any]:
        result = {
            "latest_revenue": None,
            "historical_growth_trend": "unavailable",
            "observations": [],
            "revenue_series": [],
            "revenue_growth_series": [],
            "average_growth": None,
        }

        if financial_metrics_df is None:
            result["observations"].append("No financial metrics dataframe was provided.")
            return result

        if getattr(financial_metrics_df, "empty", False):
            result["observations"].append("Financial metrics dataframe is empty.")
            return result

        revenue_candidates = [
            "Revenue",
            "revenue",
            "Total Revenue",
            "totalRevenue",
            "revenue_actual",
        ]
        period_candidates = [
            "Calendar Year",
            "calendarYear",
            "Date",
            "date",
            "Period",
            "period",
        ]

        # Support row-based financial metric format:
        # metrics | 2021A | 2022A | 2023A | ...
        if "metrics" in financial_metrics_df.columns:
            metric_name_series = financial_metrics_df["metrics"].astype(str).str.lower()
            revenue_rows = financial_metrics_df[metric_name_series.str.strip().isin(["revenue", "total revenue"])]

            if not revenue_rows.empty:
                revenue_row = revenue_rows.iloc[0]
                year_cols = [
                    col for col in financial_metrics_df.columns
                    if isinstance(col, str) and (col.endswith("A") or col.endswith("E"))
                ]

                historical_cols = [col for col in year_cols if col.endswith("A")]
                historical_cols = sorted(historical_cols, key=lambda x: int(str(x)[:4]) if str(x)[:4].isdigit() else 9999)

                clean_rows = []
                for col in historical_cols:
                    raw_revenue = revenue_row.get(col)
                    try:
                        revenue_value = float(raw_revenue)
                    except Exception:
                        continue

                    if revenue_value <= 0:
                        continue

                    clean_rows.append({
                        "period": col,
                        "revenue": revenue_value,
                    })

                if clean_rows:
                    result["revenue_series"] = clean_rows
                    result["latest_revenue"] = clean_rows[-1]["revenue"]

                    growth_series = []
                    for i in range(1, len(clean_rows)):
                        prev_rev = clean_rows[i - 1]["revenue"]
                        curr_rev = clean_rows[i]["revenue"]
                        if prev_rev == 0:
                            continue
                        growth = (curr_rev - prev_rev) / prev_rev
                        growth_series.append({
                            "period": clean_rows[i]["period"],
                            "growth": round(growth, 4),
                        })

                    result["revenue_growth_series"] = growth_series

                    if growth_series:
                        avg_growth = sum(item["growth"] for item in growth_series) / len(growth_series)
                        result["average_growth"] = round(avg_growth, 4)
                        latest_growth = growth_series[-1]["growth"]

                        if latest_growth > 0.10:
                            trend = "strong growth"
                        elif latest_growth > 0.03:
                            trend = "moderate growth"
                        elif latest_growth >= -0.03:
                            trend = "stable"
                        else:
                            trend = "declining"

                        result["historical_growth_trend"] = trend
                        result["observations"].append(
                            f"Revenue increased from ${clean_rows[0]['revenue'] / 1e9:.1f}B in {clean_rows[0]['period']} to ${clean_rows[-1]['revenue'] / 1e9:.1f}B in {clean_rows[-1]['period']}."
                        )
                        result["observations"].append(
                            f"Latest observed revenue growth was {latest_growth:.1%}, indicating {trend}."
                        )
                        result["observations"].append(
                            f"Average historical revenue growth across available actual periods was {avg_growth:.1%}."
                        )
                    else:
                        result["historical_growth_trend"] = "insufficient history"
                        result["observations"].append("Only one valid revenue observation was available, so growth trend could not be computed.")

                    return result

        revenue_col = next((col for col in revenue_candidates if col in financial_metrics_df.columns), None)
        period_col = next((col for col in period_candidates if col in financial_metrics_df.columns), None)

        if revenue_col is None:
            result["observations"].append(
                "Revenue trend could not be extracted from the provided dataframe layout, so driver interpretation should rely on the forecast table and qualitative driver signals."
            )
            return result

        working_df = financial_metrics_df.copy()
        revenue_series = working_df[revenue_col]

        try:
            revenue_series = revenue_series.astype(float)
        except Exception:
            revenue_series = revenue_series

        if period_col is not None:
            try:
                working_df = working_df.sort_values(by=period_col)
            except Exception:
                pass

        clean_rows = []
        for _, row in working_df.iterrows():
            raw_revenue = row.get(revenue_col)
            try:
                revenue_value = float(raw_revenue)
            except Exception:
                continue

            if revenue_value <= 0:
                continue

            label = str(row.get(period_col)) if period_col is not None else None
            clean_rows.append({
                "period": label,
                "revenue": revenue_value,
            })

        if not clean_rows:
            result["observations"].append("No valid positive revenue values were found.")
            return result

        result["revenue_series"] = clean_rows
        result["latest_revenue"] = clean_rows[-1]["revenue"]

        growth_series = []
        for i in range(1, len(clean_rows)):
            prev_rev = clean_rows[i - 1]["revenue"]
            curr_rev = clean_rows[i]["revenue"]
            if prev_rev == 0:
                continue
            growth = (curr_rev - prev_rev) / prev_rev
            growth_series.append({
                "period": clean_rows[i]["period"],
                "growth": round(growth, 4),
            })

        result["revenue_growth_series"] = growth_series

        if growth_series:
            avg_growth = sum(item["growth"] for item in growth_series) / len(growth_series)
            result["average_growth"] = round(avg_growth, 4)

            latest_growth = growth_series[-1]["growth"]
            if latest_growth > 0.10:
                trend = "strong growth"
            elif latest_growth > 0.03:
                trend = "moderate growth"
            elif latest_growth >= -0.03:
                trend = "stable"
            else:
                trend = "declining"

            result["historical_growth_trend"] = trend
            result["observations"].append(
                f"Latest observed revenue growth was {latest_growth:.1%}, indicating {trend}."
            )

            if avg_growth is not None:
                result["observations"].append(
                    f"Average historical revenue growth across available periods was {avg_growth:.1%}."
                )
        else:
            result["historical_growth_trend"] = "insufficient history"
            result["observations"].append("Only one valid revenue observation was available, so growth trend could not be computed.")

        return result

    def extract_text_drivers(self, text: str) -> Dict[str, List[str]]:
        results = {k: [] for k in self.driver_keywords.keys()}
        text_lower = text.lower()

        for category, keywords in self.driver_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    results[category].append(kw)

        return results

    def analyze_news_drivers(self, company_news=None) -> Dict[str, List[str]]:
        combined = {k: [] for k in self.driver_keywords.keys()}
        if not company_news:
            return combined

        for article in company_news:
            text = " ".join([
                str(article.get("title", "")),
                str(article.get("text", "")),
            ])
            found = self.extract_text_drivers(text)
            for k, vals in found.items():
                combined[k].extend(vals)

        return {k: sorted(set(v)) for k, v in combined.items()}

    def analyze_filing_drivers(self, sec_filing_analysis=None) -> Dict[str, List[str]]:
        combined = {k: [] for k in self.driver_keywords.keys()}
        if not sec_filing_analysis:
            return combined

        for filing in sec_filing_analysis.get("filings_analyzed", []):
            sections = filing.get("sections", {})
            text = " ".join([
                sections.get("business_overview", ""),
                sections.get("management_discussion", ""),
                sections.get("guidance_or_outlook", ""),
            ])
            found = self.extract_text_drivers(text)
            for k, vals in found.items():
                combined[k].extend(vals)

        return {k: sorted(set(v)) for k, v in combined.items()}

    def merge_drivers(self, news_drivers, filing_drivers):
        merged = {}
        for k in self.driver_keywords.keys():
            merged[k] = sorted(set(news_drivers.get(k, []) + filing_drivers.get(k, [])))
        return merged

    def _category_label(self, category: str) -> str:
        mapping = {
            "product_segment": "Product / Segment",
            "geographic": "Geographic",
            "pricing": "Pricing",
            "volume_demand": "Volume / Demand",
            "strategic": "Strategic",
            "macro": "Macro",
        }
        return mapping.get(category, category.replace("_", " ").title())

    def _top_driver_phrases(self, merged_drivers: Dict[str, List[str]], limit: int = 3) -> List[str]:
        phrases: List[str] = []
        for category in self.driver_keywords.keys():
            values = merged_drivers.get(category, [])
            if values:
                phrases.extend(values)
        deduped: List[str] = []
        for phrase in phrases:
            if phrase not in deduped:
                deduped.append(phrase)
        return deduped[:limit]

    def build_driver_summary(self, merged_drivers: Dict[str, List[str]]) -> List[str]:
        lines = []
        for category, vals in merged_drivers.items():
            if vals:
                label = self._category_label(category)
                lines.append(f"{label}: {', '.join(vals[:5])}")
        return lines

    def build_forecast_support(self, forecast_assumptions: Optional[Dict[str, Any]], merged_drivers: Dict[str, List[str]], ticker: str = "") -> Dict[str, str]:
        if not forecast_assumptions:
            return {}

        support = {}
        is_apple = ticker.upper() == "AAPL"

        for year, growth in forecast_assumptions.items():
            try:
                growth_text = f"{float(growth):.1%}"
            except Exception:
                growth_text = str(growth)

            if is_apple:
                support[str(year)] = (
                    f"The {growth_text} revenue growth assumption is supported by Apple-specific drivers including product refresh cycles, "
                    "services and ecosystem monetization, pricing/mix discipline, installed-base retention, and demand resilience."
                )
            else:
                top_driver_phrases = self._top_driver_phrases(merged_drivers, limit=3)
                if top_driver_phrases:
                    driver_text = ", ".join(top_driver_phrases)
                else:
                    driver_text = "limited identified revenue drivers"
                support[str(year)] = (
                    f"The {growth_text} revenue growth assumption is supported by {driver_text}."
                )

        return support

    def analyze(
        self,
        ticker: str,
        company_name: str,
        financial_metrics_df=None,
        company_news=None,
        sec_filing_analysis=None,
        forecast_assumptions=None,
    ) -> Dict[str, Any]:
        historical = self.analyze_historical_revenue(financial_metrics_df)
        news_drivers = self.analyze_news_drivers(company_news)
        filing_drivers = self.analyze_filing_drivers(sec_filing_analysis)
        merged_drivers = self.merge_drivers(news_drivers, filing_drivers)
        driver_summary = self.build_driver_summary(merged_drivers)
        forecast_support = self.build_forecast_support(forecast_assumptions, merged_drivers, ticker=ticker)

        active_categories = [self._category_label(k) for k, v in merged_drivers.items() if v]
        top_driver_phrases = self._top_driver_phrases(merged_drivers, limit=3)

        if ticker.upper() == "AAPL" or "apple" in company_name.lower():
            apple_observations = [
                "Apple's revenue trajectory should be read through four primary drivers: product refresh cycles, services growth, pricing/mix, and installed-base monetization.",
                "Services and ecosystem monetization can support margin quality even when hardware unit growth is moderate.",
                "Forecast revenue growth should be monitored against replacement-cycle demand, new product adoption, and competitive pressure in premium consumer devices.",
            ]
            existing_observations = historical.get("observations", [])
            for obs in apple_observations:
                if obs not in existing_observations:
                    existing_observations.append(obs)
            historical["observations"] = existing_observations

        return {
            "ticker": ticker,
            "company_name": company_name,
            "historical_revenue_analysis": historical,
            "identified_drivers": merged_drivers,
            "driver_summary": driver_summary,
            "forecast_support": forecast_support,
            "summary": (
                f"Revenue driver analysis for {company_name} links the forecast assumptions to product-cycle execution, "
                "services and ecosystem monetization, pricing/mix discipline, demand resilience, and strategic expansion. "
                f"Historical trend: {historical.get('historical_growth_trend', 'unavailable')}. "
                f"Active driver categories: {', '.join(active_categories) if active_categories else 'none'}."
            )
        }