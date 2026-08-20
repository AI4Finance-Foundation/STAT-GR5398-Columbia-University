import re
from datetime import datetime
from typing import Any, Dict, List, Optional


class SECFilingAnalyzer:
    def __init__(self, sec_api_key=None, openai_api_key=None, model=None):
        self.sec_api_key = sec_api_key
        self.openai_api_key = openai_api_key
        self.model = model

    def fetch_filings(self, ticker: str, filing_types=None, limit: int = 3):
        filing_types = filing_types or ["10-K", "10-Q"]
        filings: List[Dict[str, Any]] = []

        for idx, filing_type in enumerate(filing_types[:limit]):
            filings.append(
                {
                    "filing_type": filing_type,
                    "filing_date": datetime.utcnow().strftime("%Y-%m-%d"),
                    "source": "mock",
                    "filing_text": self._build_mock_filing_text(ticker=ticker, filing_type=filing_type),
                }
            )

        return filings

    def extract_key_sections(self, filing_text: str):
        patterns = {
            "business_overview": r"BUSINESS OVERVIEW:(.*?)(?:RISK FACTORS:|$)",
            "risk_factors": r"RISK FACTORS:(.*?)(?:MANAGEMENT DISCUSSION AND ANALYSIS:|$)",
            "management_discussion": r"MANAGEMENT DISCUSSION AND ANALYSIS:(.*?)(?:GUIDANCE AND OUTLOOK:|$)",
            "guidance_or_outlook": r"GUIDANCE AND OUTLOOK:(.*)$",
        }

        sections: Dict[str, str] = {}
        for section_name, pattern in patterns.items():
            match = re.search(pattern, filing_text, flags=re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_name] = self._clean_text(match.group(1))
            else:
                sections[section_name] = ""

        return sections

    def summarize_sections(self, sections: dict):
        summary_parts: List[str] = []

        if sections.get("business_overview"):
            summary_parts.append(
                f"Business overview: {self._truncate_text(sections['business_overview'], 220)}"
            )
        if sections.get("management_discussion"):
            summary_parts.append(
                f"Management discussion: {self._truncate_text(sections['management_discussion'], 220)}"
            )
        if sections.get("guidance_or_outlook"):
            summary_parts.append(
                f"Outlook: {self._truncate_text(sections['guidance_or_outlook'], 220)}"
            )

        if not summary_parts:
            return "No meaningful sections were extracted from the filing text."

        return "\n".join(summary_parts)

    def detect_risk_signals(self, sections: dict):
        risk_signal_descriptions = {
            "macroeconomic": "Macroeconomic pressure could slow premium-device replacement cycles and weaken discretionary consumer demand.",
            "inflation": "Inflation may pressure component, logistics, labor, and consumer purchasing-power dynamics, limiting margin upside.",
            "supply chain": "Supply chain disruption could affect product availability, cost structure, inventory timing, and launch execution.",
            "competition": "Competitive pressure across smartphones, wearables, services, cloud, and AI features could affect pricing power and market share.",
            "regulation": "Regulatory scrutiny around app stores, payments, privacy, and platform control could raise compliance costs or reduce ecosystem flexibility.",
            "litigation": "Litigation exposure may create legal costs, settlement risk, or reputational pressure.",
            "cybersecurity": "Cybersecurity and data-protection issues could create operational disruption, reputational damage, and customer-trust risk.",
            "uncertainty": "Uncertainty language suggests limited visibility into demand, macro conditions, product cycles, or operating performance.",
            "decline": "Decline-related language points to potential weakness in demand, revenue, margins, or profitability trends.",
            "risk": "Management highlights broad execution and operating uncertainty that could affect financial performance.",
        }

        signals: List[str] = []
        risk_text = " ".join(
            [
                sections.get("risk_factors", ""),
                sections.get("management_discussion", ""),
                sections.get("guidance_or_outlook", ""),
            ]
        ).lower()

        for keyword, description in risk_signal_descriptions.items():
            if keyword in risk_text:
                signals.append(description)

        if not signals and sections.get("risk_factors"):
            signals.append("The filing includes risk-factor disclosure, but the extracted text does not identify a more specific risk category.")

        return signals

    def detect_catalyst_signals(self, sections: dict):
        catalyst_signal_descriptions = {
            "growth": "Growth language supports the case for continued expansion across products, services, geographies, or ecosystem monetization.",
            "expansion": "Expansion language points to potential upside from new markets, service penetration, installed-base monetization, or ecosystem breadth.",
            "margin improvement": "Margin improvement could support EPS upside if favorable product/service mix and operating discipline continue.",
            "demand": "Demand references suggest customer interest and replacement-cycle resilience can support the revenue forecast.",
            "new product": "New product language supports potential product-cycle upside, customer upgrades, and ecosystem engagement.",
            "guidance": "Guidance language provides directional visibility into management's expectations and execution priorities.",
            "efficiency": "Efficiency initiatives may support operating leverage, margin expansion, and cash generation.",
            "innovation": "Innovation references may indicate upside from product differentiation, AI features, silicon capabilities, or services integration.",
            "pipeline": "Pipeline references suggest future product or service launches that could support medium-term growth.",
            "strategic": "Strategic priorities highlight where management is focusing capital, execution, and long-term positioning.",
        }

        signals: List[str] = []
        catalyst_text = " ".join(
            [
                sections.get("business_overview", ""),
                sections.get("management_discussion", ""),
                sections.get("guidance_or_outlook", ""),
            ]
        ).lower()

        for keyword, description in catalyst_signal_descriptions.items():
            if keyword in catalyst_text:
                signals.append(description)

        return signals

    def analyze_filings(self, ticker: str, company_name: str, filing_types=None, limit: int = 3):
        filings = self.fetch_filings(ticker=ticker, filing_types=filing_types, limit=limit)

        analyzed_filings: List[Dict[str, Any]] = []
        for filing in filings:
            sections = self.extract_key_sections(filing.get("filing_text", ""))
            section_summary = self.summarize_sections(sections)
            risk_signals = self.detect_risk_signals(sections)
            catalyst_signals = self.detect_catalyst_signals(sections)
            key_takeaways = self._build_key_takeaways(sections, risk_signals, catalyst_signals)

            analyzed_filings.append(
                {
                    "filing_type": filing.get("filing_type"),
                    "filing_date": filing.get("filing_date"),
                    "source": filing.get("source", "unknown"),
                    "sections": sections,
                    "section_summary": section_summary,
                    "key_takeaways": key_takeaways,
                    "risk_signals": risk_signals,
                    "catalyst_signals": catalyst_signals,
                }
            )

        overall_summary = self._build_overall_summary(company_name, analyzed_filings)

        return {
            "ticker": ticker,
            "company_name": company_name,
            "filings_analyzed": analyzed_filings,
            "summary": overall_summary,
            "generated_at": datetime.utcnow().isoformat(),
            "mode": "mock",
        }

    def _build_mock_filing_text(self, ticker: str, filing_type: str) -> str:
        ticker_upper = ticker.upper()

        if ticker_upper == "AAPL":
            return f"""
            BUSINESS OVERVIEW:
            {ticker_upper} continues to focus on product innovation, ecosystem expansion, services growth, and operational efficiency.
            The company highlighted demand resilience across its installed base, continued investment in new product development,
            and opportunities to deepen customer engagement across hardware, software, payments, cloud, media, and subscription services.

            RISK FACTORS:
            The company faces macroeconomic uncertainty, inflationary pressure, supply chain risk, cybersecurity concerns,
            increasing competition, potential regulatory changes around platform control and digital services, and execution risk around
            product launches, geographic exposure, and consumer replacement cycles.

            MANAGEMENT DISCUSSION AND ANALYSIS:
            Management noted revenue growth opportunities in priority segments, disciplined cost control, continued services momentum,
            and opportunities for margin improvement through mix shift and operating leverage. At the same time, management acknowledged
            uncertainty related to consumer demand, competitive intensity, component costs, and the broader economic environment.

            GUIDANCE AND OUTLOOK:
            Management expects continued strategic execution, selective expansion, product innovation, services monetization,
            and further efficiency gains, while remaining cautious about near-term volatility and demand normalization.
            """

        return f"""
        BUSINESS OVERVIEW:
        {ticker_upper} continues to focus on product innovation, strategic expansion, and operational efficiency.
        The company highlighted demand resilience in core segments and continued investment in new product development.

        RISK FACTORS:
        The company faces macroeconomic uncertainty, inflationary pressure, supply chain risk, cybersecurity concerns,
        increasing competition, and potential regulatory changes.

        MANAGEMENT DISCUSSION AND ANALYSIS:
        Management noted revenue growth in priority segments, disciplined cost control, and opportunities for margin improvement.
        At the same time, management acknowledged uncertainty related to consumer demand and the broader economic environment.

        GUIDANCE AND OUTLOOK:
        Management expects continued strategic execution, selective expansion, and further efficiency gains,
        while remaining cautious about near-term volatility.
        """

    def _build_key_takeaways(
        self,
        sections: Dict[str, str],
        risk_signals: List[str],
        catalyst_signals: List[str],
    ) -> List[str]:
        takeaways: List[str] = []

        filing_text = " ".join(sections.values()).lower()

        if "ecosystem" in filing_text or "services" in filing_text:
            takeaways.append(
                "The filing language supports an ecosystem-led thesis, with services, installed-base monetization, and customer retention acting as key long-term drivers."
            )
        elif sections.get("business_overview"):
            takeaways.append(
                "The filing emphasizes product innovation, strategic execution, and operating discipline as core business priorities."
            )

        if "margin" in filing_text or sections.get("management_discussion"):
            takeaways.append(
                "Margin discipline is an important value driver, especially because the forecast depends on sustained operating leverage and cost control."
            )

        if "demand" in filing_text or catalyst_signals:
            takeaways.append(
                "Demand resilience and product-cycle execution remain central to the revenue trajectory."
            )

        if risk_signals:
            takeaways.append(
                "The most relevant downside risks are macro uncertainty, inflation, competitive pressure, supply chain exposure, regulatory scrutiny, and execution risk."
            )

        if not takeaways:
            takeaways.append(
                "The filing does not provide enough extracted detail to form a differentiated analyst read-through."
            )

        return takeaways

    def _build_overall_summary(self, company_name: str, analyzed_filings: List[Dict[str, Any]]) -> str:
        if not analyzed_filings:
            return f"No SEC filings were analyzed for {company_name}."

        combined_text = " ".join(
            " ".join(filing.get("sections", {}).values()) for filing in analyzed_filings
        ).lower()

        if "ecosystem" in combined_text or "services" in combined_text:
            return (
                f"The filings point to a balanced but constructive outlook for {company_name}: management emphasizes product innovation, "
                "ecosystem expansion, services monetization, demand resilience, and margin discipline as potential catalysts. "
                "The main constraints are macroeconomic uncertainty, inflation, competitive intensity, supply chain exposure, regulatory scrutiny, "
                "and execution risk around product cycles and margin expansion."
            )

        return (
            f"The filings point to a balanced outlook for {company_name}: management emphasizes product innovation, "
            "strategic expansion, demand resilience, and margin discipline as potential catalysts, while also flagging "
            "macroeconomic uncertainty, inflation, competition, supply chain exposure, regulatory pressure, and execution risk "
            "as key constraints on the forecast."
        )

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _truncate_text(self, text: str, max_len: int = 220) -> str:
        cleaned = self._clean_text(text)
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3].rstrip() + "..."