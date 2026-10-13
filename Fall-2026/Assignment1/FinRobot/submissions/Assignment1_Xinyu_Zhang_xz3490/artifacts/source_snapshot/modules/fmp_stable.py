"""FMP Stable API transport and field compatibility for the V1 pipeline.

Only transport/schema compatibility belongs here. Forecasts and valuation
assumptions remain in the original pipeline.
"""

from urllib.parse import quote, quote_plus

import requests

BASE_URL = "https://financialmodelingprep.com/stable/"


class FMPDataError(requests.exceptions.RequestException):
    """An API failure whose message is safe to show in terminal output."""


def _redact(message, api_key):
    text = str(message)
    for secret in (api_key, quote(api_key, safe=""), quote_plus(api_key)):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text[:300]


def _compatibility_fields(endpoint, records):
    aliases = {
        "profile": {
            "marketCap": "mktCap",
            "averageVolume": "volAvg",
            "exchange": "exchangeShortName",
            "lastDividend": "lastDiv",
        },
        "income-statement": {"epsDiluted": "epsdiluted"},
        "ratios": {
            "priceToEarningsRatio": "priceEarningsRatio",
            "debtToEquityRatio": "debtEquityRatio",
            "debtToAssetsRatio": "debtRatio",
            "interestCoverageRatio": "interestCoverage",
        },
        "key-metrics": {"evToEBITDA": "enterpriseValueOverEBITDA"},
        "grades": {"date": "publishedDate"},
    }.get(endpoint, {})
    result = []
    for original in records:
        row = dict(original)
        for stable_name, legacy_name in aliases.items():
            if row.get(legacy_name) is None and stable_name in row:
                row[legacy_name] = row[stable_name]
        if "fiscalYear" in row and row.get("calendarYear") is None:
            # Legacy report CSVs use this column as the fiscal-year label.
            row["calendarYear"] = str(row["fiscalYear"])
        result.append(row)
    return result


def fetch(endpoint, ticker, api_key, **parameters):
    """Return list records; distinguish API errors from empty data.

    A finite timeout prevents an unresponsive endpoint blocking the pipeline.
    Do not include requests' exception text: it can contain the API key URL.
    """
    if not isinstance(api_key, str) or not api_key.strip():
        raise FMPDataError("FMP API key is missing.")
    params = dict(parameters)
    params.setdefault("symbols" if endpoint == "news/stock" else "symbol", ticker)
    params["apikey"] = api_key.strip()
    try:
        response = requests.get(BASE_URL + endpoint, params=params, timeout=20)
    except requests.exceptions.RequestException as error:
        raise FMPDataError(
            f"FMP {endpoint} ({ticker}): network request failed ({type(error).__name__})."
        ) from None
    try:
        data = response.json()
    except ValueError:
        raise FMPDataError(
            f"FMP {endpoint} ({ticker}): HTTP {response.status_code}, non-JSON response."
        ) from None
    if response.status_code >= 400 or isinstance(data, dict):
        detail = ""
        if isinstance(data, dict):
            detail = data.get("Error Message") or data.get("error") or data.get("message") or ""
        hint = {
            401: "Check the API key.",
            403: "Check this endpoint/symbol/history entitlement before upgrading.",
            429: "Quota or rate limit reached; retry later.",
        }.get(response.status_code, "")
        raise FMPDataError(
            f"FMP {endpoint} ({ticker}): HTTP {response.status_code}. "
            f"{hint} {_redact(detail, api_key.strip())}".strip()
        ) from None
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise FMPDataError(f"FMP {endpoint} ({ticker}): unexpected response schema.")
    return _compatibility_fields(endpoint, data)
