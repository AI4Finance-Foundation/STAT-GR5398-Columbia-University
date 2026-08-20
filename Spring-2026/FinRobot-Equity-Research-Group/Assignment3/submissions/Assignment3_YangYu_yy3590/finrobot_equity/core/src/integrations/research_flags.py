"""LLM flag extraction and cacheable research artifact handling."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.llm_gateway import LLMSettings, call_llm


ALLOWED_VALUES = {
    "risk_flag": {"low", "medium", "high"},
    "valuation_flag": {"cheap", "fair", "expensive", "extreme"},
    "catalyst_flag": {"positive", "neutral", "negative"},
    "sentiment_flag": {"positive", "mixed", "negative"},
    "thesis_strength": {"strong", "moderate", "weak"},
}


@dataclass(frozen=True)
class ResearchFlags:
    ticker: str
    as_of_date: str
    risk_flag: str
    valuation_flag: str
    catalyst_flag: str
    sentiment_flag: str
    thesis_strength: str

    def to_dict(self) -> dict[str, str]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "risk_flag": self.risk_flag,
            "valuation_flag": self.valuation_flag,
            "catalyst_flag": self.catalyst_flag,
            "sentiment_flag": self.sentiment_flag,
            "thesis_strength": self.thesis_strength,
        }


def _ensure_required_sections(base_dir: Path) -> dict[str, str]:
    section_names = [
        "investment_overview",
        "valuation_overview",
        "risks",
        "major_takeaways",
        "news_summary",
    ]
    paths = {}
    for name in section_names:
        path = base_dir / f"{name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing required research artifact: {path}")
        paths[name] = str(path)
    return paths


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _build_prompt(ticker: str, as_of_date: str, sections: dict[str, str]) -> str:
    payload = {name: _load_text(Path(path)) for name, path in sections.items()}
    return json.dumps(
        {
            "ticker": ticker,
            "as_of_date": as_of_date,
            #"instructions": "Return only strict JSON with keys risk_flag, valuation_flag, catalyst_flag, sentiment_flag, thesis_strength.",
            "sections": payload,
            "schema": {
                "risk_flag": sorted(ALLOWED_VALUES["risk_flag"]),
                "valuation_flag": sorted(ALLOWED_VALUES["valuation_flag"]),
                "catalyst_flag": sorted(ALLOWED_VALUES["catalyst_flag"]),
                "sentiment_flag": sorted(ALLOWED_VALUES["sentiment_flag"]),
                "thesis_strength": sorted(ALLOWED_VALUES["thesis_strength"]),
            },
        },
        indent=2,
        ensure_ascii=False,
    )


def _validate_payload(payload: dict[str, Any], ticker: str, as_of_date: str) -> ResearchFlags:
    required = ["risk_flag", "valuation_flag", "catalyst_flag", "sentiment_flag", "thesis_strength"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required flag field: {key}")

    normalized = {k: str(payload[k]).strip().lower() for k in required}
    for key, allowed in ALLOWED_VALUES.items():
        if normalized[key] not in allowed:
            raise ValueError(f"Invalid value for {key}: {payload[key]!r}")

    return ResearchFlags(
        ticker=ticker,
        as_of_date=as_of_date,
        **normalized,
    )


def _parse_llm_json(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object")
    return payload


def extract_research_flags(
    *,
    ticker: str,
    as_of_date: str,
    research_dir: str | Path,
    llm_settings: LLMSettings,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> tuple[ResearchFlags, dict]:
    research_dir = Path(research_dir)
    sections = _ensure_required_sections(research_dir)
    cache_root = Path(cache_dir) if cache_dir else research_dir / "_flags_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{ticker.upper()}_{as_of_date}.json"

    if cache_path.exists() and not force_refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return _validate_payload(payload["flags"], ticker, as_of_date), payload.get("meta", {})

    prompt = _build_prompt(ticker, as_of_date, sections)
    instructions = (
        "You are a professional equity research analyst. Your current job is to extract investment research labels from text-based equity information. "
        "In most cases both positive and negative flags co-exist. Use all available information for in-depth analysis. "
        "Return only strict JSON, no prose, with keys risk_flag, valuation_flag, catalyst_flag, sentiment_flag, thesis_strength."
    )
    raw_text, meta = call_llm(
        settings=llm_settings,
        instructions=instructions,
        prompt=prompt,
        max_output_tokens=256,
        temperature=0.0,
        return_meta=True,
    )

    try:
        payload = _parse_llm_json(raw_text)
    except Exception as exc:
        raise ValueError(f"LLM did not return valid JSON: {exc}; raw={raw_text!r}") from exc

    flags = _validate_payload(payload, ticker, as_of_date)
    cache_path.write_text(
        json.dumps({"flags": flags.to_dict(), "meta": meta, "raw": raw_text}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return flags, meta
