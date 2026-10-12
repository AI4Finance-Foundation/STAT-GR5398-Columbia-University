import configparser
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "finrobot_equity/core/src"))

from modules.text_generator_agents import generate_text_section

TICKER = "NVDA"
COMPANY = "NVIDIA Corporation"
SECTIONS = ["investment_overview", "risks"]
RUNS = 3

INPUT = ROOT / "output/NVDA/analysis"
OUT = ROOT / "experiments/part2_stability/NVDA"
OUT.mkdir(parents=True, exist_ok=True)

config = configparser.ConfigParser()
config.read(ROOT / "finrobot_equity/core/config/config.ini")

api = config["API_KEYS"]
key = api.get("openai_api_key", "").strip()
model = api.get("openai_model", "").strip() or None
base_url = api.get("openai_base_url", "").strip() or None

if not key:
    raise RuntimeError("OpenAI API key unavailable")

financial = pd.read_csv(
    INPUT / "financial_metrics_and_forecasts.csv"
)
peers = pd.read_csv(
    INPUT / "peer_ebitda_comparison.csv"
)
ev_peers = pd.read_csv(
    INPUT / "peer_ev_ebitda_comparison.csv"
)

# Freeze all input evidence for all three runs.
data = {
    "financial_metrics": financial,
    "peer_ebitda": peers,
    "peer_ev_ebitda": ev_peers,
    "company_news": None,
    "enhanced_news": None,
    "retail_sentiment": None,
    "sensitivity_analysis": None,
    "catalyst_analysis": None,
    "data_provenance": (
        "STABILITY EXPERIMENT: The supplied financial tables "
        "are a fixed snapshot used identically across three runs. "
        "A means historical Actual; E means illustrative Estimate. "
        "Do not invent recent news or unsupported market claims. "
        "The FMP analyst consensus target previously observed "
        "was $339.17 and rating Buy, but these are external "
        "consensus figures, not an independent FinRobot valuation. "
        "Do not invent an independently calculated price target. "
        "If discussing the target, identify its external source. "
        "Report uncertainty and key risks."
    ),
}

manifest = {
    "ticker": TICKER,
    "company": COMPANY,
    "model": model or "module default",
    "temperature": 0.7,
    "financial_snapshot": str(
        INPUT / "financial_metrics_and_forecasts.csv"
    ),
    "peer_snapshot": str(
        INPUT / "peer_ebitda_comparison.csv"
    ),
    "sections": SECTIONS,
    "runs": RUNS,
    "experiment_started_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "outputs": [],
}

for run in range(1, RUNS + 1):
    print(f"\n========== NVDA STABILITY RUN {run}/{RUNS} ==========")

    for section in SECTIONS:
        path = OUT / f"run_{run}_{section}.txt"

        if path.exists() and len(
            path.read_text(encoding="utf-8").strip()
        ) >= 150:
            print("SKIP existing:", path.name)
            content = path.read_text(encoding="utf-8")
        else:
            print("Generating:", section)

            content = generate_text_section(
                data,
                section,
                key,
                COMPANY,
                TICKER,
                base_url=base_url,
                model=model,
            )

            content = (content or "").strip()

            if len(content) < 150 or (
                "analysis not available" in content.lower()
            ):
                review = OUT / f"run_{run}_{section}.review.txt"
                review.write_text(content, encoding="utf-8")
                raise RuntimeError(
                    f"Invalid response for {path.name}; "
                    f"saved to {review}"
                )

            path.write_text(content, encoding="utf-8")
            print(
                "SAVED:",
                path.name,
                len(content),
                "characters"
            )

        manifest["outputs"].append({
            "run": run,
            "section": section,
            "file": path.name,
            "characters": len(content),
        })

        (OUT / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )

print("\nSTABILITY GENERATION COMPLETE")
