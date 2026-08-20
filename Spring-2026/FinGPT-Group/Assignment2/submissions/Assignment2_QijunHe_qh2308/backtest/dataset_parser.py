"""
Dataset parsing utilities for FinGPT backtesting.
"""

from __future__ import annotations

import re

import pandas as pd

_INST_TAG_RE = re.compile(r"\[/?INST\]|<<SYS>>|<</SYS>>", re.IGNORECASE)
_DATE_RANGE_RE = re.compile(
    r"From\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from local parquet/CSV or Hugging Face dataset ID.
    Returns a DataFrame with normalized columns: input, output, answer, ticker.
    """
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif lower.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        try:
            from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset
        except Exception as exc:
            raise ValueError(
                "Unsupported dataset source. Use .parquet/.csv or install `datasets` "
                "to load a Hugging Face dataset ID."
            ) from exc

        loaded = hf_load_dataset(path)
        if isinstance(loaded, Dataset):
            hf_ds = loaded
        elif isinstance(loaded, DatasetDict):
            split_priority = ("test", "validation", "train")
            split = next((name for name in split_priority if name in loaded), None)
            if split is None:
                split = next(iter(loaded.keys()))
            hf_ds = loaded[split]
        else:
            raise ValueError(f"Unexpected dataset object type: {type(loaded).__name__}")
        df = hf_ds.to_pandas()

    # Normalize common naming variants to required names.
    normalized_columns = {col: col.strip().lower() for col in df.columns}
    df = df.rename(columns=normalized_columns)
    column_aliases = {
        "input_text": "input",
        "prompt": "input",
        "response": "output",
        "label": "answer",
        "symbol": "ticker",
    }
    df = df.rename(columns=column_aliases)

    required = ["input", "output", "answer", "ticker"]
    normalized = pd.DataFrame(index=df.index)
    for col in required:
        if col not in df.columns:
            normalized[col] = ""
            continue

        selected = df[col]
        # Handle duplicate column labels that return a DataFrame.
        if isinstance(selected, pd.DataFrame):
            selected = selected.iloc[:, 0]
        normalized[col] = selected

    return normalized.reset_index(drop=True)


def extract_article_text(input_prompt: str) -> str:
    """
    Extract headline/news section from raw [INST] prompt content.
    """
    text = input_prompt or ""
    text = _INST_TAG_RE.sub(" ", text)
    text = text.strip()

    start_idx = text.find("[Headline]:")
    if start_idx < 0:
        return ""

    end_idx = text.find("[Basic Financials]:", start_idx)
    if end_idx < 0:
        end_idx = len(text)

    news_block = text[start_idx:end_idx]

    # Keep headline/summary pairs explicit so multi-news prompts remain structured.
    pattern = re.compile(
        r"\[Headline\]:\s*(.*?)\s*\[Summary\]:\s*(.*?)(?=\s*\[Headline\]:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    pairs = pattern.findall(news_block)
    if pairs:
        rendered: list[str] = []
        for idx, (headline, summary) in enumerate(pairs, start=1):
            h = re.sub(r"\s+", " ", headline).strip()
            s = re.sub(r"\s+", " ", summary).strip()
            if not h and not s:
                continue
            rendered.append(f"News {idx}\nHeadline: {h}\nSummary: {s}")
        return "\n\n".join(rendered).strip()

    # Fallback for prompts that don't strictly follow headline/summary pair format.
    news_block = re.sub(r"\[(Headline|Summary)\]:", " ", news_block, flags=re.IGNORECASE)
    news_block = re.sub(r"\s+", " ", news_block).strip()
    return news_block


def extract_date_range(input_prompt: str) -> tuple[str, str]:
    """
    Parse `From YYYY-MM-DD to YYYY-MM-DD` from prompt text.
    """
    match = _DATE_RANGE_RE.search(input_prompt or "")
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def parse_fingpt_label(answer: str) -> str:
    """
    Normalize FinGPT answer label to up/down/neutral.
    """
    normalized = (answer or "").strip().lower()
    if "up" in normalized:
        return "up"
    if "down" in normalized:
        return "down"
    return "neutral"


def build_backtest_rows(df: pd.DataFrame) -> list[dict]:
    """
    Convert dataset rows into normalized backtest rows.
    """
    def _row_value(row_obj: pd.Series, key: str) -> str:
        raw = row_obj.get(key, "")
        if isinstance(raw, pd.Series):
            for value in raw.tolist():
                if pd.notna(value):
                    return str(value)
            return ""
        if pd.isna(raw):
            return ""
        return str(raw)

    rows: list[dict] = []
    for _, row in df.iterrows():
        raw_input = _row_value(row, "input")
        ticker = _row_value(row, "ticker").strip().upper()
        article_text = extract_article_text(raw_input)
        if not ticker or not article_text:
            continue

        start_date, end_date = extract_date_range(raw_input)
        fingpt_label = parse_fingpt_label(_row_value(row, "answer"))

        rows.append(
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "article_text": article_text,
                "fingpt_label": fingpt_label,
            }
        )
    return rows

