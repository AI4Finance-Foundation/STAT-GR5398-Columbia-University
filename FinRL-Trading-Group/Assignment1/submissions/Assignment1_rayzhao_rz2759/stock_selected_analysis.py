from __future__ import annotations

from pathlib import Path

import pandas as pd


SECTOR_MAP = {
    10: "Energy",
    15: "Materials",
    20: "Industrials",
    25: "Consumer Discretionary",
    30: "Consumer Staples",
    35: "Health Care",
    40: "Financials",
    45: "Information Technology",
    50: "Communication Services",
    55: "Utilities",
    60: "Real Estate",
}


def _find_assignment_root() -> Path:
    start = Path.cwd().resolve()
    candidates: list[Path] = []
    for base in [start, *start.parents]:
        candidates.extend(
            [
                base,
                base / "Assignment1_rayzhao_rz2759",
                base / "Assignment1" / "submissions" / "Assignment1_rayzhao_rz2759",
                base / "FinRL-Trading-Group" / "Assignment1" / "submissions" / "Assignment1_rayzhao_rz2759",
            ]
        )

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand).lower()
        if key in seen:
            continue
        seen.add(key)
        if (cand / "outputs_step2" / "stock_selected.csv").exists() and (cand / "outputs" / "final_ratios.csv").exists():
            return cand
    raise FileNotFoundError("Cannot locate Assignment1_rayzhao_rz2759 root.")


def _load_sector_map(final_ratios_path: Path) -> pd.DataFrame:
    ratios = pd.read_csv(final_ratios_path, usecols=["tic", "gsector", "date"])
    ratios["tic"] = ratios["tic"].astype(str)
    ratios["gsector"] = pd.to_numeric(ratios["gsector"], errors="coerce")
    ratios["date"] = pd.to_datetime(ratios["date"], errors="coerce").dt.normalize()
    ratios = ratios.sort_values(["tic", "date"], kind="mergesort")

    # Keep earliest known sector per ticker to avoid future leakage.
    sec = ratios.dropna(subset=["gsector"]).drop_duplicates(subset=["tic"], keep="first")[["tic", "gsector"]].copy()
    sec["sector_name"] = sec["gsector"].map(SECTOR_MAP).fillna("Unknown")
    return sec


def build_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(root / "outputs_step2" / "stock_selected.csv")
    selected["trade_date"] = pd.to_datetime(selected["trade_date"], errors="coerce").dt.normalize()
    selected["tic"] = selected["tic"].astype(str)
    selected = selected.dropna(subset=["trade_date", "tic"])

    sec_map = _load_sector_map(root / "outputs" / "final_ratios.csv")
    merged = selected.merge(sec_map, on="tic", how="left")
    merged["sector_name"] = merged["sector_name"].fillna("Unknown")

    obs_tbl = (
        merged.groupby(["gsector", "sector_name"], dropna=False)
        .agg(observation_count=("tic", "size"), unique_tickers_in_observations=("tic", "nunique"))
        .reset_index()
        .sort_values("observation_count", ascending=False)
    )
    obs_total = int(obs_tbl["observation_count"].sum())
    obs_tbl["observation_share"] = obs_tbl["observation_count"] / obs_total if obs_total > 0 else 0.0

    uniq = merged[["tic", "gsector", "sector_name"]].drop_duplicates(subset=["tic"])
    uniq_tbl = (
        uniq.groupby(["gsector", "sector_name"], dropna=False)
        .agg(unique_ticker_count=("tic", "size"))
        .reset_index()
        .sort_values("unique_ticker_count", ascending=False)
    )
    uniq_total = int(uniq_tbl["unique_ticker_count"].sum())
    uniq_tbl["unique_ticker_share"] = uniq_tbl["unique_ticker_count"] / uniq_total if uniq_total > 0 else 0.0

    top_tickers_tbl = (
        merged.groupby("tic")
        .size()
        .sort_values(ascending=False)
        .head(30)
        .reset_index(name="observation_count")
        .merge(sec_map[["tic", "sector_name"]], on="tic", how="left")
    )
    top_tickers_tbl["sector_name"] = top_tickers_tbl["sector_name"].fillna("Unknown")

    return obs_tbl, uniq_tbl, top_tickers_tbl


def save_tables(root: Path, obs_tbl: pd.DataFrame, uniq_tbl: pd.DataFrame, top_tickers_tbl: pd.DataFrame) -> None:
    obs_path = root / "stock_selected_sector_observation_table.csv"
    uniq_path = root / "stock_selected_sector_unique_tickers_table.csv"
    top_path = root / "stock_selected_top_tickers_table.csv"

    obs_tbl.to_csv(obs_path, index=False)
    uniq_tbl.to_csv(uniq_path, index=False)
    top_tickers_tbl.to_csv(top_path, index=False)

    print(f"Wrote: {obs_path}")
    print(f"Wrote: {uniq_path}")
    print(f"Wrote: {top_path}")


def main() -> None:
    root = _find_assignment_root()
    obs_tbl, uniq_tbl, top_tickers_tbl = build_tables(root)
    save_tables(root, obs_tbl, uniq_tbl, top_tickers_tbl)

    print("\nTop sectors by observation share:")
    print(obs_tbl.head(10).to_string(index=False))
    print("\nTop selected tickers:")
    print(top_tickers_tbl.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
