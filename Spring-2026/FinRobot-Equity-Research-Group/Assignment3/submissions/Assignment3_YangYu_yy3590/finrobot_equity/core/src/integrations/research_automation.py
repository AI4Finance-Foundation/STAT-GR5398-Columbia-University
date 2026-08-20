"""Generate and reuse FinRobot research snapshots for integrated backtests."""

from __future__ import annotations

import configparser
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.llm_gateway import call_llm, load_llm_settings

from .active_setup import ActiveResearchSetup, utc_now_iso


RESEARCH_MODES = {"existing", "generate-missing", "regenerate"}

DEFAULT_SECTIONS = [
    "tagline",
    "company_overview",
    "investment_overview",
    "valuation_overview",
    "risks",
    "competitor_analysis",
    "major_takeaways",
    "news_summary",
]

REQUIRED_ANALYSIS_FILES = [
    "financial_metrics_and_forecasts.csv",
    "ratios_raw_data.csv",
    "tagline.txt",
    "company_overview.txt",
    "investment_overview.txt",
    "valuation_overview.txt",
    "risks.txt",
    "competitor_analysis.txt",
    "major_takeaways.txt",
    "news_summary.txt",
    "run_manifest.json",
]


ROLE_CRITIC_PROMPT = """You are a senior equity research critic and reviser.

Task:
Revise ONE section from an analyst draft report for {ticker} ({company_name}).
Section name: {section_name}

Requirements:
1. Keep factual consistency with the provided draft; do not invent financial figures.
2. Improve investment-thesis coherence and decision usefulness.
3. Strengthen risk/catalyst clarity where relevant.
4. Keep the section concise and analyst-facing.
5. Return plain text only.

Context:
{context_text}

Current section draft:
{section_text}
"""


@dataclass(frozen=True)
class SnapshotResult:
    ticker: str
    as_of_date: str
    analysis_dir: Path
    report_dir: Path
    status: str
    generated: bool
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "analysis_dir": str(self.analysis_dir),
            "report_dir": str(self.report_dir),
            "status": self.status,
            "generated": self.generated,
            "detail": self.detail,
        }


def project_root_from_src() -> Path:
    return Path(__file__).resolve().parents[4]


def snapshot_root(integration_output_dir: str | Path) -> Path:
    return Path(integration_output_dir) / "research_snapshots"


def snapshot_dirs(integration_output_dir: str | Path, as_of_date: str, ticker: str) -> tuple[Path, Path, Path]:
    root = snapshot_root(integration_output_dir) / as_of_date / ticker.upper()
    return root, root / "analysis", root / "report"


def snapshot_exists(integration_output_dir: str | Path, as_of_date: str, ticker: str) -> bool:
    _, analysis_dir, report_dir = snapshot_dirs(integration_output_dir, as_of_date, ticker)
    return all((analysis_dir / name).exists() for name in REQUIRED_ANALYSIS_FILES) and report_dir.exists()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_command(command: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return result.returncode


def build_analysis_command(
    *,
    python_cmd: str,
    src_dir: Path,
    ticker: str,
    company_name: str,
    analyst_config_path: Path,
    as_of_date: str,
    analysis_dir: Path,
    fmp_cache_policy: str,
    force_refresh_fmp: bool = False,
    fmp_cache_dir: str | Path | None = None,
    peer_tickers: list[str] | None = None,
    enable_enhanced_news: bool = False,
    enable_catalyst_analysis: bool = False,
    news_days_back: int = 5,
    news_limit: int = 50,
) -> list[str]:
    command = [
        python_cmd,
        str(src_dir / "generate_financial_analysis.py"),
        "--company-ticker",
        ticker.upper(),
        "--company-name",
        company_name,
        "--config-file",
        str(analyst_config_path),
        "--as-of-date",
        as_of_date,
        "--generate-text-sections",
        "--output-dir",
        str(analysis_dir),
        "--fmp-cache-policy",
        fmp_cache_policy,
        "--news-days-back",
        str(news_days_back),
        "--news-limit",
        str(news_limit),
    ]
    if enable_enhanced_news:
        command.append("--enable-enhanced-news")
    if enable_catalyst_analysis:
        command.append("--enable-catalyst-analysis")
    if force_refresh_fmp:
        command.append("--force-refresh-fmp")
    if fmp_cache_dir:
        command += ["--fmp-cache-dir", str(fmp_cache_dir)]
    if peer_tickers:
        command += ["--peer-tickers"] + peer_tickers
    return command


def _failure_detail(stage: str, return_code: int, stdout_path: Path, stderr_path: Path) -> str:
    detail = f"{stage}_rc={return_code}; stdout={stdout_path}; stderr={stderr_path}"
    try:
        stderr_tail = stderr_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-12:]
        if stderr_tail:
            detail += "; stderr_tail=" + " | ".join(stderr_tail)
    except Exception:
        pass
    try:
        stdout_tail = stdout_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-8:]
        if stdout_tail:
            detail += "; stdout_tail=" + " | ".join(stdout_tail)
    except Exception:
        pass
    return detail


def _build_provider_config(
    *,
    base_config_path: str | Path | None,
    provider: str,
    model: str,
    output_path: Path,
) -> Path:
    cfg = configparser.ConfigParser()
    if base_config_path:
        with Path(base_config_path).open("r", encoding="utf-8") as f:
            cfg.read_file(f)
    if not cfg.has_section("API_KEYS"):
        cfg.add_section("API_KEYS")
    cfg.set("API_KEYS", "llm_provider", provider)
    cfg.set("API_KEYS", "llm_model", model)
    if provider == "openai":
        cfg.set("API_KEYS", "openai_model", model)
    elif provider == "claude":
        cfg.set("API_KEYS", "claude_model", model)
    elif provider == "gemini":
        cfg.set("API_KEYS", "gemini_model", model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        cfg.write(f)
    return output_path


def _company_name_for(ticker: str, company_names: dict[str, str] | None) -> str:
    if company_names:
        name = company_names.get(ticker.upper()) or company_names.get(ticker)
        if name:
            return name
    return ticker.upper()


def _peer_tickers_for(ticker: str, peer_map: dict[str, list[str]] | None) -> list[str]:
    if not peer_map:
        return []
    return list(peer_map.get(ticker.upper()) or peer_map.get(ticker) or [])


def _make_critic_context(analysis_dir: Path) -> str:
    parts: list[str] = []
    for name in ["tagline", "investment_overview", "valuation_overview", "risks", "major_takeaways"]:
        path = analysis_dir / f"{name}.txt"
        if not path.exists():
            continue
        text = _read_text(path).strip()
        if len(text) > 1800:
            text = text[:1800] + "\n[TRUNCATED]"
        if text:
            parts.append(f"[{name}]\n{text}")
    return "\n\n".join(parts) if parts else "No additional section context available."


def _apply_role_mixed_critic(
    *,
    active_setup: ActiveResearchSetup,
    base_config_path: str | Path | None,
    runtime_config_path: Path,
    analysis_dir: Path,
    ticker: str,
    company_name: str,
) -> dict[str, Any]:
    if not active_setup.critic:
        return {"enabled": False, "errors": ["critic_missing"]}

    critic_config_path = _build_provider_config(
        base_config_path=base_config_path,
        provider=active_setup.critic.provider,
        model=active_setup.critic.model,
        output_path=runtime_config_path,
    )
    cfg = configparser.ConfigParser()
    cfg.read(str(critic_config_path), encoding="utf-8")
    critic_settings = load_llm_settings(
        cfg,
        provider=active_setup.critic.provider,
        model=active_setup.critic.model,
    )

    context = _make_critic_context(analysis_dir)
    summary = {
        "enabled": True,
        "generated_at_utc": utc_now_iso(),
        "critic": active_setup.critic.to_dict(),
        "sections": [],
        "errors": [],
    }
    for section in active_setup.critic_sections:
        section_path = analysis_dir / f"{section}.txt"
        if not section_path.exists():
            summary["errors"].append(f"missing_section:{section}")
            continue
        original = _read_text(section_path)
        prompt = ROLE_CRITIC_PROMPT.format(
            ticker=ticker,
            company_name=company_name,
            section_name=section,
            context_text=context,
            section_text=original,
        )
        try:
            revised = call_llm(
                settings=critic_settings,
                instructions="Return plain text only.",
                prompt=prompt,
                max_output_tokens=1600,
                temperature=0.2,
            )
            revised_text = (revised or "").strip() or original
            _write_text(section_path, revised_text)
            summary["sections"].append(
                {
                    "section": section,
                    "status": "success",
                    "original_chars": len(original),
                    "revised_chars": len(revised_text),
                }
            )
        except Exception as exc:
            summary["sections"].append({"section": section, "status": "failed", "error": str(exc)})
            summary["errors"].append(f"critic_failed:{section}:{exc}")

    _write_json(analysis_dir / "critic_diff_summary.json", summary)
    return summary


def generate_research_snapshot(
    *,
    integration_output_dir: str | Path,
    as_of_date: str,
    ticker: str,
    active_setup: ActiveResearchSetup,
    base_config_path: str | Path | None,
    fmp_cache_policy: str,
    force_refresh_fmp: bool,
    fmp_cache_dir: str | Path | None,
    company_names: dict[str, str] | None = None,
    peer_map: dict[str, list[str]] | None = None,
    python_executable: str | None = None,
    enable_enhanced_news: bool = False,
    enable_catalyst_analysis: bool = False,
    news_days_back: int = 5,
    news_limit: int = 50,
) -> SnapshotResult:
    project_root = project_root_from_src()
    src_dir = project_root / "finrobot_equity" / "core" / "src"
    root, analysis_dir, report_dir = snapshot_dirs(integration_output_dir, as_of_date, ticker)
    logs_dir = root / "logs"
    runtime_dir = root / "config"
    if root.exists():
        shutil.rmtree(root)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    python_cmd = python_executable or sys.executable
    company_name = _company_name_for(ticker, company_names)
    peer_tickers = _peer_tickers_for(ticker, peer_map)

    analyst_config_path = _build_provider_config(
        base_config_path=base_config_path,
        provider=active_setup.analyst.provider,
        model=active_setup.analyst.model,
        output_path=runtime_dir / "analyst_config.ini",
    )

    analysis_cmd = build_analysis_command(
        python_cmd=python_cmd,
        src_dir=src_dir,
        ticker=ticker,
        company_name=company_name,
        analyst_config_path=analyst_config_path,
        as_of_date=as_of_date,
        analysis_dir=analysis_dir,
        fmp_cache_policy=fmp_cache_policy,
        force_refresh_fmp=force_refresh_fmp,
        fmp_cache_dir=fmp_cache_dir,
        peer_tickers=peer_tickers,
        enable_enhanced_news=enable_enhanced_news,
        enable_catalyst_analysis=enable_catalyst_analysis,
        news_days_back=news_days_back,
        news_limit=news_limit,
    )

    analysis_stdout = logs_dir / "generate_stdout.log"
    analysis_stderr = logs_dir / "generate_stderr.log"
    analysis_rc = _run_command(
        analysis_cmd,
        cwd=project_root,
        stdout_path=analysis_stdout,
        stderr_path=analysis_stderr,
    )
    if analysis_rc != 0:
        return SnapshotResult(
            ticker.upper(),
            as_of_date,
            analysis_dir,
            report_dir,
            "failed",
            True,
            _failure_detail("analysis", analysis_rc, analysis_stdout, analysis_stderr),
        )

    critic_summary = None
    if active_setup.is_role_mixed:
        try:
            critic_summary = _apply_role_mixed_critic(
                active_setup=active_setup,
                base_config_path=base_config_path,
                runtime_config_path=runtime_dir / "critic_config.ini",
                analysis_dir=analysis_dir,
                ticker=ticker.upper(),
                company_name=company_name,
            )
        except Exception as exc:
            return SnapshotResult(ticker.upper(), as_of_date, analysis_dir, report_dir, "failed", True, f"critic:{exc}")

    report_cmd = [
        python_cmd,
        str(src_dir / "create_equity_report.py"),
        "--company-ticker",
        ticker.upper(),
        "--company-name",
        company_name,
        "--analysis-csv",
        str(analysis_dir / "financial_metrics_and_forecasts.csv"),
        "--ratios-csv",
        str(analysis_dir / "ratios_raw_data.csv"),
        "--tagline-file",
        str(analysis_dir / "tagline.txt"),
        "--company-overview-file",
        str(analysis_dir / "company_overview.txt"),
        "--investment-overview-file",
        str(analysis_dir / "investment_overview.txt"),
        "--valuation-overview-file",
        str(analysis_dir / "valuation_overview.txt"),
        "--risks-file",
        str(analysis_dir / "risks.txt"),
        "--competitor-analysis-file",
        str(analysis_dir / "competitor_analysis.txt"),
        "--major-takeaways-file",
        str(analysis_dir / "major_takeaways.txt"),
        "--news-summary-file",
        str(analysis_dir / "news_summary.txt"),
        "--peer-ev-ebitda-csv",
        str(analysis_dir / "peer_ev_ebitda_comparison.csv"),
        "--peer-ebitda-csv",
        str(analysis_dir / "peer_ebitda_comparison.csv"),
        "--output-dir",
        str(report_dir),
        "--config-file",
        str(analyst_config_path),
        "--fmp-cache-policy",
        fmp_cache_policy,
        "--enable-valuation-analysis",
    ]
    if force_refresh_fmp:
        report_cmd.append("--force-refresh-fmp")
    if fmp_cache_dir:
        report_cmd += ["--fmp-cache-dir", str(fmp_cache_dir)]
    report_stdout = logs_dir / "report_stdout.log"
    report_stderr = logs_dir / "report_stderr.log"
    report_rc = _run_command(
        report_cmd,
        cwd=project_root,
        stdout_path=report_stdout,
        stderr_path=report_stderr,
    )
    if report_rc != 0:
        return SnapshotResult(
            ticker.upper(),
            as_of_date,
            analysis_dir,
            report_dir,
            "failed",
            True,
            _failure_detail("report", report_rc, report_stdout, report_stderr),
        )

    manifest = {
        "generated_at_utc": utc_now_iso(),
        "ticker": ticker.upper(),
        "company_name": company_name,
        "as_of_date": as_of_date,
        "active_setup": active_setup.to_dict(),
        "critic_summary": critic_summary,
        "commands": {
            "analysis": analysis_cmd,
            "report": report_cmd,
        },
        "paths": {
            "analysis_dir": str(analysis_dir),
            "report_dir": str(report_dir),
            "logs_dir": str(logs_dir),
        },
    }
    _write_json(root / "snapshot_manifest.json", manifest)
    return SnapshotResult(ticker.upper(), as_of_date, analysis_dir, report_dir, "success", True)


def ensure_research_snapshot(
    *,
    integration_output_dir: str | Path,
    as_of_date: str,
    ticker: str,
    mode: str,
    active_setup: ActiveResearchSetup,
    base_config_path: str | Path | None,
    fmp_cache_policy: str,
    force_refresh_fmp: bool,
    fmp_cache_dir: str | Path | None,
    company_names: dict[str, str] | None = None,
    peer_map: dict[str, list[str]] | None = None,
    python_executable: str | None = None,
    enable_enhanced_news: bool = False,
    enable_catalyst_analysis: bool = False,
    news_days_back: int = 5,
    news_limit: int = 50,
) -> SnapshotResult:
    if mode not in RESEARCH_MODES:
        raise ValueError(f"research mode must be one of {sorted(RESEARCH_MODES)}")

    _, analysis_dir, report_dir = snapshot_dirs(integration_output_dir, as_of_date, ticker)
    exists = snapshot_exists(integration_output_dir, as_of_date, ticker)

    if mode == "existing":
        if not exists:
            raise FileNotFoundError(f"Missing research snapshot for {ticker} on {as_of_date}: {analysis_dir}")
        return SnapshotResult(ticker.upper(), as_of_date, analysis_dir, report_dir, "reused", False)

    if mode == "generate-missing" and exists:
        return SnapshotResult(ticker.upper(), as_of_date, analysis_dir, report_dir, "reused", False)

    return generate_research_snapshot(
        integration_output_dir=integration_output_dir,
        as_of_date=as_of_date,
        ticker=ticker,
        active_setup=active_setup,
        base_config_path=base_config_path,
        fmp_cache_policy=fmp_cache_policy,
        force_refresh_fmp=force_refresh_fmp,
        fmp_cache_dir=fmp_cache_dir,
        company_names=company_names,
        peer_map=peer_map,
        python_executable=python_executable,
        enable_enhanced_news=enable_enhanced_news,
        enable_catalyst_analysis=enable_catalyst_analysis,
        news_days_back=news_days_back,
        news_limit=news_limit,
    )
