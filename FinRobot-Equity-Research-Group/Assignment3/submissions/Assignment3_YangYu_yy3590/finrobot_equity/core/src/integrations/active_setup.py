"""Active FinRobot research setup registry for production integration runs."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_SETUP_TYPES = {"single_model", "role_mixed"}


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any], field_name: str) -> "ModelRef":
        if not isinstance(payload, dict):
            raise ValueError(f"{field_name} must be an object")
        provider = str(payload.get("provider", "")).strip().lower()
        model = str(payload.get("model", "")).strip()
        if provider not in {"openai", "claude", "gemini"}:
            raise ValueError(f"{field_name}.provider must be openai, claude, or gemini")
        if not model:
            raise ValueError(f"{field_name}.model is required")
        return cls(provider=provider, model=model)

    def to_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model}


@dataclass(frozen=True)
class ActiveResearchSetup:
    setup_type: str
    selected_setup_id: str
    analyst: ModelRef
    critic: ModelRef | None
    critic_sections: list[str]
    source_comparison_run: str | None
    selected_at_utc: str | None
    notes: str | None
    raw: dict[str, Any]

    @property
    def is_role_mixed(self) -> bool:
        return self.setup_type == "role_mixed"

    def to_dict(self) -> dict[str, Any]:
        payload = deepcopy(self.raw)
        payload["setup_type"] = self.setup_type
        payload["selected_setup_id"] = self.selected_setup_id
        payload["analyst"] = self.analyst.to_dict()
        if self.critic:
            payload["critic"] = self.critic.to_dict()
        payload["critic_sections"] = list(self.critic_sections)
        return payload


def default_active_setup_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "active_research_setup.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_active_research_setup(path: str | Path | None = None) -> ActiveResearchSetup:
    setup_path = Path(path) if path else default_active_setup_path()
    if not setup_path.exists():
        raise FileNotFoundError(f"Active research setup not found: {setup_path}")

    with setup_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Active research setup must be a JSON object")

    setup_type = str(payload.get("setup_type", "")).strip()
    if setup_type not in ALLOWED_SETUP_TYPES:
        raise ValueError(f"setup_type must be one of {sorted(ALLOWED_SETUP_TYPES)}")

    selected_setup_id = str(payload.get("selected_setup_id", "")).strip()
    if not selected_setup_id:
        raise ValueError("selected_setup_id is required")

    analyst = ModelRef.from_payload(payload.get("analyst") or {}, "analyst")
    critic = None
    critic_sections = list(payload.get("critic_sections") or [])
    if setup_type == "role_mixed":
        critic = ModelRef.from_payload(payload.get("critic") or {}, "critic")
        if not critic_sections:
            critic_sections = ["investment_overview", "risks", "major_takeaways"]
    elif payload.get("critic"):
        critic = ModelRef.from_payload(payload.get("critic") or {}, "critic")

    normalized_sections: list[str] = []
    for section in critic_sections:
        section_name = str(section).strip()
        if section_name:
            normalized_sections.append(section_name)

    return ActiveResearchSetup(
        setup_type=setup_type,
        selected_setup_id=selected_setup_id,
        analyst=analyst,
        critic=critic,
        critic_sections=normalized_sections,
        source_comparison_run=payload.get("source_comparison_run"),
        selected_at_utc=payload.get("selected_at_utc"),
        notes=payload.get("notes"),
        raw=payload,
    )
