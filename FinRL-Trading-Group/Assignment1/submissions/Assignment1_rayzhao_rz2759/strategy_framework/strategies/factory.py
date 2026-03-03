from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from strategy_framework.interfaces import AllocationStrategy

@dataclass(frozen=True)
class _StrategySpec:
    display_name: str
    strategy_cls: type
    config_cls: type | None
    aliases: set[str]


_REGISTRY_CACHE: dict[str, _StrategySpec] | None = None

_COMPAT_ALIAS_TO_DISPLAY = {
    "alpha": "alpha_rp_regime",
    "equal_weight": "equal_weight_topk",
    "eqw": "equal_weight_topk",
    "eqw_topk": "equal_weight_topk",
    "arcrra": "AdaptiveRegime_CorrRP_RankAlpha",
}

_DISPLAY_PRIORITY = [
    "alpha_rp_regime",
    "equal_weight_topk",
    "original_equal",
]


def _to_snake(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = s.replace("-", "_")
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    return s.strip("_").lower()


def _normalize_query(name: str) -> str:
    return _to_snake(name).replace("__", "_")


def _discover_specs() -> dict[str, _StrategySpec]:
    specs: dict[str, _StrategySpec] = {}
    strategy_dir = Path(__file__).resolve().parent
    module_prefix = __name__.rsplit(".", 1)[0]

    for py_file in sorted(strategy_dir.glob("*.py")):
        stem = py_file.stem
        if stem in {"__init__", "factory"}:
            continue

        module = import_module(f"{module_prefix}.{stem}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not hasattr(cls, "generate_signals") or not hasattr(cls, "name"):
                continue

            display_name = str(getattr(cls, "name", "")).strip()
            if not display_name:
                continue

            if not cls.__name__.endswith("Strategy"):
                continue
            cfg_cls_name = cls.__name__.replace("Strategy", "Config")
            cfg_cls = getattr(module, cfg_cls_name, None)
            if cfg_cls is not None and not inspect.isclass(cfg_cls):
                cfg_cls = None

            aliases = {
                display_name.lower(),
                _to_snake(display_name),
                _to_snake(stem),
                _to_snake(cls.__name__),
            }
            if hasattr(cls, "aliases"):
                aliases.update(_normalize_query(str(a)) for a in getattr(cls, "aliases", []))
            aliases = {_normalize_query(a) for a in aliases if a}

            specs[display_name] = _StrategySpec(
                display_name=display_name,
                strategy_cls=cls,
                config_cls=cfg_cls,
                aliases=aliases,
            )

    return specs


def _get_registry() -> dict[str, _StrategySpec]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _discover_specs()
    return _REGISTRY_CACHE


def _resolve_spec(strategy_name: str) -> _StrategySpec:
    registry = _get_registry()
    if not registry:
        raise RuntimeError("No strategies discovered under strategy_framework.strategies")

    q_raw = str(strategy_name).strip()
    if not q_raw:
        supported = ", ".join(list_available_strategies())
        raise ValueError(f"Unsupported strategy_name={strategy_name!r}. Supported: {supported}")

    compat_target = _COMPAT_ALIAS_TO_DISPLAY.get(_normalize_query(q_raw))
    if compat_target and compat_target in registry:
        return registry[compat_target]

    q_norm = _normalize_query(q_raw)
    for spec in registry.values():
        if q_raw == spec.display_name or q_norm in spec.aliases:
            return spec

    supported = ", ".join(list_available_strategies())
    raise ValueError(f"Unsupported strategy_name={strategy_name!r}. Supported: {supported}")


def list_available_strategies() -> list[str]:
    names = list(_get_registry().keys())
    if not names:
        return []

    ordered: list[str] = []
    for fixed in _DISPLAY_PRIORITY:
        if fixed in names:
            ordered.append(fixed)
    for name in sorted(names):
        if name not in ordered:
            ordered.append(name)
    return ordered


def create_strategy(
    strategy_name: str,
    config: Mapping[str, Any] | None = None,
    force_config_name: str | None = None,
) -> AllocationStrategy:
    spec = _resolve_spec(strategy_name)

    cfg_payload = dict(config or {})
    if force_config_name is not None:
        cfg_payload["force_config_name"] = force_config_name

    if spec.config_cls is None:
        if cfg_payload:
            try:
                return spec.strategy_cls(**cfg_payload)
            except TypeError:
                pass
        return spec.strategy_cls()

    return spec.strategy_cls(spec.config_cls(**cfg_payload))
