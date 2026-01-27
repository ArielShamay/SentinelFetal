"""Runtime config loader for windowing and strict-mode invariants."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple, Type
import warnings

import yaml

DEFAULT_RUNTIME_CONFIG_PATH = Path("config/runtime.yaml")


@dataclass(frozen=True)
class RuntimeConfig:
    fs_hz: float
    window_minutes: float
    stride_minutes: float
    min_window_minutes: float
    recommended_case_minutes: float
    min_case_minutes: float
    strict_mode: bool


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _validate_runtime_config(cfg: RuntimeConfig, source: Path) -> None:
    if cfg.fs_hz <= 0:
        raise ValueError(f"{source} fs_hz must be > 0 (got {cfg.fs_hz})")
    if cfg.window_minutes < cfg.min_window_minutes:
        raise ValueError(
            f"{source} window_minutes must be >= min_window_minutes "
            f"({cfg.window_minutes} < {cfg.min_window_minutes})"
        )
    if cfg.recommended_case_minutes < cfg.min_case_minutes:
        raise ValueError(
            f"{source} recommended_case_minutes must be >= min_case_minutes "
            f"({cfg.recommended_case_minutes} < {cfg.min_case_minutes})"
        )


@lru_cache(maxsize=None)
def load_runtime_config(path: Path | None = None) -> RuntimeConfig:
    config_path = path or DEFAULT_RUNTIME_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Missing runtime config: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}
    cfg = RuntimeConfig(
        fs_hz=float(raw.get("fs_hz", 4.0)),
        window_minutes=float(raw.get("window_minutes", 20)),
        stride_minutes=float(raw.get("stride_minutes", 5)),
        min_window_minutes=float(raw.get("min_window_minutes", 20)),
        recommended_case_minutes=float(raw.get("recommended_case_minutes", 30)),
        min_case_minutes=float(raw.get("min_case_minutes", 20)),
        strict_mode=_as_bool(raw.get("strict_mode", False), False),
    )
    _validate_runtime_config(cfg, config_path)
    return cfg


def apply_strict_warnings(
    strict_mode: bool,
    allowlist: Iterable[Tuple[Type[Warning], str]] | None = None,
) -> None:
    """Optionally treat warnings as errors, except for allowlisted ones."""
    if not strict_mode:
        return
    warnings.filterwarnings("error")
    for category, message in allowlist or ():
        warnings.filterwarnings("ignore", category=category, message=message)
