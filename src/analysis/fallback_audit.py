"""Fallback audit utilities for gauntlet runs."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Dict, List, Optional

import pandas as pd

_CASE_ID: ContextVar[Optional[str]] = ContextVar("case_id", default=None)
_FALLBACKS: List[Dict] = []


def set_case_context(case_id: Optional[str]) -> None:
    _CASE_ID.set(case_id)


def reset_fallback_audit() -> None:
    _FALLBACKS.clear()


def record_fallback(module: str, reason: str, stats: Dict) -> None:
    _FALLBACKS.append({
        "case_id": _CASE_ID.get(),
        "module": module,
        "reason": reason,
        **stats,
    })


def get_fallback_records() -> List[Dict]:
    return list(_FALLBACKS)


def write_fallback_audit(path) -> None:
    df = pd.DataFrame(_FALLBACKS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
