"""Fallback audit utilities (re-exported)."""

from __future__ import annotations

from src.analysis.fallback_audit import (  # noqa: F401
    get_case_context,
    get_fallback_records,
    get_window_context,
    raise_if_any_fallback,
    record_fallback,
    reset_fallback_audit,
    set_case_context,
    set_window_context,
    write_fallback_audit,
)

__all__ = [
    "get_case_context",
    "get_fallback_records",
    "get_window_context",
    "raise_if_any_fallback",
    "record_fallback",
    "reset_fallback_audit",
    "set_case_context",
    "set_window_context",
    "write_fallback_audit",
]
