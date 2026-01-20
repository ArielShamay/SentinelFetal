# -*- coding: utf-8 -*-
"""
Minimal Plotly utilities for the synthetic multi-patient dashboard.

Design:
- White background, black strokes.
- Semantic colors only for alert shading: green, orange, red.
- Lightweight sparkline for grid cards; richer detail plot with XAI shading.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import TYPE_CHECKING, Any, Iterable, Optional, Tuple

from src.config import COLORS, CTG

if TYPE_CHECKING:
    from src.ui.app import SyntheticEvent, SyntheticPatient  # pragma: no cover


WHITE_BG = "#FFFFFF"
BLACK = "#000000"
GREEN = "#1A8F2B"
ORANGE = "#CC7A00"
RED = "#B00020"


def _event_color(event_type: str) -> str:
    if event_type == "Late Deceleration":
        return RED
    if event_type == "Variable Deceleration":
        return ORANGE
    if event_type == "Tachysystole":
        return ORANGE
    return BLACK


def create_patient_sparkline(patient: SyntheticPatient) -> go.Figure:
    """Tiny sparkline for grid cards (last 60 samples)."""
    if not patient.timestamps:
        return go.Figure()
    x = np.array(patient.timestamps[-60:])
    y = np.array(patient.fhr[-60:])
    x_norm = x - x.min()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_norm,
            y=y,
            mode="lines",
            line=dict(color=BLACK, width=1.5),
            hovertemplate="t+%{x:.1f}s | %{y:.0f} bpm<extra></extra>",
        )
    )
    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=WHITE_BG,
        paper_bgcolor=WHITE_BG,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def create_patient_detail_plot(patient: SyntheticPatient) -> go.Figure:
    """Full detail plot with baseline and event shading."""
    if not patient.timestamps:
        return go.Figure()

    x = np.array(patient.timestamps)
    y = np.array(patient.fhr)
    t0 = x.min()
    x_rel = x - t0

    fig = go.Figure()
    # FHR line
    fig.add_trace(
        go.Scatter(
            x=x_rel,
            y=y,
            mode="lines",
            name="FHR",
            line=dict(color=BLACK, width=2),
            hovertemplate="t+%{x:.1f}s | %{y:.0f} bpm<extra></extra>",
        )
    )

    # Baseline guide
    fig.add_hline(
        y=patient.baseline,
        line=dict(color=BLACK, dash="dash", width=1),
        annotation_text="Baseline",
        annotation_position="top right",
    )

    # Completed events
    for ev in patient.events:
        fig.add_vrect(
            x0=ev.start_ts - t0,
            x1=ev.end_ts - t0,
            fillcolor=_event_color(ev.event_type) + "33",
            line_width=0,
            annotation_text=ev.event_type,
            annotation_position="top left",
        )

    # Active event shading
    if patient.active_event:
        ev = patient.active_event
        fig.add_vrect(
            x0=ev.start_ts - t0,
            x1=ev.end_ts - t0,
            fillcolor=_event_color(ev.event_type) + "44",
            line_width=0,
            annotation_text=f"{ev.event_type} (active)",
            annotation_position="top left",
        )

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=30),
        plot_bgcolor=WHITE_BG,
        paper_bgcolor=WHITE_BG,
        showlegend=False,
        xaxis=dict(title="Seconds", zeroline=False, showgrid=False),
        yaxis=dict(title="bpm", range=[90, 190], showgrid=True, gridcolor="#e5e5e5"),
    )

    return fig


def _as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.array([], dtype=np.float32)
    arr = np.asarray(values)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _downsample_minmax(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast min/max downsampling.

    Produces up to ~2 points per bin; preserves local extrema.
    """
    n = len(y)
    if n == 0 or max_points <= 0 or n <= max_points:
        return x, y

    # Each bin contributes up to 2 points (min and max), so target ~max_points.
    target_bins = max(1, max_points // 2)
    bin_size = int(np.ceil(n / target_bins))
    if bin_size <= 1:
        return x, y

    out_x: list[float] = []
    out_y: list[float] = []

    for start in range(0, n, bin_size):
        end = min(n, start + bin_size)
        seg = y[start:end]
        if seg.size == 0:
            continue

        local_min = int(np.argmin(seg)) + start
        local_max = int(np.argmax(seg)) + start

        if local_min == local_max:
            out_x.append(float(x[local_min]))
            out_y.append(float(y[local_min]))
        else:
            first, second = (local_min, local_max) if local_min < local_max else (local_max, local_min)
            out_x.append(float(x[first]))
            out_y.append(float(y[first]))
            out_x.append(float(x[second]))
            out_y.append(float(y[second]))

    return np.asarray(out_x, dtype=np.float32), np.asarray(out_y, dtype=np.float32)


def _time_axis_minutes(n_samples: int, sampling_rate: float) -> np.ndarray:
    if n_samples <= 0:
        return np.array([], dtype=np.float32)
    # Use float32 to reduce serialization size.
    return (np.arange(n_samples, dtype=np.float32) / np.float32(sampling_rate * 60.0))


def _extract_deceleration_bounds_minutes(decel: Any, sampling_rate: float) -> Optional[Tuple[float, float]]:
    if decel is None:
        return None
    for start_attr, end_attr in (
        ("start_idx", "end_idx"),
        ("start_index", "end_index"),
        ("start_sample", "end_sample"),
    ):
        if hasattr(decel, start_attr) and hasattr(decel, end_attr):
            start_idx = getattr(decel, start_attr)
            end_idx = getattr(decel, end_attr)
            try:
                x0 = float(start_idx) / (sampling_rate * 60.0)
                x1 = float(end_idx) / (sampling_rate * 60.0)
                return x0, x1
            except (TypeError, ValueError):
                return None

    return None


def create_ctg_plot(
    *,
    fhr: Any,
    uc: Any,
    decelerations: Optional[Iterable[Any]] = None,
    sampling_rate: float = CTG.SAMPLING_RATE,
    title: str = "",
    mode: str = "monitor",
    window_minutes: float = 20.0,
    max_points: int = 2000,
    use_webgl: bool = True,
    uirevision: Optional[str] = None,
) -> go.Figure:
    """Create an optimized CTG plot (FHR + UC).

    Design goals:
    - Fast rendering/serialization via downsampling.
    - Stable pan/zoom via uirevision.
    - Scattergl for large traces (WebGL) when enabled.

    Args:
        fhr: FHR samples (array-like)
        uc: UC samples (array-like)
        decelerations: iterable of objects with (start_idx, end_idx, decel_type)
        sampling_rate: Hz
        title: plot title
        mode: 'monitor' (last window) or 'review' (full)
        window_minutes: used in monitor mode
        max_points: target max points per trace (approx.)
        use_webgl: enables Scattergl for FHR when large
        uirevision: stable key to preserve UI state
    """
    fhr_arr = _as_float_array(fhr)
    uc_arr = _as_float_array(uc)

    n = int(fhr_arr.size)
    if n == 0:
        return go.Figure()

    # Ensure uc aligns; tolerate missing/short UC.
    if uc_arr.size != fhr_arr.size:
        if uc_arr.size == 0:
            uc_arr = np.zeros_like(fhr_arr)
        else:
            m = min(int(uc_arr.size), n)
            fhr_arr = fhr_arr[-m:]
            uc_arr = uc_arr[-m:]
            n = m

    # Monitor mode: slice to last N minutes before downsampling.
    if mode == "monitor" and window_minutes and window_minutes > 0:
        window_samples = int(window_minutes * 60.0 * sampling_rate)
        if window_samples > 0 and n > window_samples:
            fhr_arr = fhr_arr[-window_samples:]
            uc_arr = uc_arr[-window_samples:]
            n = int(fhr_arr.size)

    x = _time_axis_minutes(n, sampling_rate)
    x_fhr, y_fhr = _downsample_minmax(x, fhr_arr, max_points=max_points)
    x_uc, y_uc = _downsample_minmax(x, uc_arr, max_points=max_points)

    scatter_cls = go.Scattergl if (use_webgl and len(y_fhr) >= 1500) else go.Scatter

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("דופק עוברי | FHR", "צירים | UC"),
    )

    fig.add_trace(
        scatter_cls(
            x=x_fhr,
            y=y_fhr,
            mode="lines",
            name="FHR",
            line=dict(color=COLORS.FHR, width=1.5),
            hovertemplate="%{y:.0f} bpm<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Normal range band + reference lines (cheap shapes).
    fig.add_hrect(
        y0=110,
        y1=160,
        fillcolor="rgba(40, 167, 69, 0.08)",
        line_width=0,
        row=1,
        col=1,
    )
    fig.add_hline(y=110, line_dash="dash", line_color="rgba(220, 53, 69, 0.35)", row=1, col=1)
    fig.add_hline(y=160, line_dash="dash", line_color="rgba(220, 53, 69, 0.35)", row=1, col=1)

    # UC: keep as SVG Scatter for fill stability.
    fig.add_trace(
        go.Scatter(
            x=x_uc,
            y=y_uc,
            mode="lines",
            name="UC",
            line=dict(color=COLORS.UC, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 140, 0, 0.18)",
            hovertemplate="%{y:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Deceleration shading (only if provided; robust to unknown types).
    if decelerations:
        for decel in decelerations:
            bounds = _extract_deceleration_bounds_minutes(decel, sampling_rate)
            if not bounds:
                continue
            x0, x1 = bounds
            # LATE gets stronger highlight; otherwise light.
            decel_type = getattr(decel, "decel_type", None)
            name = getattr(decel_type, "name", str(decel_type) if decel_type is not None else "")
            alpha = 0.18 if name == "LATE" else 0.10
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=f"rgba(255, 0, 0, {alpha})",
                line_width=0,
                row=1,
                col=1,
            )

    fig.update_layout(
        height=450,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        title=dict(text=title, x=0.02, xanchor="left") if title else None,
        uirevision=uirevision or "ctg",
    )

    fig.update_xaxes(title_text="Minutes", row=2, col=1, showgrid=False, zeroline=False)
    fig.update_yaxes(
        title_text="BPM",
        range=[50, 200],
        dtick=30,
        gridcolor=COLORS.GRID,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="AU",
        range=[0, 100],
        gridcolor=COLORS.GRID,
        row=2,
        col=1,
    )

    return fig
