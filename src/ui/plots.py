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
from typing import TYPE_CHECKING

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
