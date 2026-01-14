"""
CTG Visualization Utilities for SentinelFetal.

This module provides Plotly-based visualizations for CTG data.
Visual requirements (Section 9.2):
- Blue line for FHR
- Orange line for UC  
- Red highlights for Decelerations

References:
    SentinelFetal Gen3.5 Technical Specification, Section 9
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.rules.decelerations import Deceleration


# Color scheme following spec
FHR_COLOR = '#1E90FF'  # Dodger Blue
UC_COLOR = '#FF8C00'   # Dark Orange
DECEL_COLOR = 'rgba(255, 0, 0, 0.3)'  # Red with transparency
GRID_COLOR = '#E5E5E5'
BACKGROUND_COLOR = '#FAFAFA'


def create_ctg_plot(
    fhr: np.ndarray,
    uc: np.ndarray,
    decelerations: Optional[List[Deceleration]] = None,
    sampling_rate: float = 4.0,
    title: str = "CTG Monitor",
    height: int = 600
) -> go.Figure:
    """
    Create an interactive CTG plot with FHR, UC, and deceleration markers.
    
    Args:
        fhr: FHR signal array.
        uc: UC (uterine contractions) signal array.
        decelerations: Optional list of detected decelerations to highlight.
        sampling_rate: Sampling rate in Hz (default 4 Hz for CTU-UHB).
        title: Plot title.
        height: Plot height in pixels.
        
    Returns:
        Plotly Figure object.
        
    Example:
        >>> fig = create_ctg_plot(fhr_data, uc_data, decel_list)
        >>> fig.show()  # Interactive display
        >>> st.plotly_chart(fig)  # Streamlit display
    """
    # Create time axis in minutes
    n_samples = len(fhr)
    time_seconds = np.arange(n_samples) / sampling_rate
    time_minutes = time_seconds / 60
    
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=('דופק עוברי (FHR)', 'צירים (UC)')
    )
    
    # === FHR Plot (top) ===
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=fhr,
            mode='lines',
            name='FHR',
            line=dict(color=FHR_COLOR, width=1.5),
            hovertemplate='זמן: %{x:.1f} דקות<br>דופק: %{y:.0f} bpm<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add normal range bands
    fig.add_hrect(
        y0=110, y1=160,
        fillcolor='rgba(0, 255, 0, 0.1)',
        line_width=0,
        row=1, col=1,
        annotation_text='טווח תקין',
        annotation_position='top right'
    )
    
    # Add deceleration highlights
    if decelerations:
        for i, decel in enumerate(decelerations):
            start_min = decel.start_idx / sampling_rate / 60
            end_min = decel.end_idx / sampling_rate / 60
            
            # Color by type
            decel_type_str = decel.decel_type.name if hasattr(decel.decel_type, 'name') else str(decel.decel_type)
            
            if 'LATE' in decel_type_str:
                fill_color = 'rgba(255, 0, 0, 0.3)'
                annotation = 'האטה מאוחרת'
            elif 'VARIABLE' in decel_type_str:
                fill_color = 'rgba(255, 165, 0, 0.3)'
                annotation = 'האטה משתנה'
            elif 'PROLONGED' in decel_type_str:
                fill_color = 'rgba(128, 0, 128, 0.3)'
                annotation = 'האטה ממושכת'
            else:
                fill_color = 'rgba(100, 100, 100, 0.2)'
                annotation = 'האטה'
            
            fig.add_vrect(
                x0=start_min, x1=end_min,
                fillcolor=fill_color,
                line_width=0,
                row=1, col=1
            )
            
            # Add annotation for first few decelerations only to avoid clutter
            if i < 5:
                fig.add_annotation(
                    x=(start_min + end_min) / 2,
                    y=decel.nadir_value if hasattr(decel, 'nadir_value') else 100,
                    text=annotation,
                    showarrow=False,
                    font=dict(size=8),
                    row=1, col=1
                )
    
    # === UC Plot (bottom) ===
    fig.add_trace(
        go.Scatter(
            x=time_minutes,
            y=uc,
            mode='lines',
            name='UC',
            line=dict(color=UC_COLOR, width=1.5),
            hovertemplate='זמן: %{x:.1f} דקות<br>צירים: %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # === Layout styling ===
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18),
            x=0.5
        ),
        height=height,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified',
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor='white'
    )
    
    # FHR y-axis
    fig.update_yaxes(
        title_text='bpm',
        range=[50, 220],
        dtick=30,
        gridcolor=GRID_COLOR,
        row=1, col=1
    )
    
    # UC y-axis
    fig.update_yaxes(
        title_text='mmHg',
        gridcolor=GRID_COLOR,
        row=2, col=1
    )
    
    # X-axis
    fig.update_xaxes(
        title_text='זמן (דקות)',
        gridcolor=GRID_COLOR,
        row=2, col=1
    )
    
    return fig


def create_category_indicator(
    category: int,
    confidence: float,
    size: int = 150
) -> go.Figure:
    """
    Create a category indicator gauge.
    
    Args:
        category: Classification category (1, 2, or 3).
        confidence: Model confidence (0-1).
        size: Size of the indicator in pixels.
        
    Returns:
        Plotly Figure object.
    """
    colors = {1: '#28a745', 2: '#fd7e14', 3: '#dc3545'}  # Green, Orange, Red
    labels = {1: 'תקין', 2: 'ביניים', 3: 'פתולוגי'}
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=category,
        title={'text': f"קטגוריה {category}<br><span style='font-size:0.6em'>{labels[category]}</span>"},
        gauge={
            'axis': {'range': [1, 3], 'tickvals': [1, 2, 3]},
            'bar': {'color': colors[category]},
            'steps': [
                {'range': [1, 1.5], 'color': 'rgba(40, 167, 69, 0.3)'},
                {'range': [1.5, 2.5], 'color': 'rgba(253, 126, 20, 0.3)'},
                {'range': [2.5, 3], 'color': 'rgba(220, 53, 69, 0.3)'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': category
            }
        },
        number={'suffix': f' ({confidence:.0%})'}
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_findings_summary(
    baseline: float,
    variability: dict,
    late_count: int,
    variable_count: int,
    tachysystole: bool
) -> go.Figure:
    """
    Create a visual summary of findings.
    
    Args:
        baseline: Baseline FHR value.
        variability: Variability dict with 'value' and 'category'.
        late_count: Number of late decelerations.
        variable_count: Number of variable decelerations.
        tachysystole: Whether tachysystole is detected.
        
    Returns:
        Plotly Figure object.
    """
    # Determine colors based on findings
    baseline_color = 'green' if 110 <= baseline <= 160 else 'red'
    
    var_cat = variability.get('category', 'Unknown')
    if var_cat in ('Moderate', 'MODERATE'):
        var_color = 'green'
    elif var_cat in ('Minimal', 'MINIMAL', 'Marked', 'MARKED'):
        var_color = 'orange'
    else:
        var_color = 'red'
    
    late_color = 'red' if late_count > 0 else 'green'
    variable_color = 'orange' if variable_count > 0 else 'green'
    tachy_color = 'red' if tachysystole else 'green'
    
    fig = go.Figure()
    
    categories = ['Baseline', 'שונות', 'האטות מאוחרות', 'האטות משתנות', 'Tachysystole']
    values = [
        baseline,
        variability.get('value', 0),
        late_count,
        variable_count,
        1 if tachysystole else 0
    ]
    colors = [baseline_color, var_color, late_color, variable_color, tachy_color]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=[1, 1, 1, 1, 1],  # Uniform height
        marker_color=colors,
        text=[
            f'{baseline:.0f} bpm',
            f'{variability.get("value", 0):.1f} bpm',
            str(late_count),
            str(variable_count),
            'כן' if tachysystole else 'לא'
        ],
        textposition='inside',
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title='סיכום ממצאים',
        showlegend=False,
        height=200,
        yaxis={'visible': False},
        xaxis={'tickangle': -45}
    )
    
    return fig
