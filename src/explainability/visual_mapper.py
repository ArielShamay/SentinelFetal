"""
Visual Mapper Module.

Maps explanation contributors to visual highlight regions
for overlay on the CTG graph.

References:
    - SentinelFetal V2.0 PRD, Section: Explainability Module
"""

from __future__ import annotations

import logging
from typing import List

from src.explainability.models import (
    Contributor,
    HighlightRegion,
)

logger = logging.getLogger(__name__)


# Color mapping from semantic names to RGBA strings
COLOR_MAP = {
    "red": "rgba(220, 53, 69, 0.3)",      # Pathological
    "orange": "rgba(253, 126, 20, 0.3)",  # Warning
    "yellow": "rgba(255, 193, 7, 0.25)",  # Caution
    "green": "rgba(40, 167, 69, 0.2)",    # Reassuring
    "blue": "rgba(0, 123, 255, 0.2)",     # Informational
}


class VisualMapper:
    """
    Creates visual highlight regions for the FHR graph.

    Converts contributor time regions to Plotly-compatible
    highlight specifications with appropriate colors.

    Example:
        >>> mapper = VisualMapper()
        >>> highlights = mapper.create_highlights(
        ...     contributors=explanation.contributors,
        ...     fhr_length=2400,
        ...     sampling_rate=4.0
        ... )
        >>> for h in highlights:
        ...     fig.add_vrect(x0=h["start"], x1=h["end"], ...)
    """

    def create_highlights(
        self,
        contributors: List[Contributor],
        fhr_length: int,
        sampling_rate: float = 4.0
    ) -> List[HighlightRegion]:
        """
        Convert contributor time regions to highlight specs.

        Args:
            contributors: List of Contributor objects with time regions.
            fhr_length: Total length of FHR signal in samples.
            sampling_rate: Sampling rate in Hz (for label generation).

        Returns:
            List of HighlightRegion objects for graph overlay.
        """
        highlights = []

        for contrib in contributors:
            if contrib.time_region is None:
                continue

            # Convert negative indices to positive
            start = contrib.time_region.start_index
            end = contrib.time_region.end_index

            if start < 0:
                start = max(0, fhr_length + start)
            if end < 0:
                end = max(0, fhr_length + end)

            # Clamp to valid range
            start = max(0, min(start, fhr_length - 1))
            end = max(0, min(end, fhr_length - 1))

            # Ensure start < end
            if start >= end:
                continue

            # Map color name to RGBA
            color = COLOR_MAP.get(
                contrib.time_region.color,
                "rgba(128, 128, 128, 0.2)"  # Gray default
            )

            # Create label from contributor name
            label = self._format_label(contrib.name)

            highlights.append(HighlightRegion(
                start=start,
                end=end,
                color=color,
                label=label,
                is_pathological=(contrib.contribution > 0)
            ))

        return highlights

    def _format_label(self, name: str) -> str:
        """
        Format contributor name into display label.

        Args:
            name: Raw contributor name (e.g., "late_decel").

        Returns:
            Formatted label (e.g., "Late Decel").
        """
        # Replace underscores with spaces and title case
        label = name.replace("_", " ").title()

        # Abbreviate long labels
        abbreviations = {
            "Late Decel": "Late Decel",
            "Variable Decel": "Var Decel",
            "Prolonged Decel": "Prolonged",
            "Sinusoidal": "Sinusoidal",
            "Variability": "Variability",
            "Accelerations": "Accel",
        }

        return abbreviations.get(label, label[:12])

    def merge_overlapping(
        self,
        highlights: List[HighlightRegion]
    ) -> List[HighlightRegion]:
        """
        Merge overlapping highlight regions of the same type.

        Prevents visual clutter when multiple events overlap.

        Args:
            highlights: List of highlight regions.

        Returns:
            List with overlapping regions merged.
        """
        if not highlights:
            return []

        # Sort by start position
        sorted_highlights = sorted(highlights, key=lambda h: h.start)

        merged = [sorted_highlights[0]]

        for current in sorted_highlights[1:]:
            last = merged[-1]

            # Check for overlap with same pathological status
            if (current.start <= last.end and
                current.is_pathological == last.is_pathological):
                # Merge: extend the end
                merged[-1] = HighlightRegion(
                    start=last.start,
                    end=max(last.end, current.end),
                    color=last.color if last.is_pathological else current.color,
                    label=last.label,
                    is_pathological=last.is_pathological
                )
            else:
                merged.append(current)

        return merged


def highlights_to_plotly_vrects(
    highlights: List[HighlightRegion],
    x_axis,  # np.ndarray or list of x values (timestamps)
) -> List[dict]:
    """
    Convert highlights to Plotly add_vrect() arguments.

    Args:
        highlights: List of HighlightRegion objects.
        x_axis: Array of x-axis values (timestamps or sample indices).

    Returns:
        List of dicts with Plotly vrect parameters.
    """
    vrects = []

    for h in highlights:
        if h.start >= len(x_axis) or h.end >= len(x_axis):
            continue

        vrects.append({
            "x0": x_axis[h.start],
            "x1": x_axis[h.end],
            "fillcolor": h.color,
            "line_width": 0,
            "layer": "below",
            "annotation_text": h.label,
            "annotation_position": "top left",
            "annotation_font_size": 8,
        })

    return vrects
