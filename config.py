"""
config.py
=========
Centralized configuration for the Structural Break Analysis Dashboard.

Holds application constants, default figure styling, export options,
and other settings shared across modules. Keeping these values in one
place makes the rest of the codebase easier to maintain and keeps
default UI behaviour consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# ----------------------------------------------------------------------
# General application metadata
# ----------------------------------------------------------------------
APP_NAME: str = "Structural Break Analysis Dashboard"
APP_VERSION: str = "1.1.0"
APP_AUTHOR: str = "Scientific Software Team"

# ----------------------------------------------------------------------
# Data handling
# ----------------------------------------------------------------------
SUPPORTED_UPLOAD_TYPES: List[str] = ["csv", "xlsx", "xls"]

# Candidate substrings used to auto-detect a date/time column by name
DATE_COLUMN_HINTS: List[str] = [
    "date", "time", "year", "yr", "period", "timestamp", "datetime"
]

# ----------------------------------------------------------------------
# Regression defaults
# ----------------------------------------------------------------------
CONFIDENCE_LEVEL: float = 0.95
MIN_POINTS_PER_SEGMENT: int = 3  # minimum points required to fit a segment

# ----------------------------------------------------------------------
# Figure / plotting defaults (publication style)
# ----------------------------------------------------------------------
@dataclass
class FigureStyle:
    """Container for all user-adjustable figure style parameters."""

    fig_width: float = 10.0
    fig_height: float = 6.0
    dpi: int = 300
    font_family: str = "DejaVu Sans"
    font_size: int = 12
    title_font_size: int = 14
    marker_style: str = "o"
    marker_size: float = 22.0
    marker_alpha: float = 0.75
    measurement_line_width: float = 1.5
    measurement_line_style: str = "Solid"
    measurement_line_color: str = "#4d4d4d"
    line_width: float = 2.2
    regression_palette: List[str] = field(default_factory=lambda: [
        "#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad",
        "#d68910", "#16a085", "#7f8c8d", "#2c3e50",
    ])
    observation_color: str = "#4d4d4d"
    show_grid: bool = True
    show_legend: bool = True
    legend_position: str = "Bottom-left (inside)"
    legend_boxed: bool = True
    legend_orientation: str = "Horizontal"
    legend_columns: int = 2
    show_ci_band: bool = True
    show_breakpoint_lines: bool = True
    axis_label_x: str = "Date"
    axis_label_y: str = "Value"
    figure_title: str = "Structural Break Analysis"
    annotate_segments: bool = True
    annotation_position: str = "Inside: Top Right"


DEFAULT_STYLE = FigureStyle()

# ----------------------------------------------------------------------
# Export defaults
# ----------------------------------------------------------------------
EXPORT_IMAGE_FORMATS: List[str] = ["png", "pdf", "svg", "jpg", "tiff", "eps"]
EXPORT_DPI_OPTIONS: List[int] = [150, 300, 600, 1200]
DEFAULT_EXPORT_DPI: int = 300

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL: str = "INFO"
