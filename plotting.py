"""
plotting.py
===========
Publication-quality (Matplotlib) and interactive (Plotly) figure
builders for the Structural Break Analysis Dashboard.

Design goals, matching a Nature / Environmental Research Letters
aesthetic:
  * White background, black axes, serif-free professional typography.
  * Minor ticks, automatic tick spacing, anti-aliased rendering.
  * Non-overlapping per-segment annotation boxes (simple vertical
    collision-avoidance algorithm).
  * Optional confidence bands and structural-break-date reference lines.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import plotly.graph_objects as go
from scipy import stats

from config import FigureStyle
from regression import SegmentResult
from policies import add_matplotlib_policy_overlays, add_policy_overlays


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def _segment_ci_band(result: SegmentResult, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a simple pointwise confidence band around the fitted line
    for a segment, based on the standard error of prediction.
    """
    t = result.t_values
    n = result.n
    dof = max(n - 2, 1)
    t_crit = stats.t.ppf(0.5 + confidence / 2, dof)

    t_mean = np.mean(t)
    ss_t = np.sum((t - t_mean) ** 2) if len(t) > 1 else 1.0
    se_fit = np.sqrt(
        result.residual_variance * (1.0 / n + (t - t_mean) ** 2 / ss_t)
    )
    y_fit = result.intercept + result.slope * t
    margin = t_crit * se_fit
    return t, y_fit - margin, y_fit + margin


def _annotation_text(result: SegmentResult, units: str = "") -> str:
    def year_label(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    return (
        f"Regime {result.segment_index} ({year_label(result.start)}–{year_label(result.end)})\n"
        f"{{S = {result.slope:.3g} ktCO₂/yr}}\n"
        f"{{R² = {result.r_squared:.3f}, RMSE = {result.rmse:.3g}}}"
    )


def _box_positions(position: str, count: int):
    """Return paper/axes coordinates and anchors for stacked boxes."""
    step = 0.16
    if position == "Inside: Top Left":
        return [(0.02, 0.98 - i * step, "left", "top") for i in range(count)]
    if position == "Inside: Top Right":
        return [(0.98, 0.98 - i * step, "right", "top") for i in range(count)]
    if position == "Inside: Bottom Left":
        return [(0.02, 0.02 + i * step, "left", "bottom") for i in range(count)]
    if position == "Inside: Bottom Right":
        return [(0.98, 0.02 + i * step, "right", "bottom") for i in range(count)]
    spacing = 0.96 / max(count, 1)
    if position == "Outside: Top Left":
        return [(0.02 + i * spacing, 1.04, "left", "bottom") for i in range(count)]
    if position == "Outside: Top Right":
        return [(0.98 - i * spacing, 1.04, "right", "bottom") for i in range(count)]
    if position == "Outside: Bottom Left":
        return [(0.02 + i * spacing, -0.16, "left", "top") for i in range(count)]
    return [(0.98 - i * spacing, -0.16, "right", "top") for i in range(count)]


def apply_plotly_legend_position(
    fig: go.Figure,
    position: str,
    *,
    boxed: bool = True,
    orientation: str = "Vertical",
    columns: int = 1,
) -> None:
    """Apply placement, orientation, columns, and box styling to Plotly."""
    mapping = {
        "Top-left (inside)": dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
        "Top-right (inside)": dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
        "Bottom-left (inside)": dict(x=0.01, y=0.01, xanchor="left", yanchor="bottom"),
        "Bottom-right (inside)": dict(x=0.99, y=0.01, xanchor="right", yanchor="bottom"),
        "Outside right": dict(x=1.02, y=1.0, xanchor="left", yanchor="top"),
        "Outside left": dict(x=-0.02, y=1.0, xanchor="right", yanchor="top"),
        "Outside top": dict(x=0.5, y=1.12, xanchor="center", yanchor="bottom"),
        "Outside bottom": dict(x=0.5, y=-0.18, xanchor="center", yanchor="top"),
    }
    legend_style = {
        **mapping[position],
        "orientation": "h" if orientation == "Horizontal" else "v",
        "bgcolor": "rgba(255,255,255,0.90)" if boxed else "rgba(0,0,0,0)",
        "bordercolor": "#b8b8b8" if boxed else "rgba(0,0,0,0)",
        "borderwidth": 1 if boxed else 0,
        "font": {"color": "#111111"},
    }
    if orientation == "Horizontal":
        legend_style.update(
            entrywidth=max(0.08, 1.0 / max(int(columns), 1)),
            entrywidthmode="fraction",
        )
    fig.update_layout(legend=legend_style)
    if position == "Outside right":
        fig.update_layout(margin=dict(r=220))
    elif position == "Outside left":
        fig.update_layout(margin=dict(l=220))
    elif position == "Outside top":
        fig.update_layout(margin=dict(t=130))
    elif position == "Outside bottom":
        fig.update_layout(margin=dict(b=130))


# ----------------------------------------------------------------------
# Matplotlib (publication-quality, static export)
# ----------------------------------------------------------------------
def build_matplotlib_figure(
    t_all: np.ndarray,
    y_all: np.ndarray,
    results: Sequence[SegmentResult],
    style: FigureStyle,
    units: str = "",
    policies=None,
    policy_categories: Optional[Sequence[str]] = None,
    policy_styles=None,
    policy_label_settings=None,
) -> plt.Figure:
    """
    Build a publication-quality static Matplotlib figure showing the
    observed data, regime-specific OLS trajectories with HAC corrections,
    optional confidence bands, structural-break-date markers, and
    per-regime annotation boxes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt.rcParams.update({
        "font.family": style.font_family,
        "font.size": style.font_size,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.1,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.default": "regular",
    })

    fig, ax = plt.subplots(figsize=(style.fig_width, style.fig_height), dpi=style.dpi)

    if policies is not None and policy_categories:
        add_matplotlib_policy_overlays(
            ax, policies, policy_categories, policy_styles=policy_styles,
            label_settings=policy_label_settings,
        )

    # Observed data and the line connecting measurements in time order.
    measurement_order = np.argsort(t_all, kind="stable")
    line_styles = {
        "Solid": "-", "Dashed": "--", "Dotted": ":", "Dash-dot": "-."
    }
    ax.plot(
        np.asarray(t_all)[measurement_order],
        np.asarray(y_all)[measurement_order],
        color=style.measurement_line_color,
        linewidth=style.measurement_line_width,
        linestyle=line_styles[style.measurement_line_style],
        alpha=style.marker_alpha,
        zorder=2.5,
        label="Point measurements",
    )
    ax.scatter(
        t_all, y_all,
        s=style.marker_size,
        marker=style.marker_style,
        color=style.observation_color,
        alpha=style.marker_alpha,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
        label="_nolegend_",
    )

    palette = style.regression_palette
    annotation_positions = _box_positions(style.annotation_position, len(results))

    for idx, result in enumerate(results):
        color = palette[idx % len(palette)]
        t_seg = result.t_values
        y_fit = result.intercept + result.slope * t_seg

        ax.plot(
            t_seg, y_fit,
            color=color,
            linewidth=style.line_width,
            solid_capstyle="round",
            zorder=4,
            label=f"Regime {result.segment_index} trajectory",
        )

        if style.show_ci_band:
            _, lo, hi = _segment_ci_band(result)
            ax.fill_between(t_seg, lo, hi, color=color, alpha=0.15, zorder=2, linewidth=0)

        if style.show_breakpoint_lines and idx > 0:
            ax.axvline(result.start, color="#888888", linestyle="--", linewidth=1.0, zorder=1)

        if style.annotate_segments:
            text = _annotation_text(result, units=units)
            box_x, box_y, box_ha, box_va = annotation_positions[idx]
            ax.text(
                box_x,
                box_y,
                text,
                transform=ax.transAxes,
                fontsize=max(style.font_size - 3, 7),
                ha=box_ha,
                va=box_va,
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.0,
                    alpha=0.92,
                ),
                clip_on=False,
                zorder=5,
            )

    ax.set_xlabel(style.axis_label_x, fontsize=style.font_size + 1, fontweight="medium")
    ax.set_ylabel(style.axis_label_y, fontsize=style.font_size + 1, fontweight="medium")
    ax.set_title(style.figure_title, fontsize=style.title_font_size, fontweight="bold", pad=14)

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="major", direction="out", length=6, width=1.0, colors="black")
    ax.tick_params(which="minor", direction="out", length=3, width=0.8, colors="black")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.1)

    if style.show_grid:
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, color="#cccccc", zorder=0)
        ax.set_axisbelow(True)

    if style.show_legend:
        legend_mapping = {
            "Top-left (inside)": dict(loc="upper left"),
            "Top-right (inside)": dict(loc="upper right"),
            "Bottom-left (inside)": dict(loc="lower left"),
            "Bottom-right (inside)": dict(loc="lower right"),
            "Outside right": dict(loc="upper left", bbox_to_anchor=(1.02, 1.0)),
            "Outside left": dict(loc="upper right", bbox_to_anchor=(-0.02, 1.0)),
            "Outside top": dict(loc="lower center", bbox_to_anchor=(0.5, 1.02)),
            "Outside bottom": dict(loc="upper center", bbox_to_anchor=(0.5, -0.14)),
        }
        ax.legend(
            frameon=style.legend_boxed,
            framealpha=0.9 if style.legend_boxed else 0.0,
            facecolor="white" if style.legend_boxed else "none",
            edgecolor="#b8b8b8" if style.legend_boxed else "none",
            labelcolor="#111111",
            ncol=max(int(style.legend_columns), 1)
            if style.legend_orientation == "Horizontal" else 1,
            fontsize=max(style.font_size - 2, 8),
            **legend_mapping[style.legend_position],
        )

    annotation_outside_top = style.annotation_position.startswith("Outside: Top")
    annotation_outside_bottom = style.annotation_position.startswith("Outside: Bottom")
    rect = [
        0.24 if style.legend_position == "Outside left" else 0.0,
        0.24 if style.legend_position == "Outside bottom" or annotation_outside_bottom else 0.0,
        0.76 if style.legend_position == "Outside right" else 1.0,
        0.76 if style.legend_position == "Outside top" or annotation_outside_top else 1.0,
    ]
    fig.tight_layout(rect=rect)
    return fig


# ----------------------------------------------------------------------
# Plotly (interactive exploration)
# ----------------------------------------------------------------------
def build_plotly_figure(
    t_all: np.ndarray,
    y_all: np.ndarray,
    results: Sequence[SegmentResult],
    style: FigureStyle,
    units: str = "",
    policies=None,
    policy_categories: Optional[Sequence[str]] = None,
    policy_styles=None,
    policy_label_settings=None,
) -> go.Figure:
    """
    Build an interactive Plotly figure mirroring the Matplotlib static
    version, supporting zoom, pan, hover tooltips, and legend toggling.
    """
    fig = go.Figure()

    measurement_order = np.argsort(t_all, kind="stable")
    plotly_line_styles = {
        "Solid": "solid", "Dashed": "dash", "Dotted": "dot", "Dash-dot": "dashdot"
    }
    fig.add_trace(go.Scatter(
        x=np.asarray(t_all)[measurement_order],
        y=np.asarray(y_all)[measurement_order],
        mode="lines+markers",
        name="Point measurements",
        line=dict(
            color=style.measurement_line_color,
            width=style.measurement_line_width,
            dash=plotly_line_styles[style.measurement_line_style],
        ),
        marker=dict(size=7, color=style.observation_color, opacity=0.75,
                    line=dict(width=0.5, color="white")),
        hovertemplate="Time=%{x:.2f}<br>Value=%{y:.4g}<extra></extra>",
    ))

    palette = style.regression_palette
    for idx, result in enumerate(results):
        color = palette[idx % len(palette)]
        t_seg = result.t_values
        y_fit = result.intercept + result.slope * t_seg

        if style.show_ci_band:
            _, lo, hi = _segment_ci_band(result)
            fig.add_trace(go.Scatter(
                x=np.concatenate([t_seg, t_seg[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.12,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

        hover = (
            f"Regime {result.segment_index}<br>"
            f"{result.equation}<br>"
            f"R\u00b2={result.r_squared:.3f}<br>"
            f"RMSE={result.rmse:.3g}<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=t_seg, y=y_fit,
            mode="lines",
            name=f"Regime {result.segment_index} trajectory",
            line=dict(color=color, width=3),
            hovertemplate=hover,
        ))

        if style.show_breakpoint_lines and idx > 0:
            fig.add_vline(x=result.start, line_dash="dash", line_color="#999999", opacity=0.7)

    if style.annotate_segments:
        for result, (x, y, xanchor, yanchor) in zip(
            results, _box_positions(style.annotation_position, len(results))
        ):
            annotation_lines = _annotation_text(result, units).splitlines()
            annotation_html = "<b>" + annotation_lines[0] + "</b><br>" + "<br>".join(annotation_lines[1:])
            annotation_color = palette[(result.segment_index - 1) % len(palette)]
            fig.add_annotation(
                x=x, y=y, xref="paper", yref="paper",
                xanchor=xanchor, yanchor=yanchor,
                text=annotation_html,
                showarrow=False, align="left",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=annotation_color,
                borderwidth=1, borderpad=5,
                font=dict(size=max(style.font_size - 2, 8), color=annotation_color),
            )

    if policies is not None and policy_categories:
        add_policy_overlays(
            fig, policies, policy_categories,
            float(np.nanmin(y_all)), float(np.nanmax(y_all)),
            policy_styles=policy_styles,
            label_settings=policy_label_settings,
        )

    fig.update_layout(
        title=style.figure_title,
        xaxis_title=style.axis_label_x,
        yaxis_title=style.axis_label_y,
        template="plotly_white",
        font=dict(family=style.font_family, size=13),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#dddddd", borderwidth=1),
        hovermode="closest",
        dragmode="zoom",
        margin=dict(l=60, r=30, t=60, b=50),
    )
    fig.update_xaxes(showgrid=style.show_grid, gridcolor="#eeeeee", zeroline=False)
    fig.update_yaxes(showgrid=style.show_grid, gridcolor="#eeeeee", zeroline=False)
    if style.show_legend:
        apply_plotly_legend_position(
            fig,
            style.legend_position,
            boxed=style.legend_boxed,
            orientation=style.legend_orientation,
            columns=style.legend_columns,
        )
    else:
        fig.update_layout(showlegend=False)
    if style.annotation_position.startswith("Outside: Top"):
        fig.update_layout(margin=dict(t=170))
    elif style.annotation_position.startswith("Outside: Bottom"):
        fig.update_layout(margin=dict(b=170))

    return fig


# ----------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------
def build_diagnostics_figure(results: Sequence[SegmentResult], style: FigureStyle) -> plt.Figure:
    """
    Build a 2x2 diagnostics panel (pooled across all segments):
    residuals vs fitted, residual histogram, Q-Q plot, observed vs
    predicted.
    """
    all_resid = np.concatenate([r.residuals for r in results])
    all_fitted = np.concatenate([r.y_predicted for r in results])
    all_observed = np.concatenate([r.y_values for r in results])

    plt.rcParams.update({"font.family": style.font_family, "font.size": max(style.font_size - 1, 9)})
    fig, axes = plt.subplots(2, 2, figsize=(style.fig_width, style.fig_height), dpi=150)

    ax = axes[0, 0]
    ax.scatter(all_fitted, all_resid, s=18, color="#1b6ca8", alpha=0.75, edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs. Fitted", fontweight="bold", fontsize=11)

    ax = axes[0, 1]
    ax.hist(all_resid, bins=min(20, max(5, len(all_resid) // 3)), color="#2e8b57", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Histogram", fontweight="bold", fontsize=11)

    ax = axes[1, 0]
    (osm, osr), (slope, intercept, r) = stats.probplot(all_resid, dist="norm")
    ax.scatter(osm, osr, s=18, color="#8e44ad", alpha=0.75, edgecolors="white", linewidths=0.3)
    ax.plot(osm, intercept + slope * np.array(osm), color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered residuals")
    ax.set_title("Normal Q-Q Plot", fontweight="bold", fontsize=11)

    ax = axes[1, 1]
    ax.scatter(all_observed, all_fitted, s=18, color="#d68910", alpha=0.75, edgecolors="white", linewidths=0.3)
    lims = [min(all_observed.min(), all_fitted.min()), max(all_observed.max(), all_fitted.max())]
    ax.plot(lims, lims, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Observed vs. Predicted", fontweight="bold", fontsize=11)

    for a in axes.flat:
        a.grid(True, linestyle=":", linewidth=0.5, color="#dddddd")
        a.set_axisbelow(True)
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)

    fig.tight_layout()
    return fig
