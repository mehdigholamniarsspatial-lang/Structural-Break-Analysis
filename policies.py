"""Climate-policy timeline loading and Plotly overlay helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import plotly.graph_objects as go


POLICY_FILE = Path(__file__).resolve().parent / "data" / "INPACT_Policy_Timeline.csv"
POLICY_STYLES = {
    "EU": {"color": "#2563eb", "dash": "dash", "width": 1.6},
    "National": {"color": "#dc2626", "dash": "solid", "width": 1.6},
    "International": {"color": "#7c3aed", "dash": "dot", "width": 1.6},
}

MATPLOTLIB_LINESTYLES = {
    "solid": "-", "dash": "--", "dot": ":", "dashdot": "-.",
}


def _brief_policy_label(policy: pd.Series, max_length: int = 46) -> str:
    """Return a compact one-line policy label suitable for dense charts."""
    name = " ".join(str(policy["Policy / Instrument"]).split())
    if len(name) > max_length:
        name = name[: max_length - 1].rstrip() + "…"
    return f"{policy['Year']} · {name}"


def load_policy_timeline(path: Path = POLICY_FILE) -> pd.DataFrame:
    """Load and validate the bundled INPACT climate-policy timeline."""
    try:
        policies = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not load the policy timeline: {exc}") from exc

    required = {"Year", "Policy / Instrument", "Level", "Key target / provision"}
    missing = required.difference(policies.columns)
    if missing:
        raise ValueError(f"Policy timeline is missing columns: {', '.join(sorted(missing))}")

    policies = policies.copy()
    policies["Year"] = pd.to_numeric(policies["Year"], errors="coerce")
    policies["Level"] = policies["Level"].astype(str).str.strip()
    policies = policies.dropna(subset=["Year", "Policy / Instrument"])
    policies = policies[policies["Level"].isin(POLICY_STYLES)].sort_values("Year")
    policies["Year"] = policies["Year"].astype(int)
    return policies.reset_index(drop=True)


def policy_entry_text(policies: pd.DataFrame, category: str) -> str:
    """Format one editable ``year | policy`` entry per line."""
    subset = policies[policies["Level"] == category]
    return "\n".join(
        f"{int(row['Year'])} | {row['Policy / Instrument']}"
        for _, row in subset.iterrows()
    )


def filter_policy_timeline(
    policies: pd.DataFrame,
    entries_by_category: Mapping[str, str],
) -> pd.DataFrame:
    """Keep only policies whose editable sidebar entry remains present."""
    selected: set[tuple[str, str]] = set()
    for category, raw_entries in entries_by_category.items():
        for raw_line in raw_entries.splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#"):
                selected.add((category, line.casefold()))

    keep = []
    for _, row in policies.iterrows():
        token = f"{int(row['Year'])} | {row['Policy / Instrument']}".casefold()
        keep.append((row["Level"], token) in selected)
    return policies.loc[keep].reset_index(drop=True)


def add_policy_overlays(
    fig: go.Figure,
    policies: pd.DataFrame,
    enabled_categories: Iterable[str],
    y_min: float,
    y_max: float,
    policy_styles: Mapping[str, Mapping[str, object]] | None = None,
    label_settings: Mapping[str, object] | None = None,
) -> go.Figure:
    """Add hoverable vertical policy markers to a Plotly figure."""
    enabled = set(enabled_categories)
    span = y_max - y_min
    if span == 0:
        span = max(abs(y_max), 1.0)
    low, high = y_min - span * 0.03, y_max + span * 0.03

    shown_categories: set[str] = set()
    year_counts: dict[int, int] = {}
    for _, policy in policies.iterrows():
        category = policy["Level"]
        if category not in enabled:
            continue
        style = (policy_styles or POLICY_STYLES)[category]
        name = policy["Policy / Instrument"]
        description = policy["Key target / provision"]
        hover = (
            f"<b>{name}</b><br>Implementation date: {policy['Year']}<br>"
            f"Category: {category}<br>{description}<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[policy["Year"], policy["Year"]],
            y=[low, high],
            mode="lines",
            name=f"{category} policies",
            legendgroup=f"policy-{category}",
            showlegend=category not in shown_categories,
            line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
            opacity=0.82,
            hovertemplate=hover,
        ))
        if label_settings and label_settings.get("show", False):
            year = int(policy["Year"])
            same_year_index = year_counts.get(year, 0)
            year_counts[year] = same_year_index + 1
            fig.add_annotation(
                x=year,
                y=high,
                text=_brief_policy_label(policy),
                textangle=-90,
                xshift=float(label_settings.get("gap", 6))
                       + same_year_index * (float(label_settings.get("size", 9)) + 3),
                yshift=-2,
                xanchor="left",
                yanchor="top",
                showarrow=False,
                font=dict(
                    family=label_settings.get("font", "DejaVu Sans"),
                    size=label_settings.get("size", 9),
                    color=style["color"],
                ),
                opacity=0.9,
            )
        shown_categories.add(category)

    return fig


def add_matplotlib_policy_overlays(
    ax,
    policies: pd.DataFrame,
    enabled_categories: Iterable[str],
    policy_styles: Mapping[str, Mapping[str, object]] | None = None,
    label_settings: Mapping[str, object] | None = None,
) -> None:
    """Add publication-quality policy lines to a Matplotlib axis."""
    enabled = set(enabled_categories)
    styles = policy_styles or POLICY_STYLES
    shown_categories: set[str] = set()
    year_counts: dict[int, int] = {}
    for _, policy in policies.iterrows():
        category = policy["Level"]
        if category not in enabled:
            continue
        category_style = styles[category]
        ax.axvline(
            policy["Year"],
            color=category_style["color"],
            linestyle=MATPLOTLIB_LINESTYLES[category_style["dash"]],
            linewidth=category_style["width"],
            alpha=0.82,
            zorder=1,
            label=f"{category} policies" if category not in shown_categories else "_nolegend_",
        )
        if label_settings and label_settings.get("show", False):
            year = int(policy["Year"])
            same_year_index = year_counts.get(year, 0)
            year_counts[year] = same_year_index + 1
            ax.annotate(
                _brief_policy_label(policy),
                xy=(year, 0.98),
                xycoords=ax.get_xaxis_transform(),
                xytext=(
                    float(label_settings.get("gap", 6))
                    + same_year_index * (float(label_settings.get("size", 9)) + 3),
                    0,
                ),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=label_settings.get("size", 9),
                fontfamily=label_settings.get("font", "DejaVu Sans"),
                color=category_style["color"],
                alpha=0.9,
                clip_on=True,
                zorder=6,
            )
        shown_categories.add(category)
