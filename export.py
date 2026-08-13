"""
export.py
=========
Export helpers for figures (PNG, JPEG, SVG, PDF, EPS, TIFF at
configurable DPI) and statistics tables (CSV, Excel, JSON).

All functions return raw bytes so they can be wired directly into
Streamlit's `st.download_button`.
"""

from __future__ import annotations

import io
import json
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def figure_to_bytes(fig: plt.Figure, fmt: str, dpi: int) -> bytes:
    """
    Render a Matplotlib figure to raw bytes in the requested format.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    fmt : str
        One of "png", "jpg"/"jpeg", "svg", "pdf", "eps", "tiff".
    dpi : int
        Output resolution in dots per inch (ignored for vector formats
        such as SVG/PDF/EPS, but still passed through harmlessly).

    Returns
    -------
    bytes
    """
    fmt = fmt.lower().replace("jpeg", "jpg")
    buf = io.BytesIO()

    save_kwargs = dict(dpi=dpi, bbox_inches="tight", facecolor="white")

    if fmt == "jpg":
        # Matplotlib needs 'jpg' handled via PIL pipeline through savefig(format="jpg")
        fig.savefig(buf, format="jpg", **save_kwargs)
    elif fmt == "tiff":
        fig.savefig(buf, format="tiff", **save_kwargs)
    elif fmt in ("png", "svg", "pdf", "eps"):
        fig.savefig(buf, format=fmt, **save_kwargs)
    else:
        raise ValueError(f"Unsupported export format: {fmt}")

    buf.seek(0)
    return buf.getvalue()


def stats_table_to_csv(rows: List[dict]) -> bytes:
    """Serialize statistics rows to CSV bytes."""
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def stats_table_to_json(rows: List[dict]) -> bytes:
    """Serialize statistics rows to pretty-printed JSON bytes."""
    return json.dumps(rows, indent=2, default=str).encode("utf-8")


def stats_table_to_excel(rows: List[dict], sheet_name: str = "Regression Statistics") -> bytes:
    """Serialize statistics rows to an in-memory Excel workbook (xlsx)."""
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        header_fmt = workbook.add_format({
            "bold": True, "bg_color": "#1b6ca8", "font_color": "white", "border": 1
        })
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_fmt)
            max_len = max(df[col_name].astype(str).map(len).max(), len(col_name)) + 2
            worksheet.set_column(col_idx, col_idx, min(max_len, 40))

    buf.seek(0)
    return buf.getvalue()


def cleaned_data_to_csv(t: "pd.Series", y: "pd.Series", t_label: str, y_label: str) -> bytes:
    """Export the cleaned (t, y) series used for regression as CSV bytes."""
    df = pd.DataFrame({t_label: t, y_label: y})
    return df.to_csv(index=False).encode("utf-8")
