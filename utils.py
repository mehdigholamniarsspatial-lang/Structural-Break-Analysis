"""
utils.py
========
General-purpose helper functions: dataset loading, date-column
auto-detection, date-to-numeric conversion, breakpoint parsing and
validation, and data-cleaning utilities.

All functions are pure (no Streamlit calls) so they can be unit
tested independently of the UI layer.
"""

from __future__ import annotations

import glob
import io
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DATE_COLUMN_HINTS, MIN_POINTS_PER_SEGMENT

logger = logging.getLogger(__name__)

# Tokens that represent "no data" in the GHG inventory sheets, e.g. " -   "
_MISSING_TOKENS = {"-", "--", "na", "n/a", "nan", ""}


class DataLoadError(Exception):
    """Raised when an uploaded dataset cannot be parsed or is unusable."""


class BreakpointError(Exception):
    """Raised when user-supplied breakpoints are invalid."""


# ----------------------------------------------------------------------
# File loading
# ----------------------------------------------------------------------
def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load a CSV or Excel file (as provided by Streamlit's file_uploader)
    into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The file-like object returned by ``st.file_uploader``.

    Returns
    -------
    pd.DataFrame
        The parsed dataset with columns unmodified.

    Raises
    ------
    DataLoadError
        If the file cannot be parsed or is empty.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            raise DataLoadError(
                f"Unsupported file extension for '{uploaded_file.name}'. "
                "Please upload a .csv, .xlsx, or .xls file."
            )
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to parse '{uploaded_file.name}': {exc}") from exc

    if df.empty:
        raise DataLoadError("The uploaded file contains no data rows.")

    # Drop fully empty columns/rows that sometimes appear from Excel exports
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    logger.info("Loaded dataset with shape %s from '%s'", df.shape, uploaded_file.name)
    return df


# ----------------------------------------------------------------------
# Date column detection & conversion
# ----------------------------------------------------------------------
def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Guess which column represents the date/time axis.

    Strategy (in order of priority):
      1. Column name matches a known hint (case-insensitive substring).
      2. Column dtype is already datetime64.
      3. Column can be successfully parsed as a datetime for >90% of rows.
      4. First numeric column that looks like a 4-digit year.

    Returns
    -------
    Optional[str]
        The best-guess column name, or None if no candidate is found.
    """
    columns = list(df.columns)

    # 1. Name-based hints
    for col in columns:
        lowered = str(col).lower()
        if any(hint in lowered for hint in DATE_COLUMN_HINTS):
            return col

    # 2. Already datetime dtype
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 3. Parseable as datetime
    for col in columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                return col

    # 4. Numeric column that looks like a year (e.g., 1950-2100)
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if not series.empty and series.between(1500, 2200).mean() > 0.9:
                return col

    return None


def to_numeric_time(series: pd.Series) -> Tuple[np.ndarray, str]:
    """
    Convert a date-like series into a numeric axis suitable for
    regression (fractional years), and report the inferred kind.

    Parameters
    ----------
    series : pd.Series
        The raw date/time column.

    Returns
    -------
    (np.ndarray, str)
        Numeric values (float, one per row) and a label describing the
        conversion applied ("datetime" or "numeric-year").
    """
    if pd.api.types.is_numeric_dtype(series):
        # Already numeric -- assume it represents years directly.
        return series.astype(float).to_numpy(), "numeric-year"

    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() < 0.5:
        # Fall back: try to coerce directly to float (e.g. "2004" as text)
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().mean() > 0.5:
            return coerced.astype(float).to_numpy(), "numeric-year"
        raise DataLoadError(
            "Could not interpret the selected date column as either a "
            "date/time value or a numeric year."
        )

    # Convert to a fractional-year float: year + (day_of_year / days_in_year)
    year = parsed.dt.year.astype(float)
    day_of_year = parsed.dt.dayofyear.astype(float)
    is_leap = parsed.dt.is_leap_year
    days_in_year = np.where(is_leap, 366.0, 365.0)
    fractional = year + (day_of_year - 1) / days_in_year
    return fractional.to_numpy(), "datetime"


def dates_from_numeric(values: np.ndarray, kind: str, reference_series: pd.Series) -> np.ndarray:
    """
    Convert numeric fractional-year values back to display-friendly
    labels for plotting axes when the original column was a real date.

    For "numeric-year" data this is a no-op (values are already years).
    """
    if kind == "numeric-year":
        return values
    # For datetime kind, convert fractional year back to a timestamp
    years = np.floor(values)
    remainder = values - years
    timestamps = []
    for y, r in zip(years, remainder):
        y = int(y)
        is_leap = (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)
        days = 366 if is_leap else 365
        day_offset = int(round(r * days))
        try:
            ts = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=day_offset)
        except (ValueError, OverflowError):
            ts = pd.Timestamp(year=y, month=1, day=1)
        timestamps.append(ts)
    return np.array(timestamps)


# ----------------------------------------------------------------------
# Breakpoint parsing / validation
# ----------------------------------------------------------------------
def parse_breakpoints(raw_text: str) -> List[float]:
    """
    Parse user-entered breakpoints from a free-text block (one value
    per line, or comma/space separated) into a sorted list of unique
    floats.

    Parameters
    ----------
    raw_text : str
        Raw text from the sidebar text area.

    Returns
    -------
    List[float]
        Sorted, de-duplicated breakpoint values. Empty list if no
        valid breakpoints were found.
    """
    if not raw_text or not raw_text.strip():
        return []

    # Split on newlines, commas, semicolons, or whitespace
    tokens = [
        tok.strip()
        for chunk in raw_text.replace(",", "\n").replace(";", "\n").splitlines()
        for tok in chunk.split()
        if tok.strip()
    ]

    values: List[float] = []
    for tok in tokens:
        try:
            # Support both plain years ("2004") and date strings ("2004-06-01")
            if any(sep in tok for sep in ("-", "/")) and len(tok) > 4:
                parsed = pd.to_datetime(tok, errors="raise")
                year = parsed.year + (parsed.dayofyear - 1) / 365.25
                values.append(float(year))
            else:
                values.append(float(tok))
        except (ValueError, TypeError):
            raise BreakpointError(f"Could not interpret structural break date '{tok}'.")

    return sorted(set(values))


def validate_breakpoints(
    breakpoints: Sequence[float], time_values: np.ndarray
) -> List[float]:
    """
    Validate breakpoints against the actual time range of the data and
    ensure each resulting segment has enough points to fit a
    regression.

    Raises
    ------
    BreakpointError
        If a breakpoint is out of range or produces an under-populated
        segment.
    """
    if not len(breakpoints):
        return []

    t_min, t_max = float(np.min(time_values)), float(np.max(time_values))
    for bp in breakpoints:
        if bp <= t_min or bp >= t_max:
            raise BreakpointError(
                f"Structural break date {bp:g} is outside the data range "
                f"({t_min:g} - {t_max:g}) and was ignored."
            )

    edges = [t_min] + list(breakpoints) + [t_max]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (time_values >= lo) & (time_values <= hi)
        else:
            mask = (time_values > lo) & (time_values <= hi)
        n_points = int(mask.sum())
        if n_points < MIN_POINTS_PER_SEGMENT:
            raise BreakpointError(
                f"Regime [{lo:g}, {hi:g}] only has {n_points} point(s); "
                f"at least {MIN_POINTS_PER_SEGMENT} are required. "
                "Please adjust the structural break dates."
            )

    return list(breakpoints)


# ----------------------------------------------------------------------
# Data cleaning
# ----------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Summary of cleaning operations applied to a dataset."""

    n_missing_dropped: int = 0
    n_duplicates_dropped: int = 0
    n_inf_dropped: int = 0
    was_sorted: bool = False


def clean_series_pair(
    time_values: np.ndarray, y_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, CleaningReport]:
    """
    Remove NaN/Inf pairs, duplicate time points (keeping the first),
    and sort by time. Returns cleaned arrays plus a report describing
    what was removed.
    """
    report = CleaningReport()
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    finite_mask = np.isfinite(t) & np.isfinite(y)
    n_before = len(t)
    t, y = t[finite_mask], y[finite_mask]
    report.n_missing_dropped = n_before - len(t)

    order = np.argsort(t, kind="mergesort")
    if not np.array_equal(order, np.arange(len(t))):
        report.was_sorted = True
    t, y = t[order], y[order]

    _, unique_idx = np.unique(t, return_index=True)
    n_before_dupes = len(t)
    unique_idx = np.sort(unique_idx)
    t, y = t[unique_idx], y[unique_idx]
    report.n_duplicates_dropped = n_before_dupes - len(t)

    return t, y, report


# ----------------------------------------------------------------------
# GHG inventory data-folder loading (wide format: sector x year)
# ----------------------------------------------------------------------
def discover_ghg_datasets(data_dir: str = "data") -> Dict[str, str]:
    """
    Scan a folder for GHG inventory CSV files and return a mapping of
    ``{gas_label: file_path}``, where the gas label is derived from
    the file name (e.g. ``data/CO2.csv`` -> ``"CO2"``).

    Parameters
    ----------
    data_dir : str
        Path to the folder containing one CSV per GHG parameter
        (e.g. CH4.csv, CO2.csv, N2O.csv, Total.csv).

    Returns
    -------
    Dict[str, str]
        Sorted mapping of gas label -> file path. Empty if the folder
        does not exist or contains no CSV files.
    """
    if not os.path.isdir(data_dir):
        return {}
    paths = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*.csv"))
        if os.path.basename(p).lower() != "inpact_policy_timeline.csv"
    )
    return {os.path.splitext(os.path.basename(p))[0]: p for p in paths}


def load_ghg_wide_csv(path: str) -> pd.DataFrame:
    """
    Load a GHG inventory CSV in "wide" format: first column is the
    sector name, every subsequent column is a year of emissions for
    that sector. Handles a UTF-8 BOM in the header, blank/dash
    missing-value tokens (e.g. ``" -   "``), and stray whitespace in
    both headers and cell values.

    Parameters
    ----------
    path : str
        Path to the CSV file (e.g. ``data/CO2.csv``).

    Returns
    -------
    pd.DataFrame
        Columns: ``sector`` (str) followed by one column per year
        (numeric, float, NaN where data is missing).

    Raises
    ------
    DataLoadError
        If the file cannot be parsed or has no usable sector column.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(f"Failed to read GHG inventory file '{path}': {exc}") from exc

    if df.empty or df.shape[1] < 2:
        raise DataLoadError(f"'{path}' does not look like a valid sector-by-year inventory file.")

    df.columns = [str(c).strip() for c in df.columns]
    sector_col = df.columns[0]
    df = df.rename(columns={sector_col: "sector"})
    df["sector"] = df["sector"].astype(str).str.strip()

    year_cols = [c for c in df.columns if c != "sector"]
    for col in year_cols:
        cleaned = df[col].astype(str).str.strip()
        cleaned = cleaned.where(~cleaned.str.lower().isin(_MISSING_TOKENS), other=np.nan)
        df[col] = pd.to_numeric(cleaned, errors="coerce")

    logger.info("Loaded GHG inventory '%s' with %d sectors and %d years", path, len(df), len(year_cols))
    return df


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Melt a wide sector-by-year GHG DataFrame (as returned by
    ``load_ghg_wide_csv``) into long format with columns
    ``sector``, ``year`` (float), and ``value`` (float).
    """
    year_cols = [c for c in df_wide.columns if c != "sector"]
    long_df = df_wide.melt(id_vars="sector", value_vars=year_cols, var_name="year", value_name="value")
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    return long_df


def get_sector_series(df_wide: pd.DataFrame, sector: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the (year, value) time series for a single sector from a
    wide-format GHG DataFrame, dropping years with missing data.
    """
    long_df = wide_to_long(df_wide)
    row = long_df[long_df["sector"] == sector].dropna(subset=["value"]).sort_values("year")
    if row.empty:
        raise DataLoadError(f"Sector '{sector}' has no numeric data to plot.")
    return row["year"].to_numpy(dtype=float), row["value"].to_numpy(dtype=float)
