"""
app.py
======
Structural Break Analysis Dashboard — main Streamlit application.

Run with:
    streamlit run app.py

This module is intentionally UI-only: all numerical logic lives in
`utils.py`, `regression.py`, `plotting.py`, and `export.py`. The app
wires user input (sidebar) to those functions and renders results
across a set of tabs (Home, Data, Inventory Overview, Regression,
Statistics, Diagnostics, Export, About).

Data sources
------------
Primary: pre-loaded GHG inventory CSVs bundled in the `data/` folder.
Each file represents one GHG parameter (its name is the file's stem,
e.g. `data/CO2.csv` -> "CO2"). Each file is a wide table: the first
column is the emission sector, and every other column is a year.

Secondary: any CSV/Excel file the user uploads on the fly (first
column = date, other columns = variables), handled by the original
generic single-variable time-series pipeline.
"""

from __future__ import annotations

import base64
import logging
import inspect
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import streamlit as st


from config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_EXPORT_DPI,
    EXPORT_DPI_OPTIONS,
    EXPORT_IMAGE_FORMATS,
    FigureStyle,
    SUPPORTED_UPLOAD_TYPES,
)
from export import (
    cleaned_data_to_csv,
    figure_to_bytes,
    stats_table_to_csv,
    stats_table_to_excel,
    stats_table_to_json,
)
from plotting import (
    apply_plotly_legend_position,
    build_diagnostics_figure,
    build_matplotlib_figure,
    build_plotly_figure,
)
from policies import (
    POLICY_STYLES,
    add_policy_overlays,
    filter_policy_timeline,
    load_policy_timeline,
    policy_entry_text,
)
from regression import fit_piecewise_regression, mann_kendall_test, sens_slope
from utils import (
    BreakpointError,
    DataLoadError,
    clean_series_pair,
    detect_date_column,
    discover_ghg_datasets,
    get_sector_series,
    load_dataset,
    load_ghg_wide_csv,
    parse_breakpoints,
    to_numeric_time,
    validate_breakpoints,
    wide_to_long,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data"


def run_mann_kendall(y: np.ndarray, alpha: float) -> dict:
    """Run MK across both the current and legacy regression module APIs."""
    parameters = inspect.signature(mann_kendall_test).parameters
    if "alpha" in parameters:
        result = mann_kendall_test(y, alpha=alpha)
    else:
        # Streamlit Cloud can briefly serve app.py with a cached/older
        # regression.py during a deployment. Its MK function accepted only y.
        result = mann_kendall_test(y)

    result = dict(result)
    n = len(y)
    result.setdefault("tau", result["S"] / (n * (n - 1) / 2) if n > 1 else 0.0)
    result["alpha"] = float(alpha)
    result["significant"] = bool(result["p_value"] < alpha)
    result["trend"] = (
        "increasing" if result["Z"] > 0 and result["significant"]
        else "decreasing" if result["Z"] < 0 and result["significant"]
        else "no significant monotonic change"
    )
    return result

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _asset_data_uri(filename: str) -> str:
    """Return a local PNG asset as an embeddable data URI."""
    asset_path = Path(__file__).resolve().parent / "assets" / filename
    encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_brand_header() -> None:
    """Render the INPACT/EPA brand banner at the top of every app view."""
    try:
        inpact_logo = _asset_data_uri("inpact-mark.png")
        epa_logo = _asset_data_uri("epa-logo.png")
    except OSError as exc:
        logger.warning("Could not load header assets: %s", exc)
        return

    st.markdown(
        textwrap.dedent(f"""
        <style>
        .block-container {{padding-top: 1.25rem;}}
        .inpact-app-header {{
            align-items:center; background:#fff; border:1px solid #e2e8ee;
            box-shadow:0 4px 14px rgba(22,50,79,.10); display:flex;
            gap:1.5rem; margin:0 0 1.25rem; min-height:132px;
            padding:1rem 1.6rem; width:100%;
        }}
        .inpact-brand-logo {{flex:0 0 auto; max-height:96px; width:auto; max-width:230px;}}
        .inpact-header-copy {{
            border-left:1px solid #dce3e9; flex:1 1 auto; min-width:0;
            padding-left:1.4rem;
        }}
        .inpact-header-title {{
            color:#16324f; font-size:1.55rem; font-weight:800;
            line-height:1.14; margin:0;
        }}
        .inpact-header-title .accent {{color:#2d9994;}}
        .inpact-header-subtitle {{
            color:#667085; font-size:.95rem; line-height:1.35;
            margin:.4rem 0 0; max-width:760px;
        }}
        .inpact-funder {{
            align-items:center; border-left:1px solid #dce3e9; display:flex;
            flex:0 0 auto; flex-direction:column; gap:.3rem; padding-left:1.4rem;
        }}
        .inpact-funded-by {{
            color:#667085; font-size:.68rem; font-weight:700;
            letter-spacing:.09em; text-transform:uppercase;
        }}
        .inpact-epa-logo {{height:auto; max-height:58px; width:190px;}}
        @media (max-width: 900px) {{
            .inpact-app-header {{min-height:110px; gap:1rem; padding:.8rem 1rem;}}
            .inpact-brand-logo {{max-height:78px; max-width:185px;}}
            .inpact-header-title {{font-size:1.2rem;}}
            .inpact-header-subtitle {{font-size:.8rem;}}
            .inpact-epa-logo {{width:145px;}}
        }}
        @media (max-width: 650px) {{
            .inpact-header-copy {{display:none;}}
            .inpact-app-header {{justify-content:space-between; min-height:92px;}}
            .inpact-funder {{border-left:0; padding-left:0;}}
            .inpact-funded-by {{display:none;}}
            .inpact-brand-logo {{max-height:68px; max-width:165px;}}
            .inpact-epa-logo {{width:125px;}}
        }}
        </style>
        <header class="inpact-app-header">
          <img class="inpact-brand-logo" src="{inpact_logo}" alt="INPACT project logo">
          <div class="inpact-header-copy">
            <h1 class="inpact-header-title"><span class="accent">Ireland</span> Greenhouse Gas Policy Impact Explorer</h1>
            <p class="inpact-header-subtitle">Explore greenhouse gas emissions, atmospheric pollutants, climate policies and policy effectiveness across Ireland.</p>
          </div>
          <div class="inpact-funder">
            <span class="inpact-funded-by">Funded by</span>
            <img class="inpact-epa-logo" src="{epa_logo}" alt="Funded by EPA Research">
          </div>
        </header>
        """),
        unsafe_allow_html=True,
    )


render_brand_header()


# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "data_mode": "Bundled GHG inventory (data/ folder)",
        "wide_df": None,
        "gas_label": None,
        "sector": None,
        "df": None,               # generic upload preview
        "filename": None,
        "date_col": None,
        "value_col": None,
        "results": None,
        "t_clean": None,
        "y_clean": None,
        "t_analysis": None,
        "y_analysis": None,
        "time_kind": None,
        "cleaning_report": None,
        "error_message": None,
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


_init_state()


def render_analysis_period(t_values: np.ndarray, time_kind: str, identity: str) -> tuple[float, float]:
    """Render synchronized start/end controls for the regression subset."""
    t_values = np.asarray(t_values, dtype=float)
    t_min, t_max = float(np.min(t_values)), float(np.max(t_values))
    fingerprint = f"{identity}|{time_kind}|{len(t_values)}|{t_min:.12g}|{t_max:.12g}"
    if st.session_state.get("analysis_period_fingerprint") != fingerprint:
        st.session_state["analysis_start"] = t_min
        st.session_state["analysis_end"] = t_max
        st.session_state["analysis_period_fingerprint"] = fingerprint

    st.sidebar.subheader("2. Analysis Period")
    st.sidebar.caption(
        "Only observations within this period are used for structural-break, "
        "regression and trend calculations. The chart still displays the full series."
    )
    start_col, end_col = st.sidebar.columns(2)
    number_format = "%.0f" if time_kind == "numeric-year" and np.allclose(t_values, np.round(t_values)) else "%.3f"
    step = 1.0 if number_format == "%.0f" else 0.01
    start = start_col.number_input(
        "Start", min_value=t_min, max_value=t_max,
        step=step, format=number_format, key="analysis_start",
    )
    end = end_col.number_input(
        "End", min_value=t_min, max_value=t_max,
        step=step, format=number_format, key="analysis_end",
    )
    return float(start), float(end)


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def render_sidebar() -> FigureStyle:
    st.sidebar.title("📈 " + APP_NAME)
    st.sidebar.caption(f"Version {APP_VERSION}")
    st.sidebar.markdown("---")

    st.sidebar.subheader("1. Data Source")
    ghg_datasets = discover_ghg_datasets(DATA_DIR)
    mode_options = ["Bundled GHG inventory (data/ folder)", "Upload custom CSV/Excel"]
    if not ghg_datasets:
        mode_options = ["Upload custom CSV/Excel"]
    data_mode = st.sidebar.radio("Choose a data source", mode_options)
    st.session_state["data_mode"] = data_mode

    units = ""
    breakpoints_raw = ""
    analysis_start = analysis_end = None
    date_col = value_col = None

    if data_mode.startswith("Bundled"):
        # ---- GHG inventory data folder --------------------------------
        gas_options = list(ghg_datasets.keys())
        default_gas_idx = gas_options.index("CO2") if "CO2" in gas_options else 0
        gas_label = st.sidebar.selectbox(
            "Greenhouse gas / parameter", gas_options, index=default_gas_idx
        )
        try:
            wide_df = load_ghg_wide_csv(ghg_datasets[gas_label])
            st.session_state["wide_df"] = wide_df
            st.session_state["gas_label"] = gas_label
            st.session_state["error_message"] = None
        except DataLoadError as exc:
            st.session_state["error_message"] = str(exc)
            st.session_state["wide_df"] = None
            return FigureStyle()

        sectors = sorted(wide_df["sector"].unique())
        if st.session_state["sector"] in sectors:
            default_sector_idx = sectors.index(st.session_state["sector"])
        elif "Energy Industries" in sectors:
            default_sector_idx = sectors.index("Energy Industries")
        else:
            default_sector_idx = 0
        sector = st.sidebar.selectbox("Sector", sectors, index=default_sector_idx)
        st.session_state["sector"] = sector

        units = st.sidebar.text_input("Units (for annotation labels)", value="kt CO2-eq")
        date_col, value_col = "Year", f"{gas_label} — {sector}"

        range_t, range_y = get_sector_series(wide_df, sector)
        range_t, _, _ = clean_series_pair(range_t, range_y)
        analysis_start, analysis_end = render_analysis_period(
            range_t, "numeric-year", f"bundled:{gas_label}:{sector}"
        )

        st.sidebar.subheader(r"3. Structural Break Dates ($T_b$)")
        st.sidebar.caption("One structural break year per line. Leave empty to estimate a single-regime trajectory over the whole period.")
        breakpoints_raw = st.sidebar.text_area(
            r"Structural Break Dates ($T_b$)", value="2001\n2015", height=100,
            placeholder="2001\n2015",
        )

    else:
        # ---- Generic upload -----------------------------------------
        st.sidebar.caption("First column = date, other columns = variables.")
        uploaded_file = st.sidebar.file_uploader("CSV or Excel file", type=SUPPORTED_UPLOAD_TYPES)
        if uploaded_file is not None:
            try:
                if st.session_state["filename"] != uploaded_file.name:
                    df = load_dataset(uploaded_file)
                    st.session_state["df"] = df
                    st.session_state["filename"] = uploaded_file.name
                    st.session_state["date_col"] = detect_date_column(df)
                    st.session_state["results"] = None
                    st.session_state["error_message"] = None
            except DataLoadError as exc:
                st.session_state["error_message"] = str(exc)
                st.session_state["df"] = None

        df = st.session_state["df"]
        if df is None:
            st.sidebar.info("Upload a dataset to begin.")
            return FigureStyle()

        columns = list(df.columns)
        default_date_idx = columns.index(st.session_state["date_col"]) if st.session_state["date_col"] in columns else 0
        date_col = st.sidebar.selectbox("Date / time column", columns, index=default_date_idx)
        st.session_state["date_col"] = date_col

        numeric_candidates = [c for c in columns if c != date_col]
        default_val_idx = numeric_candidates.index(st.session_state["value_col"]) if st.session_state["value_col"] in numeric_candidates else 0
        value_col = st.sidebar.selectbox("Variable to analyze", numeric_candidates, index=default_val_idx)
        st.session_state["value_col"] = value_col

        units = st.sidebar.text_input("Units (optional, for annotation labels)", value="")

        range_t, range_kind = to_numeric_time(df[date_col])
        range_y = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
        range_t, _, _ = clean_series_pair(range_t, range_y)
        analysis_start, analysis_end = render_analysis_period(
            range_t, range_kind, f"upload:{st.session_state['filename']}:{date_col}:{value_col}"
        )

        st.sidebar.subheader(r"3. Structural Break Dates ($T_b$)")
        st.sidebar.caption("One structural break date per line. Leave empty to estimate a single-regime trajectory.")
        breakpoints_raw = st.sidebar.text_area(r"Structural Break Dates ($T_b$)", value="", height=100, placeholder="1998\n2007\n2016")

    st.sidebar.subheader("Policy overlays")
    st.sidebar.caption("Show climate-policy markers on interactive emissions charts.")
    enabled_policies = []
    for category in POLICY_STYLES:
        if st.sidebar.checkbox(f"{category} policies", value=True, key=f"policy_{category}"):
            enabled_policies.append(category)
    st.session_state["policy_categories"] = enabled_policies

    try:
        sidebar_policies = load_policy_timeline()
    except ValueError:
        sidebar_policies = pd.DataFrame()

    st.sidebar.caption(
        "Edit the three lists below. Delete a `year | policy` row to remove "
        "that policy marker from all plots."
    )
    policy_defaults = {
        category: policy_entry_text(sidebar_policies, category)
        if not sidebar_policies.empty else ""
        for category in POLICY_STYLES
    }
    for category, default_text in policy_defaults.items():
        st.session_state.setdefault(f"policy_entries_{category}", default_text)
    if st.sidebar.button("Restore all policy entries", use_container_width=True):
        for category, default_text in policy_defaults.items():
            st.session_state[f"policy_entries_{category}"] = default_text

    policy_entries = {}
    for category in POLICY_STYLES:
        policy_entries[category] = st.sidebar.text_area(
            f"{category} policy timeline",
            height=150,
            key=f"policy_entries_{category}",
            help="One entry per line. Delete an entry to hide its policy line.",
        )
    st.session_state["policy_entries"] = policy_entries

    policy_styles = {}
    line_type_labels = {
        "Solid": "solid", "Dashed": "dash", "Dotted": "dot", "Dash-dot": "dashdot",
    }
    with st.sidebar.expander("Customize policy lines"):
        for category, defaults in POLICY_STYLES.items():
            st.markdown(f"**{category}**")
            color = st.color_picker(
                f"{category} colour", defaults["color"], key=f"policy_color_{category}",
            )
            default_line_label = next(
                label for label, value in line_type_labels.items() if value == defaults["dash"]
            )
            line_label = st.selectbox(
                f"{category} line type", list(line_type_labels),
                index=list(line_type_labels).index(default_line_label),
                key=f"policy_dash_{category}",
            )
            width = st.number_input(
                f"{category} line width", min_value=0.5, max_value=8.0,
                value=float(defaults["width"]), step=0.1, key=f"policy_width_{category}",
            )
            policy_styles[category] = {
                "color": color, "dash": line_type_labels[line_label], "width": width,
            }
    st.session_state["policy_styles"] = policy_styles

    with st.sidebar.expander("Customize policy labels"):
        show_policy_labels = st.checkbox(
            "Display brief labels beside policy lines", value=False,
            key="show_policy_labels",
        )
        policy_label_gap = st.slider(
            "Text gap from line (points)", 0, 40, 6,
            key="policy_label_gap",
        )
        policy_label_size = st.slider(
            "Policy label text size", 6, 20, 9,
            key="policy_label_size",
        )
        policy_label_font = st.selectbox(
            "Policy label font",
            ["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono", "Arial"],
            key="policy_label_font",
        )
    st.session_state["policy_label_settings"] = {
        "show": show_policy_labels,
        "gap": policy_label_gap,
        "size": policy_label_size,
        "font": policy_label_font,
    }

    # ---- Regression options -----------------------------------------
    st.sidebar.subheader("4. Regression Options")
    method_label = st.sidebar.selectbox(
        "Regression method",
        ["Ordinary Least Squares (OLS)", "Robust Regression (Huber)", "Theil-Sen"],
    )
    method_map = {
        "Ordinary Least Squares (OLS)": "ols",
        "Robust Regression (Huber)": "huber",
        "Theil-Sen": "theilsen",
    }
    method = method_map[method_label]
    confidence_pct = st.sidebar.slider("Confidence level (%)", 80, 99, 95)
    show_mk = st.sidebar.checkbox("Compute Mann-Kendall test", value=False)

    # ---- Figure options -----------------------------------------------
    st.sidebar.subheader("5. Figure Options")
    style = FigureStyle()
    default_title = value_col if value_col else "Structural Break Analysis"
    style.figure_title = st.sidebar.text_input("Figure title", value=f"Structural Break Analysis: {default_title}")
    style.axis_label_x = st.sidebar.text_input("X-axis label", value=date_col or "Date")
    style.axis_label_y = st.sidebar.text_input("Y-axis label", value=f"{value_col or 'Value'}{f' ({units})' if units else ''}")

    with st.sidebar.expander("Advanced figure customization"):
        style.fig_width = st.number_input("Figure width (in)", 4.0, 20.0, 10.0, 0.5)
        style.fig_height = st.number_input("Figure height (in)", 3.0, 16.0, 6.0, 0.5)
        style.font_size = st.number_input("Font size", 6, 24, 12, 1)
        style.marker_size = st.number_input("Marker size", 5.0, 100.0, 22.0, 1.0)
        style.marker_alpha = st.slider("Marker transparency", 0.1, 1.0, 0.75)
        st.markdown("**Point measurement line**")
        style.measurement_line_width = st.number_input(
            "Measurement line width", 0.5, 6.0, 1.5, 0.1
        )
        style.measurement_line_style = st.selectbox(
            "Measurement line style", ["Solid", "Dashed", "Dotted", "Dash-dot"]
        )
        style.measurement_line_color = st.color_picker(
            "Measurement line color", "#4d4d4d"
        )
        style.line_width = st.number_input("Regression line width", 0.5, 6.0, 2.2, 0.1)
        style.show_grid = st.checkbox("Show grid", value=True)
        style.show_legend = st.checkbox("Show legend", value=True)
        placement_options = [
            "Top-left (inside)", "Top-right (inside)",
            "Bottom-left (inside)", "Bottom-right (inside)",
            "Outside right", "Outside left", "Outside top", "Outside bottom",
        ]
        style.legend_position = st.selectbox(
            "Legend position", placement_options,
            index=placement_options.index(style.legend_position),
        )
        style.legend_boxed = st.radio(
            "Legend background", [True, False],
            format_func=lambda value: "Boxed" if value else "Transparent",
            horizontal=True,
        )
        style.legend_orientation = st.radio(
            "Legend orientation", ["Vertical", "Horizontal"],
            index=["Vertical", "Horizontal"].index(style.legend_orientation),
            horizontal=True,
        )
        style.legend_columns = st.number_input(
            "Legend columns", min_value=1, max_value=8,
            value=style.legend_columns, step=1,
            help="Controls item columns when Horizontal orientation is selected.",
        )
        style.show_ci_band = st.checkbox("Show 95% confidence band", value=True)
        style.show_breakpoint_lines = st.checkbox(r"Show structural break dates ($T_b$)", value=True)
        style.annotate_segments = st.checkbox("Show regime annotations (slope, R², RMSE)", value=True)
        annotation_placement_options = [
            "Inside: Top Left", "Inside: Top Right",
            "Inside: Bottom Left", "Inside: Bottom Right",
            "Outside: Top Left", "Outside: Top Right",
            "Outside: Bottom Left", "Outside: Bottom Right",
        ]
        style.annotation_position = st.selectbox(
            r"Regime trajectory ($\Delta\mathrm{Mt\ CO_2e}/\mathrm{year}$) annotation position", annotation_placement_options,
            index=annotation_placement_options.index(style.annotation_position),
        )

    st.session_state["_analysis_inputs"] = dict(
        date_col=date_col, value_col=value_col, units=units,
        breakpoints_raw=breakpoints_raw, method=method,
        confidence=confidence_pct / 100.0, show_mk=show_mk,
        analysis_start=analysis_start, analysis_end=analysis_end,
    )
    return style


style = render_sidebar()

try:
    all_policy_timeline = load_policy_timeline()
    policy_timeline = filter_policy_timeline(
        all_policy_timeline,
        st.session_state.get("policy_entries", {}),
    )
    policy_error = None
except ValueError as exc:
    all_policy_timeline = pd.DataFrame()
    policy_timeline = pd.DataFrame()
    policy_error = str(exc)


# ----------------------------------------------------------------------
# Run analysis (whenever inputs are ready)
# ----------------------------------------------------------------------
def run_analysis() -> None:
    inputs = st.session_state.get("_analysis_inputs")
    if not inputs:
        return

    data_mode = st.session_state["data_mode"]
    st.session_state["error_message"] = None
    try:
        if data_mode.startswith("Bundled"):
            wide_df = st.session_state["wide_df"]
            sector = st.session_state["sector"]
            if wide_df is None or sector is None:
                return
            t_numeric, y_raw = get_sector_series(wide_df, sector)
            time_kind = "numeric-year"
        else:
            df = st.session_state["df"]
            if df is None:
                return
            date_series = df[inputs["date_col"]]
            value_series = pd.to_numeric(df[inputs["value_col"]], errors="coerce")
            t_numeric, time_kind = to_numeric_time(date_series)
            y_raw = value_series.to_numpy()

        t_clean, y_clean, cleaning_report = clean_series_pair(t_numeric, y_raw)

        analysis_start = float(inputs["analysis_start"])
        analysis_end = float(inputs["analysis_end"])
        if analysis_start > analysis_end:
            raise ValueError("Analysis period start must be earlier than or equal to its end.")
        analysis_mask = (t_clean >= analysis_start) & (t_clean <= analysis_end)
        t_analysis = t_clean[analysis_mask]
        y_analysis = y_clean[analysis_mask]
        if len(t_analysis) < 3:
            raise ValueError(
                "The selected analysis period contains fewer than 3 valid observations. "
                "Please widen the start/end range."
            )

        breakpoints = parse_breakpoints(inputs["breakpoints_raw"])
        breakpoints = validate_breakpoints(breakpoints, t_analysis)

        results = fit_piecewise_regression(
            t_analysis, y_analysis, breakpoints,
            method=inputs["method"], confidence=inputs["confidence"],
        )
        whole_series_result = fit_piecewise_regression(
            t_analysis, y_analysis, [],
            method=inputs["method"], confidence=inputs["confidence"],
        )[0]
        mk_alpha = 1.0 - inputs["confidence"]

        st.session_state["t_clean"] = t_clean
        st.session_state["y_clean"] = y_clean
        st.session_state["t_analysis"] = t_analysis
        st.session_state["y_analysis"] = y_analysis
        st.session_state["time_kind"] = time_kind
        st.session_state["cleaning_report"] = cleaning_report
        st.session_state["results"] = results
        st.session_state["whole_series_result"] = whole_series_result
        st.session_state["whole_series_mk"] = (
            run_mann_kendall(y_analysis, alpha=mk_alpha) if inputs["show_mk"] else None
        )
        st.session_state["segment_mk"] = (
            {r.segment_index: run_mann_kendall(r.y_values, alpha=mk_alpha) for r in results}
            if inputs["show_mk"] else {}
        )

    except (BreakpointError, DataLoadError, ValueError) as exc:
        st.session_state["error_message"] = str(exc)
        st.session_state["results"] = None
        st.session_state["whole_series_result"] = None
        st.session_state["whole_series_mk"] = None
        st.session_state["segment_mk"] = {}
        st.session_state["t_analysis"] = None
        st.session_state["y_analysis"] = None


run_analysis()


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_labels = [
    "🏠 Home", "📁 Data", "🌍 Inventory Overview", "📉 Regression",
    "📊 Statistics", "🔍 Diagnostics", "💾 Export", "ℹ️ About",
]
tab_home, tab_data, tab_overview, tab_regression, tab_stats, tab_diag, tab_export, tab_about = st.tabs(
    tab_labels, default=tab_labels[-1]
)

# ---- HOME -------------------------------------------------------------
with tab_home:
    st.title(APP_NAME)
    st.markdown(
        """
        An interactive tool for **structural break analysis** of
        greenhouse gas emission inventories and other environmental,
        climate, and scientific time series. Choose a GHG parameter and
        sector from the bundled inventory data (or upload your own file),
        optionally define structural break dates ($T_b$), and get publication-quality
        regime-specific OLS trajectory figures with full statistics for every regime.
        """
    )
    if st.session_state["data_mode"].startswith("Bundled"):
        col1, col2, col3 = st.columns(3)
        col1.metric("GHG parameter", st.session_state["gas_label"] or "—")
        col2.metric("Sector", st.session_state["sector"] or "—")
        col3.metric("Regimes estimated", len(st.session_state["results"]) if st.session_state["results"] else 0)
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset loaded", "Yes" if st.session_state["df"] is not None else "No")
        col2.metric("Rows", len(st.session_state["df"]) if st.session_state["df"] is not None else 0)
        col3.metric("Regimes estimated", len(st.session_state["results"]) if st.session_state["results"] else 0)
    st.markdown("---")
    st.markdown(
        """
        **Workflow**
        1. In the sidebar, pick a bundled GHG parameter (CH4 / CO2 / N2O / Total) and sector, or upload your own CSV/Excel file.
        2. Optionally list structural break dates ($T_b$) — one per line.
        3. Browse the full inventory in the **Inventory Overview** tab.
        4. Review the fitted regression lines and statistics in the **Regression** and **Statistics** tabs.
        5. Check the **Diagnostics** tab for residual analysis.
        6. Export the figure (up to 1200 DPI) and statistics tables in the **Export** tab.
        """
    )

# ---- DATA ---------------------------------------------------------------
with tab_data:
    st.header("Dataset Preview")
    if st.session_state["data_mode"].startswith("Bundled"):
        wide_df = st.session_state["wide_df"]
        if wide_df is None:
            st.warning("No GHG inventory data could be loaded from the data/ folder.")
        else:
            st.caption(f"Source file: `data/{st.session_state['gas_label']}.csv` — sector-by-year wide format.")
            st.dataframe(wide_df, width='stretch', height=420)
            c1, c2, c3 = st.columns(3)
            c1.metric("Sectors", wide_df.shape[0])
            c2.metric("Years covered", wide_df.shape[1] - 1)
            n_missing = wide_df.drop(columns="sector").isna().sum().sum()
            c3.metric("Missing cells", int(n_missing))

            if st.session_state["cleaning_report"] is not None:
                rep = st.session_state["cleaning_report"]
                st.markdown(f"**Cleaning summary for sector `{st.session_state['sector']}`**")
                st.write(
                    f"- Missing/Inf years dropped: `{rep.n_missing_dropped}`\n"
                    f"- Duplicate years dropped: `{rep.n_duplicates_dropped}`\n"
                    f"- Data re-sorted chronologically: `{rep.was_sorted}`"
                )
    else:
        df = st.session_state["df"]
        if df is None:
            st.warning("No dataset loaded yet. Use the sidebar to upload a CSV/Excel file.")
        else:
            st.dataframe(df.head(50), width='stretch')
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Detected date column", st.session_state["date_col"] or "None")

            if st.session_state["cleaning_report"] is not None:
                rep = st.session_state["cleaning_report"]
                st.markdown("**Cleaning summary**")
                st.write(
                    f"- Missing/Inf values dropped: `{rep.n_missing_dropped}`\n"
                    f"- Duplicate time points dropped: `{rep.n_duplicates_dropped}`\n"
                    f"- Data re-sorted chronologically: `{rep.was_sorted}`"
                )

    if st.session_state["error_message"]:
        st.error(st.session_state["error_message"])

    st.markdown("---")
    st.subheader("Climate Policy Timeline")
    if policy_error:
        st.warning(policy_error)
    else:
        st.caption(
            f"Automatically loaded from `data/INPACT_Policy_Timeline.csv` — "
            f"showing {len(policy_timeline)} of {len(all_policy_timeline)} policies "
            "after applying the editable sidebar lists."
        )
        st.dataframe(policy_timeline, width='stretch', height=320)

# ---- INVENTORY OVERVIEW -----------------------------------------------
with tab_overview:
    st.header("GHG Inventory Overview")
    if not st.session_state["data_mode"].startswith("Bundled") or st.session_state["wide_df"] is None:
        st.info("Select 'Bundled GHG inventory (data/ folder)' as the data source to see a multi-sector overview.")
    else:
        wide_df = st.session_state["wide_df"]
        gas_label = st.session_state["gas_label"]
        long_df = wide_to_long(wide_df).dropna(subset=["value"])

        all_sectors = sorted(wide_df["sector"].unique())
        totals = long_df.groupby("sector")["value"].sum().sort_values(ascending=False)
        default_sectors = list(totals.head(6).index)

        chosen = st.multiselect(
            "Sectors to compare",
            all_sectors,
            default=[s for s in default_sectors if s in all_sectors],
        )

        if chosen:
            import plotly.graph_objects as go

            fig = go.Figure()
            for sector in chosen:
                sub = long_df[long_df["sector"] == sector].sort_values("year")
                fig.add_trace(go.Scatter(
                    x=sub["year"], y=sub["value"],
                    mode="lines+markers", name=sector,
                    hovertemplate="%{y:.2f}<extra>" + sector + "</extra>",
                ))
            fig.update_layout(
                title=f"{gas_label} emissions by sector",
                xaxis_title="Year",
                yaxis_title=f"{gas_label} emissions",
                template="plotly_white",
                legend=dict(bgcolor="rgba(255,255,255,0.85)"),
                hovermode="closest",
                dragmode="zoom",
                margin=dict(l=60, r=30, t=60, b=50),
            )
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
            if not policy_timeline.empty:
                add_policy_overlays(
                    fig, policy_timeline,
                    st.session_state.get("policy_categories", []),
                    float(long_df[long_df["sector"].isin(chosen)]["value"].min()),
                    float(long_df[long_df["sector"].isin(chosen)]["value"].max()),
                    policy_styles=st.session_state.get("policy_styles"),
                    label_settings=st.session_state.get("policy_label_settings"),
                )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Pick one or more sectors above to compare their time series.")

        st.markdown("---")
        st.subheader("Total emissions by sector (summed over all years)")
        st.bar_chart(totals.head(15))

# ---- REGRESSION -----------------------------------------------------------
with tab_regression:
    st.header("Structural Break Analysis")
    if st.session_state["error_message"] and st.session_state["results"] is None:
        st.error(st.session_state["error_message"])
    elif st.session_state["results"] is None:
        st.info("Configure a GHG parameter/sector (or upload a dataset) in the sidebar to see results.")
    else:
        results = st.session_state["results"]
        t_clean, y_clean = st.session_state["t_clean"], st.session_state["y_clean"]
        t_analysis, y_analysis = st.session_state["t_analysis"], st.session_state["y_analysis"]
        units = st.session_state["_analysis_inputs"]["units"]

        st.caption(
            f"Regression/trend analysis uses {len(t_analysis)} observations from "
            f"{t_analysis.min():g} to {t_analysis.max():g}; all {len(t_clean)} cleaned "
            "observations remain visible in the chart."
        )

        view_mode = st.radio(
            "Figure mode",
            ["Publication (Matplotlib)", "Interactive (Plotly)"],
            horizontal=True,
        )

        if view_mode == "Interactive (Plotly)":
            fig = build_plotly_figure(
                t_clean, y_clean, results, style, units=units,
                policies=policy_timeline,
                policy_categories=st.session_state.get("policy_categories", []),
                policy_styles=st.session_state.get("policy_styles"),
                policy_label_settings=st.session_state.get("policy_label_settings"),
            )
            st.plotly_chart(fig, width='stretch')
        else:
            mpl_fig = build_matplotlib_figure(
                t_clean, y_clean, results, style, units=units,
                policies=policy_timeline,
                policy_categories=st.session_state.get("policy_categories", []),
                policy_styles=st.session_state.get("policy_styles"),
                policy_label_settings=st.session_state.get("policy_label_settings"),
            )
            st.pyplot(mpl_fig, width='stretch')
            st.session_state["_last_mpl_fig"] = mpl_fig

        st.markdown("---")
        st.subheader("Regime Summary")
        cols = st.columns(min(len(results), 4))
        for i, r in enumerate(results):
            with cols[i % len(cols)]:
                st.metric(f"Regime {r.segment_index}", r.equation, f"R²={r.r_squared:.3f}")

        if st.session_state["_analysis_inputs"]["show_mk"]:
            st.markdown("---")
            st.subheader("Mann-Kendall Test (whole series)")
            mk = st.session_state["whole_series_mk"]
            sen = sens_slope(t_analysis, y_analysis)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Monotonic-change classification", mk["trend"])
            c2.metric("Kendall's Tau", f"{mk['tau']:.3f}")
            c3.metric("Z-score", f"{mk['Z']:.3f}")
            c4.metric("p-value", f"{mk['p_value']:.4f}")
            c5.metric("Sen's slope", f"{sen['slope']:.4g}")
            st.caption(
                f"Significance level α = {mk['alpha']:.3g}; "
                f"significant: {'Yes' if mk['significant'] else 'No'}."
            )

# ---- STATISTICS -----------------------------------------------------------
with tab_stats:
    st.header("Regression Statistics")
    if st.session_state["results"] is None:
        st.info("No results yet — configure your analysis in the sidebar.")
    else:
        show_mk = st.session_state["_analysis_inputs"]["show_mk"]
        whole_result = st.session_state["whole_series_result"]

        st.markdown("#### Whole-series regression statistics")
        whole_row = whole_result.as_table_row()
        whole_row["Scope"] = "Whole series"
        if show_mk:
            mk = st.session_state["whole_series_mk"]
            whole_row.update({
                "MK monotonic-change classification": mk["trend"],
                "MK Kendall's Tau": round(mk["tau"], 6),
                "MK Z-score": round(mk["Z"], 6),
                "MK p-value": round(mk["p_value"], 6),
                "MK Alpha": round(mk["alpha"], 6),
                "MK Significant": "Yes" if mk["significant"] else "No",
            })
        st.dataframe(pd.DataFrame([whole_row]), width='stretch')

        st.markdown("#### Regime regression statistics")
        rows = [r.as_table_row() for r in st.session_state["results"]]
        if show_mk:
            for row, result in zip(rows, st.session_state["results"]):
                mk = st.session_state["segment_mk"][result.segment_index]
                row.update({
                    "MK monotonic-change classification": mk["trend"],
                    "MK Kendall's Tau": round(mk["tau"], 6),
                    "MK Z-score": round(mk["Z"], 6),
                    "MK p-value": round(mk["p_value"], 6),
                    "MK Alpha": round(mk["alpha"], 6),
                    "MK Significant": "Yes" if mk["significant"] else "No",
                })
        stats_df = pd.DataFrame(rows)
        st.dataframe(stats_df, width='stretch')
        st.session_state["_stats_rows"] = rows

        st.markdown("#### Regime equations")
        for r in st.session_state["results"]:
            st.markdown(f"**Regime {r.segment_index}: {r.start:g}–{r.end:g}**")
            st.latex(
                f"y = {r.slope:.4g}x {'+' if r.intercept >= 0 else '-'} {abs(r.intercept):.4g} "
                rf"\quad (R^2={r.r_squared:.3f},\ \mathrm{{RMSE}}={r.rmse:.3g},\ n={r.n})"
            )
            st.caption(
                f"Slope = {r.slope:.6g}; intercept = {r.intercept:.6g}; "
                f"adjusted R² = {r.adj_r_squared:.4f}; MAE = {r.mae:.4g}; "
                f"p-value = {r.p_value:.4g}."
            )
            if show_mk:
                mk = st.session_state["segment_mk"][r.segment_index]
                st.write(
                    f"**Mann–Kendall:** monotonic-change classification = **{mk['trend']}**; "
                    f"Kendall's Tau = `{mk['tau']:.4f}`; Z-score = `{mk['Z']:.4f}`; "
                    f"p-value = `{mk['p_value']:.4g}`; α = `{mk['alpha']:.3g}`; "
                    f"significant = **{'Yes' if mk['significant'] else 'No'}**."
                )

# ---- DIAGNOSTICS -----------------------------------------------------------
with tab_diag:
    st.header("Regression Diagnostics")
    if st.session_state["results"] is None:
        st.info("No results yet — configure your analysis in the sidebar.")
    else:
        diag_fig = build_diagnostics_figure(st.session_state["results"], style)
        st.pyplot(diag_fig, width='stretch')
        st.caption(
            "Panels show pooled residuals across all estimated regimes: "
            "residuals vs. fitted values, residual distribution, a normal "
            "Q-Q plot, and observed vs. predicted values."
        )

# ---- EXPORT -----------------------------------------------------------
with tab_export:
    st.header("Export")
    if st.session_state["results"] is None:
        st.info("No results yet — configure your analysis in the sidebar.")
    else:
        results = st.session_state["results"]
        t_clean, y_clean = st.session_state["t_clean"], st.session_state["y_clean"]
        t_analysis, y_analysis = st.session_state["t_analysis"], st.session_state["y_analysis"]
        units = st.session_state["_analysis_inputs"]["units"]

        st.subheader("Figure Export")
        c1, c2 = st.columns(2)
        with c1:
            export_format = st.selectbox("Image format", EXPORT_IMAGE_FORMATS, index=0)
        with c2:
            export_dpi = st.selectbox("Resolution (DPI)", EXPORT_DPI_OPTIONS, index=EXPORT_DPI_OPTIONS.index(DEFAULT_EXPORT_DPI))

        export_fig = build_matplotlib_figure(
            t_clean, y_clean, results, style, units=units,
            policies=policy_timeline,
            policy_categories=st.session_state.get("policy_categories", []),
            policy_styles=st.session_state.get("policy_styles"),
            policy_label_settings=st.session_state.get("policy_label_settings"),
        )
        try:
            fig_bytes = figure_to_bytes(export_fig, export_format, export_dpi)
            fname_stub = "piecewise_regression"
            if st.session_state["data_mode"].startswith("Bundled"):
                fname_stub = f"{st.session_state['gas_label']}_{st.session_state['sector']}".replace(" ", "_")
            st.download_button(
                f"Download figure (.{export_format}, {export_dpi} DPI)",
                data=fig_bytes,
                file_name=f"{fname_stub}_{export_dpi}dpi.{export_format}",
                mime="application/octet-stream",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not render figure in '{export_format}' format: {exc}")

        st.markdown("---")
        st.subheader("Statistics Table Export")
        rows = [r.as_table_row() for r in results]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Download CSV", data=stats_table_to_csv(rows),
                                file_name="regression_statistics.csv", mime="text/csv")
        with c2:
            st.download_button("Download Excel", data=stats_table_to_excel(rows),
                                file_name="regression_statistics.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3:
            st.download_button("Download JSON", data=stats_table_to_json(rows),
                                file_name="regression_statistics.json", mime="application/json")

        st.markdown("---")
        st.subheader("Analysis Data Export")
        date_col = st.session_state["_analysis_inputs"]["date_col"] or "Year"
        value_col = st.session_state["_analysis_inputs"]["value_col"] or "Value"
        st.download_button(
            "Download analyzed (t, y) subset as CSV",
            data=cleaned_data_to_csv(pd.Series(t_analysis), pd.Series(y_analysis), date_col, value_col),
            file_name="analysis_period_series.csv", mime="text/csv",
        )

# ---- ABOUT -----------------------------------------------------------
with tab_about:
    about_path = Path(__file__).resolve().parent / "data" / "about.md"
    try:
        about_text = about_path.read_text(encoding="utf-8-sig")
        section_headings = {
            "About the INPACT Project", "Mission", "Explorer Dashboard",
            "Data Sources", "Scientific Methodology", "Why This Dashboard Matters",
            "Technology Stack", "Research Team", "Funding",
        }
        about_lines = about_text.splitlines()
        heading_rows = [(idx, line) for idx, line in enumerate(about_lines) if line.strip()][:4]
        if len(heading_rows) == 4:
            st.title(heading_rows[1][1])
            st.subheader(heading_rows[2][1])
            st.caption(heading_rows[3][1])
            body_lines = [
                f"## {line}" if line.strip() in section_headings else line
                for line in about_lines[heading_rows[3][0] + 1:]
            ]
            st.markdown("\n".join(body_lines))
        else:
            st.markdown(about_text)
    except OSError as exc:
        st.error(f"Could not load the About page content: {exc}")
