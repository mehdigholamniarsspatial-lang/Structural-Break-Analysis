# Structural Break Analysis Dashboard

An interactive Streamlit application for **structural break analysis** of greenhouse gas emission inventories and other environmental, climate, atmospheric, hydrology, ecology, GIS, and remote-sensing time series.

The emissions charts automatically load `data/INPACT_Policy_Timeline.csv`. Sidebar controls independently toggle EU, National (Ireland), and International policy markers and let users customize each category's colour, line type, and width. Optional compact labels beside the policy lines can also be configured for gap, text size, and font. The settings apply consistently to interactive Plotly charts, publication-quality Matplotlib figures, and exported figures. Hovering an interactive vertical marker shows the policy name, implementation year, category, and key provision.

Legend and regime-trajectory-statistics annotations can be independently positioned in any plot corner or outside the figure on the right, left, top, or bottom. Outside placements automatically reserve additional margins for publication and export output.

Legends can use a bordered white box or a transparent background with high-contrast text. Users can also select vertical or horizontal arrangement and configure the number of columns used by horizontal legends.

Three editable sidebar timeline boxes list the EU, National, and International policies as `year | policy name`. Removing a row suppresses that individual policy across all charts and exports; a restore button returns all bundled policies.

The app reads its primary dataset directly from CSV files bundled in the `data/` folder — no upload needed. Pick a GHG parameter (the file name) and a sector, optionally define any number of Structural Break Dates ($T_b$), and get publication-quality regression figures with a full statistics suite for every regime. A generic "upload your own file" mode is also available for one-off datasets.

---

## Features

- **Bundled GHG inventory data** — `data/CH4.csv`, `data/CO2.csv`, `data/N2O.csv`, `data/Total.csv` load automatically; the GHG parameter name is taken directly from the file name and appears in the sidebar dropdown. Drop in another CSV with the same layout and it shows up automatically — no code changes needed.
- **Sector-by-year wide-format parsing** — each file's first column is the emission sector, every other column is a year. The loader handles a UTF-8 BOM in the header, blank/dash missing-value cells (e.g. `" -   "`), and stray whitespace.
- **Inventory Overview tab** — compare multiple sectors' time series on one interactive chart, plus a bar chart of total emissions by sector, for presenting the full inventory at a glance.
- **User-defined Structural Break Dates ($T_b$)** — enter any number of dates, one per line. They are automatically sorted, de-duplicated, and validated against the data range. With no structural break dates, a single-regime trajectory spans the entire period.
- **Full regression statistics per regime** — slope, intercept, regression equation, R², adjusted R², RMSE, MAE, MSE, standard error, 95% confidence interval, t-statistic, p-value, residual variance, and N.
- **Optional regression methods** — Ordinary Least Squares, Huber robust regression, and Theil-Sen. Optional Mann-Kendall test and Sen's slope estimator for the whole series.
- **Publication-quality figures** — white background, black axes, minor ticks, auto-legend, optional 95% confidence bands, Structural Break Date ($T_b$) reference lines, and automatically positioned, non-overlapping per-regime annotation boxes (slope, R², RMSE, equation).
- **Interactive exploration** — a Plotly view with zoom, pan, and hover tooltips alongside the static Matplotlib publication view.
- **Diagnostics** — residuals vs. fitted, residual histogram, normal Q-Q plot, and observed vs. predicted, pooled across all estimated regimes.
- **Multi-format export** — figures as PNG, JPEG, SVG, PDF, EPS, or TIFF at 150/300/600/1200 DPI; statistics tables as CSV, Excel, or JSON; cleaned series as CSV.
- **Generic upload mode** — switch the sidebar's data source to "Upload custom CSV/Excel" to analyze any other date-indexed dataset with the same tool.

---

## Installation

```bash
git clone <this-repository-url>
cd PiecewiseRegressionDashboard
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

No code changes are required — the app runs immediately after `pip install -r requirements.txt`.

---

## Example workflow

1. In the sidebar, keep the default data source **"Bundled GHG inventory (data/ folder)"**, pick a **GHG parameter** (CH4 / CO2 / N2O / Total) and a **sector**.
2. In **Structural Break Dates ($T_b$)**, optionally type one or more years, one per line (e.g. `2007`) — or leave blank to estimate a single-regime trajectory.
3. Choose a regression method (OLS / Huber / Theil-Sen) and confidence level.
4. Open the **Inventory Overview** tab to compare several sectors at once and see total emissions by sector.
5. Open the **Regression** tab to view the interactive or publication figure for the selected sector.
6. Open **Statistics** for the full per-regime statistics table, and **Diagnostics** for residual analysis.
7. Open **Export** to download the figure (choose format + DPI) and the statistics table (CSV / Excel / JSON).

To analyze a different dataset, switch the sidebar's data source to **"Upload custom CSV/Excel"** and upload any file where the first column is a date and other columns are variables.

## Data format

### Bundled GHG inventory files (`data/` folder)

Each file is one GHG parameter, named after the gas (e.g. `CO2.csv` → "CO2" in the sidebar). Rows are emission sectors; columns are years:

| sector                    | 1990    | 1991    | ... | 2025   |
|---------------------------|---------|---------|-----|--------|
| Energy Industries         | 11145.01| 11604.44| ... | 6454.84|
| Public electricity...     | 10876.49| 11361.81| ... | 6157.51|
| ...                       | ...     | ...     | ... | ...    |

Missing values may be blank or `-`; these are automatically treated as missing data and excluded from the regression for that year. To add another gas or an updated inventory, drop a CSV with this same layout into `data/` — it will appear in the sidebar automatically, with the file name (without `.csv`) as its label.

### Custom upload mode

The first column should be a date, datetime, or year value; every other column is treated as a candidate variable to analyze. Example:

| Date       | CO2_MtCO2 | Temp_Anomaly_C |
|------------|-----------|-----------------|
| 1990-06-15 | 350.46    | 0.249           |
| 1991-06-15 | 350.24    | 0.310           |
| ...        | ...       | ...             |

(A synthetic example of this format ships in `example_data/co2_emissions_example.csv` for testing the upload mode.)

## Export instructions

Go to the **Export** tab, choose an image format (PNG/JPEG/SVG/PDF/EPS/TIFF) and a DPI (150/300/600/1200 — 300 DPI is the common publication default), then click **Download figure**. Statistics tables can be downloaded separately as CSV, Excel, or JSON from the same tab.

## Project architecture

```
PiecewiseRegressionDashboard/
├── app.py            # Streamlit UI: sidebar, tabs, orchestration
├── config.py          # App constants and default figure style
├── utils.py           # Data loading, date detection/conversion, structural-break-date parsing, cleaning
├── regression.py       # Piecewise OLS/Huber/Theil-Sen fitting, full statistics, Mann-Kendall/Sen's slope
├── plotting.py         # Matplotlib (publication) and Plotly (interactive) figure builders, diagnostics
├── export.py           # Figure and statistics-table export helpers
├── requirements.txt
├── data/                        # Bundled GHG inventory CSVs (primary data source)
│   ├── CH4.csv
│   ├── CO2.csv
│   ├── N2O.csv
│   └── Total.csv
├── example_data/                # Sample file for the generic "upload" mode
│   └── co2_emissions_example.csv
├── LICENSE
└── .github/
    └── workflows/ci.yml
```

Each numerical module (`utils.py`, `regression.py`, `plotting.py`, `export.py`) is UI-independent — no Streamlit calls — so it can be imported and unit tested on its own.

## Troubleshooting

- **"Could not interpret the selected date column..."** — pick a different column in the sidebar, or ensure your date column contains recognizable dates (e.g. `YYYY-MM-DD`) or plain 4-digit years.
- **"Structural break date X is outside the data range"** — structural break dates must fall strictly between the minimum and maximum time values in your cleaned dataset.
- **"Regime [...] only has N point(s)..."** — each regime needs at least 3 points for estimation; add more data or reduce the number of structural break dates.
- **TIFF export is large** — TIFF is an uncompressed raster format; prefer PNG for raster or PDF/SVG/EPS for vector publication figures.

## Putting this on GitHub

```bash
cd PiecewiseRegressionDashboard
git init
git add .
git commit -m "Initial commit: Structural Break Analysis Dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

The `data/` folder (with `CH4.csv`, `CO2.csv`, `N2O.csv`, `Total.csv`) is committed alongside the code, since `.gitignore` only excludes caches, virtual environments, and the `outputs/` scratch folder — so the app works immediately for anyone who clones the repo, no manual data upload required.

## Deployment — important note about Vercel

**Vercel cannot host this app.** Vercel is built for static sites, Next.js/React frontends, and short-lived serverless functions; Streamlit needs a persistent Python process with an open WebSocket connection to the browser, which Vercel's serverless model doesn't support. Attempts to wrap Streamlit in a Vercel serverless function (`@vercel/python`) are unofficial workarounds that routinely break on cold starts, memory limits, and the WebSocket requirement — not something to rely on for a real presentation.

Platforms that natively support this kind of app (all can deploy straight from the GitHub repo above):

| Platform | Notes |
|---|---|
| **[Streamlit Community Cloud](https://streamlit.io/cloud)** | Free, purpose-built for Streamlit, deploys directly from a GitHub repo in a few clicks. The easiest option for this project. |
| **[Hugging Face Spaces](https://huggingface.co/spaces)** | Free tier, has a "Streamlit" Space type, deploys from a GitHub repo or direct push. |
| **[Render](https://render.com)** | Free/paid tiers, deploy as a "Web Service" with the start command `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`. |
| **[Railway](https://railway.app)** | Similar to Render; auto-detects Python and lets you set the same start command. |

**Recommended path:** push to GitHub, then connect the repo on Streamlit Community Cloud (Sign in → "New app" → pick the repo/branch → set the main file to `app.py`) and it will be live within a couple of minutes, including the bundled `data/` CSVs.

## License

Released under the MIT License — see [LICENSE](LICENSE).
