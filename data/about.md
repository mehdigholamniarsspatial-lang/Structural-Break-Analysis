About
Ireland Greenhouse Gas Policy Impact Explorer
INPACT — Investigating National Policy Impacts on Atmospheric Climate Targets

Version 2.0 (2026)

The Ireland Greenhouse Gas Policy Impact Explorer is an interactive scientific dashboard developed as part of the INPACT (Investigating National Policy Impacts on Atmospheric Climate Targets) research project. The platform enables researchers, policymakers, environmental agencies, and the public to explore long-term greenhouse gas emission inventories, atmospheric observations, climate policy timelines, and policy effectiveness across Ireland.

The dashboard combines advanced statistical analysis, interactive visualisation, and evidence-based climate data to support transparent evaluation of Ireland's progress towards national and international climate commitments.

About the INPACT Project

INPACT is a multidisciplinary research project funded by the Environmental Protection Agency (EPA) Ireland that investigates how national climate and environmental policies have influenced greenhouse gas emissions, atmospheric composition, and climate outcomes since Ireland's engagement with the United Nations Framework Convention on Climate Change (UNFCCC) in 1992.

As Ireland works towards achieving its legally binding climate objectives—including a 51% reduction in greenhouse gas emissions by 2030 and net-zero emissions by 2050—INPACT provides the scientific evidence needed to determine:

Which climate policies have been effective.
Which sectors have responded most strongly.
Where policy gaps remain.
How future climate policies can be improved using evidence-based analysis.

The project integrates atmospheric science, environmental economics, climate policy, geospatial analysis, machine learning, and data science to provide a comprehensive assessment of Ireland's climate transition.

Mission

INPACT aims to evaluate the real-world effectiveness of Irish climate policies by linking policy interventions with changes in greenhouse gas emissions, atmospheric observations, and climate indicators.

Rather than relying solely on reported emission inventories, the project combines multiple independent datasets to understand whether policy interventions have produced measurable atmospheric and environmental outcomes.

Explorer Dashboard

The dashboard provides an integrated platform for analysing greenhouse gas emissions, policy interventions, and environmental indicators through interactive visualisation and statistical modelling.

Key capabilities include:

Interactive exploration of Ireland's greenhouse gas emission inventories (1990–present)
Multi-sector and sub-sector emissions analysis
Climate policy timeline visualisation aligned with emissions trajectories
Structural Break Analysis for identifying regime shifts in emissions
Structural Break Date ($T_b$) analysis using user-defined policy intervention dates
Regime trajectory diagnostics using Ordinary Least Squares (OLS), Huber robust regression, Theil–Sen regression, the Mann–Kendall test, and Sen's slope estimator
Statistical evaluation of policy impacts using:
Slope and intercept estimates
R² and Adjusted R²
RMSE, MAE and MSE
Standard Error
95% Confidence Intervals
t-statistics and p-values
Residual variance
Multi-sector comparison dashboards
Publication-quality static figures and interactive Plotly visualisations
Export of figures and analytical results in PNG, JPEG, SVG, PDF, EPS, TIFF (150–1200 DPI), CSV, Excel and JSON formats
Data Sources

The platform integrates multiple national and international environmental datasets, including:

Ireland National Greenhouse Gas Inventory
Environmental Protection Agency (EPA) Ireland
United Nations Framework Convention on Climate Change (UNFCCC)
European Environment Agency (EEA)
Copernicus Atmosphere Monitoring Service (CAMS)
Copernicus Earth Observation Programme
Integrated Carbon Observation System (ICOS)
European Centre for Medium-Range Weather Forecasts (ECMWF)
Earth Observation and atmospheric monitoring datasets
National climate policy and legislative records
Scientific Methodology

The Explorer combines multiple analytical approaches to evaluate policy effectiveness, including:

Structural Break Analysis
Regime-Shift Detection
Robust Regression Methods
Time-Series Analysis
Mann–Kendall analysis of monotonic temporal change
Sen's Slope Estimation
Statistical Significance Testing
Causal Inference Frameworks
Geospatial Analysis
Machine Learning and Environmental Data Integration

The platform supports comparison between bottom-up national emission inventories and top-down atmospheric observations to improve understanding of how policy interventions translate into measurable environmental outcomes.

Why This Dashboard Matters

Climate policies are traditionally evaluated using reported emissions inventories. However, the ultimate measure of success is whether these policies produce observable improvements in atmospheric greenhouse gas concentrations and climate indicators.

The INPACT Explorer bridges this gap by combining emissions data, atmospheric observations, and policy timelines into a single evidence-based platform that enables users to:

investigate long-term emissions trajectories;
identify periods of significant change associated with policy interventions;
evaluate the statistical significance of observed changes;
compare sectoral responses to climate policies;
support evidence-based climate policy development.

The platform promotes transparency, reproducibility, and open access to climate information for policymakers, researchers, students, and the wider public.

Technology Stack

Built using modern open-source scientific computing technologies:

Streamlit
Python
Pandas
NumPy
SciPy
Statsmodels
Scikit-learn
Matplotlib
Plotly

The dashboard automatically discovers and processes greenhouse gas inventory datasets using a standard sector-by-year CSV structure, allowing new gases or updated inventories to be added with minimal configuration.

Research Team

The INPACT project is led by the University of Galway in collaboration with the Insight Research Ireland Centre for Data Analytics, bringing together expertise in atmospheric science, climate policy, environmental economics, artificial intelligence, geospatial analysis, and public engagement.

Core research team:

Dr. Liz Coleman — Principal Investigator
Professor Karyn Morrissey — Co-Principal Investigator
Dr. Mehdi Gholamnia — Postdoctoral Researcher
Andy Donald — Data Science & Applied Innovation
Dr. Darius Ceburnis — Atmospheric Science
Dr. Damien Martin — Atmospheric Measurements & Emissions Analysis
David Nganga — PhD Researcher
Funding

The INPACT project is funded by the Environmental Protection Agency (EPA) Ireland and supports Ireland's transition towards evidence-based climate policy and net-zero greenhouse gas emissions.
