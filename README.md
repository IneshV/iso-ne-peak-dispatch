# ISO New England Peak Demand Forecasting

This project forecasts daily ISO New England system peak demand using archived Global Forecast System temperature forecasts retrieved through the Open Meteo Previous Runs API. Each temperature forecast was issued 24 hours before its valid hour.

The selected model combines 75 percent histogram gradient boosting with 25 percent Ridge regression. Model selection used validation with expanding training windows across 2022 through 2024. The selected model was then evaluated retrospectively on 2025.

## Final results

| Metric | Result |
|---|---:|
| Validation mean absolute error | 460 MW |
| 2025 mean absolute error | 421 MW |
| 2025 root mean squared error | 610 MW |
| Monthly peaks captured with three dispatches per month | 12 of 12 |

## Reproduce the analysis

Create an environment and install the required packages:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The repository already contains the raw data used for the final analysis. Start Jupyter and run the notebook from top to bottom:

```bash
.venv/bin/python -m pip install jupyter
.venv/bin/python -m jupyter notebook ISO_New_England_Peak_Demand_Analysis.ipynb
```

The notebook presents the complete workflow as a single narrative: data loading, feature construction, temporal validation, final 2025 evaluation, dispatch analysis, and charts. It does not generate or modify `dashboard/project_summary.html`; that file is the manually edited final summary.

## Files retained

```text
artifacts/model_metrics.csv                 Validation and 2025 metrics
artifacts/test_predictions_2025.csv         Daily 2025 predictions
dashboard/project_summary.html              Final one page summary
data/raw/isone_load/                         Official ISO New England load files
data/raw/weather/                            Archived 24 hour ahead GFS temperatures
ISO_New_England_Peak_Demand_Analysis.ipynb   Complete reproducible analysis
```

## Important limitations

The model predicts regional ISO New England load rather than the load of a specific municipal utility. It predicts peak magnitude and likely peak days but not the intraday peak hour. The dispatch analysis selects the highest forecasts after each month is complete and does not simulate operating costs, state of charge, charging time, degradation, outages, or cycle limits.
