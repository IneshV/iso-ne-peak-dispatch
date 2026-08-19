# ISO-NE Peak Demand and Battery Dispatch Lab

A portfolio-ready decision-support project that forecasts ISO New England daily system peaks and backtests a hypothetical municipal battery dispatch strategy.

## Results

The model was trained through 2023, selected using 2024, and tested once on the 2025 holdout year.

| Metric | 2025 result |
|---|---:|
| Daily peak MAE | 513 MW |
| Daily peak RMSE | 733 MW |
| Top-five annual peak recall | 80% |
| Monthly peaks captured at five dispatches/month | 11 of 12 |

The battery-dollar result is deliberately presented as a scenario, not a claim. With the default assumptions—5 MW, four hours, 90% efficiency, $25/kW-month peak value, and $50/MWh variable cost—the illustrative 2025 net value is $1.178 million.

## Data source

Hourly system load comes from ISO New England's official **System Loads in EEI Format** reports:

https://www.iso-ne.com/isoexpress/web/reports/load-and-demand/-/tree/sys-load-eei-fmt

The raw EEI files are retained unchanged in `data/raw/isone_load/`. The included manifest records the direct source URL and ISO-NE publication timestamp for every downloaded file.

## Download or refresh

From the project root, run:

```bash
python3 scripts/download_isone_load.py --start-year 2015 --end-year 2026
```

The 2026 file is year-to-date and may be revised by ISO-NE. Historical files can also be downloaded manually from the source page above.

## Next data step

The fixed-width EEI format stores two 12-hour records for each date. The parser converts these into a tidy hourly table and explicitly flags/interpolates the zero used for the 2015 spring daylight-saving gap.

Weather comes from the [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api). It is averaged across six representative cities: Boston, Hartford, Providence, Concord, Portland, and Burlington.

## Run the analysis

Create an environment and install the two required libraries:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then run:

```bash
.venv/bin/python scripts/run_pipeline.py
.venv/bin/python scripts/build_dashboard.py
.venv/bin/python -m unittest discover -s tests -v
```

Open `dashboard/index.html` in any browser. It has no server or JavaScript-package dependency.

## Project structure

```text
data/raw/                 Original ISO-NE and Open-Meteo files
data/processed/           Tidy hourly load and daily modeling table
src/data.py               Parsing, validation, weather, feature engineering
src/modeling.py           Time-aware ridge model selection and holdout test
src/dispatch.py           Battery dispatch and economic backtest
artifacts/                Model, predictions, metrics, dispatch outputs
dashboard/index.html      Self-contained interactive results dashboard
reports/                  Executive summary and data dictionary
tests/                    Parser and economic-logic tests
```

## Modeling design

- Target: daily ISO-NE system peak MW.
- Predictors: regional weather, nonlinear heating/cooling terms, calendar variables, prior-day load, seven-day lag, and trailing seven-day load.
- Baseline: same day from the previous week.
- Selection: ridge penalty chosen using 2024 MAE.
- Final evaluation: untouched 2025 holdout.
- Dispatch policy: select the five highest predicted days in each month.

## Important limitations

- Historical realized weather is used, so the backtest measures model and decision logic—not day-ahead weather forecast error.
- ISO-NE system load is not an individual municipal utility's load.
- The battery model does not yet simulate hourly state of charge, outages, degradation, or energy-market arbitrage.
- Peak value is user-adjustable and must be replaced with actual utility tariff/cost determinants before operational use.
- A five-day-per-month dispatch policy is intentionally simple. Production use should make sequential decisions using information available at the time.

These boundaries are part of the project’s design: assumptions are visible, outputs are reproducible, and no illustrative value is presented as realized MMWEC savings.
