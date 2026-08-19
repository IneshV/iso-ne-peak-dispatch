# ISO-NE Peak Demand and Battery Dispatch: Executive Summary

## Decision

This project evaluates whether a weather-informed daily peak forecast can improve the timing of a hypothetical municipal battery used to reduce monthly coincident-peak exposure.

## Method

- Official ISO-NE hourly system load, 2015–2026 year-to-date.
- Hourly temperature, humidity, dew point, and apparent temperature averaged across Boston, Hartford, Providence, Concord, Portland, and Burlington.
- Daily ridge-regression peak model using nonlinear weather terms, calendar indicators, and lagged load.
- Training through 2023, model selection on 2024, and a locked 2025 holdout test.
- Battery backtest dispatching on the five highest forecast days in each month.

## Economics

The default illustration assumes a 5 MW / 20 MWh battery, 90% efficiency, $25/kW-month value for capturing a monthly peak, and $50/MWh variable dispatch cost. These are scenario inputs—not MMWEC costs, tariffs, or claimed realized savings.

## 2025 holdout results

- Daily peak MAE: **513 MW**; RMSE: **733 MW**.
- Four of the five largest annual peak days identified by the model's five highest predictions.
- Eleven of twelve monthly peaks captured when permitting five candidate dispatches per month.
- Sixty dispatches, of which 49 were not the monthly peak day.
- Illustrative net value under the default inputs: **$1.178 million**.

The high number of false dispatches is an important operating tradeoff. The next decision improvement should optimize the number of monthly calls against degradation, staffing, customer-fatigue, and missed-peak costs rather than maximizing capture alone.

## Appropriate interpretation

The project is a decision-support prototype. It demonstrates reproducible data ingestion, time-aware validation, operational backtesting, and transparent economics. Production deployment would require utility-specific load, actual tariff determinants, day-ahead weather forecasts rather than realized weather, outage/state-of-charge constraints, and stakeholder-approved dispatch rules.
