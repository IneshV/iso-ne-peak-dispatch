# Data Dictionary

## `data/processed/isone_hourly_load.csv`

| Column | Meaning |
|---|---|
| `operating_date` | ISO-NE operating date |
| `hour_ending` | Hour ending 1–24 |
| `timestamp_local` | Naive local clock timestamp representing the beginning of the hour-ending interval |
| `load_mw` | ISO-NE system load in MW; the single 2015 DST zero is linearly interpolated |
| `source_file` | Original ISO-NE EEI annual file |
| `is_dst_gap` | True when the source used zero for the skipped spring-forward hour |

## `data/processed/daily_peak_features.csv`

| Group | Columns |
|---|---|
| Target | `peak_load_mw`, `hour_ending`, `energy_mwh` |
| Weather | `temp_max_f`, `temp_min_f`, `temp_mean_f`, `humidity_mean_pct`, `dew_point_max_f`, `apparent_temp_max_f` |
| Weather transforms | `cooling_degrees`, `heating_degrees`, `temp_sq` |
| Calendar | `year`, `month`, `dayofweek`, `dayofyear`, `is_weekend`, `sin_doy`, `cos_doy` |
| Lagged load | `lag_1_peak_mw`, `lag_7_peak_mw`, `rolling_7_peak_mw` |

## Artifacts

- `model_metrics.csv`: validation and holdout performance.
- `peak_model.json`: deployable ridge coefficients and feature normalization.
- `test_predictions_2025.csv`: locked holdout predictions.
- `dispatch_backtest_2025.csv`: daily dispatch flags, peak capture, and economics.
- `dispatch_summary.json`: headline operating and financial scenario results.
