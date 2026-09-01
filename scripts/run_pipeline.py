#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_daily_dataset, build_hourly_load, load_or_download_weather, write_metadata
from src.modeling import train_and_evaluate


def main() -> None:
    processed = ROOT / "data" / "processed"
    artifacts = ROOT / "artifacts"
    processed.mkdir(parents=True, exist_ok=True)

    load = build_hourly_load(ROOT / "data" / "raw" / "isone_load")
    load.to_csv(processed / "isone_hourly_load.csv", index=False)
    weather = load_or_download_weather(
        ROOT / "data" / "raw" / "weather" / "new_england_hourly_gfs_day_ahead_forecast.csv",
        str(load["operating_date"].min().date()),
        str(load["operating_date"].max().date()),
    )
    daily = build_daily_dataset(load, weather)
    daily.to_csv(processed / "daily_peak_features.csv", index=False)
    write_metadata(processed / "data_metadata.json", load, weather, daily)

    _, test, results = train_and_evaluate(daily)
    results["metrics"].to_csv(artifacts / "model_metrics.csv", index=False)
    test.to_csv(artifacts / "test_predictions_2025.csv", index=False)
    print(results["metrics"].to_string(index=False))


if __name__ == "__main__":
    main()
