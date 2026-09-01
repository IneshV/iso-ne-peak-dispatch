from __future__ import annotations

import json
import subprocess
import urllib.parse
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


CITIES = {
    "boston": (42.3601, -71.0589),
    "hartford": (41.7658, -72.6734),
    "providence": (41.8240, -71.4128),
    "concord_nh": (43.2081, -71.5376),
    "portland_me": (43.6591, -70.2568),
    "burlington_vt": (44.4759, -73.2121),
}

FORECAST_ARCHIVE_START = "2021-03-23"


def parse_eei_file(path: Path) -> pd.DataFrame:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        if len(line) < 60:
            raise ValueError(f"{path}:{line_number}: record is too short")
        prefix = line[:-60].strip()
        period = int(prefix[-1])
        date_text = prefix[:-1]
        fmt = "%m%d%Y" if len(date_text) == 8 else "%m%d%y"
        operating_date = pd.to_datetime(date_text, format=fmt)
        values = [int(line[i : i + 5]) for i in range(len(line) - 60, len(line), 5)]
        if period not in (1, 2) or len(values) != 12:
            raise ValueError(f"{path}:{line_number}: malformed EEI record")
        first_hour = 1 if period == 1 else 13
        for offset, load_mw in enumerate(values):
            hour_ending = first_hour + offset
            rows.append(
                {
                    "operating_date": operating_date,
                    "hour_ending": hour_ending,
                    "timestamp_local": operating_date + pd.Timedelta(hours=hour_ending - 1),
                    "load_mw": load_mw,
                    "source_file": path.name,
                }
            )
    return pd.DataFrame(rows)


def build_hourly_load(raw_dir: Path) -> pd.DataFrame:
    frames = [parse_eei_file(path) for path in sorted(raw_dir.glob("*_eei_loads.txt"))]
    data = pd.concat(frames, ignore_index=True).sort_values("timestamp_local")
    duplicates = data.duplicated(["operating_date", "hour_ending"]).sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicate operating-date/hour records")
    if not data["hour_ending"].between(1, 24).all() or (data["load_mw"] < 0).any():
        raise ValueError("Invalid hour-ending or negative load values")
    # EEI encodes the skipped spring-forward hour as zero in some older files.
    data["is_dst_gap"] = data["load_mw"].eq(0)
    data["load_mw"] = data["load_mw"].replace(0, float("nan")).astype(float)
    data["load_mw"] = data.groupby("operating_date")["load_mw"].transform(
        lambda series: series.interpolate(limit_direction="both")
    )
    return data


def download_weather(output_path: Path, start: str, end: str) -> pd.DataFrame:
    endpoint = "https://previous-runs-api.open-meteo.com/v1/forecast"
    start = max(start, FORECAST_ARCHIVE_START)
    params = {
        "latitude": ",".join(str(coords[0]) for coords in CITIES.values()),
        "longitude": ",".join(str(coords[1]) for coords in CITIES.values()),
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m_previous_day1",
        "models": "gfs_seamless",
        "timezone": "America/New_York",
    }
    url = f"{endpoint}?{urllib.parse.urlencode(params)}"
    response = subprocess.run(
        ["curl", "--fail", "--location", "--silent", "--show-error", url],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(response.stdout)
    payloads = payload if isinstance(payload, list) else [payload]
    frames = []
    for city, city_payload in zip(CITIES, payloads):
        hourly = city_payload["hourly"]
        frame = pd.DataFrame(hourly).rename(columns={
            "time": "timestamp_local",
            "temperature_2m_previous_day1": "temperature_2m",
        })
        frame["timestamp_local"] = pd.to_datetime(frame["timestamp_local"])
        frame["city"] = city
        frames.append(frame)
    weather = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(output_path, index=False)
    return weather


def load_or_download_weather(output_path: Path, start: str, end: str) -> pd.DataFrame:
    start = max(start, FORECAST_ARCHIVE_START)
    if output_path.exists():
        weather = pd.read_csv(output_path, parse_dates=["timestamp_local"])
        if weather["timestamp_local"].min() <= pd.Timestamp(start) and weather["timestamp_local"].max() >= pd.Timestamp(end):
            return weather
    return download_weather(output_path, start, end)


def build_daily_dataset(load: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    daily_load = (
        load.groupby("operating_date")
        .agg(peak_load_mw=("load_mw", "max"), energy_mwh=("load_mw", "sum"))
        .reset_index()
    )
    peak_hours = load.loc[load.groupby("operating_date")["load_mw"].idxmax(), ["operating_date", "hour_ending"]]
    daily_load = daily_load.merge(peak_hours, on="operating_date", how="left")

    hourly_region = (
        weather.groupby("timestamp_local")
        .agg(
            temperature_f=("temperature_2m", lambda x: x.mean() * 9 / 5 + 32),
        )
        .reset_index()
    )
    hourly_region["operating_date"] = hourly_region["timestamp_local"].dt.normalize()
    daily_weather = (
        hourly_region.groupby("operating_date")
        .agg(
            temp_max_f=("temperature_f", "max"),
            temp_min_f=("temperature_f", "min"),
            temp_mean_f=("temperature_f", "mean"),
            forecast_hour_count=("temperature_f", "count"),
        )
        .reset_index()
    )
    daily_weather = daily_weather[daily_weather["forecast_hour_count"] == 24].drop(
        columns="forecast_hour_count"
    )
    # Preserve the complete calendar before constructing lags. Missing forecast
    # days must remain gaps rather than making nonadjacent load days look adjacent.
    daily = daily_load.merge(daily_weather, on="operating_date", how="left").sort_values("operating_date")
    daily["year"] = daily["operating_date"].dt.year
    daily["month"] = daily["operating_date"].dt.month
    daily["dayofweek"] = daily["operating_date"].dt.dayofweek
    daily["dayofyear"] = daily["operating_date"].dt.dayofyear
    daily["is_weekend"] = (daily["dayofweek"] >= 5).astype(int)
    daily["cooling_degrees"] = (daily["temp_max_f"] - 65).clip(lower=0)
    daily["heating_degrees"] = (55 - daily["temp_mean_f"]).clip(lower=0)
    daily["temp_sq"] = daily["temp_mean_f"] ** 2
    daily["sin_doy"] = __import__("numpy").sin(2 * __import__("numpy").pi * daily["dayofyear"] / 365.25)
    daily["cos_doy"] = __import__("numpy").cos(2 * __import__("numpy").pi * daily["dayofyear"] / 365.25)
    daily["lag_1_peak_mw"] = daily["peak_load_mw"].shift(1)
    daily["lag_7_peak_mw"] = daily["peak_load_mw"].shift(7)
    daily["rolling_7_peak_mw"] = daily["peak_load_mw"].shift(1).rolling(7).mean()

    # ISO-NE's published peak-load methodology uses nonlinear weather,
    # multi-day weather response, holiday/weekend flags, and autoregressive
    # errors. These features reproduce those ideas without using future load.
    holiday_dates = USFederalHolidayCalendar().holidays(
        start=daily["operating_date"].min(), end=daily["operating_date"].max()
    )
    daily["is_holiday"] = daily["operating_date"].isin(holiday_dates).astype(int)
    daily["is_business_day"] = (
        (daily["dayofweek"] < 5) & (daily["is_holiday"] == 0)
    ).astype(int)
    daily["trend_days"] = (
        daily["operating_date"] - daily["operating_date"].min()
    ).dt.days

    for lag in (2, 3, 14, 21, 28):
        daily[f"lag_{lag}_peak_mw"] = daily["peak_load_mw"].shift(lag)
    for window in (3, 14, 28):
        history = daily["peak_load_mw"].shift(1).rolling(window)
        daily[f"rolling_{window}_mean_mw"] = history.mean()
        daily[f"rolling_{window}_std_mw"] = history.std()
        daily[f"rolling_{window}_max_mw"] = history.max()

    daily["weighted_temp_3d_f"] = (
        10 * daily["temp_max_f"]
        + 5 * daily["temp_max_f"].shift(1)
        + 2 * daily["temp_max_f"].shift(2)
    ) / 17
    daily["cooling_sq"] = daily["cooling_degrees"] ** 2
    daily["heating_sq"] = daily["heating_degrees"] ** 2
    return daily.dropna().reset_index(drop=True)


def write_metadata(path: Path, load: pd.DataFrame, weather: pd.DataFrame, daily: pd.DataFrame) -> None:
    metadata = {
        "load_rows": len(load),
        "load_start": str(load["operating_date"].min().date()),
        "load_end": str(load["operating_date"].max().date()),
        "weather_rows": len(weather),
        "weather_cities": sorted(weather["city"].unique().tolist()),
        "daily_rows": len(daily),
        "daily_start": str(daily["operating_date"].min().date()),
        "daily_end": str(daily["operating_date"].max().date()),
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")
