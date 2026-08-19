from __future__ import annotations

import pandas as pd


def backtest_monthly_dispatch(
    predictions: pd.DataFrame,
    battery_mw: float = 5.0,
    duration_hours: float = 4.0,
    efficiency: float = 0.90,
    peak_value_per_kw_month: float = 25.0,
    variable_cost_per_mwh: float = 50.0,
    max_dispatches_per_month: int = 5,
) -> tuple[pd.DataFrame, dict]:
    frame = predictions.copy()
    frame["month_period"] = frame["operating_date"].dt.to_period("M").astype(str)
    frame["dispatch"] = False
    for _, indices in frame.groupby("month_period").groups.items():
        selected = frame.loc[indices].nlargest(max_dispatches_per_month, "predicted_peak_mw").index
        frame.loc[selected, "dispatch"] = True

    actual_peak_indices = frame.groupby("month_period")["peak_load_mw"].idxmax()
    frame["actual_monthly_peak"] = frame.index.isin(actual_peak_indices)
    frame["captured_peak"] = frame["dispatch"] & frame["actual_monthly_peak"]
    frame["false_dispatch"] = frame["dispatch"] & ~frame["actual_monthly_peak"]

    effective_mw = battery_mw * efficiency
    monthly_savings = effective_mw * 1000 * peak_value_per_kw_month
    dispatch_cost = battery_mw * duration_hours * variable_cost_per_mwh
    frame["gross_savings_usd"] = frame["captured_peak"].astype(float) * monthly_savings
    frame["dispatch_cost_usd"] = frame["dispatch"].astype(float) * dispatch_cost
    frame["net_savings_usd"] = frame["gross_savings_usd"] - frame["dispatch_cost_usd"]

    summary = {
        "battery_mw": battery_mw,
        "duration_hours": duration_hours,
        "efficiency": efficiency,
        "peak_value_per_kw_month": peak_value_per_kw_month,
        "max_dispatches_per_month": max_dispatches_per_month,
        "months": int(frame["month_period"].nunique()),
        "monthly_peaks_captured": int(frame["captured_peak"].sum()),
        "capture_rate": float(frame["captured_peak"].sum() / frame["month_period"].nunique()),
        "dispatches": int(frame["dispatch"].sum()),
        "false_dispatches": int(frame["false_dispatch"].sum()),
        "gross_savings_usd": float(frame["gross_savings_usd"].sum()),
        "dispatch_cost_usd": float(frame["dispatch_cost_usd"].sum()),
        "net_savings_usd": float(frame["net_savings_usd"].sum()),
    }
    return frame, summary
