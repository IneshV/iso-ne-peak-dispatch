from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "dashboard" / "project_summary.html"


def line_svg(frame: pd.DataFrame, width: int = 920, height: int = 230) -> str:
    actual = frame["peak_load_mw"].to_numpy(float)
    predicted = frame["predicted_peak_mw"].to_numpy(float)
    low = min(actual.min(), predicted.min()) * 0.96
    high = max(actual.max(), predicted.max()) * 1.03
    left, right, top, bottom = 62, 16, 18, 48

    def points(values) -> str:
        coords = []
        for i, value in enumerate(values):
            x = left + i * (width - left - right) / (len(values) - 1)
            y = top + (high - value) * (height - top - bottom) / (high - low)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    grid = []
    for i in range(4):
        value = low + i * (high - low) / 3
        y = top + (high - value) * (height - top - bottom) / (high - low)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" />'
            f'<text x="{left-8}" y="{y+3:.1f}" text-anchor="end">{value/1000:.0f}k</text>'
        )

    month_ticks = []
    dates = pd.to_datetime(frame["operating_date"])
    for month in range(1, 13):
        idx = dates[dates.dt.month == month].index[0] - frame.index[0]
        x = left + idx * (width - left - right) / (len(frame) - 1)
        month_ticks.append(f'<text x="{x:.1f}" y="{height-22}" text-anchor="middle">{dates.iloc[idx]:%b}</text>')

    return f'''<svg class="forecast-chart" viewBox="0 0 {width} {height}" role="img" aria-label="Actual and predicted daily ISO New England peak load in 2025">
      <g class="grid">{''.join(grid)}</g>
      <polyline class="actual" points="{points(actual)}" />
      <polyline class="predicted" points="{points(predicted)}" />
      <g class="ticks">{''.join(month_ticks)}</g>
      <text x="{(left + width - right) / 2:.1f}" y="{height-5}" text-anchor="middle">Operating month</text>
      <text x="13" y="{(top + height - bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 13 {(top + height - bottom) / 2:.1f})">Daily peak load (MW)</text>
    </svg>'''


def monthly_capture(frame: pd.DataFrame, selections: int) -> int:
    scored = frame.copy()
    scored["month"] = pd.to_datetime(scored["operating_date"]).dt.to_period("M")
    captured = 0
    for _, group in scored.groupby("month"):
        actual_peak = group.nlargest(1, "peak_load_mw").index[0]
        selected = set(group.nlargest(selections, "predicted_peak_mw").index)
        captured += int(actual_peak in selected)
    return captured


def build() -> None:
    predictions = pd.read_csv(ARTIFACTS / "test_predictions_2025.csv")
    metrics = pd.read_csv(ARTIFACTS / "model_metrics.csv")
    test = metrics[metrics["split"] == "retrospective_test_2025"].iloc[0]
    cv = metrics[metrics["split"] == "rolling_cv_2022_2024"].copy()
    selected_cv = cv[cv["model"] == "ensemble_hgb75_ridge25"].iloc[0]
    captures = {k: monthly_capture(predictions, k) for k in (1, 2, 3)}
    html = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ISO New England Peak Demand Forecasting</title>
  <style>
    :root {{
      --ink:#173047; --muted:#5e7080; --blue:#246b91; --teal:#268f84;
      --orange:#e98a2e; --paper:#ffffff; --wash:#eef4f6; --line:#d7e1e6;
    }}
    * {{ box-sizing:border-box; }}
    html {{ background:#dfe6e9; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font-family:Arial, Helvetica, sans-serif; }}
    .page {{ width:8.5in; min-height:11in; margin:28px auto; padding:.48in .55in .42in; background:white; box-shadow:0 12px 35px rgba(20,45,64,.16); }}
    header {{ border-bottom:2px solid var(--blue); padding-bottom:15px; margin-bottom:16px; }}
    .eyebrow {{ color:var(--blue); font-size:10px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }}
    h1 {{ margin:6px 0 6px; font-size:28px; line-height:1.08; letter-spacing:-.025em; }}
    .dek {{ max-width:6.7in; margin:0; color:var(--muted); font-size:12px; line-height:1.42; }}
    .byline {{ display:flex; justify-content:space-between; margin-top:11px; font-size:9px; color:var(--muted); }}
    .metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:11px; margin:0 0 16px; }}
    .metric {{ padding:12px 13px 10px; background:var(--wash); border:1px solid var(--line); border-radius:7px; min-height:72px; }}
    .metric strong {{ display:block; font-size:22px; line-height:1; margin-bottom:7px; }}
    .metric span {{ display:block; color:var(--muted); font-size:8.5px; line-height:1.25; }}
    .columns {{ display:grid; grid-template-columns:1.02fr .98fr; gap:22px; }}
    h2 {{ margin:0 0 5px; color:var(--blue); font-size:10px; letter-spacing:.07em; text-transform:uppercase; }}
    p {{ margin:0; font-size:9px; line-height:1.42; }}
    .block + .block {{ margin-top:13px; }}
    .decision {{ background:#f7f3ea; border:1px solid #ead8b6; border-radius:7px; padding:11px 12px; }}
    .calls {{ display:grid; gap:7px; margin-top:9px; }}
    .call-row {{ display:grid; grid-template-columns:70px 1fr 48px; gap:8px; align-items:center; font-size:8px; }}
    .bar {{ height:9px; border-radius:5px; background:#e5eaec; overflow:hidden; }}
    .fill {{ height:100%; background:var(--teal); border-radius:5px; }}
    .call-row b {{ text-align:right; font-size:8px; }}
    figure {{ margin:16px 0 0; border-top:1px solid var(--line); padding-top:12px; }}
    .figure-head {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:4px; }}
    .figure-head h2 {{ margin:0; }}
    .legend {{ display:flex; gap:13px; color:var(--muted); font-size:8px; }}
    .legend span::before {{ content:""; display:inline-block; width:13px; height:3px; margin-right:5px; vertical-align:middle; background:var(--blue); }}
    .legend .pred::before {{ background:var(--orange); }}
    .forecast-chart {{ width:100%; height:154px; display:block; }}
    .forecast-chart .grid line {{ stroke:#e3eaee; stroke-width:1; }}
    .forecast-chart text {{ fill:#71818e; font-size:10px; font-family:Arial, Helvetica, sans-serif; }}
    .forecast-chart polyline {{ fill:none; stroke-width:2.2; stroke-linejoin:round; stroke-linecap:round; }}
    .forecast-chart .actual {{ stroke:var(--blue); }}
    .forecast-chart .predicted {{ stroke:var(--orange); opacity:.9; }}
    figcaption {{ margin-top:2px; color:var(--muted); font-size:7.5px; line-height:1.35; }}
    .bottom {{ display:grid; grid-template-columns:.9fr 1.05fr .95fr; gap:18px; margin-top:13px; padding-top:12px; border-top:1px solid var(--line); }}
    .source {{ color:var(--muted); font-size:7.4px; }}
    footer {{ display:flex; justify-content:space-between; margin-top:12px; padding-top:8px; border-top:1px solid var(--line); color:var(--muted); font-size:7px; }}
    @media (max-width:850px) {{ .page {{ width:100%; min-height:auto; margin:0; box-shadow:none; }} }}
    @page {{ size:letter portrait; margin:0; }}
    @media print {{
      html,body {{ background:white; }}
      .page {{ width:8.5in; height:11in; min-height:11in; margin:0; box-shadow:none; break-after:page; overflow:hidden; }}
    }}
  </style>
</head>
<body>
<main class="page">
  <header>
    <div class="eyebrow">Independent project summary</div>
    <h1>Forecasting ISO New England Peak Demand</h1>
    <div class="byline"><strong>Inesh A. Vytheswaran</strong><span>September 2026</span></div>
  </header>

  <section class="metrics" aria-label="Key results">
    <div class="metric"><strong>{selected_cv.mae_mw:,.0f} MW</strong><span>mean absolute error across 2022 through 2024 validation</span></div>
    <div class="metric"><strong>{captures[3]} of 12</strong><span>monthly peaks captured with 3 dispatches per month</span></div>
    <div class="metric"><strong>{test.mae_mw:,.0f} MW</strong><span>2025 daily peak mean absolute error</span></div>
  </section>

  <section class="columns">
    <div>
      <div class="block">
        <h2>Objective</h2>
        <p>Can we create a prototype model using public ISO New England load and regional weather forecasts to predict each day's peak demand and identify the days most likely to contain the monthly system peak?</p>
      </div>
      <div class="block">
        <h2>Approach</h2>
        <p>I combined official ISO New England hourly system load with GFS temperature forecasts issued 24 hours before each valid hour. Forecasts were averaged across Boston, Hartford, Providence, Concord, Portland, and Burlington. The model uses nonlinear temperature effects, a weighted measure of forecast temperature across three days, holidays, and calendar seasonality. Historical load features include the daily peak from 1, 2, 3, 7, 14, 21, and 28 days earlier; averages over the trailing 3, 7, 14, and 28 days; and the variability and maximum peak over the prior 3, 14, and 28 days. All load features are shifted so that no future load information enters a forecast.</p>
      </div>
      <div class="block">
        <h2>Evaluation</h2>
        <p>I compared several models using validation with expanding training windows to determine which performed best across 2022 through 2024. Each validation year was predicted using only data from prior years. After selecting the best model, I evaluated it on 2025 data reserved for testing and not used to fit the model. The selected ensemble combines 75% gradient boosting with 25% Ridge regression. It had the lowest validation error and achieved a {test.mae_mw:,.0f} MW MAE in the retrospective 2025 evaluation.</p>
      </div>
    </div>
    <div class="decision">
      <h2>From forecast to dispatch</h2>
      <p>For each completed month in 2025, I ranked days by predicted peak. The single highest forecast captured {captures[1]} of 12 monthly peaks, two candidate days captured {captures[2]} of 12, and three candidate days captured all 12. The fourth and fifth selections added no benefit in this backtest.</p>
      <div class="calls" aria-label="Monthly peak capture by allowed selections">
        <div class="call-row"><span>1 selection</span><div class="bar"><div class="fill" style="width:{captures[1]/12:.1%}"></div></div><b>{captures[1]}/12</b></div>
        <div class="call-row"><span>2 selections</span><div class="bar"><div class="fill" style="width:{captures[2]/12:.1%}"></div></div><b>{captures[2]}/12</b></div>
        <div class="call-row"><span>3 selections</span><div class="bar"><div class="fill" style="width:{captures[3]/12:.1%}"></div></div><b>{captures[3]}/12</b></div>
      </div>
    </div>
  </section>

  <figure>
    <div class="figure-head">
      <h2>Actual vs. predicted daily peak load in 2025</h2>
      <div class="legend"><span>Actual</span><span class="pred">Predicted</span></div>
    </div>
    {line_svg(predictions)}
    <figcaption>Retrospective 2025 evaluation. The model follows seasonal demand patterns and identified four of the year's five highest load days among its five highest forecasts.</figcaption>
  </figure>

  <section class="bottom">
    <div>
      <h2>What the result shows</h2>
      <p>In a live setting, the model could estimate tomorrow's peak demand each day using the latest weather forecast and recent load. An operator could compare that estimate with demand already observed during the month and a threshold that reflects the days and dispatch opportunities remaining. If tomorrow appears likely to set the monthly peak and the battery is available, the operator could schedule a discharge once a separate model identifies the likely peak hour.</p>
    </div>
    <div>
      <h2>Limitations</h2>
      <p>The model predicts total electricity demand across New England rather than demand for a specific municipal utility. It predicts daily peak magnitude and ranks likely peak days, but it does not predict the hour of the peak. The dispatch test selects each month's three highest forecasts after the month is complete. A real operating strategy would need to decide whether to dispatch each day without knowing the forecasts or conditions for the rest of the month. The test also does not account for operating costs or limits on dispatch availability, including charging time, state of charge, degradation, outages, or maximum cycle limits.</p>
    </div>
    <div>
      <h2>Future improvements</h2>
      <p>Given more time, I would add archived humidity, dew point, and apparent temperature forecasts, build a separate model for the intraday peak hour, and produce prediction intervals rather than a single point forecast. I would then test a sequential dispatch rule using load data from the utility, battery state of charge, degradation costs, the billing rules that determine which regional peaks affect the utility's costs, and only the information available before each operating decision.</p>
    </div>
  </section>

  <footer>
    <span>Sources: ISO New England System Loads in EEI Format and Open Meteo Previous Runs API.</span>
  </footer>
</main>
</body>
</html>'''

    OUTPUT.write_text(html)
    print(OUTPUT)


if __name__ == "__main__":
    build()
