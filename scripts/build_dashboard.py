#!/usr/bin/env python3
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def line_svg(frame: pd.DataFrame, width: int = 960, height: int = 300) -> str:
    actual = frame["peak_load_mw"].tolist()
    predicted = frame["predicted_peak_mw"].tolist()
    values = actual + predicted
    lo, hi = min(values), max(values)
    left, right, top, bottom = 58, 20, 24, 52
    def points(series):
        return " ".join(
            f"{left + i * (width - left - right) / (len(series) - 1):.1f},{height - bottom - (value - lo) * (height - top - bottom) / (hi - lo):.1f}"
            for i, value in enumerate(series)
        )
    month_ticks = []
    for month, group in frame.groupby(frame["operating_date"].dt.month, sort=True):
        index = frame.index.get_loc(group.index[0])
        x = left + index * (width - left - right) / (len(frame) - 1)
        label = group.iloc[0]["operating_date"].strftime("%b")
        month_ticks.append(
            f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+6}" stroke="#627084"/>'
            f'<text x="{x:.1f}" y="{height-bottom+21}" text-anchor="middle" fill="#627084" font-size="12">{label}</text>'
        )
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="Actual and predicted daily peak load">
      <rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}" fill="none" stroke="#d7dde7"/>
      <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#9aa4b2"/>
      {''.join(month_ticks)}
      <polyline points="{points(actual)}" fill="none" stroke="#1f5eff" stroke-width="2" opacity=".85"/>
      <polyline points="{points(predicted)}" fill="none" stroke="#f59e0b" stroke-width="2" opacity=".9"/>
      <text x="{left-8}" y="{top+4}" text-anchor="end" fill="#627084" font-size="12">{hi:,.0f}</text>
      <text x="{left-8}" y="{height-bottom+4}" text-anchor="end" fill="#627084" font-size="12">{lo:,.0f}</text>
      <text x="{(left+width-right)/2:.1f}" y="{height-5}" text-anchor="middle" fill="#627084" font-size="12">Time</text>
      <text x="14" y="{(top+height-bottom)/2:.1f}" text-anchor="middle" transform="rotate(-90 14 {(top+height-bottom)/2:.1f})" fill="#627084" font-size="12">Daily peak load (MW)</text>
    </svg>'''


def main() -> None:
    artifacts = ROOT / "artifacts"
    dashboard = ROOT / "dashboard"
    dashboard.mkdir(exist_ok=True)
    predictions = pd.read_csv(artifacts / "test_predictions_2025.csv", parse_dates=["operating_date"])
    metrics = pd.read_csv(artifacts / "model_metrics.csv")
    summary = json.loads((artifacts / "dispatch_summary.json").read_text())
    test_metric = metrics[metrics["split"] == "test"].iloc[0]
    monthly = predictions.set_index("operating_date").resample("MS").agg(actual=("peak_load_mw", "max"), predicted=("predicted_peak_mw", "max"))
    monthly_rows = "".join(f"<tr><td>{idx.strftime('%B')}</td><td>{row.actual:,.0f}</td><td>{row.predicted:,.0f}</td></tr>" for idx, row in monthly.iterrows())
    metric_rows = "".join(f"<tr><td>{html.escape(str(row.model))}</td><td>{row.split}</td><td>{row.mae_mw:,.0f}</td><td>{row.rmse_mw:,.0f}</td><td>{row.top5_recall:.0%}</td></tr>" for row in metrics.itertuples())
    content = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ISO-NE Peak Demand &amp; Dispatch</title><style>
:root{{--ink:#14213d;--muted:#627084;--blue:#1f5eff;--gold:#f59e0b;--bg:#f5f7fb;--card:#fff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--ink)}}header{{padding:48px max(5vw,24px);background:linear-gradient(120deg,#0c1f3d,#164e63);color:#fff}}header>p{{max-width:780px;color:#dbeafe}}.model-glance{{max-width:980px;margin-top:22px;padding:16px 18px;border:1px solid #ffffff35;background:#ffffff12}}.model-glance strong{{display:block;margin-bottom:6px}}.model-glance p{{margin:4px 0;color:#e8f1ff}}main{{max-width:1180px;margin:auto;padding:30px 24px 60px}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}.card{{background:var(--card);border:1px solid #e2e8f0;border-radius:14px;padding:20px;box-shadow:0 4px 16px #14213d0d}}.value{{font-size:28px;font-weight:750}}.label{{color:var(--muted);font-size:13px}}section{{margin-top:28px}}h2{{margin-bottom:10px}}.legend span{{margin-right:20px}}.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #e5e7eb}}th:first-child,td:first-child{{text-align:left}}.controls{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}input{{width:100%}}.note{{padding:14px;border-left:4px solid var(--gold);background:#fffbeb}}@media(max-width:800px){{.grid,.controls{{grid-template-columns:1fr 1fr}}}}@media(max-width:520px){{.grid,.controls{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>ISO-NE Peak Demand & Dispatch</h1><p>A reproducible 2025 holdout backtest connecting weather-driven daily peak forecasts to a hypothetical municipal battery dispatch decision.</p><div class="model-glance"><strong>Model at a glance</strong><p><b>Target:</b> daily ISO-NE system peak MW. <b>Data:</b> official hourly ISO-NE load plus regional weather averaged across six New England cities.</p><p><b>Features:</b> temperature, humidity, dew point, apparent temperature, heating/cooling effects, calendar seasonality, and 1-day, 7-day, and rolling 7-day peak-load lags.</p><p><b>Time split:</b> trained on 2015–2023, selected on 2024, tested on 2025. The backtest uses observed historical weather rather than archived day-ahead forecasts.</p></div></header><main>
<div class="grid"><div class="card"><div class="label">Test MAE</div><div class="value">{test_metric.mae_mw:,.0f} MW</div></div><div class="card"><div class="label">Top-5 peak recall</div><div class="value">{test_metric.top5_recall:.0%}</div></div><div class="card"><div class="label">Monthly peaks captured</div><div class="value">{summary['monthly_peaks_captured']} / {summary['months']}</div></div><div class="card"><div class="label">Illustrative net savings</div><div class="value">${summary['net_savings_usd']/1000:,.0f}K</div></div></div>
<section class="card"><h2>Actual vs predicted daily peak load</h2><div class="legend"><span><i class="dot" style="background:#1f5eff"></i>Actual</span><span><i class="dot" style="background:#f59e0b"></i>Predicted</span></div>{line_svg(predictions)}</section>
<section class="card"><h2>Battery scenario calculator</h2><div class="controls"><label>Battery power: <b><span id="mwOut">5</span> MW</b><input id="mw" type="range" min="1" max="20" step="1" value="5"></label><label>Peak value: <b>$<span id="valueOut">25</span>/kW-month</b><input id="value" type="range" min="5" max="50" step="1" value="25"></label><label>Variable cost: <b>$<span id="costOut">50</span>/MWh</b><input id="cost" type="range" min="0" max="200" step="10" value="50"></label></div><p>Illustrative 2025 net value: <span class="value" id="net"></span></p><div class="note">This calculator changes economic assumptions only. It holds the tested dispatch dates, 90% efficiency, four-hour duration, and {summary['monthly_peaks_captured']} captured monthly peaks constant.</div></section>
<section class="card"><h2>Model comparison</h2><table><thead><tr><th>Model</th><th>Split</th><th>MAE (MW)</th><th>RMSE (MW)</th><th>Top-5 recall</th></tr></thead><tbody>{metric_rows}</tbody></table></section>
<section class="card"><h2>Monthly peak comparison</h2><table><thead><tr><th>Month</th><th>Actual peak MW</th><th>Highest predicted MW</th></tr></thead><tbody>{monthly_rows}</tbody></table></section>
</main><script>const captured={summary['monthly_peaks_captured']},dispatches={summary['dispatches']},eff=.9,hours=4;function calc(){{let mw=+document.querySelector('#mw').value,v=+document.querySelector('#value').value,c=+document.querySelector('#cost').value;mwOut.textContent=mw;valueOut.textContent=v;costOut.textContent=c;let net=captured*mw*1000*eff*v-dispatches*mw*hours*c;document.querySelector('#net').textContent='$'+Math.round(net).toLocaleString();}}document.querySelectorAll('input').forEach(x=>x.addEventListener('input',calc));calc();</script></body></html>'''
    (dashboard / "index.html").write_text(content)
    print(f"Wrote {dashboard / 'index.html'}")


if __name__ == "__main__":
    main()
