import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import parse_eei_file
from src.dispatch import backtest_monthly_dispatch


class DataTests(unittest.TestCase):
    def test_parse_one_eei_day(self):
        line1 = "010120251" + " " * 11 + "".join(f"{value:05d}" for value in range(10001, 10013))
        line2 = "010120252" + " " * 11 + "".join(f"{value:05d}" for value in range(10013, 10025))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.txt"
            path.write_text(line1 + "\n" + line2 + "\n")
            result = parse_eei_file(path)
        self.assertEqual(len(result), 24)
        self.assertEqual(result.iloc[0]["hour_ending"], 1)
        self.assertEqual(result.iloc[-1]["hour_ending"], 24)
        self.assertEqual(result["load_mw"].max(), 10024)


class DispatchTests(unittest.TestCase):
    def test_dispatch_capture_and_economics(self):
        frame = pd.DataFrame({
            "operating_date": pd.date_range("2025-01-01", periods=4),
            "peak_load_mw": [10, 20, 15, 12],
            "predicted_peak_mw": [11, 19, 16, 13],
        })
        _, summary = backtest_monthly_dispatch(frame, battery_mw=1, efficiency=1, peak_value_per_kw_month=10, variable_cost_per_mwh=0, max_dispatches_per_month=1)
        self.assertEqual(summary["monthly_peaks_captured"], 1)
        self.assertEqual(summary["gross_savings_usd"], 10000)


if __name__ == "__main__":
    unittest.main()
