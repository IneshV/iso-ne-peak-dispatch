from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_FEATURES = [
    "temp_max_f", "temp_min_f", "temp_mean_f", "humidity_mean_pct",
    "dew_point_max_f", "apparent_temp_max_f", "cooling_degrees",
    "heating_degrees", "temp_sq", "sin_doy", "cos_doy",
    "lag_1_peak_mw", "lag_7_peak_mw", "rolling_7_peak_mw", "is_weekend",
]


def design_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame[NUMERIC_FEATURES].astype(float).copy()
    categories = pd.get_dummies(frame[["month", "dayofweek"]].astype(str), prefix=["month", "dow"], dtype=float)
    return pd.concat([numeric, categories], axis=1)


@dataclass
class RidgePeakModel:
    alpha: float
    columns: list[str] | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "RidgePeakModel":
        matrix = design_matrix(frame)
        self.columns = matrix.columns.tolist()
        x = matrix.to_numpy(float)
        self.means = x.mean(axis=0)
        self.scales = x.std(axis=0)
        self.scales[self.scales == 0] = 1
        z = (x - self.means) / self.scales
        y = target.to_numpy(float)
        self.intercept = float(y.mean())
        centered = y - self.intercept
        penalty = self.alpha * np.eye(z.shape[1])
        self.coefficients = np.linalg.solve(z.T @ z + penalty, z.T @ centered)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = design_matrix(frame).reindex(columns=self.columns, fill_value=0)
        z = (matrix.to_numpy(float) - self.means) / self.scales
        return self.intercept + z @ self.coefficients

    def as_dict(self) -> dict:
        return {
            "model_type": "ridge_regression",
            "alpha": self.alpha,
            "columns": self.columns,
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }


def mae(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.abs(actual.to_numpy() - predicted).mean())


def rmse(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.square(actual.to_numpy() - predicted).mean()))


def top_peak_recall(frame: pd.DataFrame, prediction_col: str, n: int = 5) -> float:
    actual = set(frame.nlargest(n, "peak_load_mw").index)
    predicted = set(frame.nlargest(n, prediction_col).index)
    return len(actual & predicted) / n


def train_and_evaluate(daily: pd.DataFrame, output_dir: Path) -> tuple[RidgePeakModel, pd.DataFrame, dict]:
    train = daily[daily["year"] <= 2023].copy()
    validation = daily[daily["year"] == 2024].copy()
    test = daily[daily["year"] == 2025].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Expected training through 2023, validation in 2024, and test in 2025")

    rows = []
    baseline = validation["lag_7_peak_mw"].to_numpy()
    baseline_frame = validation.assign(prediction=baseline)
    rows.append({"model": "seven_day_baseline", "split": "validation", "mae_mw": mae(validation["peak_load_mw"], baseline), "rmse_mw": rmse(validation["peak_load_mw"], baseline), "top5_recall": top_peak_recall(baseline_frame, "prediction")})

    for alpha in (0.1, 1.0, 10.0, 100.0, 500.0):
        model = RidgePeakModel(alpha).fit(train, train["peak_load_mw"])
        prediction = model.predict(validation)
        scored = validation.assign(prediction=prediction)
        rows.append({"model": f"ridge_{alpha:g}", "split": "validation", "mae_mw": mae(validation["peak_load_mw"], prediction), "rmse_mw": rmse(validation["peak_load_mw"], prediction), "top5_recall": top_peak_recall(scored, "prediction")})

    metrics = pd.DataFrame(rows)
    best_row = metrics[metrics["model"].str.startswith("ridge")].sort_values(["mae_mw", "rmse_mw"]).iloc[0]
    best_alpha = float(best_row["model"].split("_")[1])
    train_final = daily[daily["year"] <= 2024]
    final_model = RidgePeakModel(best_alpha).fit(train_final, train_final["peak_load_mw"])
    test["predicted_peak_mw"] = final_model.predict(test)
    train_residuals = train_final["peak_load_mw"].to_numpy() - final_model.predict(train_final)
    residual_std = float(train_residuals.std(ddof=1))
    test_row = {"model": f"ridge_{best_alpha:g}", "split": "test", "mae_mw": mae(test["peak_load_mw"], test["predicted_peak_mw"].to_numpy()), "rmse_mw": rmse(test["peak_load_mw"], test["predicted_peak_mw"].to_numpy()), "top5_recall": top_peak_recall(test, "predicted_peak_mw")}
    metrics = pd.concat([metrics, pd.DataFrame([test_row])], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = final_model.as_dict()
    (output_dir / "peak_model.json").write_text(json.dumps(model_payload, indent=2) + "\n")
    metadata = {"best_model": f"ridge_{best_alpha:g}", "residual_std_mw": residual_std, "train_end": "2024-12-31", "test_year": 2025, "features": NUMERIC_FEATURES + ["month", "dayofweek"]}
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return final_model, test, {"metrics": metrics, **metadata}
