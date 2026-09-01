from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


EXCLUDED_COLUMNS = {
    "operating_date", "peak_load_mw", "energy_mwh", "hour_ending",
    "year", "dayofyear",
}
CV_YEARS = (2022, 2023, 2024)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in EXCLUDED_COLUMNS]


def design_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    return frame[columns or feature_columns(frame)].astype(float).copy()


@dataclass
class RidgePeakModel:
    alpha: float
    columns: list[str] | None = None
    means: np.ndarray | None = None
    scales: np.ndarray | None = None
    coefficients: np.ndarray | None = None
    intercept: float = 0.0

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "RidgePeakModel":
        self.columns = feature_columns(frame)
        x = design_matrix(frame, self.columns).to_numpy(float)
        self.means = x.mean(axis=0)
        self.scales = x.std(axis=0)
        self.scales[self.scales == 0] = 1
        z = (x - self.means) / self.scales
        y = target.to_numpy(float)
        self.intercept = float(y.mean())
        penalty = self.alpha * np.eye(z.shape[1])
        self.coefficients = np.linalg.solve(
            z.T @ z + penalty, z.T @ (y - self.intercept)
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = design_matrix(frame, self.columns).to_numpy(float)
        return self.intercept + ((x - self.means) / self.scales) @ self.coefficients


@dataclass
class HGBPeakModel:
    params: dict
    columns: list[str] | None = None
    estimator: HistGradientBoostingRegressor | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "HGBPeakModel":
        self.columns = feature_columns(frame)
        self.estimator = HistGradientBoostingRegressor(**self.params)
        self.estimator.fit(design_matrix(frame, self.columns), target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(design_matrix(frame, self.columns))


@dataclass
class EnsemblePeakModel:
    hgb_weight: float = 0.75
    hgb: HGBPeakModel | None = None
    ridge: RidgePeakModel | None = None

    def fit(self, frame: pd.DataFrame, target: pd.Series) -> "EnsemblePeakModel":
        self.hgb = hgb_factory(15)().fit(frame, target)
        self.ridge = RidgePeakModel(10.0).fit(frame, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (
            self.hgb_weight * self.hgb.predict(frame)
            + (1 - self.hgb_weight) * self.ridge.predict(frame)
        )


def mae(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.abs(actual.to_numpy() - predicted).mean())


def rmse(actual: pd.Series, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.square(actual.to_numpy() - predicted).mean()))


def top_peak_recall(frame: pd.DataFrame, prediction_col: str, n: int = 5) -> float:
    actual = set(frame.nlargest(n, "peak_load_mw").index)
    predicted = set(frame.nlargest(n, prediction_col).index)
    return len(actual & predicted) / n


def score_predictions(frame: pd.DataFrame, predicted: np.ndarray) -> dict:
    scored = frame.assign(_prediction=predicted)
    return {
        "mae_mw": mae(frame["peak_load_mw"], predicted),
        "rmse_mw": rmse(frame["peak_load_mw"], predicted),
        "top5_recall": top_peak_recall(scored, "_prediction"),
    }


def rolling_origin_score(
    daily: pd.DataFrame,
    model_name: str,
    factory: Callable[[], object] | None,
) -> tuple[dict, list[dict]]:
    folds = []
    for year in CV_YEARS:
        train = daily[daily["year"] < year]
        validation = daily[daily["year"] == year]
        if factory is None:
            prediction = validation["lag_7_peak_mw"].to_numpy()
        else:
            model = factory().fit(train, train["peak_load_mw"])
            prediction = model.predict(validation)
        folds.append({
            "model": model_name,
            "year": year,
            **score_predictions(validation, prediction),
        })
    fold_frame = pd.DataFrame(folds)
    summary = {
        "model": model_name,
        "split": "rolling_cv_2022_2024",
        "mae_mw": float(fold_frame["mae_mw"].mean()),
        "rmse_mw": float(fold_frame["rmse_mw"].mean()),
        "top5_recall": float(fold_frame["top5_recall"].mean()),
    }
    return summary, folds


def hgb_factory(max_leaf_nodes: int) -> Callable[[], HGBPeakModel]:
    return lambda: HGBPeakModel({
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": 20,
        "l2_regularization": 10,
        "random_state": 42,
    })


def train_and_evaluate(daily: pd.DataFrame) -> tuple[object, pd.DataFrame, dict]:
    test = daily[daily["year"] == 2025].copy()
    if test.empty:
        raise ValueError("Expected a complete 2025 retrospective holdout")

    candidates: list[tuple[str, Callable[[], object] | None]] = [
        ("seven_day_baseline", None),
        ("ridge_10", lambda: RidgePeakModel(10.0)),
        ("hgb_7_leaf", hgb_factory(7)),
        ("hgb_15_leaf", hgb_factory(15)),
        ("ensemble_hgb75_ridge25", lambda: EnsemblePeakModel()),
    ]

    summaries, fold_rows = [], []
    for name, factory in candidates:
        summary, folds = rolling_origin_score(daily, name, factory)
        summaries.append(summary)
        fold_rows.extend(folds)
    cv_metrics = pd.DataFrame(summaries)
    eligible = cv_metrics[cv_metrics["model"] != "seven_day_baseline"]
    best_name = eligible.sort_values(["mae_mw", "rmse_mw"]).iloc[0]["model"]
    best_factory = dict(candidates)[best_name]

    train_final = daily[daily["year"] <= 2024]
    final_model = best_factory().fit(train_final, train_final["peak_load_mw"])
    test["predicted_peak_mw"] = final_model.predict(test)
    test_metrics = {
        "model": best_name,
        "split": "retrospective_test_2025",
        **score_predictions(test, test["predicted_peak_mw"].to_numpy()),
    }
    metrics = pd.concat([cv_metrics, pd.DataFrame([test_metrics])], ignore_index=True)
    metadata = {
        "best_model": best_name,
        "selection_method": "expanding-window rolling-origin validation, 2022-2024",
        "train_end": "2024-12-31",
        "test_year": 2025,
        "test_status": "retrospective holdout previously examined during project development",
        "features": feature_columns(daily),
        "model_params": getattr(
            final_model,
            "params",
            {"hgb_weight": final_model.hgb_weight, "ridge_weight": 1 - final_model.hgb_weight}
            if isinstance(final_model, EnsemblePeakModel)
            else {"alpha": 10.0},
        ),
    }
    return final_model, test, {"metrics": metrics, **metadata}
