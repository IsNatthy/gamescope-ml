"""Fit the recorded XGBoost log model without tags; save metrics only."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.scripts.run_train_success import model_and_space


TRAIN_PATH = ROOT / "src" / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "src" / "data" / "processed" / "test.csv"
REPORT_PATH = ROOT / "src" / "models" / "success_models.json"
FULL_RESULTS_PATH = ROOT / "results" / "pendientes" / "test_metrics.csv"
OUTPUT_PATH = ROOT / "results" / "pendientes" / "tag_ablation.json"


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        r2_score,
    )

    y_true = np.asarray(y_true, dtype="float64")
    prediction = np.maximum(0, np.asarray(prediction, dtype="float64"))
    true_log, pred_log = np.log1p(y_true), np.log1p(prediction)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** 0.5),
        "R2": float(r2_score(y_true, prediction)),
        "MedAE": float(median_absolute_error(y_true, prediction)),
        "RMSE_log": float(mean_squared_error(true_log, pred_log) ** 0.5),
        "MAE_log": float(mean_absolute_error(true_log, pred_log)),
    }


def main() -> None:
    train_columns = pd.read_csv(TRAIN_PATH, nrows=0).columns
    test_columns = pd.read_csv(TEST_PATH, nrows=0).columns
    ignored = {"owners_midpoint", "appid", "name"}
    features = [
        column for column in train_columns
        if column not in ignored and not column.startswith("tag_") and column != "n_tags"
    ]
    if [
        column for column in test_columns
        if column not in ignored and not column.startswith("tag_") and column != "n_tags"
    ] != features:
        raise ValueError("Train y test no tienen el mismo orden de features.")

    train = pd.read_csv(TRAIN_PATH, usecols=features + ["owners_midpoint"])
    test = pd.read_csv(TEST_PATH, usecols=features + ["owners_midpoint"])
    x_train = train[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    x_test = test[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_train = train["owners_midpoint"].to_numpy(dtype="float64")
    y_test = test["owners_midpoint"].to_numpy(dtype="float64")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    selected = next(
        item for item in report["models"]
        if item["model"] == "xgboost" and item["target_variant"] == "log"
    )
    estimator, _ = model_and_space("xgboost", threads=2, trees=200)
    estimator.set_params(**selected["best_params"])
    estimator.fit(x_train, np.log1p(y_train))
    prediction = np.maximum(0, np.expm1(estimator.predict(x_test)))
    ablation_metrics = metrics(y_test, prediction)

    full_results = pd.read_csv(FULL_RESULTS_PATH).set_index("Modelo")
    full_metrics = full_results.loc["XGBoost (B)"].to_dict()
    result = {
        "model": "xgboost",
        "target_variant": "log",
        "features_used": len(features),
        "excluded_tag_columns": len([
            column for column in train_columns
            if column.startswith("tag_") or column == "n_tags"
        ]),
        "fixed_best_params": selected["best_params"],
        "test_ablation": ablation_metrics,
        "test_full_model": {key: float(value) for key, value in full_metrics.items()},
        "metric_difference_ablation_minus_full": {
            key: ablation_metrics[key] - float(full_metrics[key])
            for key in ablation_metrics
        },
        "selection_note": (
            "Hyperparameters and variant were fixed from the train-CV report; "
            "test was used only for evaluation."
        ),
    }
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
