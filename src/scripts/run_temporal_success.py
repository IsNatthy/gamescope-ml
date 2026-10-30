"""Evalúa la regresión con una partición temporal para juegos futuros."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError as error:
    raise SystemExit(
        "Falta XGBoost. Ejecuta: python -m pip install -r requirements.txt"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "models"


def metrics(y_true, prediction):
    prediction = np.maximum(0, prediction)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)),
        "mae_log": float(mean_absolute_error(np.log1p(y_true), np.log1p(prediction))),
        "rmse_log": float(
            mean_squared_error(np.log1p(y_true), np.log1p(prediction)) ** 0.5
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa Random Forest y XGBoost con juegos recientes como test."
    )
    parser.add_argument("--input", type=Path, default=PROCESSED / "model_dataset.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--without-tags", action="store_true")
    args = parser.parse_args()
    if not 0 < args.test_fraction < 0.5:
        raise ValueError("test-fraction debe estar entre 0 y 0.5.")

    df = pd.read_csv(args.input)
    if not {"owners_midpoint", "release_year", "release_month"}.issubset(df.columns):
        raise ValueError("Faltan owners_midpoint, release_year o release_month.")
    df["time_key"] = (
        pd.to_numeric(df["release_year"], errors="coerce") * 12
        + pd.to_numeric(df["release_month"], errors="coerce")
    )
    df = df.dropna(subset=["time_key", "owners_midpoint"]).sort_values("time_key")
    time_values = np.sort(df["time_key"].unique())
    cutoff_index = max(1, int(len(time_values) * (1 - args.test_fraction)))
    cutoff = time_values[min(cutoff_index, len(time_values) - 1)]
    train = df[df["time_key"] < cutoff].copy()
    test = df[df["time_key"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("La partición temporal no produjo train y test no vacíos.")
    ignored = {"appid", "name", "owners_midpoint", "time_key"}
    features = [column for column in df.columns if column not in ignored]
    if args.without_tags:
        features = [
            column for column in features
            if not column.startswith("tag_") and column != "n_tags"
        ]
    x_train = train[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    x_test = test[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_train = train["owners_midpoint"].to_numpy(dtype="float64")
    y_test = test["owners_midpoint"].to_numpy(dtype="float64")
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    y_train_log = np.log1p(y_train)

    models = {
        "random_forest_log": RandomForestRegressor(
            n_estimators=args.trees, max_depth=20, max_samples=0.8,
            n_jobs=args.threads, random_state=42,
        ),
        "xgboost_log": XGBRegressor(
            n_estimators=args.trees, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=args.threads,
            tree_method="hist", objective="reg:squarederror", random_state=42,
        ),
    }
    results = {
        "split": {
            "method": "sorted by release_year/release_month",
            "train_rows": len(train),
            "test_rows": len(test),
            "train_latest": int(train["time_key"].max()),
            "test_earliest": int(test["time_key"].min()),
            "without_tags": args.without_tags,
        },
        "models": [],
    }
    for name, model in models.items():
        model.fit(x_train, y_train_log)
        prediction = np.maximum(0, np.expm1(model.predict(x_test)))
        results["models"].append({"model": name, **metrics(y_test, prediction)})

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "success_models_temporal.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
