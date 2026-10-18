"""Entrena y compara Random Forest y XGBoost para predecir jugadores.

El target oficial es ``owners_midpoint`` y el ajuste se hace sobre
``log1p(owners_midpoint)`` para reducir el efecto de valores extremos.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as error:
    raise SystemExit(
        "Falta XGBoost. Ejecuta: python -m pip install -r requirements.txt"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "models"


def evaluate(name, model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    prediction_log = model.predict(x_test)
    prediction = np.maximum(0, np.expm1(prediction_log))
    real = np.expm1(y_test)
    result = {
        "model": name,
        "mae": float(mean_absolute_error(real, prediction)),
        "rmse": float(mean_squared_error(real, prediction) ** 0.5),
        "r2": float(r2_score(real, prediction)),
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS / f"{name}.joblib")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compara regresores para owners_midpoint."
    )
    parser.add_argument("--train", type=Path, default=PROCESSED / "train.csv")
    parser.add_argument("--test", type=Path, default=PROCESSED / "test.csv")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()
    if args.threads < 1 or args.trees < 10:
        raise ValueError("threads debe ser >= 1 y trees debe ser >= 10.")

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    target = "owners_midpoint"
    if target not in train.columns or target not in test.columns:
        raise ValueError(
            "Falta owners_midpoint. Ejecuta primero run_build_dataset y run_split."
        )
    if list(train.drop(columns=[target, "appid", "name"], errors="ignore").columns) != list(
        test.drop(columns=[target, "appid", "name"], errors="ignore").columns
    ):
        raise ValueError("Train y test no tienen las mismas columnas de features.")

    ignored = {target, "appid", "name"}
    features = [column for column in train.columns if column not in ignored]
    x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    y_train = np.log1p(train[target].astype("float64"))
    y_test = np.log1p(test[target].astype("float64"))

    models = {
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=args.trees, max_depth=20, n_jobs=args.threads,
            random_state=42, max_samples=0.8,
        ),
        "xgboost_regressor": XGBRegressor(
            n_estimators=args.trees, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=args.threads,
            tree_method="hist", random_state=42,
        ),
    }
    results = [
        evaluate(name, model, x_train, y_train, x_test, y_test)
        for name, model in models.items()
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "success_models.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
