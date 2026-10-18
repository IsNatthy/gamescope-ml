"""Entrena y compara Random Forest y XGBoost para predecir jugadores.

El entrenamiento no se ejecuta al importar el módulo. Usa pocos hilos por
defecto para evitar saturar la máquina.
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


def evaluate(name, model, x_train, y_train, x_test, y_test, output_dir):
    model.fit(x_train, y_train)
    pred_log = model.predict(x_test)
    pred = np.maximum(0, np.expm1(pred_log))
    real = np.expm1(y_test)
    result = {
        "model": name,
        "mae": float(mean_absolute_error(real, pred)),
        "rmse": float(mean_squared_error(real, pred) ** 0.5),
        "r2": float(r2_score(real, pred)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / f"{name}.joblib")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=PROCESSED / "games_train.csv")
    parser.add_argument("--test", type=Path, default=PROCESSED / "games_test.csv")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trees", type=int, default=200)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    target = "target_players_log"
    if target not in train or target not in test:
        raise ValueError("Falta target_players_log. Ejecuta primero el constructor de regresión.")

    ignored = {"target_players", target, "appid", "name"}
    features = [c for c in train.columns if c not in ignored]
    x_train = train[features].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    x_test = test[features].apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
    y_train, y_test = train[target], test[target]

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
    results = [evaluate(n, m, x_train, y_train, x_test, y_test, MODELS) for n, m in models.items()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "success_models.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
