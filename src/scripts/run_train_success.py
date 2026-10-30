"""Selecciona y evalúa regresores para ``owners_midpoint``.

El script compara Random Forest, XGBoost, una línea base Ridge y dos variantes
del target: escala original y ``log1p``. La regla operativa de 20.000 no se
evalúa aquí; pertenece a la capa de aplicación.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
RANDOM_STATE = 42


def transform_target(values, variant):
    values = np.asarray(values, dtype="float64")
    return np.log1p(values) if variant == "log" else values


def inverse_target(values, variant):
    values = np.asarray(values, dtype="float64")
    if variant == "log":
        return np.maximum(0, np.expm1(values))
    return np.maximum(0, values)


def original_mae_score(y_true_model, y_pred_model, *, variant):
    true_raw = inverse_target(y_true_model, variant)
    pred_raw = inverse_target(y_pred_model, variant)
    return -mean_absolute_error(true_raw, pred_raw)


def metric_summary(y_true, prediction):
    y_true = np.asarray(y_true, dtype="float64")
    prediction = np.maximum(0, np.asarray(prediction, dtype="float64"))
    y_true_log = np.log1p(y_true)
    prediction_log = np.log1p(prediction)
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(mean_squared_error(y_true, prediction) ** 0.5),
        "medae": float(median_absolute_error(y_true, prediction)),
        "r2": float(r2_score(y_true, prediction)),
        "mae_log": float(mean_absolute_error(y_true_log, prediction_log)),
        "rmse_log": float(mean_squared_error(y_true_log, prediction_log) ** 0.5),
    }


def segment_metrics(y_true, prediction):
    """Métricas descriptivas por tamaño del target; no aplica el umbral operativo."""
    y_true = np.asarray(y_true, dtype="float64")
    prediction = np.asarray(prediction, dtype="float64")
    segments = {
        "0_20k": (0, 20_000),
        "20k_100k": (20_000, 100_000),
        "100k_1m": (100_000, 1_000_000),
        "over_1m": (1_000_000, np.inf),
    }
    result = {}
    for name, (lower, upper) in segments.items():
        mask = (y_true >= lower) & (y_true < upper)
        if int(mask.sum()) >= 2:
            result[name] = {
                "rows": int(mask.sum()),
                **metric_summary(y_true[mask], prediction[mask]),
            }
    return result


def make_pipeline(model, *, scale=False):
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def model_and_space(name, threads, trees):
    if name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=trees, random_state=RANDOM_STATE,
            n_jobs=threads, max_samples=0.8,
        )
        space = {
            "model__n_estimators": sorted({max(50, trees // 2), trees, trees * 2}),
            "model__max_depth": [None, 12, 20, 30],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": [0.7, 1.0],
        }
        return make_pipeline(model), space

    if name == "xgboost":
        model = XGBRegressor(
            n_estimators=trees, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=threads,
            tree_method="hist", objective="reg:squarederror",
            random_state=RANDOM_STATE, verbosity=0,
        )
        space = {
            "model__n_estimators": sorted({max(50, trees // 2), trees, trees * 2}),
            "model__max_depth": [3, 5, 6, 8],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
            "model__min_child_weight": [1, 5, 10],
            "model__reg_lambda": [1.0, 5.0, 10.0],
        }
        return make_pipeline(model), space

    raise ValueError(f"Modelo no soportado: {name}")


def load_data(train_path, test_path, without_tags):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = "owners_midpoint"
    if target not in train.columns or target not in test.columns:
        raise ValueError("Falta owners_midpoint. Ejecuta primero run_build_dataset y run_split.")
    ignored = {target, "appid", "name"}
    features = [column for column in train.columns if column not in ignored]
    if without_tags:
        features = [
            column for column in features
            if not column.startswith("tag_") and column != "n_tags"
        ]
    if list(train[features].columns) != list(test[features].columns):
        raise ValueError("Train y test no tienen las mismas columnas de features.")
    x_train = train[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    x_test = test[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_train = train[target].astype("float64").to_numpy()
    y_test = test[target].astype("float64").to_numpy()
    return x_train, x_test, y_train, y_test, features


def fit_search(name, variant, x_train, y_train, args):
    estimator, space = model_and_space(name, args.threads, args.trees)
    y_model = transform_target(y_train, variant)
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    scorer = make_scorer(
        original_mae_score, greater_is_better=True, variant=variant
    )
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=space,
        n_iter=args.search_iterations,
        scoring=scorer,
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        pre_dispatch=1,
        return_train_score=False,
        error_score="raise",
        verbose=1,
    )
    search.fit(x_train, y_model)
    return search


def evaluate_search(name, variant, search, x_test, y_test, features, args):
    prediction_model = search.predict(x_test)
    prediction = inverse_target(prediction_model, variant)
    best_index = search.best_index_
    result = {
        "model": name,
        "target_variant": variant,
        "cv_mae_original": float(-search.best_score_),
        "cv_mae_original_std": float(search.cv_results_["std_test_score"][best_index]),
        "cv_folds": args.cv,
        "best_params": search.best_params_,
        "test": metric_summary(y_test, prediction),
        "test_by_segment": segment_metrics(y_test, prediction),
    }
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": search.best_estimator_, "target_variant": variant,
         "features": features},
        MODELS / f"{name}_{variant}.joblib",
    )
    return result, prediction


def main():
    parser = argparse.ArgumentParser(
        description="Compara regresores con CV, tuning y métricas completas."
    )
    parser.add_argument("--train", type=Path, default=PROCESSED / "train.csv")
    parser.add_argument("--test", type=Path, default=PROCESSED / "test.csv")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--search-iterations", type=int, default=3)
    parser.add_argument("--variants", nargs="+", choices=["raw", "log"],
                        default=["raw", "log"])
    parser.add_argument("--without-tags", action="store_true",
                        help="Ejecuta una ablación sin tags para estudiar su disponibilidad temporal.")
    parser.add_argument("--importance-rows", type=int, default=5000)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--skip-importance", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.trees < 10 or args.cv < 2:
        raise ValueError("threads >= 1, trees >= 10 y cv >= 2 son obligatorios.")
    if args.search_iterations < 1:
        raise ValueError("search-iterations debe ser >= 1.")

    x_train, x_test, y_train, y_test, features = load_data(
        args.train, args.test, args.without_tags
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"Train: {len(x_train):,}; test: {len(x_test):,}; features: {len(features)}")

    median_prediction = np.full(len(y_test), np.median(y_train))
    mean_prediction = np.full(len(y_test), np.mean(y_train))
    results = {
        "configuration": {
            "cv": args.cv,
            "search_iterations": args.search_iterations,
            "threads": args.threads,
            "trees_base": args.trees,
            "target_variants": args.variants,
            "without_tags": args.without_tags,
            "threshold_evaluation": "not included; belongs to application",
        },
        "baselines": {
            "median_train": metric_summary(y_test, median_prediction),
            "mean_train": metric_summary(y_test, mean_prediction),
        },
        "models": [],
    }
    ridge = make_pipeline(Ridge(alpha=1.0), scale=True)
    ridge.fit(x_train, transform_target(y_train, "log"))
    ridge_prediction = inverse_target(ridge.predict(x_test), "log")
    ridge_result = {
        "model": "ridge",
        "target_variant": "log",
        "test": metric_summary(y_test, ridge_prediction),
        "test_by_segment": segment_metrics(y_test, ridge_prediction),
    }
    results["models"].append(ridge_result)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": ridge, "target_variant": "log", "features": features},
        MODELS / "ridge_log.joblib",
    )

    fitted = []
    for name in ("random_forest", "xgboost"):
        for variant in args.variants:
            print(f"Ajustando {name} ({variant})...", flush=True)
            search = fit_search(name, variant, x_train, y_train, args)
            result, prediction = evaluate_search(
                name, variant, search, x_test, y_test, features, args
            )
            results["models"].append(result)
            fitted.append((result, search, prediction))

    candidates = [result for result in results["models"] if "cv_mae_original" in result]
    if candidates:
        winner = min(candidates, key=lambda item: item["cv_mae_original"])
        results["winner_by_cv_mae_original"] = {
            "model": winner["model"],
            "target_variant": winner["target_variant"],
            "cv_mae_original": winner["cv_mae_original"],
        }
        selected = next(
            item for item in fitted
            if item[0]["model"] == winner["model"]
            and item[0]["target_variant"] == winner["target_variant"]
        )
        if not args.skip_importance:
            rows = min(args.importance_rows, len(x_test))
            sample = x_test.iloc[:rows]
            sample_y = transform_target(y_test[:rows], winner["target_variant"])
            importance = permutation_importance(
                selected[1].best_estimator_, sample, sample_y,
                scoring="neg_mean_absolute_error",
                n_repeats=args.importance_repeats,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
            importance_df = pd.DataFrame({
                "feature": features,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }).sort_values("importance_mean", ascending=False)
            importance_df.to_csv(REPORTS / "success_feature_importance.csv", index=False)
            results["importance"] = {
                "model": winner["model"],
                "target_variant": winner["target_variant"],
                "rows": rows,
                "repeats": args.importance_repeats,
                "file": str((REPORTS / "success_feature_importance.csv").resolve()),
            }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "success_models.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
