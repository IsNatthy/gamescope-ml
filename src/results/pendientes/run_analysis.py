"""Reproduce the pending GameScope regression results without overwriting artifacts."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, ParameterSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.scripts.run_train_success import make_pipeline, model_and_space
from sklearn.linear_model import Ridge


TRAIN_PATH = ROOT / "src" / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "src" / "data" / "processed" / "test.csv"
MODEL_DIR = ROOT / "src" / "models"
REPORT_PATH = MODEL_DIR / "success_models.json"
OUTPUT_DIR = ROOT / "results" / "pendientes"
CV_CACHE_PATH = OUTPUT_DIR / "cv_fold_metrics.partial.csv"
THRESHOLD = 20_000
RANDOM_STATE = 42
METRICS = ("MAE", "RMSE", "R2", "MedAE", "RMSE_log", "MAE_log")
MODEL_NAMES = (
    "Ridge (log)",
    "Random Forest (A)",
    "Random Forest (B)",
    "XGBoost (A)",
    "XGBoost (B)",
)


def postprocess(prediction: np.ndarray, variant: str) -> np.ndarray:
    prediction = np.asarray(prediction, dtype="float64")
    if variant == "log":
        prediction = np.expm1(prediction)
    return np.maximum(0, prediction)


def regression_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    prediction = np.maximum(0, np.asarray(prediction, dtype="float64"))
    true_log = np.log1p(y_true)
    pred_log = np.log1p(prediction)
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** 0.5),
        "R2": float(r2_score(y_true, prediction)),
        "MedAE": float(median_absolute_error(y_true, prediction)),
        "RMSE_log": float(mean_squared_error(true_log, pred_log) ** 0.5),
        "MAE_log": float(mean_absolute_error(true_log, pred_log)),
    }


def read_split() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str]]:
    train_columns = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
    test_columns = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
    ignored = {"owners_midpoint", "appid", "name"}
    features = [column for column in train_columns if column not in ignored]
    if [column for column in test_columns if column not in ignored] != features:
        raise ValueError("Train y test no tienen el mismo orden de features.")

    train = pd.read_csv(TRAIN_PATH, usecols=features + ["owners_midpoint"])
    test = pd.read_csv(TEST_PATH, usecols=features + ["owners_midpoint"])
    x_train = train[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    x_test = test[features].apply(pd.to_numeric, errors="coerce").astype("float32")
    y_train = train["owners_midpoint"].to_numpy(dtype="float64")
    y_test = test["owners_midpoint"].to_numpy(dtype="float64")
    return x_train, y_train, x_test, y_test, features


def load_report() -> dict:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"No existe el reporte de entrenamiento: {REPORT_PATH}")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def reported_models(report: dict) -> dict[tuple[str, str], dict]:
    return {
        (item["model"], item["target_variant"]): item
        for item in report["models"]
        if "best_params" in item
    }


def fit_final_estimator(
    model: str,
    variant: str,
    params: dict,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
) -> object:
    estimator, _ = model_and_space(model, threads=2, trees=200)
    estimator.set_params(**params)
    y_model = np.log1p(y_train) if variant == "log" else y_train
    estimator.fit(x_train, y_model)
    return estimator


def fit_ridge(x_train: pd.DataFrame, y_train: np.ndarray):
    pipeline = make_pipeline(Ridge(alpha=1.0), scale=True)
    pipeline.fit(x_train, np.log1p(y_train))
    return pipeline


def cross_validate(
    name: str,
    model: str | None,
    variant: str,
    params: dict | None,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    cached_rows: list[dict],
) -> list[dict]:
    folds = [row for row in cached_rows if row["modelo"] == name]
    completed_folds = {int(row["fold"]) for row in folds}
    for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
        if fold_number in completed_folds:
            continue
        x_fold_train = x_train.iloc[train_idx]
        x_fold_valid = x_train.iloc[valid_idx]
        y_fold_train, y_fold_valid = y_train[train_idx], y_train[valid_idx]

        if name == "Constante: mediana":
            prediction = np.full(len(valid_idx), np.median(y_fold_train))
        elif name == "Constante: media":
            prediction = np.full(len(valid_idx), np.mean(y_fold_train))
        elif name == "Ridge (log)":
            estimator = fit_ridge(x_fold_train, y_fold_train)
            prediction = postprocess(estimator.predict(x_fold_valid), "log")
            del estimator
        else:
            assert model is not None and params is not None
            estimator, _ = model_and_space(model, threads=2, trees=200)
            estimator.set_params(**params)
            y_model = np.log1p(y_fold_train) if variant == "log" else y_fold_train
            estimator.fit(x_fold_train, y_model)
            prediction = postprocess(estimator.predict(x_fold_valid), variant)
            del estimator

        folds.append({
            "modelo": name,
            "fold": fold_number,
            **regression_metrics(y_fold_valid, prediction),
        })
        cached_rows.extend(row for row in folds if row not in cached_rows)
        pd.DataFrame(cached_rows).to_csv(CV_CACHE_PATH, index=False)
        gc.collect()
        print(f"CV {name}: fold {fold_number}/{len(splits)} completado", flush=True)
    return folds


def threshold_metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    y_true_bin = np.asarray(y_true) >= THRESHOLD
    prediction = np.maximum(0, np.asarray(prediction, dtype="float64"))
    prediction_bin = prediction >= THRESHOLD
    (tp, fn), (fp, tn) = confusion_matrix(
        y_true_bin, prediction_bin, labels=[True, False]
    )
    result = {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "Precision": float(precision_score(y_true_bin, prediction_bin, zero_division=0)),
        "Recall": float(recall_score(y_true_bin, prediction_bin, zero_division=0)),
        "F1": float(f1_score(y_true_bin, prediction_bin, zero_division=0)),
        "Accuracy": float(accuracy_score(y_true_bin, prediction_bin)),
    }
    if np.unique(prediction).size > 1:
        result["ROC_AUC"] = float(roc_auc_score(y_true_bin, prediction))
        result["PR_AUC"] = float(average_precision_score(y_true_bin, prediction))
    return result


def save_curves(
    label: str, y_true: np.ndarray, prediction: np.ndarray, output_dir: Path
) -> None:
    y_true_bin = (y_true >= THRESHOLD).astype("int8")
    fpr, tpr, _ = roc_curve(y_true_bin, prediction)
    roc_auc = roc_auc_score(y_true_bin, prediction)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
             title=f"Curva ROC — {label}")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_{label.replace(' ', '_').lower()}.png", dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true_bin, prediction)
    pr_auc = average_precision_score(y_true_bin, prediction)
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision, label=f"AP = {pr_auc:.4f}")
    axis.axhline(y_true_bin.mean(), linestyle="--", color="gray", label="Prevalencia")
    axis.set(xlabel="Recall", ylabel="Precisión", title=f"Curva PR — {label}")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"pr_{label.replace(' ', '_').lower()}.png", dpi=160)
    plt.close(fig)


def importance_score(variant: str):
    def score(estimator, x, y):
        prediction = postprocess(estimator.predict(x), variant)
        return -mean_absolute_error(y, prediction)
    return score


def run_importance(
    model: str,
    variant: str,
    label: str,
    params: dict,
    features: list[str],
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> pd.DataFrame:
    estimator = fit_final_estimator(model, variant, params, x_train, y_train)
    result = permutation_importance(
        estimator,
        x_test,
        y_test,
        scoring=importance_score(variant),
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    del estimator
    gc.collect()
    return (
        pd.DataFrame({
            "modelo": label,
            "feature": features,
            "media": result.importances_mean,
            "desviacion": result.importances_std,
        })
        .sort_values("media", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    curves_dir = OUTPUT_DIR / "curvas"
    curves_dir.mkdir(parents=True, exist_ok=True)

    report = load_report()
    best = reported_models(report)
    x_train, y_train, x_test, y_test, features = read_split()
    splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(splitter.split(x_train))

    candidates = {}
    for model in ("random_forest", "xgboost"):
        _, space = model_and_space(model, threads=2, trees=200)
        candidates[model] = list(ParameterSampler(space, n_iter=3, random_state=42))

    test_rows = []
    segment_rows = []
    threshold_rows = []
    cv_rows = (
        pd.read_csv(CV_CACHE_PATH).to_dict("records")
        if CV_CACHE_PATH.exists()
        else []
    )
    predictions: dict[str, np.ndarray] = {}

    median_pred = np.full(len(y_test), np.median(y_train))
    mean_pred = np.full(len(y_test), np.mean(y_train))
    predictions["Constante: mediana"] = median_pred
    predictions["Constante: media"] = mean_pred
    for label, prediction in predictions.items():
        test_rows.append({"Modelo": label, **regression_metrics(y_test, prediction)})
        threshold_rows.append({"Modelo": label, **threshold_metrics(y_test, prediction)})
        cross_validate(
            label, None, "raw", None, x_train, y_train, cv_splits, cv_rows
        )

    for model, variant, label in [
        (None, "log", "Ridge (log)"),
        ("random_forest", "raw", "Random Forest (A)"),
        ("random_forest", "log", "Random Forest (B)"),
        ("xgboost", "raw", "XGBoost (A)"),
        ("xgboost", "log", "XGBoost (B)"),
    ]:
        print(f"Evaluando {label}...", flush=True)
        if model is None:
            estimator = fit_ridge(x_train, y_train)
            params = None
        else:
            item = best[(model, variant)]
            params = item["best_params"]
            estimator = fit_final_estimator(
                model, variant, params, x_train, y_train
            )
        prediction = postprocess(estimator.predict(x_test), variant)
        predictions[label] = prediction
        test_rows.append({"Modelo": label, **regression_metrics(y_test, prediction)})
        threshold_rows.append({"Modelo": label, **threshold_metrics(y_test, prediction)})
        for segment, mask in (
            ("y < 20.000", y_test < THRESHOLD),
            ("y >= 20.000", y_test >= THRESHOLD),
        ):
            segment_rows.append({
                "Segmento": segment,
                "n": int(mask.sum()),
                "Modelo": label,
                **regression_metrics(y_test[mask], prediction[mask]),
            })

        del estimator
        gc.collect()
        folds = cross_validate(
            label, model, variant, params, x_train, y_train, cv_splits, cv_rows
        )

        if model in ("random_forest", "xgboost"):
            save_curves(label, y_test, prediction, curves_dir)
    for label, prediction in predictions.items():
        if label.startswith("Constante"):
            for segment, mask in (
                ("y < 20.000", y_test < THRESHOLD),
                ("y >= 20.000", y_test >= THRESHOLD),
            ):
                segment_rows.append({
                    "Segmento": segment,
                    "n": int(mask.sum()),
                    "Modelo": label,
                    **regression_metrics(y_test[mask], prediction[mask]),
                })

    cv_frame = pd.DataFrame(cv_rows)
    cv_summary = []
    for label, group in cv_frame.groupby("modelo", sort=False):
        row = {"Modelo": label}
        for metric in METRICS:
            row[f"{metric}_media"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        cv_summary.append(row)

    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(OUTPUT_DIR / "test_metrics_by_segment.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(OUTPUT_DIR / "threshold_metrics.csv", index=False)
    cv_frame.to_csv(OUTPUT_DIR / "cv_fold_metrics.csv", index=False)
    if CV_CACHE_PATH.exists():
        CV_CACHE_PATH.unlink()
    pd.DataFrame(cv_summary).to_csv(OUTPUT_DIR / "cv_metrics_summary.csv", index=False)
    (OUTPUT_DIR / "sampled_search_configs.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    importance_frames = []
    winner = report["winner_by_cv_mae_original"]
    winner_model, winner_variant = winner["model"], winner["target_variant"]
    importance_frames.append(run_importance(
        winner_model, winner_variant,
        f"{winner_model} ({winner_variant})",
        best[(winner_model, winner_variant)]["best_params"],
        features, x_train, y_train, x_test, y_test,
    ))
    if winner_model != "random_forest":
        importance_frames.append(run_importance(
            "random_forest", "log", "random_forest (log)",
            best[("random_forest", "log")]["best_params"],
            features, x_train, y_train, x_test, y_test,
        ))
    pd.concat(importance_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / "permutation_importance.csv", index=False
    )

    test_y = pd.Series(y_test)
    summary = {
        "source_report": str(REPORT_PATH),
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "test_below_threshold": int((test_y < THRESHOLD).sum()),
        "test_at_or_above_threshold": int((test_y >= THRESHOLD).sum()),
        "threshold": THRESHOLD,
        "cv": {"n_splits": 5, "shuffle": True, "random_state": RANDOM_STATE,
               "standard_deviation_ddof": 1},
        "importance": {"n_repeats": 5, "random_state": RANDOM_STATE,
                       "scoring": "negative MAE in original target scale"},
        "test_estimators": (
            "refitted on current train with the best_params recorded in "
            "success_models.json; the serialized Random Forest feature metadata "
            "differs from the current CSV at genre_episodic vs genre_tutorial"
        ),
        "winner_by_training_report": winner,
        "note": (
            "CV folds reuse the already encoded train features; imputation and "
            "scaling are fitted per fold, but the multi-hot vocabulary is not."
        ),
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
