"""Evaluate TF-IDF and DistilBERT on an appid-disjoint GroupShuffleSplit."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "src" / "data" / "processed"
OUTPUT = ROOT / "results" / "pendientes_resenas" / "grouped"
MODEL_NAME = "distilbert-base-uncased"
SEED = 42
ROWS_PER_PARTITION = 5_000
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 2
PATIENCE = 1


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)


def build_partitions(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = frame["appid"].astype(str).to_numpy()
    labels = frame["voted_up"].to_numpy()
    outer = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    train_validation_idx, test_idx = next(outer.split(frame, labels, groups))
    train_validation = frame.iloc[train_validation_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=SEED)
    train_idx, validation_idx = next(
        inner.split(
            train_validation,
            train_validation["voted_up"],
            train_validation["appid"].astype(str),
        )
    )
    parts = {
        "train": train_validation.iloc[train_idx],
        "validation": train_validation.iloc[validation_idx],
        "test": frame.iloc[test_idx],
    }
    sampled = {
        name: part.sample(n=ROWS_PER_PARTITION, random_state=SEED).reset_index(drop=True)
        for name, part in parts.items()
    }
    appids = {name: set(part["appid"].astype(str)) for name, part in sampled.items()}
    if (
        appids["train"] & appids["validation"]
        or appids["train"] & appids["test"]
        or appids["validation"] & appids["test"]
    ):
        raise ValueError("GroupShuffleSplit produjo appid compartidos.")
    if any(len(part) != ROWS_PER_PARTITION for part in sampled.values()):
        raise ValueError("No se pudo obtener el tamaño de piloto en cada partición.")
    return sampled


def class_metrics(y_true: np.ndarray, p_negative: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    p_negative = np.asarray(p_negative, dtype=float)
    predicted_negative = p_negative >= 0.5
    y_negative = y_true == 0
    matrix = confusion_matrix(
        y_negative.astype(int), predicted_negative.astype(int), labels=[1, 0]
    )
    tp, fn = (int(value) for value in matrix[0])
    fp, tn = (int(value) for value in matrix[1])
    y_predicted = np.where(predicted_negative, 0, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "precision_negative": float(
            precision_score(y_negative, predicted_negative, zero_division=0)
        ),
        "recall_negative": float(
            recall_score(y_negative, predicted_negative, zero_division=0)
        ),
        "f1_negative": float(f1_score(y_negative, predicted_negative, zero_division=0)),
        "f1_macro": float(
            f1_score(y_true, y_predicted, average="macro", labels=[0, 1], zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_predicted)),
        "roc_auc_negative": (
            float(roc_auc_score(y_negative, p_negative))
            if np.unique(y_negative).size == 2
            else None
        ),
        "pr_auc_negative": (
            float(average_precision_score(y_negative, p_negative))
            if np.any(y_negative)
            else None
        ),
        "ece_negative": expected_calibration_error(y_negative, p_negative),
        "brier_negative": float(brier_score_loss(y_negative, p_negative)),
        "confusion_matrix_real_negative_tp_fn_real_positive_fp_tn": [tp, fn, fp, tn],
        "n": int(len(y_true)),
    }


def expected_calibration_error(y_binary: np.ndarray, probabilities: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    total = len(probabilities)
    error = 0.0
    for index in range(10):
        if index == 0:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities > edges[index]) & (probabilities <= edges[index + 1])
        if mask.any():
            observed = float(y_binary[mask].mean())
            predicted = float(probabilities[mask].mean())
            error += mask.sum() / total * abs(observed - predicted)
    return float(error)


def plot_curves(y_true: np.ndarray, probabilities: np.ndarray,
                model_name: str, split_name: str) -> None:
    y_negative = (np.asarray(y_true) == 0).astype(int)
    fpr, tpr, _ = roc_curve(y_negative, probabilities)
    precision, recall, _ = precision_recall_curve(y_negative, probabilities)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_negative, probabilities):.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC negativa: {model_name} ({split_name})")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"roc_negative_{model_name}_{split_name}.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, label=f"AP={average_precision_score(y_negative, probabilities):.4f}")
    axis.set(xlabel="Recall negativo", ylabel="Precisión negativa", title=f"PR negativa: {model_name} ({split_name})")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"pr_negative_{model_name}_{split_name}.png", dpi=150)
    plt.close(fig)

    edges = np.linspace(0.0, 1.0, 11)
    mean_probability = []
    observed_fraction = []
    for index in range(10):
        if index == 0:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities > edges[index]) & (probabilities <= edges[index + 1])
        if mask.any():
            mean_probability.append(float(probabilities[mask].mean()))
            observed_fraction.append(float(y_negative[mask].mean()))
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(mean_probability, observed_fraction, marker="o", label="Modelo")
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Calibración perfecta")
    axis.set(xlabel="Probabilidad media de clase negativa", ylabel="Fracción negativa observada", title=f"Calibración: {model_name} ({split_name})")
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / f"calibration_{model_name}_{split_name}.png", dpi=150)
    plt.close(fig)


def predict_tfidf(parts: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    vectorizer = TfidfVectorizer(
        max_features=100_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    train_x = vectorizer.fit_transform(parts["train"]["review"])
    model = LogisticRegression(
        max_iter=200, C=1.0, solver="liblinear", random_state=SEED
    )
    model.fit(train_x, parts["train"]["voted_up"])
    return {
        split: model.predict_proba(vectorizer.transform(part["review"]))[:, 0]
        for split, part in parts.items()
    }


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer):
        self.texts = texts
        self.labels = np.asarray(labels, dtype=int)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def predict_bert(parts: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando DistilBERT en {device}; 5.000 reseñas, hasta 2 épocas.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    loaders = {
        split: DataLoader(
            ReviewDataset(
                part["review"].astype(str).tolist(),
                part["voted_up"].to_numpy(),
                tokenizer,
            ),
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for split, part in parts.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_state = None
    best_validation_f1 = -np.inf
    no_improvement = 0

    def predict(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_labels, all_negative_probabilities = [], []
        with torch.inference_mode():
            for batch in loader:
                labels = batch.pop("labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                logits = model(**inputs).logits
                all_negative_probabilities.extend(
                    torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
                )
                all_labels.extend(labels.cpu().numpy())
        return np.asarray(all_labels), np.asarray(all_negative_probabilities)

    for epoch in range(EPOCHS):
        print(f"Época {epoch + 1}/{EPOCHS}", flush=True)
        model.train()
        for step, batch in enumerate(loaders["train"], start=1):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"  batch {step}/{len(loaders['train'])}", flush=True)
        y_validation, p_validation = predict(loaders["validation"])
        validation_f1_negative = class_metrics(y_validation, p_validation)["f1_negative"]
        print(f"  validation F1 negativa={validation_f1_negative:.4f}", flush=True)
        if validation_f1_negative > best_validation_f1:
            best_validation_f1 = validation_f1_negative
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("No se obtuvo ningún checkpoint basado en validación.")
    model.load_state_dict(best_state)
    results = {}
    for split in ("validation", "test"):
        _, results[split] = predict(loaders[split])
    return results


def bootstrap_by_appid(test_frame: pd.DataFrame, p_negative: np.ndarray,
                       repetitions: int = 1_000) -> dict:
    appids = test_frame["appid"].astype(str).to_numpy()
    unique_appids = np.unique(appids)
    rows_by_appid = {
        appid: np.flatnonzero(appids == appid) for appid in unique_appids
    }
    rng = np.random.default_rng(SEED)
    sampled_metrics = {key: [] for key in ("f1_negative", "f1_macro", "mcc", "pr_auc_negative")}
    labels = test_frame["voted_up"].to_numpy()
    for _ in range(repetitions):
        sampled_games = rng.choice(unique_appids, size=len(unique_appids), replace=True)
        indices = np.concatenate([rows_by_appid[appid] for appid in sampled_games])
        metrics = class_metrics(labels[indices], p_negative[indices])
        for key in sampled_metrics:
            if metrics[key] is not None:
                sampled_metrics[key].append(metrics[key])
    return {
        metric: {
            "lower_95": float(np.percentile(values, 2.5)) if values else None,
            "upper_95": float(np.percentile(values, 97.5)) if values else None,
            "repetitions": repetitions,
            "defined_repetitions": len(values),
            "seed": SEED,
            "games": int(len(unique_appids)),
        }
        for metric, values in sampled_metrics.items()
    }


def per_game_metrics(test_frame: pd.DataFrame, p_negative: np.ndarray) -> dict:
    frame = test_frame[["appid", "voted_up"]].copy()
    frame["p_negative"] = p_negative
    rows = []
    for appid, group in frame.groupby("appid"):
        if len(group) < 20:
            continue
        rows.append(class_metrics(group["voted_up"].to_numpy(), group["p_negative"].to_numpy()))
    if not rows:
        return {"eligible_games": 0}
    metrics = pd.DataFrame(rows)
    result = {
        "eligible_games": int(len(metrics)),
    }
    for column in (
        "accuracy", "precision_negative", "recall_negative", "f1_negative",
        "f1_macro", "mcc", "roc_auc_negative", "pr_auc_negative",
        "ece_negative", "brier_negative",
    ):
        numeric = pd.to_numeric(metrics[column], errors="coerce")
        result[f"{column}_mean_per_game"] = (
            float(numeric.mean()) if numeric.notna().any() else None
        )
    result["games_with_defined_roc_auc"] = int(metrics["roc_auc_negative"].notna().sum())
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        PROCESSED / "reviews_model_dataset.csv",
        usecols=["appid", "review", "voted_up"],
        dtype={"appid": "string", "review": "string"},
        keep_default_na=False,
    )
    frame["voted_up"] = frame["voted_up"].astype(int)
    parts = build_partitions(frame)
    partition_summary = {
        name: {
            "rows": int(len(part)),
            "negative": int((part["voted_up"] == 0).sum()),
            "positive": int((part["voted_up"] == 1).sum()),
            "games": int(part["appid"].nunique()),
        }
        for name, part in parts.items()
    }
    tfidf_predictions = predict_tfidf(parts)
    bert_predictions = predict_bert(parts)
    output = {"split": partition_summary, "models": {}, "provenance": "New GroupShuffleSplit experiment, not the missing original BERT pilot split"}

    for model_name, predictions in (
        ("tfidf_logistic_regression", tfidf_predictions),
        ("distilbert_standard", bert_predictions),
    ):
        output["models"][model_name] = {}
        for split in ("validation", "test"):
            metrics = class_metrics(parts[split]["voted_up"].to_numpy(), predictions[split])
            output["models"][model_name][split] = metrics
            if split == "test":
                plot_curves(
                    parts[split]["voted_up"].to_numpy(),
                    predictions[split],
                    model_name,
                    split,
                )
        test_frame = parts["test"].reset_index(drop=True)
        output["models"][model_name]["test_bootstrap_by_appid"] = bootstrap_by_appid(
            test_frame, predictions["test"]
        )
        output["models"][model_name]["test_mean_per_game_min_20_reviews"] = per_game_metrics(
            test_frame, predictions["test"]
        )
        pd.DataFrame(
            {
                "appid": test_frame["appid"],
                "voted_up": test_frame["voted_up"],
                "p_negative": predictions["test"],
            }
        ).to_csv(OUTPUT / f"{model_name}_test_predictions.csv", index=False)

    (OUTPUT / "grouped_evaluation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
