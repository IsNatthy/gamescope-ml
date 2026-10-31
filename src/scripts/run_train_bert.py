"""Entrena y evalúa BERT para recomendar/no recomendar una reseña.

Usa particiones 80/10/10 agrupadas por appid. ``voted_up`` es una etiqueta de
recomendación global y se documenta como proxy de sentimiento.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "models"


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= 0.5).astype(int)
    confidence = np.maximum(probabilities, 1 - probabilities)
    error = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            accuracy = (predicted[mask] == labels[mask]).mean()
            error += mask.mean() * abs(accuracy - confidence[mask].mean())
    return float(error)


def classification_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_negative": float(precision[0]),
        "recall_negative": float(recall[0]),
        "f1_negative": float(f1[0]),
        "f1_macro": float(f1_score(labels, predicted, average="macro")),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "classification_report": classification_report(
            labels, predicted, output_dict=True, zero_division=0
        ),
    }


def load_partition(path, max_rows):
    frame = pd.read_csv(
        path, usecols=["review", "voted_up"], nrows=max_rows,
        dtype={"review": "string", "voted_up": "string"}, keep_default_na=False,
    )
    frame["review"] = frame["review"].fillna("").str.strip()
    labels = frame["voted_up"].str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    frame = frame.loc[frame["review"].str.len().ge(10) & labels.notna()].copy()
    frame["label"] = labels.loc[frame.index].astype("int64")
    return frame


def evaluate_tfidf(train_df, validation_df, test_df, args):
    vectorizer = TfidfVectorizer(
        max_features=args.tfidf_features, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True,
    )
    train_x = vectorizer.fit_transform(train_df["review"])
    validation_x = vectorizer.transform(validation_df["review"])
    test_x = vectorizer.transform(test_df["review"])
    model = LogisticRegression(max_iter=200, C=1.0, solver="liblinear", random_state=42)
    model.fit(train_x, train_df["label"])
    validation_probability = model.predict_proba(validation_x)[:, 1]
    test_probability = model.predict_proba(test_x)[:, 1]
    return {
        "model": "tfidf_logistic_regression",
        "validation": classification_metrics(validation_df["label"], validation_probability),
        "test": classification_metrics(test_df["label"], test_probability),
        "features": len(vectorizer.vocabulary_),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=PROCESSED / "reviews_train.csv")
    parser.add_argument("--validation", type=Path, default=PROCESSED / "reviews_validation.csv")
    parser.add_argument("--test", type=Path, default=PROCESSED / "reviews_test.csv")
    parser.add_argument("--max-rows", type=int, default=20_000,
                        help="Máximo de reseñas por partición para controlar RAM.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--weighted", action="store_true",
                        help="Pondera la pérdida por frecuencia de clase.")
    parser.add_argument("--no-tfidf", action="store_true")
    parser.add_argument("--tfidf-features", type=int, default=100_000)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Faltan dependencias de BERT. Ejecuta: "
            "python -m pip install -r requirements-modeling.txt"
        ) from error

    if min(args.batch_size, args.max_length, args.max_rows, args.epochs) < 1:
        raise ValueError("batch-size, max-length, max-rows y epochs deben ser positivos.")
    if args.patience < 0:
        raise ValueError("patience no puede ser negativo.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_df = load_partition(args.train, args.max_rows)
    validation_df = load_partition(args.validation, args.max_rows)
    test_df = load_partition(args.test, args.max_rows)
    for name, frame in (("train", train_df), ("validation", validation_df), ("test", test_df)):
        if len(frame) < 100 or frame["label"].nunique() < 2:
            raise ValueError(f"No hay suficientes reseñas o clases en {name}.")

    results = {
        "configuration": {
            "device": str(device), "weighted": args.weighted,
            "epochs": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "max_length": args.max_length,
            "train_rows": len(train_df), "validation_rows": len(validation_df),
            "test_rows": len(test_df),
        },
        "baselines": {},
    }
    if not args.no_tfidf:
        print("Entrenando baseline TF-IDF + regresión logística...", flush=True)
        results["baselines"]["tfidf_logistic_regression"] = evaluate_tfidf(
            train_df, validation_df, test_df, args
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    class ReviewDataset(Dataset):
        def __init__(self, frame):
            self.texts = frame["review"].tolist()
            self.labels = frame["label"].tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            encoded = tokenizer(
                self.texts[index], truncation=True, padding="max_length",
                max_length=args.max_length, return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    train_loader = DataLoader(
        ReviewDataset(train_df), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    validation_loader = DataLoader(
        ReviewDataset(validation_df), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ReviewDataset(test_df), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    class_weights = None
    if args.weighted:
        counts = np.bincount(train_df["label"], minlength=2).astype("float64")
        class_weights = torch.tensor(
            counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device
        )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    def predict(loader, include_loss=False):
        model.eval()
        labels, probabilities, losses = [], [], []
        with torch.inference_mode():
            for batch in loader:
                labels_batch = batch.pop("labels").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                loss = criterion(output.logits, labels_batch)
                probabilities.extend(torch.softmax(output.logits, dim=1)[:, 1].cpu().tolist())
                labels.extend(labels_batch.cpu().tolist())
                losses.append(float(loss.cpu()))
        return np.asarray(labels), np.asarray(probabilities), float(np.mean(losses))

    best_state = None
    best_validation_f1 = -np.inf
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader, start=1):
            labels_batch = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                print(f"epoch {epoch + 1}, batch {step}, loss {float(loss.detach().cpu()):.4f}")
        validation_labels, validation_probability, validation_loss = predict(validation_loader)
        validation_result = classification_metrics(validation_labels, validation_probability)
        print(
            f"epoch {epoch + 1}: validation f1_macro={validation_result['f1_macro']:.4f}, "
            f"loss={validation_loss:.4f}"
        )
        if validation_result["f1_macro"] > best_validation_f1:
            best_validation_f1 = validation_result["f1_macro"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > args.patience:
                print("Parada temprana por falta de mejora en validation.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    validation_labels, validation_probability, _ = predict(validation_loader)
    test_labels, test_probability, _ = predict(test_loader)
    results["bert"] = {
        "model": args.model_name,
        "variant": "weighted" if args.weighted else "standard",
        "validation": classification_metrics(validation_labels, validation_probability),
        "test": classification_metrics(test_labels, test_probability),
    }

    model_dir = ROOT / "models" / ("bert_sentiment_weighted" if args.weighted else "bert_sentiment")
    model_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    report_name = "bert_sentiment_weighted.json" if args.weighted else "bert_sentiment_standard.json"
    (REPORT_DIR / report_name).write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
