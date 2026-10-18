"""Entrena un clasificador BERT de sentimiento con reseñas de Steam.

Usa ``review`` como texto y ``voted_up`` como etiqueta inicial. Esta etiqueta
es un proxy de sentimiento y debe documentarse como tal en el informe.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "bert_sentiment"
REPORT_DIR = ROOT / "reports" / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROCESSED / "steam_game_reviews_clean.csv")
    parser.add_argument("--max-rows", type=int, default=20_000,
                        help="Máximo de reseñas; reduce RAM y tiempo de entrenamiento.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "Faltan dependencias de BERT. Ejecuta: "
            "python -m pip install -r requirements-modeling.txt"
        ) from error

    if args.batch_size < 1 or args.max_length < 16 or args.max_rows < 100:
        raise ValueError("max-rows, batch-size y max-length deben ser valores razonables.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    df = pd.read_csv(
        args.input, usecols=["review", "voted_up"], nrows=args.max_rows,
        dtype={"review": "string", "voted_up": "string"}, keep_default_na=False,
    )
    df["review"] = df["review"].fillna("").str.strip()
    labels = df["voted_up"].str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    df = df.loc[df["review"].str.len().ge(10) & labels.notna()].copy()
    df["label"] = labels.loc[df.index].astype("int64")
    if len(df) < 100 or df["label"].nunique() < 2:
        raise ValueError("No hay suficientes reseñas o clases para entrenar BERT.")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
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
    test_loader = DataLoader(
        ReviewDataset(test_df), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss.detach().cpu())
            if step % 100 == 0:
                print(f"época {epoch + 1}, lote {step}, pérdida media {total_loss / step:.4f}")

    model.eval()
    actual, predicted = [], []
    with torch.inference_mode():
        for batch in test_loader:
            labels_batch = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            actual.extend(labels_batch.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "model": args.model_name,
        "device": str(device),
        "rows": len(df),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1": float(f1_score(actual, predicted)),
        "classification_report": classification_report(
            actual, predicted, output_dict=True, zero_division=0
        ),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    (REPORT_DIR / "bert_sentiment.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
