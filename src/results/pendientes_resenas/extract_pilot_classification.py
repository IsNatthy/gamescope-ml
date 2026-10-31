"""Extract reproducible pilot metrics from the saved BERT report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "src" / "models" / "bert_sentiment_standard.json"
OUTPUT = ROOT / "results" / "pendientes_resenas"


def reconstruct_negative_confusion(metrics: dict) -> tuple[int, int, int, int]:
    supports = metrics["classification_report"]
    actual_negative = int(supports["0"]["support"])
    actual_positive = int(supports["1"]["support"])
    tp = round(actual_negative * metrics["recall_negative"])
    fn = actual_negative - tp
    fp = round(tp / metrics["precision_negative"] - tp)
    tn = actual_positive - fp
    if (tp + tn) / (actual_negative + actual_positive) != metrics["accuracy"]:
        raise ValueError("Reconstructed confusion matrix does not match reported accuracy.")
    return tp, fn, fp, tn


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = []
    model_entries = [
        ("TF-IDF + logistic regression", report["baselines"]["tfidf_logistic_regression"]),
        ("DistilBERT standard", report["bert"]),
    ]
    for model_name, model in model_entries:
        for split in ("validation", "test"):
            metrics = model[split]
            tp, fn, fp, tn = reconstruct_negative_confusion(metrics)
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "accuracy": metrics["accuracy"],
                    "precision_negative": metrics["precision_negative"],
                    "recall_negative": metrics["recall_negative"],
                    "f1_negative": metrics["f1_negative"],
                    "f1_macro": metrics["f1_macro"],
                    "mcc": metrics["mcc"],
                    "roc_auc_negative": metrics["roc_auc"],
                    "pr_auc_negative": None,
                    "ece_reported_top_label": metrics["ece"],
                    "ece_negative": None,
                    "brier_negative": None,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "tn": tn,
                    "probability_vectors_saved": False,
                }
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "pilot_classification_metrics_available.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
