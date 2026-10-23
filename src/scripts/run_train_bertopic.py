"""Descubre temas de reseñas con BERTopic usando un límite de filas."""

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "bertopic_reviews"
REPORT_DIR = ROOT / "reports" / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROCESSED / "reviews_train.csv")
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--min-topic-size", type=int, default=50)
    args = parser.parse_args()
    if args.max_rows < 100 or args.min_topic_size < 2:
        raise ValueError("Los límites deben ser valores razonables.")

    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as error:
        raise SystemExit(
            "Faltan dependencias de BERTopic. Ejecuta: "
            "python -m pip install -r requirements-modeling.txt"
        ) from error

    df = pd.read_csv(
        args.input, usecols=["appid", "review"], nrows=args.max_rows,
        dtype={"appid": "string", "review": "string"}, keep_default_na=False,
    )
    df["review"] = df["review"].fillna("").str.strip()
    df = df[df["review"].str.len().ge(20)].reset_index(drop=True)
    if len(df) < args.min_topic_size * 2:
        raise ValueError("No hay suficientes reseñas para BERTopic.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo para embeddings: {device}")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=args.min_topic_size,
        calculate_probabilities=False,
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(df["review"].tolist())
    df["topic"] = topics
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    topic_model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True)
    topic_model.get_topic_info().to_csv(REPORT_DIR / "bertopic_topics.csv", index=False)
    df[["appid", "topic", "review"]].to_csv(
        REPORT_DIR / "bertopic_review_topics.csv", index=False
    )
    print(f"Temas guardados en {REPORT_DIR}.")


if __name__ == "__main__":
    main()
