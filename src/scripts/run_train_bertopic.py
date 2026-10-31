"""Entrena y evalúa BERTopic con coherencia, diversidad y estabilidad."""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "bertopic_reviews"
REPORT_DIR = ROOT / "reports" / "models"


def top_words_from_model(model, top_n):
    topics = {}
    for topic_id in model.get_topics():
        if topic_id == -1:
            continue
        words = [word for word, _ in model.get_topic(topic_id)[:top_n]]
        if words:
            topics[int(topic_id)] = words
    return topics


def diversity(topic_words):
    words = [word for words in topic_words.values() for word in words]
    return float(len(set(words)) / len(words)) if words else 0.0


def topic_npmi(documents, topic_words, min_df=2):
    """Calcula NPMI promedio de pares de palabras dentro de cada tópico."""
    if not topic_words or not documents:
        return 0.0
    vectorizer = CountVectorizer(stop_words="english", binary=True, min_df=min_df)
    matrix = vectorizer.fit_transform(documents)
    vocabulary = vectorizer.vocabulary_
    total_docs = matrix.shape[0]
    values = []
    for words in topic_words.values():
        indices = [vocabulary[word] for word in words if word in vocabulary]
        for left, right in combinations(indices, 2):
            p_left = matrix[:, left].sum() / total_docs
            p_right = matrix[:, right].sum() / total_docs
            p_both = matrix[:, left].multiply(matrix[:, right]).sum() / total_docs
            if p_both and p_left and p_right:
                pmi = np.log(p_both / (p_left * p_right))
                values.append(float(pmi / -np.log(p_both)))
    return float(np.mean(values)) if values else 0.0


def topic_summary(documents, topic_ids, topic_words, top_n):
    topic_ids = np.asarray(topic_ids)
    non_outlier = topic_ids[topic_ids != -1]
    return {
        "topics": len(topic_words),
        "npmi": topic_npmi(documents, topic_words),
        "diversity": diversity(topic_words),
        "outlier_fraction": float(np.mean(topic_ids == -1)),
        "topic_sizes": {
            str(int(topic)): int((topic_ids == topic).sum())
            for topic in sorted(set(non_outlier.tolist()))
        },
        "top_n_words": top_n,
    }


def word_jaccard(topic_words_a, topic_words_b):
    if not topic_words_a or not topic_words_b:
        return 0.0
    scores = []
    sets_b = [set(words) for words in topic_words_b.values()]
    for words in topic_words_a.values():
        current = set(words)
        scores.append(max(
            len(current & other) / len(current | other)
            for other in sets_b if current | other
        ))
    return float(np.mean(scores)) if scores else 0.0


def build_topic_model(
    embedding_model, seed, min_topic_size, n_neighbors, n_components, min_samples,
    top_words,
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        min_dist=0.0, metric="cosine", random_state=seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size, min_samples=min_samples,
        metric="euclidean", prediction_data=True,
    )
    return BERTopic(
        embedding_model=embedding_model, umap_model=umap_model,
        hdbscan_model=hdbscan_model, calculate_probabilities=False,
        top_n_words=top_words, verbose=True,
    )


def baseline_topics(documents, topic_count, max_features, top_n):
    count_vectorizer = CountVectorizer(stop_words="english", min_df=2, max_features=max_features)
    count_matrix = count_vectorizer.fit_transform(documents)
    vocabulary = np.asarray(count_vectorizer.get_feature_names_out())
    topic_count = max(2, min(topic_count, max(2, count_matrix.shape[0] // 20)))
    lda = LatentDirichletAllocation(
        n_components=topic_count, random_state=42, learning_method="batch", max_iter=10,
    )
    lda_topics = lda.fit_transform(count_matrix).argmax(axis=1)
    lda_words = {
        topic: vocabulary[component.argsort()[-top_n:][::-1]].tolist()
        for topic, component in enumerate(lda.components_)
    }
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=max_features)
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    tfidf_vocabulary = np.asarray(tfidf_vectorizer.get_feature_names_out())
    nmf = NMF(n_components=topic_count, init="nndsvda", random_state=42, max_iter=200)
    nmf_topics = nmf.fit_transform(tfidf_matrix).argmax(axis=1)
    nmf_words = {
        topic: tfidf_vocabulary[component.argsort()[-top_n:][::-1]].tolist()
        for topic, component in enumerate(nmf.components_)
    }
    return {
        "lda": {
            "model": "LDA", "topic_count": topic_count,
            "npmi": topic_npmi(documents, lda_words),
            "diversity": diversity(lda_words), "outlier_fraction": 0.0,
        },
        "nmf": {
            "model": "NMF", "topic_count": topic_count,
            "npmi": topic_npmi(documents, nmf_words),
            "diversity": diversity(nmf_words), "outlier_fraction": 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=PROCESSED / "reviews_train.csv")
    parser.add_argument("--validation", type=Path, default=PROCESSED / "reviews_validation.csv")
    parser.add_argument("--test", type=Path, default=PROCESSED / "reviews_test.csv")
    parser.add_argument("--max-rows", type=int, default=30_000)
    parser.add_argument("--eval-rows", type=int, default=5_000)
    parser.add_argument("--stability-rows", type=int, default=5_000)
    parser.add_argument("--min-topic-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--top-words", type=int, default=10)
    parser.add_argument("--embedding-batch-size", type=int, default=32,
                        help="Lote usado al calcular embeddings para controlar VRAM/RAM.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--baseline-rows", type=int, default=5_000)
    parser.add_argument("--baseline-features", type=int, default=30_000)
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()
    if args.max_rows < 100 or args.eval_rows < 100 or args.stability_rows < 100:
        raise ValueError("Los límites de filas deben ser >= 100.")
    if args.embedding_batch_size < 1 or args.top_words < 1:
        raise ValueError("embedding-batch-size y top-words deben ser positivos.")

    try:
        import torch
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise SystemExit(
            "Faltan dependencias de BERTopic. Ejecuta: "
            "python -m pip install -r requirements-modeling.txt"
        ) from error

    def load(path, limit):
        frame = pd.read_csv(
            path, usecols=["appid", "review"], nrows=limit,
            dtype={"appid": "string", "review": "string"}, keep_default_na=False,
        )
        frame["review"] = frame["review"].fillna("").str.strip()
        return frame[frame["review"].str.len().ge(20)].reset_index(drop=True)

    train = load(args.input, args.max_rows)
    validation = load(args.validation, args.eval_rows)
    test = load(args.test, args.eval_rows)
    if len(train) < args.min_topic_size * 2:
        raise ValueError("No hay suficientes reseñas para BERTopic.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo para embeddings: {device}")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    train_documents = train["review"].tolist()
    validation_documents = validation["review"].tolist()
    test_documents = test["review"].tolist()
    train_embeddings = embedding_model.encode(
        train_documents, show_progress_bar=True,
        batch_size=args.embedding_batch_size, convert_to_numpy=True,
    )
    validation_embeddings = embedding_model.encode(
        validation_documents, show_progress_bar=True,
        batch_size=args.embedding_batch_size, convert_to_numpy=True,
    )
    test_embeddings = embedding_model.encode(
        test_documents, show_progress_bar=True,
        batch_size=args.embedding_batch_size, convert_to_numpy=True,
    )
    model = build_topic_model(
        embedding_model, args.seeds[0], args.min_topic_size,
        args.n_neighbors, args.n_components, args.min_samples, args.top_words,
    )
    train_topics, _ = model.fit_transform(train_documents, embeddings=train_embeddings)
    train_words = top_words_from_model(model, args.top_words)
    results = {
        "configuration": {
            "train_rows": len(train), "validation_rows": len(validation),
            "test_rows": len(test), "device": device,
            "min_topic_size": args.min_topic_size, "seeds": args.seeds,
        },
        "train": topic_summary(train_documents, train_topics, train_words, args.top_words),
    }
    validation_topics, _ = model.transform(validation_documents, embeddings=validation_embeddings)
    test_topics, _ = model.transform(test_documents, embeddings=test_embeddings)
    results["validation_assignment"] = topic_summary(
        validation_documents, validation_topics, train_words, args.top_words
    )
    results["test_assignment"] = topic_summary(
        test_documents, test_topics, train_words, args.top_words
    )

    stability_docs = train_documents[:args.stability_rows]
    stability_embeddings = embedding_model.encode(
        stability_docs, show_progress_bar=False, batch_size=32,
    )
    stability_topics = []
    stability_words = []
    for seed in args.seeds:
        stable_model = build_topic_model(
            embedding_model, seed, args.min_topic_size,
            args.n_neighbors, args.n_components, args.min_samples, args.top_words,
        )
        topics, _ = stable_model.fit_transform(stability_docs, embeddings=stability_embeddings)
        stability_topics.append(np.asarray(topics))
        stability_words.append(top_words_from_model(stable_model, args.top_words))
    ari_values, jaccard_values = [], []
    for left, right in combinations(range(len(stability_topics)), 2):
        ari_values.append(adjusted_rand_score(stability_topics[left], stability_topics[right]))
        jaccard_values.append(word_jaccard(stability_words[left], stability_words[right]))
    results["stability"] = {
        "ari_mean": float(np.mean(ari_values)) if ari_values else 0.0,
        "ari_std": float(np.std(ari_values)) if ari_values else 0.0,
        "word_jaccard_mean": float(np.mean(jaccard_values)) if jaccard_values else 0.0,
        "word_jaccard_std": float(np.std(jaccard_values)) if jaccard_values else 0.0,
        "pairs": len(ari_values),
    }

    if not args.skip_baselines:
        baseline_docs = train["review"].head(args.baseline_rows).tolist()
        results["baselines"] = baseline_topics(
            baseline_docs, max(2, len(train_words)), args.baseline_features, args.top_words
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR, serialization="safetensors", save_ctfidf=True)
    pd.DataFrame({"appid": train["appid"], "topic": train_topics, "review": train["review"]}).to_csv(
        REPORT_DIR / "bertopic_train_topics.csv", index=False
    )
    pd.DataFrame({"appid": validation["appid"], "topic": validation_topics, "review": validation["review"]}).to_csv(
        REPORT_DIR / "bertopic_validation_topics.csv", index=False
    )
    pd.DataFrame({"appid": test["appid"], "topic": test_topics, "review": test["review"]}).to_csv(
        REPORT_DIR / "bertopic_test_topics.csv", index=False
    )
    model.get_topic_info().to_csv(REPORT_DIR / "bertopic_topics.csv", index=False)
    (REPORT_DIR / "bertopic_evaluation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
