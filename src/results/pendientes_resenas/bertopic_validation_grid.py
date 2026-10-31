"""Tune BERTopic on saved train/validation reviews; evaluate test only at the end."""

from __future__ import annotations

import ast
import argparse
import gc
import json
import time
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "src" / "models"
PROCESSED = ROOT / "src" / "data" / "processed"
OUTPUT = ROOT / "results" / "pendientes_resenas"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RANDOM_STATE = 42
TOP_N = 10
EMBEDDING_BATCH_SIZE = 16
MAX_FEATURES = 30_000


def load_topic_reviews(name: str) -> pd.DataFrame:
    frame = pd.read_csv(
        MODEL_ROOT / f"bertopic_{name}_topics.csv",
        usecols=["appid", "review"],
        dtype={"appid": "string", "review": "string"},
        keep_default_na=False,
    )
    frame["review"] = frame["review"].fillna("").str.strip()
    return frame.loc[frame["review"].str.len().ge(20)].reset_index(drop=True)


def top_words_from_model(model) -> dict[int, list[str]]:
    result = {}
    for topic_id, pairs in model.get_topics().items():
        if topic_id == -1:
            continue
        words = [word for word, _ in pairs[:TOP_N]]
        if words:
            result[int(topic_id)] = words
    return result


def topic_npmi(documents: list[str], topic_words: dict[int, list[str]]) -> float:
    if not documents or not topic_words:
        return 0.0
    vectorizer = CountVectorizer(stop_words="english", binary=True, min_df=2)
    matrix = vectorizer.fit_transform(documents)
    vocabulary = vectorizer.vocabulary_
    total_docs = matrix.shape[0]
    values = []
    for words in topic_words.values():
        indices = [vocabulary[word] for word in words if word in vocabulary]
        for left, right in combinations(indices, 2):
            p_left = float(matrix[:, left].sum()) / total_docs
            p_right = float(matrix[:, right].sum()) / total_docs
            p_both = float(matrix[:, left].multiply(matrix[:, right]).sum()) / total_docs
            if p_left and p_right and p_both:
                pmi = np.log(p_both / (p_left * p_right))
                values.append(float(pmi / -np.log(p_both)))
    return float(np.mean(values)) if values else 0.0


def diversity(topic_words: dict[int, list[str]]) -> float:
    words = [word for topic in topic_words.values() for word in topic]
    return float(len(set(words)) / len(words)) if words else 0.0


def make_model(seed: int, n_neighbors: int, n_components: int,
               min_cluster_size: int, min_samples: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    return BERTopic(
        umap_model=UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
            n_jobs=1,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            prediction_data=True,
        ),
        vectorizer_model=CountVectorizer(
            stop_words="english", ngram_range=(1, 1), min_df=2
        ),
        nr_topics=None,
        calculate_probabilities=False,
        top_n_words=TOP_N,
        verbose=False,
    )


def summarize_assignments(documents: list[str], labels: np.ndarray,
                          topic_words: dict[int, list[str]]) -> dict:
    labels = np.asarray(labels)
    sizes = {
        str(int(topic)): int((labels == topic).sum())
        for topic in sorted(set(labels.tolist()))
        if topic != -1
    }
    return {
        "topic_count": len(topic_words),
        "npmi": topic_npmi(documents, topic_words),
        "topic_diversity": diversity(topic_words),
        "outlier_fraction": float(np.mean(labels == -1)),
        "topic_sizes": sizes,
        "largest_topic_share_non_outlier": (
            max(sizes.values()) / sum(sizes.values()) if sizes else 1.0
        ),
    }


def best_topic_jaccard(left: dict[int, list[str]],
                       right: dict[int, list[str]]) -> float:
    if not left or not right:
        return 0.0
    right_sets = [set(words) for words in right.values()]
    values = []
    for words in left.values():
        current = set(words)
        overlaps = [
            len(current & candidate) / len(current | candidate)
            for candidate in right_sets
            if current | candidate
        ]
        if overlaps:
            values.append(max(overlaps))
    return float(np.mean(values)) if values else 0.0


def short_review(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    return compact[:limit].rstrip()


def baseline_topics(train_documents: list[str], topic_count: int) -> pd.DataFrame:
    vectorizer = CountVectorizer(
        stop_words="english", ngram_range=(1, 1), min_df=2,
        max_features=MAX_FEATURES,
    )
    matrix = vectorizer.fit_transform(train_documents)
    words = np.asarray(vectorizer.get_feature_names_out())
    topic_count = max(2, min(topic_count, max(2, matrix.shape[0] // 20)))

    lda = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=10,
    )
    lda.fit(matrix)

    nmf = NMF(
        n_components=topic_count,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=200,
    )
    nmf.fit(matrix)

    rows = []
    for name, components in (("LDA", lda.components_), ("NMF", nmf.components_)):
        topic_words = {
            topic: words[component.argsort()[-TOP_N:][::-1]].tolist()
            for topic, component in enumerate(components)
        }
        npmi = topic_npmi(train_documents, topic_words)
        topic_diversity = diversity(topic_words)
        for topic, top_words in topic_words.items():
            rows.append(
                {
                    "model": name,
                    "topic": topic,
                    "top_words": " | ".join(top_words),
                    "npmi_train": npmi,
                    "topic_diversity": topic_diversity,
                }
            )
    return pd.DataFrame(rows)


def integrate_topics(test_assignments: pd.DataFrame) -> pd.DataFrame:
    catalog = pd.read_csv(
        PROCESSED / "games_fixed_clean.csv",
        usecols=["appid", "genres"],
        dtype={"appid": "string", "genres": "string"},
        keep_default_na=False,
    ).drop_duplicates("appid", keep="first")

    def primary_genre(value: str) -> str | None:
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(parsed, list):
            return str(parsed[0]).strip() if parsed else None
        return str(parsed).strip() if parsed else None

    catalog["primary_genre"] = catalog["genres"].map(primary_genre)
    data = test_assignments.merge(
        catalog[["appid", "primary_genre"]], on="appid", how="left"
    )
    data = data.loc[
        data["primary_genre"].notna()
        & data["primary_genre"].ne("")
    ].copy()
    data["negative"] = data["voted_up"].eq(0).astype(int)
    segment_totals = data.groupby("primary_genre").agg(
        N_s=("negative", "size"), negatives_s=("negative", "sum")
    )
    eligible_segments = segment_totals.index[segment_totals["N_s"] >= 100]

    rows = []
    z = 1.959963984540054
    for genre in eligible_segments:
        group = data.loc[data["primary_genre"] == genre]
        n_s = int(len(group))
        q_s = float(group["negative"].mean())
        topics_group = group.loc[group["topic"].ne(-1)]
        topic_phi = topics_group["topic"].value_counts(normalize=False) / n_s
        top_three = set(topic_phi.sort_values(ascending=False).head(3).index)
        for topic_id, topic_frame in topics_group.groupby("topic"):
            n_ks = int(len(topic_frame))
            if n_ks < 10:
                continue
            n_negative = int(topic_frame["negative"].sum())
            q_ks = n_negative / n_ks
            phi_ks = n_ks / n_s
            impact = phi_ks * q_ks
            q_smooth = (n_ks * q_ks + 20 * q_s) / (n_ks + 20)
            denominator = 1 + z * z / n_ks
            center = (q_ks + z * z / (2 * n_ks)) / denominator
            margin = (
                z
                * np.sqrt(q_ks * (1 - q_ks) / n_ks + z * z / (4 * n_ks * n_ks))
                / denominator
            )
            lower, upper = max(0.0, center - margin), min(1.0, center + margin)
            rows.append(
                {
                    "topic": int(topic_id),
                    "primary_genre": genre,
                    "n_k_s": n_ks,
                    "negative_count": n_negative,
                    "N_s": n_s,
                    "phi_k_s": phi_ks,
                    "q_k_s": q_ks,
                    "q_s": q_s,
                    "I_k_s": impact,
                    "q_tilde_m20": q_smooth,
                    "wilson_lower_95": lower,
                    "wilson_upper_95": upper,
                    "top_3_phi": topic_id in top_three,
                    "highlight": lower > q_s and topic_id in top_three,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["highlight", "wilson_lower_95", "I_k_s"],
        ascending=[False, False, False],
    ).head(10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Evaluate four configurations on train/validation and stop before final evaluation.",
    )
    args = parser.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    OUTPUT.mkdir(parents=True, exist_ok=True)
    train = load_topic_reviews("train")
    validation = load_topic_reviews("validation")
    train_documents = train["review"].tolist()
    validation_documents = validation["review"].tolist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    train_embeddings = encoder.encode(
        train_documents,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    validation_embeddings = encoder.encode(
        validation_documents,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    grid_rows = []
    started = time.perf_counter()
    for n_neighbors, n_components, min_cluster_size, min_samples in product(
        (15,), (5,), (10, 50), (5, 10)
    ):
        config = {
            "n_neighbors": n_neighbors,
            "n_components": n_components,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "seed": RANDOM_STATE,
        }
        model = make_model(**config)
        train_topics, _ = model.fit_transform(
            train_documents, embeddings=train_embeddings
        )
        words = top_words_from_model(model)
        validation_topics, _ = model.transform(
            validation_documents, embeddings=validation_embeddings
        )
        summary = summarize_assignments(
            validation_documents, np.asarray(validation_topics), words
        )
        accepted = (
            summary["topic_count"] >= 5
            and summary["largest_topic_share_non_outlier"] <= 0.50
        )
        grid_rows.append(
            {
                **config,
                **summary,
                "accepted": accepted,
                "topic_words": json.dumps(words, ensure_ascii=False),
            }
        )
        del model
        gc.collect()

    grid = pd.DataFrame(grid_rows).sort_values(
        ["accepted", "npmi"], ascending=[False, False]
    )
    grid.to_csv(OUTPUT / "bertopic_grid_validation.csv", index=False)
    eligible = grid.loc[grid["accepted"]]
    if eligible.empty:
        result = {
            "status": "no_configuration_met_filters",
            "criterion": "highest validation NPMI among K >= 5 and largest non-outlier topic <= 50%",
            "grid_rows": len(grid),
        }
        (OUTPUT / "bertopic_selected.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return

    best = eligible.iloc[0]
    best_config = {
        "seed": int(best["seed"]),
        "n_neighbors": int(best["n_neighbors"]),
        "n_components": int(best["n_components"]),
        "min_cluster_size": int(best["min_cluster_size"]),
        "min_samples": int(best["min_samples"]),
    }
    if args.grid_only:
        result = {
            "status": "grid_only_selection",
            "embedding_model": EMBEDDING_MODEL,
            "device": device,
            "grid_configurations": len(grid),
            "selection_criterion": (
                "highest validation NPMI among K >= 5 and "
                "largest non-outlier topic <= 50%"
            ),
            "configuration": {
                key: value for key, value in best_config.items() if key != "seed"
            },
            "validation": {
                key: best[key]
                for key in (
                    "topic_count",
                    "npmi",
                    "topic_diversity",
                    "outlier_fraction",
                    "topic_sizes",
                    "largest_topic_share_non_outlier",
                )
            },
            "topic_words": json.loads(best["topic_words"]),
            "test_evaluated": False,
            "next_steps_not_run": [
                "test transform and evaluation",
                "five-seed stability",
                "LDA and NMF baselines",
                "topic-segment integration",
            ],
            "elapsed_grid_seconds": float(time.perf_counter() - started),
        }
        (OUTPUT / "bertopic_selected.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    best_model = make_model(**best_config)
    best_train_topics, _ = best_model.fit_transform(
        train_documents, embeddings=train_embeddings
    )
    best_train_topics = np.asarray(best_train_topics)
    best_words = top_words_from_model(best_model)
    train_summary = summarize_assignments(
        train_documents, best_train_topics, best_words
    )
    validation_topics, _ = best_model.transform(
        validation_documents, embeddings=validation_embeddings
    )
    validation_summary = summarize_assignments(
        validation_documents, np.asarray(validation_topics), best_words
    )

    stability_scores = []
    stability_words = []
    for seed in (42, 43, 44, 45, 46):
        stable_model = make_model(
            seed=seed,
            n_neighbors=int(best["n_neighbors"]),
            n_components=int(best["n_components"]),
            min_cluster_size=int(best["min_cluster_size"]),
            min_samples=int(best["min_samples"]),
        )
        labels, _ = stable_model.fit_transform(
            train_documents, embeddings=train_embeddings
        )
        stability_scores.append(np.asarray(labels))
        stability_words.append(top_words_from_model(stable_model))
    ari = [
        adjusted_rand_score(stability_scores[left], stability_scores[right])
        for left, right in combinations(range(len(stability_scores)), 2)
    ]
    jaccard = [
        best_topic_jaccard(stability_words[left], stability_words[right])
        for left, right in combinations(range(len(stability_words)), 2)
    ]

    train_ids = best_train_topics
    counts = pd.Series(train_ids).value_counts()
    train_assignments = train[["appid"]].copy()
    train_assignments["topic"] = train_ids
    word_rows = []
    train_matrix = normalize(train_embeddings)
    for topic_id, words in best_words.items():
        indices = np.flatnonzero(train_ids == topic_id)
        centroid = train_matrix[indices].mean(axis=0)
        similarities = train_matrix[indices] @ centroid
        representative_indices = indices[np.argsort(similarities)[-5:][::-1]]
        representatives = [
            short_review(train_documents[index]) for index in representative_indices
        ]
        word_rows.append(
            {
                "topic": topic_id,
                "train_size": int(counts.get(topic_id, 0)),
                "top_words": " | ".join(words),
                "representative_review_snippets": json.dumps(
                    representatives, ensure_ascii=False
                ),
            }
        )
    pd.DataFrame(word_rows).to_csv(OUTPUT / "bertopic_best_topics.csv", index=False)
    train_assignments.to_csv(OUTPUT / "bertopic_best_train_assignments.csv", index=False)

    # Test is first loaded and transformed only after validation selection is fixed.
    test = load_topic_reviews("test")
    test_documents = test["review"].tolist()
    test_embeddings = encoder.encode(
        test_documents,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    test_topics, _ = best_model.transform(test_documents, embeddings=test_embeddings)
    test_summary = summarize_assignments(
        test_documents, np.asarray(test_topics), best_words
    )

    target_data = pd.read_csv(
        PROCESSED / "reviews_model_dataset.csv",
        usecols=["appid", "review", "voted_up"],
        dtype={"appid": "string", "review": "string"},
        keep_default_na=False,
    )
    target_data["review"] = target_data["review"].str.strip()
    test_assignments = test[["appid", "review"]].copy()
    test_assignments["topic"] = np.asarray(test_topics, dtype=int)
    test_assignments = test_assignments.merge(
        target_data[["appid", "review", "voted_up"]],
        on=["appid", "review"],
        how="left",
        validate="one_to_one",
    )
    if test_assignments["voted_up"].isna().any():
        raise ValueError(
            "No se pudieron recuperar todas las etiquetas reales del test de BERTopic."
        )
    test_assignments.to_csv(
        OUTPUT / "bertopic_best_test_assignments.csv",
        columns=["appid", "topic", "voted_up"],
        index=False,
    )

    baseline = baseline_topics(train_documents, int(best["topic_count"]))
    baseline.to_csv(OUTPUT / "topic_baselines_lda_nmf.csv", index=False)
    integration = integrate_topics(test_assignments)
    integration.to_csv(OUTPUT / "topic_segment_integration.csv", index=False)

    selected = {
        "status": "selected",
        "embedding_model": EMBEDDING_MODEL,
        "device": device,
        "grid_configurations": len(grid),
        "selection_criterion": "highest validation NPMI among K >= 5 and largest non-outlier topic <= 50%",
        "configuration": {
            "n_neighbors": int(best["n_neighbors"]),
            "n_components": int(best["n_components"]),
            "min_cluster_size": int(best["min_cluster_size"]),
            "min_samples": int(best["min_samples"]),
            "metric_umap": "cosine",
            "metric_hdbscan": "euclidean",
            "random_state": RANDOM_STATE,
            "vectorizer_stop_words": "english",
            "vectorizer_ngram_range": [1, 1],
            "nr_topics": None,
        },
        "train": train_summary,
        "validation": validation_summary,
        "test": test_summary,
        "stability": {
            "seeds": [42, 43, 44, 45, 46],
            "pair_count": len(ari),
            "ari_mean": float(np.mean(ari)),
            "ari_std": float(np.std(ari)),
            "word_jaccard_mean": float(np.mean(jaccard)),
            "word_jaccard_std": float(np.std(jaccard)),
        },
        "elapsed_grid_seconds": float(time.perf_counter() - started),
        "baseline_models": ["LDA", "NMF"],
        "integration_highlights": int(
            integration["highlight"].sum() if not integration.empty else 0
        ),
    }
    (OUTPUT / "bertopic_selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
