"""Reproduce data, language, token-length, game, and genre checks for reviews."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "src" / "data" / "processed"
MODELS = ROOT / "src" / "models"
REPORTS = ROOT / "src" / "reports" / "features"
OUTPUT = ROOT / "results" / "pendientes_resenas"
TOKENIZER_FILE = MODELS / "bert_sentiment" / "tokenizer.json"
SAMPLE_SIZE = 2_000
RANDOM_STATE = 42


def target_values(values: pd.Series) -> pd.Series:
    mapped = values.astype("string").str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    return mapped.astype("Int8")


def clean_stage_counts(frame: pd.DataFrame) -> dict:
    review = frame["review"].fillna("").astype("string")
    target = target_values(frame["voted_up"])
    nonblank = review.str.count(r"\S").clip(lower=1)
    letter_ratio = review.str.count(r"[A-Za-z]") / nonblank
    counts = {"before_cleaning": int(len(frame))}

    nonempty = review.str.strip().str.len().gt(0)
    counts["empty_text_rows"] = int((~nonempty).sum())
    keep = letter_ratio.ge(0.5) & target.notna()
    filtered = frame.loc[keep, ["review"]].copy()
    filtered["target"] = target.loc[keep].astype("int8").to_numpy()
    counts["after_letter_ratio_and_valid_target"] = int(len(filtered))
    labels_per_text = filtered.groupby("review", sort=False)["target"].transform("nunique")
    consistent = labels_per_text.eq(1)
    counts["contradictory_text_rows_removed"] = int((~consistent).sum())
    filtered = filtered.loc[consistent]
    unique = ~filtered["review"].duplicated(keep="first")
    counts["duplicate_review_text_rows_removed"] = int((~unique).sum())
    counts["after_cleaning_and_text_deduplication"] = int(unique.sum())
    return counts


def load_language_sample(dataset: pd.DataFrame) -> pd.DataFrame:
    DetectorFactory.seed = RANDOM_STATE
    sample = dataset.sample(
        n=min(SAMPLE_SIZE, len(dataset)), random_state=RANDOM_STATE
    )
    languages = []
    for text in sample["review"].astype(str):
        try:
            languages.append(detect(text))
        except LangDetectException:
            languages.append("und")
    return (
        pd.Series(languages, name="language")
        .value_counts()
        .rename_axis("language")
        .reset_index(name="reviews")
        .assign(percent=lambda frame: 100 * frame["reviews"] / len(sample))
    )


def token_length_table(reviews: pd.Series) -> pd.DataFrame:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    texts = reviews.astype(str).tolist()
    lengths: list[int] = []
    batch_size = 512
    for start in range(0, len(texts), batch_size):
        encodings = tokenizer.encode_batch(texts[start : start + batch_size])
        lengths.extend(len(encoding.ids) for encoding in encodings)
    values = np.asarray(lengths, dtype=np.int64)
    rows = []
    for limit in (128, 256):
        rows.append(
            {
                "max_length": limit,
                "reviews": int(len(values)),
                "mean_tokens": float(values.mean()),
                "median_tokens": float(np.median(values)),
                "p90_tokens": float(np.percentile(values, 90)),
                "p95_tokens": float(np.percentile(values, 95)),
                "p99_tokens": float(np.percentile(values, 99)),
                "max_tokens": int(values.max()),
                "truncated_reviews": int((values > limit).sum()),
                "truncated_percent": float(100 * (values > limit).mean()),
            }
        )
    return pd.DataFrame(rows)


def parse_genres(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    labels = target_values(frame["voted_up"])
    return {str(int(key)): int(value) for key, value in labels.value_counts().sort_index().items()}


def class_percentages(distribution: dict[str, int]) -> dict[str, float]:
    total = sum(distribution.values())
    return {
        key: 100 * value / total for key, value in distribution.items()
    } if total else {}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        PROCESSED / "steam_game_reviews_clean.csv",
        usecols=["appid", "review", "voted_up"],
        dtype={"appid": "string", "review": "string", "voted_up": "string"},
        keep_default_na=False,
    )
    dataset = pd.read_csv(
        PROCESSED / "reviews_model_dataset.csv",
        usecols=["appid", "review", "voted_up"],
        dtype={"appid": "string", "review": "string", "voted_up": "string"},
        keep_default_na=False,
    )
    data_report = json.loads((REPORTS / "reviews_model_dataset.json").read_text(encoding="utf-8"))
    split_report = json.loads((REPORTS / "reviews_split.json").read_text(encoding="utf-8"))
    target_data = dataset[["appid", "review", "voted_up"]].copy()
    target_data["review"] = target_data["review"].str.strip()
    bertopic_partitions = {}
    for name in ("train", "validation", "test"):
        assigned = pd.read_csv(
            MODELS / f"bertopic_{name}_topics.csv",
            usecols=["appid", "review"],
            dtype={"appid": "string", "review": "string"},
            keep_default_na=False,
        )
        assigned["review"] = assigned["review"].str.strip()
        matched = assigned.merge(
            target_data, on=["appid", "review"], how="left", validate="one_to_one"
        )
        bertopic_partitions[name] = {
            "rows": int(len(matched)),
            "matched_labels": int(matched["voted_up"].notna().sum()),
            "games": int(matched["appid"].nunique()),
            "class_counts": class_distribution(matched),
        }

    partitions = {}
    for name in ("train", "test"):
        partition = pd.read_csv(
            PROCESSED / f"reviews_{name}.csv",
            usecols=["review", "voted_up"],
            dtype={"review": "string", "voted_up": "string"},
            nrows=5_000,
            keep_default_na=False,
        )
        partition["review"] = partition["review"].fillna("").str.strip()
        labels = target_values(partition["voted_up"])
        keep = partition["review"].str.len().ge(10) & labels.notna()
        partitions[f"bert_pilot_{name}_first_5000"] = class_distribution(
            partition.loc[keep].assign(voted_up=labels.loc[keep].astype(str))
        )

    raw_counts = clean_stage_counts(raw)
    lang_counts = load_language_sample(dataset)
    token_lengths = token_length_table(dataset["review"])

    game_counts = dataset.groupby("appid", dropna=False).size().rename("review_count")
    catalog = pd.read_csv(
        PROCESSED / "games_fixed_clean.csv",
        usecols=["appid", "genres"],
        dtype={"appid": "string", "genres": "string"},
        keep_default_na=False,
    ).drop_duplicates("appid", keep="first")
    catalog["genre_list"] = catalog["genres"].map(parse_genres)
    catalog["primary_genre"] = catalog["genre_list"].map(
        lambda values: values[0] if values else "Unknown"
    )
    game_table = game_counts.reset_index().merge(
        catalog[["appid", "primary_genre"]], on="appid", how="left", indicator=True
    )
    game_table["primary_genre"] = game_table["primary_genre"].fillna("Unknown")
    genre_counts = (
        game_table.groupby("primary_genre", dropna=False)["review_count"]
        .sum()
        .sort_values(ascending=False)
        .rename("reviews")
        .reset_index()
    )
    genre_counts.to_csv(OUTPUT / "genre_review_counts.csv", index=False)
    lang_counts.to_csv(OUTPUT / "language_sample_counts.csv", index=False)
    token_lengths.to_csv(OUTPUT / "wordpiece_lengths.csv", index=False)

    appid_stats = {
        "games": int(game_counts.size),
        "reviews_per_game_min": int(game_counts.min()),
        "reviews_per_game_median": float(game_counts.median()),
        "reviews_per_game_max": int(game_counts.max()),
    }
    known_games = int((game_table["_merge"] == "both").sum())
    unique_raw_appids = int(raw["appid"].nunique())

    summary = {
        "cleaning": {
            "recomputed_raw_stages": raw_counts,
            "reported_builder_metadata": data_report,
        },
        "class_distribution": {
            "raw_cleaned_reviews": class_distribution(raw),
            "balanced_model_dataset": class_distribution(dataset),
            "full_train_partition": split_report["train"]["distribucion"],
            "full_test_partition": split_report["test"]["distribucion"],
            "pilot_first_5000_after_run_train_bert_filter": partitions,
            "pilot_validation_support_from_existing_bert_report": {
                "0": 2503,
                "1": 2497,
                "source": "src/models/bert_sentiment_standard.json",
            },
            "pilot_test_support_from_existing_bert_report": {
                "0": 2474,
                "1": 2526,
                "source": "src/models/bert_sentiment_standard.json",
            },
            "validation_csv_available": (PROCESSED / "reviews_validation.csv").exists(),
        },
        "class_percentages": {
            "raw_cleaned_reviews": class_percentages(class_distribution(raw)),
            "balanced_model_dataset": class_percentages(class_distribution(dataset)),
            "current_full_train_partition": class_percentages(split_report["train"]["distribucion"]),
            "current_full_test_partition": class_percentages(split_report["test"]["distribucion"]),
            "pilot_validation_from_bert_report": class_percentages({"0": 2503, "1": 2497}),
            "pilot_test_from_bert_report": class_percentages({"0": 2474, "1": 2526}),
            "bertopic_saved_assignments": {
                name: class_percentages(values["class_counts"])
                for name, values in bertopic_partitions.items()
            },
        },
        "language_sample": {
            "size": int(lang_counts["reviews"].sum()),
            "seed": RANDOM_STATE,
            "method": "langdetect.detect on a random sample from the balanced model dataset",
            "english_percent": float(
                lang_counts.loc[lang_counts["language"] == "en", "percent"].sum()
            ),
            "counts_file": "language_sample_counts.csv",
        },
        "wordpiece_lengths": {
            "tokenizer": str(TOKENIZER_FILE.relative_to(ROOT)),
            "method": "Tokenizer JSON encode with special tokens; truncated iff length > L",
            "file": "wordpiece_lengths.csv",
        },
        "game_coverage": {
            "unique_appids_in_clean_review_source": unique_raw_appids,
            "unique_appids_in_balanced_model_dataset": int(dataset["appid"].nunique()),
            "model_dataset_appids_found_in_game_catalog": known_games,
            "model_dataset_appids_missing_from_game_catalog": int(
                (game_table["_merge"] == "left_only").sum()
            ),
            "games_review_distribution": appid_stats,
            "genres_with_at_least_100_reviews": int(
                (
                    (genre_counts["reviews"] >= 100)
                    & genre_counts["primary_genre"].ne("Unknown")
                ).sum()
            ),
            "genre_counts_file": "genre_review_counts.csv",
        },
        "pilot_bertopic": {
            "train_rows": 4999,
            "validation_rows": 2000,
            "test_rows": 1999,
            "note": "Counts found in src/models/bertopic_evaluation.json; topic assignment files are present.",
            "assignments_with_target_join": bertopic_partitions,
        },
    }
    (OUTPUT / "data_checks.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
