"""Compute real-class summaries and prepare a per-game proportional sample."""

from __future__ import annotations

import ast
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "src" / "data" / "processed"
OUTPUT_ROOT = ROOT / "results" / "pendientes_resenas"
RANDOM_STATE = 42


def parse_genres(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def review_seed(appid: str) -> int:
    digest = hashlib.sha256(appid.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") ^ RANDOM_STATE


def clean_reviews() -> pd.DataFrame:
    frame = pd.read_csv(
        PROCESSED / "steam_game_reviews_clean.csv",
        usecols=["appid", "review", "voted_up"],
        dtype={"appid": "string", "review": "string", "voted_up": "string"},
        keep_default_na=False,
    )
    target = frame["voted_up"].str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}
    )
    text = frame["review"].fillna("")
    nonblank = text.str.count(r"\S").clip(lower=1)
    keep = (text.str.count(r"[A-Za-z]") / nonblank).ge(0.5) & target.notna()
    frame = frame.loc[keep, ["appid", "review"]].copy()
    frame["voted_up"] = target.loc[keep].astype("int8").to_numpy()
    consistent = frame.groupby("review", sort=False)["voted_up"].transform("nunique").eq(1)
    frame = frame.loc[consistent].drop_duplicates("review", keep="first")
    return frame.reset_index(drop=True)


def proportional_sample(frame: pd.DataFrame, max_per_game: int) -> pd.DataFrame:
    sampled_groups = []
    for appid, game in frame.groupby("appid", sort=True, observed=True):
        target_n = min(max_per_game, len(game))
        counts = game["voted_up"].value_counts().reindex([0, 1], fill_value=0)
        quotas_float = counts.to_numpy(dtype=float) * target_n / len(game)
        quotas = np.floor(quotas_float).astype(int)
        remainder = target_n - int(quotas.sum())
        order = np.argsort(-(quotas_float - quotas))
        for label in order:
            if remainder == 0:
                break
            if quotas[label] < counts.iloc[label]:
                quotas[label] += 1
                remainder -= 1
        seed = review_seed(str(appid))
        for label, quota in zip((0, 1), quotas, strict=True):
            if quota:
                sampled_groups.append(
                    game.loc[game["voted_up"].eq(label)].sample(
                        n=int(quota), random_state=seed
                    )
                )
    result = pd.concat(sampled_groups, ignore_index=True)
    return result.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def class_summary(frame: pd.DataFrame, group: str | None = None) -> pd.DataFrame:
    if group:
        counts = frame.groupby([group, "voted_up"], dropna=False).size().unstack(
            "voted_up", fill_value=0
        )
    else:
        counts = frame.groupby("voted_up", dropna=False).size().to_frame().T
    counts = counts.reindex(columns=[0, 1], fill_value=0)
    counts.columns = ["negative_n", "positive_n"]
    counts["total_n"] = counts["negative_n"] + counts["positive_n"]
    counts["negative_pct"] = 100 * counts["negative_n"] / counts["total_n"]
    counts["positive_pct"] = 100 * counts["positive_n"] / counts["total_n"]
    return counts.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-game", type=int, default=300)
    args = parser.parse_args()
    if args.max_per_game < 1:
        raise ValueError("--max-per-game must be positive.")
    output = OUTPUT_ROOT / f"descriptive_cap_{args.max_per_game}"
    output.mkdir(parents=True, exist_ok=True)
    cleaned = clean_reviews()
    catalog = pd.read_csv(
        PROCESSED / "games_fixed_clean.csv",
        usecols=["appid", "genres"],
        dtype={"appid": "string", "genres": "string"},
        keep_default_na=False,
    ).drop_duplicates("appid", keep="first")
    catalog["primary_genre"] = catalog["genres"].map(
        lambda value: (parse_genres(value) or ["Unknown"])[0]
    )
    joined = cleaned.merge(
        catalog[["appid", "primary_genre"]],
        on="appid",
        how="left",
        validate="many_to_one",
    )
    joined["primary_genre"] = joined["primary_genre"].fillna("Unknown")

    all_counts = class_summary(cleaned)
    genre_counts = class_summary(joined, "primary_genre")
    genre_counts = genre_counts.loc[
        genre_counts["total_n"].ge(100)
        & genre_counts["primary_genre"].ne("Unknown")
    ].sort_values("total_n", ascending=False)
    all_counts.to_csv(output / "real_class_distribution.csv", index=False)
    genre_counts.to_csv(output / "real_class_by_primary_genre.csv", index=False)

    balanced = pd.read_csv(
        PROCESSED / "reviews_model_dataset.csv",
        usecols=["appid", "review", "voted_up"],
        dtype={"appid": "string"},
        keep_default_na=False,
    )
    balanced["voted_up"] = balanced["voted_up"].astype("int8")
    sample = proportional_sample(cleaned, args.max_per_game)
    sample.to_csv(output / "descriptive_corpus_sample.csv", index=False)

    clean_games = cleaned["appid"].nunique()
    balanced_games = balanced["appid"].nunique()
    per_game_counts = cleaned.groupby("appid").size()
    sample_meta = {
        "clean_rows": int(len(cleaned)),
        "clean_games": int(clean_games),
        "balanced_rows": int(len(balanced)),
        "balanced_games": int(balanced_games),
        "rows_removed_by_global_class_undersampling": int(len(cleaned) - len(balanced)),
        "clean_games_absent_from_balanced_dataset": int(
            cleaned["appid"].nunique()
            - cleaned.loc[cleaned["appid"].isin(balanced["appid"]), "appid"].nunique()
        ),
        "clean_game_counts_min_median_max": [
            int(per_game_counts.min()),
            float(per_game_counts.median()),
            int(per_game_counts.max()),
        ],
        "balancing_method": (
            "Global random undersampling of the positive majority class to the "
            "negative-class count; random_state=42. No per-game cap."
        ),
        "descriptive_sample_rows": int(len(sample)),
        "descriptive_sample_games": int(sample["appid"].nunique()),
        "max_reviews_per_game": args.max_per_game,
        "sample_method": (
            f"Within each appid, retain at most {args.max_per_game} reviews, allocating class "
            "quotas proportionally to that game's observed labels (largest "
            "remainder rounding), then sample within class with deterministic "
            "SHA-256-derived seeds and shuffle with random_state=42."
        ),
        "descriptive_sample_classes": {
            str(int(label)): int(count)
            for label, count in sample["voted_up"].value_counts().sort_index().items()
        },
        "descriptive_sample_class_percentages": {
            str(int(label)): float(100 * count / len(sample))
            for label, count in sample["voted_up"].value_counts().sort_index().items()
        },
    }
    (output / "descriptive_corpus_prep.json").write_text(
        json.dumps(sample_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sample_meta, ensure_ascii=False, indent=2))
    print("\nReal class distribution:\n", all_counts.to_string(index=False))
    print("\nBy primary genre:\n", genre_counts.to_string(index=False))


if __name__ == "__main__":
    main()
