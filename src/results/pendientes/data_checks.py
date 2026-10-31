"""Verify threshold counts and source owner intervals without changing inputs."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "src" / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "src" / "data" / "processed" / "test.csv"
SOURCE_PATH = ROOT / "src" / "data" / "processed" / "games_fixed_clean.csv"
OUTPUT_PATH = ROOT / "results" / "pendientes" / "data_checks.json"
THRESHOLD = 20_000


def main() -> None:
    y_train = pd.read_csv(TRAIN_PATH, usecols=["owners_midpoint"])["owners_midpoint"]
    y_test = pd.read_csv(TEST_PATH, usecols=["owners_midpoint"])["owners_midpoint"]
    source = pd.read_csv(
        SOURCE_PATH, usecols=["estimated_owners"], dtype=str, keep_default_na=False
    )["estimated_owners"]
    bounds = source.str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$").apply(
        pd.to_numeric, errors="coerce"
    )
    valid = bounds.notna().all(axis=1)
    midpoint_zero = valid & ((bounds[0] + bounds[1]) / 2 == 0)
    crossing = valid & (bounds[0] < THRESHOLD) & (bounds[1] > THRESHOLD)
    touching = valid & (
        (bounds[0] == THRESHOLD) | (bounds[1] == THRESHOLD)
    )

    result = {
        "threshold": THRESHOLD,
        "train_rows": int(len(y_train)),
        "train_target_counts": [
            {"owners_midpoint": float(value), "count": int(count),
             "percent": float(count / len(y_train) * 100)}
            for value, count in y_train.value_counts().sort_index().items()
        ],
        "test_rows": int(len(y_test)),
        "test_segments": {
            "y_below_threshold": {
                "count": int((y_test < THRESHOLD).sum()),
                "percent": float((y_test < THRESHOLD).mean() * 100),
            },
            "y_at_or_above_threshold": {
                "count": int((y_test >= THRESHOLD).sum()),
                "percent": float((y_test >= THRESHOLD).mean() * 100),
            },
        },
        "source_interval_rows": int(len(source)),
        "valid_source_intervals": int(valid.sum()),
        "source_intervals_with_midpoint_zero": {
            str(interval): int(count)
            for interval, count in source.loc[midpoint_zero].value_counts().items()
        },
        "intervals_strictly_crossing_threshold": int(crossing.sum()),
        "intervals_with_boundary_equal_to_threshold": int(touching.sum()),
        "crossing_interval_values": {
            str(interval): int(count)
            for interval, count in source.loc[crossing].value_counts().items()
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
