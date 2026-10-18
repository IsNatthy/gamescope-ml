"""Punto de entrada para generar el EDA de reseñas sobre train."""

import argparse
from pathlib import Path

from src.reporting.reviews_eda_report import ReviewsEDAReport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="EDA de reseñas sobre el conjunto de train")
    parser.add_argument("--train-file", type=Path, default=PROCESSED / "reviews_train.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "eda" / "reviews")
    args = parser.parse_args()

    ReviewsEDAReport().generate(args.train_file.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
