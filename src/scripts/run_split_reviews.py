"""Punto de entrada para dividir reseñas en train, validation y test."""

import argparse
from pathlib import Path

from src.features.split_reviews_dataset import ReviewsSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(
        description="Split 80/10/10 de reseñas, estratificado y agrupado por appid"
    )
    parser.add_argument("--input-file", type=Path,
                        default=PROCESSED / "reviews_model_dataset.csv")
    parser.add_argument("--train-file", type=Path, default=PROCESSED / "reviews_train.csv")
    parser.add_argument("--validation-file", type=Path,
                        default=PROCESSED / "reviews_validation.csv")
    parser.add_argument("--test-file", type=Path, default=PROCESSED / "reviews_test.csv")
    parser.add_argument("--report-file", type=Path,
                        default=PROJECT_ROOT / "reports" / "features" / "reviews_split.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescribe salidas existentes")
    args = parser.parse_args()

    ReviewsSplitter().split(
        args.input_file.resolve(), args.train_file.resolve(),
        args.validation_file.resolve(), args.test_file.resolve(),
        args.report_file.resolve(), overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
