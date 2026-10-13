"""Punto de entrada para dividir el dataset de modelado en train y test."""

import argparse
from pathlib import Path

from src.features.split_dataset import DatasetSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Split train/test (80/20, estratificado)")
    parser.add_argument("--input-file", type=Path, default=PROCESSED / "model_dataset.csv")
    parser.add_argument("--train-file", type=Path, default=PROCESSED / "train.csv")
    parser.add_argument("--test-file", type=Path, default=PROCESSED / "test.csv")
    parser.add_argument("--report-file", type=Path,
                        default=PROJECT_ROOT / "reports" / "features" / "split.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescribe las salidas si ya existen")
    args = parser.parse_args()

    DatasetSplitter().split(
        args.input_file.resolve(), args.train_file.resolve(),
        args.test_file.resolve(), args.report_file.resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()