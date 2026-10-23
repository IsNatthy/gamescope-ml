"""Punto de entrada para construir el dataset de modelado de reseñas."""

import argparse
from pathlib import Path

from src.features.build_reviews_dataset import ReviewsDatasetBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(
        description="Construye reviews_model_dataset.csv (texto depurado + target balanceado)"
    )
    parser.add_argument("--input-file", type=Path,
                        default=PROCESSED / "steam_game_reviews_clean.csv")
    parser.add_argument("--output-file", type=Path,
                        default=PROCESSED / "reviews_model_dataset.csv")
    parser.add_argument("--report-file", type=Path,
                        default=PROJECT_ROOT / "reports" / "features" / "reviews_model_dataset.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescribe las salidas si ya existen")
    args = parser.parse_args()

    ReviewsDatasetBuilder().build(
        args.input_file.resolve(),
        args.output_file.resolve(),
        args.report_file.resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
