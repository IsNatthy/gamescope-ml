"""Punto de entrada para generar los gráficos del EDA sobre train."""

import argparse
from pathlib import Path

from src.reporting.eda_report import EDAReport

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="EDA de regresión sobre el conjunto de train")
    parser.add_argument("--train-file", type=Path, default=PROCESSED / "train.csv")
    parser.add_argument("--reviews-file", type=Path,
                        default=PROCESSED / "steam_game_reviews_clean.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "eda" / "games")
    args = parser.parse_args()

    EDAReport().generate(
        args.train_file.resolve(), args.output_dir.resolve(),
        reviews_file=args.reviews_file.resolve(),
    )


if __name__ == "__main__":
    main()