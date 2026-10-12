"""Punto de entrada para construir el dataset de modelado."""

import argparse
from pathlib import Path

from src.features.build_dataset import DatasetBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Construye model_dataset.csv (features + target balanceado)"
    )
    parser.add_argument(
        "--input-file", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "games_fixed_clean.csv",
    )
    parser.add_argument(
        "--output-file", type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "model_dataset.csv",
    )
    parser.add_argument(
        "--report-file", type=Path,
        default=PROJECT_ROOT / "reports" / "features" / "model_dataset.json",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Sobrescribe las salidas si ya existen")
    args = parser.parse_args()

    DatasetBuilder().build(
        args.input_file.resolve(),
        args.output_file.resolve(),
        args.report_file.resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()