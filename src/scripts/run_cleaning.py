"""Punto de entrada para ejecutar la limpieza desde la carpeta src."""

import argparse
from pathlib import Path

from cleaning.config import CONFIG
from cleaning.dataset_cleaner import DatasetCleaner

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Limpieza independiente de juegos y reseñas")
    parser.add_argument("--dataset", choices=["games", "reviews", "all"], default="all")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "cleaning")
    args = parser.parse_args()

    cleaner = DatasetCleaner()
    datasets = list(CONFIG) if args.dataset == "all" else [args.dataset]
    for dataset in datasets:
        cleaner.clean(dataset, args.input_dir.resolve(), args.output_dir.resolve(), args.report_dir.resolve())


if __name__ == "__main__":
    main()
