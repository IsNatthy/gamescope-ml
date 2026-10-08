from pathlib import Path

import pandas as pd

from reporting.report_generator import ReportGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "profiling"

REVIEWS_FILE = DATA_DIR / "steam_game_reviews.csv"
GAMES_FILE = DATA_DIR / "games_fixed.csv"


def main():
    print("Iniciando generación de reportes...")

    generator = ReportGenerator(REPORT_DIR)

    datasets = {
        "reviews": REVIEWS_FILE,
        "games_fixed": GAMES_FILE,
    }

    for dataset_name, file_path in datasets.items():
        print(f"\nGenerando reporte de {dataset_name}...")

        dataframe = pd.read_csv(
            file_path,
            low_memory=False
        )

        generator.generate_report(
            dataframe,
            dataset_name
        )

    print("\nProceso terminado.")


if __name__ == "__main__":
    main()