
"""Coordina la limpieza y el profiling de los datasets."""

from pathlib import Path

from cleaning.config import CONFIG
from cleaning.dataset_cleaner import DatasetCleaner


class Pipeline:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        cleaning_report_dir: Path,
        profiler,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.cleaning_report_dir = cleaning_report_dir
        self.cleaner = DatasetCleaner()
        self.profiler = profiler

    def run(self, dataset="all"):
        """Limpia los datasets y genera sus perfiles."""

        datasets = list(CONFIG) if dataset == "all" else [dataset]

        if any(name not in CONFIG for name in datasets):
            raise ValueError(
                f"Dataset no válido: {dataset}. "
                "Usa 'games', 'reviews' o 'all'."
            )

        results = {}

        for name in datasets:
            self.cleaner.clean(
                dataset=name,
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                report_dir=self.cleaning_report_dir,
            )

            cleaned_file = (
                self.output_dir / f"{Path(CONFIG[name]['file']).stem}_clean.csv"
            )

            results[name] = cleaned_file

            if self.profiler is not None:
                self.profiler.generate_report(
                    cleaned_file,
                    name,
                )

        return results