from pathlib import Path

import pandas as pd
from ydata_profiling import ProfileReport


class ReportGenerator:
    """Genera reportes de perfilamiento para datasets."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        dataframe: pd.DataFrame,
        dataset_name: str
    ) -> Path:
        """Genera y guarda un reporte HTML del dataset."""

        report_path = self.output_dir / f"reporte_{dataset_name}.html"

        profile = ProfileReport(
            dataframe,
            title=f"Reporte de perfilamiento - {dataset_name}",
            explorative=True,
            minimal=True
        )

        profile.to_file(str(report_path))

        print(f"Reporte generado: {report_path}")

        return report_path

