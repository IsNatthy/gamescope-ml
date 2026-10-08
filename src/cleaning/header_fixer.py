from pathlib import Path

import pandas as pd


class HeaderFixer:
    """Corrige el encabezado del dataset de juegos."""

    OLD_HEADER = "DiscountDLC count"
    NEW_HEADER = "Discount,DLC count"

    def __init__(self, input_file: Path, output_file: Path):
        self.input_file = input_file
        self.output_file = output_file

    def fix_header(self) -> Path:
        """Corrige el encabezado y guarda una copia del CSV."""

        print("Iniciando limpieza de games.csv...")

        with self.input_file.open(
            "r", encoding="utf-8", newline=""
        ) as source:
            header = source.readline()
            remaining_content = source.read()

        if self.OLD_HEADER not in header:
            raise ValueError(
                f"No se encontró el encabezado que necesita corrección: "
                f"{self.OLD_HEADER}"
            )

        header_fixed = header.replace(
            self.OLD_HEADER,
            self.NEW_HEADER,
            1
        )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with self.output_file.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            target.write(header_fixed)
            target.write(remaining_content)

        print(f"Archivo corregido: {self.output_file}")

        return self.output_file

    def show_summary(self) -> None:
        """Muestra información básica del archivo corregido."""

        dataframe = pd.read_csv(
            self.output_file,
            low_memory=False
        )

        print(f"\nFilas: {dataframe.shape[0]}")
        print(f"Columnas: {dataframe.shape[1]}")

        print("\nColumnas del dataset corregido:")
        print(dataframe.columns.tolist())

        print("\nSample del dataset corregido:")
        print(dataframe.head())

        print("\nLimpieza finalizada.")

