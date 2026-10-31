"""Divide el dataset de regresión en train y test (80/20, aleatorio)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from src.cleaning.normalizer import file_hash
from src.features.split_config import RANDOM_STATE, TEST_SIZE, split_indices

TARGET = "owners_midpoint"


class DatasetSplitter:
    """Genera train.csv y test.csv antes de cualquier transformación con fit."""

    def split(self, input_file: Path, train_file: Path, test_file: Path,
              report_file: Path, overwrite: bool = False) -> dict:
        for target in (train_file, test_file, report_file):
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"La salida ya existe; usa --overwrite o elige otra ruta: {target}"
                )

        print(f"Leyendo {input_file.name}...", flush=True)
        # Solo las celdas vacías cuentan como faltantes (un nombre "NA" se conserva).
        df = pd.read_csv(
            input_file, dtype={"appid": str}, na_values=[""], keep_default_na=False
        )
        if TARGET not in df.columns:
            raise ValueError(f"Falta la columna objetivo: {TARGET}")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="raise")
        if df[TARGET].isna().any():
            raise ValueError(f"El target {TARGET} contiene valores faltantes")

        train_indices, test_indices = split_indices(len(df))
        train, test = df.iloc[train_indices], df.iloc[test_indices]
        if set(train["appid"]) & set(test["appid"]):
            raise ValueError("Hay appid repetidos entre train y test")

        for path in (train_file, test_file, report_file):
            path.parent.mkdir(parents=True, exist_ok=True)
        train.reset_index(drop=True).to_csv(
            train_file, index=False, encoding="utf-8", lineterminator="\n")
        test.reset_index(drop=True).to_csv(
            test_file, index=False, encoding="utf-8", lineterminator="\n")

        def target_summary(part: pd.DataFrame) -> dict:
            summary = part[TARGET].describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]
            )
            return {
                str(key): float(value) if key != "count" else int(value)
                for key, value in summary.items()
            }

        report = {
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "entrada": str(input_file.resolve()),
            "sha256_entrada": file_hash(input_file),
            "metodo": f"train_test_split, test_size={TEST_SIZE}, "
                      f"random_state={RANDOM_STATE}, sin estratificación",
            "filas_entrada": len(df),
            "train": {"archivo": str(train_file.resolve()),
                      "sha256": file_hash(train_file),
                      "filas": len(train), "estadisticas_target": target_summary(train)},
            "test": {"archivo": str(test_file.resolve()),
                     "sha256": file_hash(test_file),
                     "filas": len(test), "estadisticas_target": target_summary(test)},
            "appid_compartidos_train_test": 0,
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Listo: train={len(train):,} filas, test={len(test):,} filas")
        return report
