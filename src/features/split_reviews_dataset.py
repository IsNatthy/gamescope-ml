"""Divide el dataset de reseñas en train y test (80/20, estratificado y por juego)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.cleaning.normalizer import file_hash

TARGET = "voted_up"
N_SPLITS = 5  # un fold de 5 = 20 % para test
RANDOM_STATE = 42


class ReviewsSplitter:
    """Genera reviews_train.csv y reviews_test.csv antes de cualquier transformación con fit.

    Cada juego tiene ~1000 reseñas; si se dividiera al azar, un mismo juego quedaría
    en train y test y el modelo podría aprender vocabulario propio del juego.
    Por eso el split es estratificado por target y agrupado por appid.
    """

    def split(self, input_file: Path, train_file: Path, test_file: Path,
              report_file: Path, overwrite: bool = False) -> dict:
        for target in (train_file, test_file, report_file):
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"La salida ya existe; usa --overwrite o elige otra ruta: {target}"
                )

        print(f"Leyendo {input_file.name}...", flush=True)
        df = pd.read_csv(
            input_file, dtype={"appid": str}, na_values=[""], keep_default_na=False
        )
        if TARGET not in df.columns:
            raise ValueError(f"Falta la columna objetivo: {TARGET}")

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df["appid"]))
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        if set(train["appid"]) & set(test["appid"]):
            raise ValueError("Hay appid repetidos entre train y test")

        for path in (train_file, test_file, report_file):
            path.parent.mkdir(parents=True, exist_ok=True)
        train.reset_index(drop=True).to_csv(
            train_file, index=False, encoding="utf-8", lineterminator="\n")
        test.reset_index(drop=True).to_csv(
            test_file, index=False, encoding="utf-8", lineterminator="\n")

        def describe(part: pd.DataFrame, path: Path) -> dict:
            return {
                "archivo": str(path.resolve()),
                "sha256": file_hash(path),
                "filas": len(part),
                "proporcion": round(len(part) / len(df), 4),
                "juegos": int(part["appid"].nunique()),
                "distribucion": {
                    str(k): int(v) for k, v in part[TARGET].value_counts().items()
                },
            }

        report = {
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "entrada": str(input_file.resolve()),
            "sha256_entrada": file_hash(input_file),
            "metodo": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, "
                      f"random_state={RANDOM_STATE}), primer fold como test; "
                      f"estratificado por {TARGET}, agrupado por appid",
            "filas_entrada": len(df),
            "train": describe(train, train_file),
            "test": describe(test, test_file),
            "appid_compartidos_train_test": 0,
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Listo: train={len(train):,} filas, test={len(test):,} filas")
        return report
