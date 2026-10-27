"""Divide reseñas en train/validation/test, estratificado y agrupado por appid."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.cleaning.normalizer import file_hash

TARGET = "voted_up"
RANDOM_STATE = 42


class ReviewsSplitter:
    """Genera particiones 80/10/10 sin compartir juegos entre particiones."""

    def split(self, input_file: Path, train_file: Path, validation_file: Path,
              test_file: Path, report_file: Path, overwrite: bool = False) -> dict:
        outputs = [train_file, validation_file, test_file, report_file]
        for target in outputs:
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"La salida ya existe; elige otro directorio o usa --overwrite: {target}"
                )

        print(f"Leyendo {input_file.name}...", flush=True)
        df = pd.read_csv(
            input_file, dtype={"appid": str}, na_values=[""], keep_default_na=False
        )
        if TARGET not in df.columns or "appid" not in df.columns:
            raise ValueError("La entrada necesita las columnas appid y voted_up")

        # Primer fold: aproximadamente 10% para test.
        outer = StratifiedGroupKFold(
            n_splits=10, shuffle=True, random_state=RANDOM_STATE
        )
        train_validation_idx, test_idx = next(
            outer.split(df, df[TARGET], groups=df["appid"])
        )
        train_validation = df.iloc[train_validation_idx]
        test = df.iloc[test_idx]

        # Un fold de 9 sobre el 90% restante produce aproximadamente 10% global.
        inner = StratifiedGroupKFold(
            n_splits=9, shuffle=True, random_state=RANDOM_STATE + 1
        )
        train_idx, validation_idx = next(
            inner.split(
                train_validation,
                train_validation[TARGET],
                groups=train_validation["appid"],
            )
        )
        train = train_validation.iloc[train_idx]
        validation = train_validation.iloc[validation_idx]

        partitions = {"train": train, "validation": validation, "test": test}
        appids = [set(part["appid"]) for part in partitions.values()]
        if appids[0] & appids[1] or appids[0] & appids[2] or appids[1] & appids[2]:
            raise ValueError("Hay appid compartidos entre particiones")

        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        train.reset_index(drop=True).to_csv(train_file, index=False, encoding="utf-8", lineterminator="\n")
        validation.reset_index(drop=True).to_csv(validation_file, index=False, encoding="utf-8", lineterminator="\n")
        test.reset_index(drop=True).to_csv(test_file, index=False, encoding="utf-8", lineterminator="\n")

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
            "metodo": "StratifiedGroupKFold 80/10/10, agrupado por appid, random_state=42",
            "filas_entrada": len(df),
            "train": describe(train, train_file),
            "validation": describe(validation, validation_file),
            "test": describe(test, test_file),
            "appid_compartidos": 0,
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Listo: train={len(train):,}, validation={len(validation):,}, "
            f"test={len(test):,} filas"
        )
        return report
