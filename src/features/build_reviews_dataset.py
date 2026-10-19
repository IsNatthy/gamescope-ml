"""Construye el dataset de modelado de reseñas: texto depurado + sentimiento balanceado."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cleaning.normalizer import file_hash

TARGET = "voted_up"  # 1 = reseña positiva (recomendada), 0 = negativa
# Proporción mínima de letras A-Z sobre los caracteres no blancos. Descarta arte
# ASCII/braille, alfabetos no latinos, spam de emojis y reseñas solo con números.
MIN_LETTER_RATIO = 0.5
MIN_ROWS = 35_000  # mínimo exigido por la rúbrica
RANDOM_STATE = 42

# appid y word_count son de trazabilidad/EDA; el modelo recibe únicamente `review`.
USE_COLUMNS = ["appid", "review", "word_count", TARGET]
TRACE_COLUMNS = ["appid", "word_count"]
# Quedan fuera: votos y horas jugadas no existen al redactar una reseña nueva
# (data leakage); el resto no aporta al análisis del texto.
EXCLUDED_COLUMNS = [
    "votes_up", "votes_funny", "author_playtime_forever", "timestamp_created",
    "review_created_at_utc", "name", "price",
]


def letter_ratio(text: pd.Series) -> pd.Series:
    """Proporción de letras A-Z entre los caracteres que no son espacios."""
    non_blank = text.str.count(r"\S").clip(lower=1)
    return text.str.count(r"[A-Za-z]") / non_blank


class ReviewsDatasetBuilder:
    """Genera reviews_model_dataset.csv: texto depurado + target balanceado."""

    def build(self, input_file: Path, output_file: Path, report_file: Path,
              overwrite: bool = False) -> dict:
        for target in (output_file, report_file):
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"La salida ya existe; usa --overwrite o elige otra ruta: {target}"
                )

        print(f"Leyendo {input_file.name}...", flush=True)
        df = pd.read_csv(
            input_file, usecols=USE_COLUMNS, dtype={"appid": str},
            na_values=[""], keep_default_na=False,
        )
        df[TARGET] = df[TARGET].astype("int8")
        rows_input = len(df)

        # --- Depuración ---
        # Solo se descartan filas (no se aprende nada), por eso puede ir antes del split.
        # El texto se conserva crudo: BERT y BERTopic necesitan el contexto completo.
        discarded = {}

        before = len(df)
        df = df.dropna(subset=["review"])
        discarded["texto_vacio"] = before - len(df)

        before = len(df)
        df = df[letter_ratio(df["review"]) >= MIN_LETTER_RATIO]
        discarded["texto_ruido_o_alfabeto_no_latino"] = before - len(df)

        before = len(df)
        labels_per_text = df.groupby("review")[TARGET].transform("nunique")
        df = df[labels_per_text == 1]
        discarded["texto_con_etiquetas_contradictorias"] = before - len(df)

        before = len(df)
        df = df.drop_duplicates(subset="review", keep="first")
        discarded["texto_duplicado"] = before - len(df)

        # --- Balanceo por submuestreo de la clase mayoritaria ---
        distribution_before = df[TARGET].value_counts().to_dict()
        per_class = df[TARGET].value_counts().min()
        balanced = (
            df.groupby(TARGET, group_keys=False)
            .sample(n=per_class, random_state=RANDOM_STATE)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)[USE_COLUMNS]
        )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        balanced.to_csv(output_file, index=False, encoding="utf-8", lineterminator="\n")

        report = {
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "entrada": str(input_file.resolve()),
            "sha256_entrada": file_hash(input_file),
            "salida": str(output_file.resolve()),
            "sha256_salida": file_hash(output_file),
            "filas_entrada": rows_input,
            "filas_descartadas_por_regla": discarded,
            "min_letter_ratio": MIN_LETTER_RATIO,
            "target": f"{TARGET} = 1 si la reseña es positiva, 0 si es negativa",
            "distribucion_antes_de_balancear": {
                str(k): int(v) for k, v in distribution_before.items()
            },
            "balanceo": "submuestreo aleatorio de la clase mayoritaria (random_state=42)",
            "filas_salida": len(balanced),
            "distribucion_salida": {
                str(k): int(v) for k, v in balanced[TARGET].value_counts().items()
            },
            "juegos_distintos": int(balanced["appid"].nunique()),
            "cumple_minimo_35000": len(balanced) > MIN_ROWS,
            "columna_texto": "review",
            "columnas_trazabilidad": TRACE_COLUMNS,
            "columnas_excluidas": EXCLUDED_COLUMNS,
            "preprocesamiento_texto": "ninguno: se conserva el texto crudo para BERT/BERTopic",
            "faltantes_en_salida": {
                c: int(v) for c, v in balanced.isna().sum().items() if v
            },
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        print(f"Listo: {output_file} ({len(balanced):,} filas)")
        if not report["cumple_minimo_35000"]:
            print(f"ADVERTENCIA: se esperaban más de {MIN_ROWS:,} filas.")
        return report
