"""Construye el dataset de modelado a partir del dataset de juegos limpio."""

import ast
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cleaning.normalizer import file_hash, snake_case

# exito = 1 si el límite inferior de estimated_owners es >= este umbral.
OWNERS_THRESHOLD = 20_000
TOP_GENRES = 30
TOP_CATEGORIES = 20
TOP_TAGS = 50
# Categorías que Steam asigna tras el lanzamiento: se excluyen por data leakage.
EXCLUDED_CATEGORIES = {"Steam Trading Cards", "Family Sharing", "Remote Play Together"}
MIN_ROWS = 35_000  # mínimo exigido por la rúbrica
RANDOM_STATE = 42

# Solo se leen estas columnas: lo conocido ANTES del lanzamiento.
USE_COLUMNS = [
    "appid", "name", "release_date", "estimated_owners", "price",
    "required_age", "windows", "mac", "linux", "supported_languages",
    "full_audio_languages", "categories", "genres", "tags",
]

# Quedan fuera por data leakage (miden lo ocurrido tras el lanzamiento).
LEAKAGE_COLUMNS = [
    "peak_ccu", "positive", "negative", "user_score", "recommendations",
    "metacritic_score", "score_rank", "average_playtime_forever",
    "average_playtime_two_weeks", "median_playtime_forever",
    "median_playtime_two_weeks", "discount", "dlc_count",
]

TRACE_COLUMNS = ["appid", "name"]
TARGET = "exito"


def parse_list(value) -> list[str]:
    """Convierte 'A,B,C' (o una lista/diccionario en texto) en lista de nombres."""
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, dict)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def multi_hot(lists: pd.Series, size: int, prefix: str):
    """Crea una columna 0/1 por cada uno de los `size` nombres más frecuentes."""
    counts = Counter(name for items in lists for name in set(items))
    names = [name for name, _ in counts.most_common(size)]
    sets = lists.map(set)
    columns = {}
    for name in names:
        column = f"{prefix}_{snake_case(name)}"
        if column == f"{prefix}_" or column in columns:
            raise ValueError(f"Nombre de columna vacío o repetido: {column!r}")
        columns[column] = sets.map(lambda items, n=name: int(n in items)).astype("int8")
    return pd.DataFrame(columns, index=lists.index), names


class DatasetBuilder:
    """Genera model_dataset.csv: features pre-lanzamiento + target balanceado."""

    def build(self, input_file: Path, output_file: Path, report_file: Path,
              overwrite: bool = False) -> dict:
        for target in (output_file, report_file):
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"La salida ya existe; usa --overwrite o elige otra ruta: {target}"
                )

        print(f"Leyendo {input_file.name}...", flush=True)
        df = pd.read_csv(
            input_file, usecols=USE_COLUMNS, dtype=str, keep_default_na=False
        )
        rows_input = len(df)

        # --- Target: límite inferior del rango de dueños estimados ---
        owners = df["estimated_owners"].str.extract(r"^\s*(\d+)\s*-")[0]
        df = df[owners.notna()].copy()
        owners = owners[owners.notna()].astype("int64")
        rows_without_target = rows_input - len(df)
        df[TARGET] = (owners >= OWNERS_THRESHOLD).astype("int8")

        # --- Features estructuradas ---
        genres = df["genres"].map(parse_list)
        categories = df["categories"].map(parse_list).map(
            lambda items: [i for i in items if i not in EXCLUDED_CATEGORIES]
        )
        tags = df["tags"].map(parse_list)
        languages = df["supported_languages"].map(parse_list)
        audio = df["full_audio_languages"].map(parse_list)
        release = pd.to_datetime(df["release_date"], format="%Y-%m-%d")

        # Los faltantes numéricos se dejan como NaN: la imputación se aprende
        # solo con train, en el preprocesamiento (paso 5).
        data = pd.DataFrame({
            "appid": df["appid"],
            "name": df["name"],
            "price": pd.to_numeric(df["price"], errors="coerce"),
            "required_age": pd.to_numeric(df["required_age"], errors="coerce"),
            "windows": df["windows"].eq("true").astype("int8"),
            "mac": df["mac"].eq("true").astype("int8"),
            "linux": df["linux"].eq("true").astype("int8"),
            "n_languages": languages.map(len),
            "n_audio_languages": audio.map(len),
            "release_year": release.dt.year,
            "release_month": release.dt.month,
            "n_genres": genres.map(len),
            "n_categories": categories.map(len),
            "n_tags": tags.map(len),
        })

        print("Generando variables multi-hot...", flush=True)
        genre_cols, genre_names = multi_hot(genres, TOP_GENRES, "genre")
        category_cols, category_names = multi_hot(categories, TOP_CATEGORIES, "cat")
        tag_cols, tag_names = multi_hot(tags, TOP_TAGS, "tag")
        data = pd.concat([data, genre_cols, category_cols, tag_cols], axis=1)
        data[TARGET] = df[TARGET]

        # --- Balanceo por submuestreo de la clase mayoritaria ---
        # Solo descarta filas (no aprende nada), por eso puede ir antes del split.
        distribution_before = data[TARGET].value_counts().to_dict()
        positives = data[data[TARGET] == 1]
        negatives = data[data[TARGET] == 0]
        if len(negatives) < len(positives):
            raise ValueError("La clase 0 debería ser la mayoritaria")
        balanced = pd.concat(
            [positives, negatives.sample(n=len(positives), random_state=RANDOM_STATE)]
        ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        balanced.to_csv(output_file, index=False, encoding="utf-8", lineterminator="\n")

        feature_columns = [
            c for c in balanced.columns if c not in TRACE_COLUMNS + [TARGET]
        ]
        report = {
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "entrada": str(input_file.resolve()),
            "sha256_entrada": file_hash(input_file),
            "salida": str(output_file.resolve()),
            "sha256_salida": file_hash(output_file),
            "filas_entrada": rows_input,
            "filas_sin_target_descartadas": rows_without_target,
            "target": f"{TARGET} = 1 si el límite inferior de estimated_owners >= {OWNERS_THRESHOLD}",
            "distribucion_antes_de_balancear": {str(k): int(v) for k, v in distribution_before.items()},
            "balanceo": "submuestreo aleatorio de la clase 0 (random_state=42)",
            "filas_salida": len(balanced),
            "distribucion_salida": {str(k): int(v) for k, v in balanced[TARGET].value_counts().items()},
            "cumple_minimo_35000": len(balanced) > MIN_ROWS,
            "columnas_trazabilidad": TRACE_COLUMNS,
            "columnas_features": feature_columns,
            "columnas_excluidas_por_leakage": LEAKAGE_COLUMNS,
            "genres_incluidos": genre_names,
            "categories_incluidas": category_names,
            "tags_incluidos": tag_names,
            "faltantes_en_salida": {
                c: int(v) for c, v in balanced.isna().sum().items() if v
            },
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        print(f"Listo: {output_file} ({len(balanced):,} filas, {len(feature_columns)} features)")
        if not report["cumple_minimo_35000"]:
            print(f"ADVERTENCIA: se esperaban más de {MIN_ROWS:,} filas.")
        return report