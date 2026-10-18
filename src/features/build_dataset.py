"""Construye el dataset de modelado a partir del dataset de juegos limpio."""

import ast
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cleaning.normalizer import file_hash, snake_case
from src.features.split_config import RANDOM_STATE, TEST_SIZE, split_indices

# La regla se aplica después, sobre la predicción continua.
SUCCESS_THRESHOLD = 20_000
TOP_GENRES = 30
TOP_CATEGORIES = 20
TOP_TAGS = 50
# Categorías que Steam asigna tras el lanzamiento: se excluyen por data leakage.
EXCLUDED_CATEGORIES = {"Steam Trading Cards", "Family Sharing", "Remote Play Together"}
MIN_ROWS = 35_000  # mínimo exigido por la rúbrica

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
TARGET = "owners_midpoint"


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


def multi_hot(lists: pd.Series, fit_lists: pd.Series, size: int, prefix: str):
    """Codifica los nombres más frecuentes en train sobre todas las filas."""
    counts = Counter(name for items in fit_lists for name in set(items))
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
    """Genera features pre-lanzamiento y un target numérico para regresión."""

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

        # Steam proporciona intervalos; el punto medio es una estimación más
        # representativa que usar siempre el límite inferior.
        owner_bounds = df["estimated_owners"].str.extract(
            r"^\s*(\d+)\s*-\s*(\d+)\s*$"
        ).apply(pd.to_numeric, errors="coerce")
        valid_bounds = (
            owner_bounds.notna().all(axis=1)
            & owner_bounds[0].le(owner_bounds[1])
        )
        df = df.loc[valid_bounds].copy()
        owner_bounds = owner_bounds.loc[valid_bounds]
        rows_without_target = rows_input - len(df)
        df[TARGET] = (owner_bounds[0] + owner_bounds[1]) / 2

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
        train_indices, _ = split_indices(len(df))
        train_index = df.iloc[train_indices].index
        genre_cols, genre_names = multi_hot(
            genres, genres.loc[train_index], TOP_GENRES, "genre"
        )
        category_cols, category_names = multi_hot(
            categories, categories.loc[train_index], TOP_CATEGORIES, "cat"
        )
        tag_cols, tag_names = multi_hot(
            tags, tags.loc[train_index], TOP_TAGS, "tag"
        )
        data = pd.concat([data, genre_cols, category_cols, tag_cols], axis=1)
        data[TARGET] = df[TARGET]

        output_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_file, index=False, encoding="utf-8", lineterminator="\n")

        feature_columns = [
            c for c in data.columns if c not in TRACE_COLUMNS + [TARGET]
        ]
        target_statistics = data[TARGET].describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]
        )
        report = {
            "generado_utc": datetime.now(timezone.utc).isoformat(),
            "entrada": str(input_file.resolve()),
            "sha256_entrada": file_hash(input_file),
            "salida": str(output_file.resolve()),
            "sha256_salida": file_hash(output_file),
            "filas_entrada": rows_input,
            "filas_sin_target_descartadas": rows_without_target,
            "target": (
                f"{TARGET} = (límite inferior + límite superior) / 2 "
                "del intervalo estimated_owners"
            ),
            "vocabulario_multi_hot": (
                f"frecuencias calculadas solo con train (test_size={TEST_SIZE}, "
                f"random_state={RANDOM_STATE}); las columnas se aplican a todas "
                "las filas"
            ),
            "regla_exito_posterior": (
                f"exitoso si la predicción >= {SUCCESS_THRESHOLD}; "
                "esta regla no se usa para construir el target"
            ),
            "balanceo": "ninguno; se conservan todas las filas válidas",
            "filas_salida": len(data),
            "estadisticas_target": {
                str(k): float(v) if k != "count" else int(v)
                for k, v in target_statistics.items()
            },
            "cumple_minimo_35000": len(data) > MIN_ROWS,
            "columnas_trazabilidad": TRACE_COLUMNS,
            "columnas_features": feature_columns,
            "columnas_excluidas_por_leakage": LEAKAGE_COLUMNS,
            "genres_incluidos": genre_names,
            "categories_incluidas": category_names,
            "tags_incluidos": tag_names,
            "faltantes_en_salida": {
                c: int(v) for c, v in data.isna().sum().items() if v
            },
        }
        report_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        print(f"Listo: {output_file} ({len(data):,} filas, {len(feature_columns)} features)")
        if not report["cumple_minimo_35000"]:
            print(f"ADVERTENCIA: se esperaban más de {MIN_ROWS:,} filas.")
        return report