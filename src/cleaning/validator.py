"""Validación estructural y de contenido del CSV exportado."""

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from cleaning.config import CONFIG
from cleaning.normalizer import normalize_number, row_hash


def validate_output(path: Path, dataset: str, expected_rows: int, expected_columns: list) -> dict:
    config = CONFIG[dataset]
    fingerprints = set()
    ids = set()
    count = 0
    missing = Counter()

    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != expected_columns:
            raise ValueError("Encabezado exportado incorrecto")

        for row in reader:
            count += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Fila exportada mal formada: {count}")

            fingerprint = row_hash(list(row.values()))
            if fingerprint in fingerprints:
                raise ValueError(f"Duplicado restante en la fila {count}")
            fingerprints.add(fingerprint)

            if dataset == "games" and row["appid"] in ids:
                raise ValueError("appid duplicado en el catálogo; requiere revisión")
            ids.add(row["appid"])

            for column, value in row.items():
                if not value:
                    missing[column] += 1
                    continue
                if value != value.strip():
                    raise ValueError(f"Espacios exteriores restantes: {column}")
                if column in config["integers"] | config["decimals"]:
                    if value != normalize_number(value, column in config["integers"]):
                        raise ValueError(f"Número no normalizado: {column}")
                if column in config["booleans"] and value not in {"true", "false"}:
                    raise ValueError(f"Booleano inválido: {column}")
                if column in config["lists"]:
                    languages = json.loads(value)
                    if not isinstance(languages, list) or not all(isinstance(item, str) for item in languages):
                        raise ValueError(f"Lista inválida: {column}")

            if dataset == "games":
                datetime.strptime(row["release_date"], "%Y-%m-%d")
            elif row["timestamp_created"]:
                expected = datetime.fromtimestamp(int(row["timestamp_created"]), timezone.utc)
                actual = datetime.fromisoformat(row["review_created_at_utc"].replace("Z", "+00:00"))
                if actual != expected:
                    raise ValueError("La fecha UTC no corresponde al timestamp original")

        if count != expected_rows:
            raise ValueError(f"Conteo incorrecto: {count} != {expected_rows}")

    return {
        "filas": count,
        "appid_distintos": len(ids),
        "faltantes": dict(missing),
        "duplicados_restantes": 0,
        "tipos_y_estructura_validos": True,
    }
