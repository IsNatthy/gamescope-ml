"""Funciones reutilizables para normalizar valores de los CSV."""

import ast
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from cleaning.config import KICHE, MONTHS


def snake_case(name: str) -> str:
    """Convierte un encabezado a snake_case."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lstrip("\ufeff").lower()).strip("_")


def file_hash(path: Path) -> str:
    """Calcula el SHA-256 de un archivo sin cargarlo entero en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_hash(row) -> bytes:
    """Calcula una huella estable para detectar filas duplicadas."""
    return hashlib.sha256(
        json.dumps(row, ensure_ascii=False).encode("utf-8")
    ).digest()


def normalize_number(value: str, integer: bool = False) -> str:
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("Número no finito")
    if integer:
        if number != number.to_integral_value():
            raise ValueError("Se esperaba un entero")
        return str(int(number))
    return format(number.normalize(), "f")


def normalize_date(value: str) -> str:
    match = re.fullmatch(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", value)
    if not match or match[1] not in MONTHS:
        raise ValueError("Fecha de lanzamiento no reconocida")
    return datetime(
        int(match[3]), MONTHS[match[1]], int(match[2])
    ).date().isoformat()


def normalize_languages(value: str) -> tuple[str, bool]:
    repaired = False
    try:
        languages = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # No se ejecuta código; solo se intenta reparar el caso conocido.
        fixed = KICHE.sub(' "K\'iche"', value)
        languages = ast.literal_eval(fixed)
        repaired = fixed != value

    if not isinstance(languages, list) or not all(
        isinstance(item, str) for item in languages
    ):
        raise ValueError("Se esperaba una lista de idiomas")

    return json.dumps(languages, ensure_ascii=False), repaired
