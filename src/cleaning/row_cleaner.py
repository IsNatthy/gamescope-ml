"""Limpia y revisa un registro individual."""

from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal

from cleaning.config import CONFIG
from cleaning.normalizer import normalize_date, normalize_languages, normalize_number


def clean_row(raw: dict, dataset: str, counts: Counter) -> tuple[dict, list]:
    config = CONFIG[dataset]
    cleaned = {}
    warnings = []

    for column, original in raw.items():
        value = original.strip()
        if value != original:
            counts[f"espacios_exteriores:{column}"] += 1

        if column in config["drop"]:
            if value:
                raise ValueError(
                    f"La columna {column} ya contiene datos; no se puede eliminar"
                )
            continue

        if value:
            if column in config["integers"] | config["decimals"]:
                value = normalize_number(value, column in config["integers"])
                number = Decimal(value)
                if number < 0:
                    warnings.append((column, "valor_negativo", "Conservado para revisión"))
                if column in {"discount", "metacritic_score", "user_score"} and not 0 <= number <= 100:
                    warnings.append((column, "fuera_de_rango", "Se esperaba un valor entre 0 y 100; conservado"))
            elif column in config["booleans"]:
                if value.lower() not in {"true", "false"}:
                    raise ValueError(f"Booleano no reconocido en {column}: {value!r}")
                value = value.lower()
            elif dataset == "games" and column == "release_date":
                value = normalize_date(value)
            elif column in config["lists"]:
                value, repaired = normalize_languages(value)
                if repaired:
                    counts[f"comillas_idioma_corregidas:{column}"] += 1
            elif column == "estimated_owners":
                import re
                match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", value)
                if not match or int(match[1]) > int(match[2]):
                    warnings.append((column, "intervalo_invalido", "Conservado para revisión"))

        if value != original:
            counts[f"celdas_modificadas:{column}"] += 1
        cleaned[column] = value

    if not cleaned.get("appid") or int(cleaned["appid"]) <= 0:
        raise ValueError("appid vacío o no positivo")
    if not cleaned.get("name"):
        warnings.append(("name", "nombre_faltante", "Fila conservada; no se inventa un nombre"))

    if dataset == "reviews":
        timestamp = cleaned["timestamp_created"]
        cleaned["review_created_at_utc"] = (
            datetime.fromtimestamp(int(timestamp), timezone.utc).isoformat().replace("+00:00", "Z")
            if timestamp else ""
        )
        if not cleaned["review"]:
            warnings.append(("review", "texto_faltante", "Fila conservada para revisión"))
        elif cleaned["word_count"] != str(len(cleaned["review"].split())):
            warnings.append(("word_count", "conteo_diferente", "No coincide con el conteo por espacios; conservado"))

    return cleaned, warnings
