"""Limpieza independiente de los CSV de juegos y reseñas, sin dependencias externas."""

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
csv.field_size_limit(10_000_000)
MONTHS = dict(zip("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), range(1, 13)))
KICHE = re.compile(r"(?<=[,\[])\s*K'iche'(?=\s*[,\]])")
CONFIG = {
    "games": {
        "file": "games_fixed.csv",
        "drop": {"movies"},
        "integers": set("appid peak_ccu required_age discount dlc_count metacritic_score positive negative score_rank achievements recommendations average_playtime_forever average_playtime_two_weeks median_playtime_forever median_playtime_two_weeks".split()),
        "decimals": {"price", "user_score"},
        "booleans": {"windows", "mac", "linux"},
        "lists": {"supported_languages", "full_audio_languages"},
    },
    "reviews": {
        "file": "steam_game_reviews.csv",
        "drop": {"release_date"},
        "integers": set("appid word_count votes_up votes_funny timestamp_created author_playtime_forever".split()),
        "decimals": {"price"},
        "booleans": {"voted_up"},
        "lists": set(),
    },
}


def snake_case(name):
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lstrip("\ufeff").lower()).strip("_")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_hash(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False).encode("utf-8")).digest()


def normalize_number(value, integer=False):
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("Número no finito")
    if integer:
        if number != number.to_integral_value():
            raise ValueError("Se esperaba un entero")
        return str(int(number))
    return format(number.normalize(), "f")


def normalize_date(value):
    match = re.fullmatch(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})", value)
    if not match or match[1] not in MONTHS:
        raise ValueError("Fecha de lanzamiento no reconocida")
    return datetime(int(match[3]), MONTHS[match[1]], int(match[2])).date().isoformat()


def normalize_languages(value):
    repaired = False
    try:
        languages = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Reparación acotada al defecto observado; nunca se evalúa código.
        fixed = KICHE.sub(' "K\'iche"', value)
        languages = ast.literal_eval(fixed)
        repaired = fixed != value
    if not isinstance(languages, list) or not all(isinstance(item, str) for item in languages):
        raise ValueError("Se esperaba una lista de idiomas")
    return json.dumps(languages, ensure_ascii=False), repaired


def clean_row(raw, dataset, counts):
    config = CONFIG[dataset]
    cleaned = {}
    warnings = []
    for column, original in raw.items():
        value = original.strip()
        if value != original:
            counts[f"espacios_exteriores:{column}"] += 1
        if column in config["drop"]:
            if value:
                raise ValueError(f"La columna {column} ya contiene datos; no se puede eliminar")
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


def validate_output(path, dataset, expected_rows, expected_columns):
    """Relee el archivo exportado y comprueba estructura, tipos, claves y duplicados."""
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
    return {"filas": count, "appid_distintos": len(ids), "faltantes": dict(missing), "duplicados_restantes": 0, "tipos_y_estructura_validos": True}


def clean_dataset(dataset, input_dir, output_dir, report_dir):
    config = CONFIG[dataset]
    source_path = input_dir / config["file"]
    stem = source_path.stem + "_clean"
    output_path = output_dir / (stem + ".csv")
    report_path = report_dir / (stem + ".json")
    markdown_path = report_dir / (stem + ".md")
    issues_path = report_dir / (stem + "_observaciones.csv")
    if source_path.resolve() == output_path.resolve():
        raise ValueError("La salida no puede reemplazar el archivo original")
    for target in [output_path, report_path, markdown_path, issues_path]:
        if target.exists():
            raise FileExistsError(f"La salida ya existe; elige otro directorio: {target}")
    source_sha = file_hash(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    staging = output_path.with_suffix(".csv.tmp")
    counts = Counter()
    missing_before = Counter()
    seen_raw, seen_clean = set(), set()
    rows = written = duplicates_raw = duplicates_normalized = observations = 0
    print(f"Limpiando {source_path.name} por separado...", flush=True)
    with source_path.open(encoding="utf-8-sig", newline="") as source, staging.open("x", encoding="utf-8", newline="") as target, issues_path.open("x", encoding="utf-8", newline="") as issues_file:
        reader = csv.reader(source, strict=True)
        original_columns = next(reader)
        columns = [snake_case(column) for column in original_columns]
        if len(set(columns)) != len(columns):
            raise ValueError("Hay encabezados repetidos tras normalizarlos")
        required = config["integers"] | config["decimals"] | config["booleans"] | config["lists"] | config["drop"] | {"appid", "name"}
        required |= {"release_date", "estimated_owners"} if dataset == "games" else {"review"}
        if not required.issubset(columns):
            raise ValueError(f"Faltan columnas: {sorted(required - set(columns))}")
        output_columns = [column for column in columns if column not in config["drop"]]
        if dataset == "reviews":
            if "review_created_at_utc" in output_columns:
                raise ValueError("La entrada ya tiene review_created_at_utc")
            output_columns.append("review_created_at_utc")
        writer = csv.DictWriter(target, fieldnames=output_columns, lineterminator="\n")
        writer.writeheader()
        issues_writer = csv.writer(issues_file, lineterminator="\n")
        issues_writer.writerow(["registro_origen", "appid", "columna", "observacion", "detalle"])
        for row_number, values in enumerate(reader, start=2):
            rows += 1
            if len(values) != len(columns):
                raise ValueError(f"Registro {row_number}: número de columnas incorrecto")
            raw = dict(zip(columns, values))
            missing_before.update(column for column, value in raw.items() if not value.strip())
            fingerprint = row_hash(values)
            if fingerprint in seen_raw:
                duplicates_raw += 1
                continue
            seen_raw.add(fingerprint)
            try:
                cleaned, warnings = clean_row(raw, dataset, counts)
            except (ValueError, SyntaxError, InvalidOperation, OverflowError) as error:
                raise ValueError(f"Registro {row_number}, appid={raw.get('appid')}: {error}") from error
            fingerprint = row_hash(list(cleaned.values()))
            if fingerprint in seen_clean:
                duplicates_normalized += 1
                continue
            seen_clean.add(fingerprint)
            writer.writerow(cleaned)
            written += 1
            for column, code, detail in warnings:
                issues_writer.writerow([row_number, cleaned["appid"], column, code, detail])
                observations += 1
            if rows % 100_000 == 0:
                print(f"  {rows:,} registros leídos", flush=True)
    # Libera las huellas antes de la relectura completa de validación.
    del seen_raw, seen_clean
    if written + duplicates_raw + duplicates_normalized != rows:
        raise ValueError("No concuerdan las filas de entrada, salida y duplicados")
    print(f"Validando {output_path.name}...", flush=True)
    validation = validate_output(staging, dataset, written, output_columns)
    if file_hash(source_path) != source_sha:
        raise ValueError("El archivo de entrada cambió durante la ejecución")
    types = {column: "texto" for column in output_columns}
    types.update({column: "entero" for column in config["integers"]})
    types["appid"] = "identificador; se recomienda leer como string"
    types.update({column: "decimal, escala original" for column in config["decimals"]})
    types.update({column: "booleano true/false" for column in config["booleans"]})
    types.update({column: "lista JSON de cadenas" for column in config["lists"]})
    types["release_date" if dataset == "games" else "review_created_at_utc"] = "fecha ISO 8601" if dataset == "games" else "fecha y hora ISO 8601 UTC"
    report = {
        "dataset": dataset, "entrada": str(source_path.resolve()), "salida": str(output_path.resolve()),
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_entrada": source_sha, "sha256_salida": file_hash(staging),
        "filas_entrada": rows, "filas_salida": written,
        "duplicados_exactos_eliminados": duplicates_raw,
        "duplicados_adicionales_tras_normalizar": duplicates_normalized,
        "columnas_eliminadas_por_estar_totalmente_vacias": sorted(config["drop"]),
        "columnas_entrada": len(columns), "columnas_salida": len(output_columns),
        "mapeo_columnas": dict(zip(original_columns, columns)), "tipos_salida": types,
        "faltantes_entrada": dict(missing_before), "cambios_en_registros_no_duplicados": dict(counts),
        "observaciones_para_revision": observations, "archivo_observaciones": str(issues_path.resolve()),
        "validacion": validation,
        "criterios": [
            "Cada dataset se procesa de forma independiente. No se cruzan ni unen datos.",
            "Originales conservados; entrada UTF-8 con BOM opcional, salida UTF-8 sin BOM.",
            "Se eliminan filas idénticas completas; no se deduplica por appid en reseñas.",
            "Se conservan faltantes parciales como celdas vacías, sin imputar ni inventar valores.",
            "Se conservan ceros y valores extremos; los negativos o fuera de rango se registran para revisión.",
            "Los precios conservan su escala original; no se presupone moneda ni se divide entre 100.",
            "Los textos conservan mayúsculas, acentos, puntuación y espacios internos; solo se recortan extremos.",
            "Las listas de idiomas pasan a JSON; únicamente se reparan las comillas defectuosas de K'iche.",
            "Las reseñas conservan timestamp_created y añaden review_created_at_utc; word_count no se recalcula.",
        ],
    }
    staging.replace(output_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    details = [f"# Limpieza de {config['file']}", "", f"- Entrada: {rows:,} filas y {len(columns)} columnas.", f"- Salida: {written:,} filas y {len(output_columns)} columnas.", f"- Filas idénticas eliminadas: {duplicates_raw}.", f"- Duplicados adicionales tras normalizar: {duplicates_normalized}.", f"- Columnas totalmente vacías eliminadas: {', '.join(sorted(config['drop']))}.", f"- Observaciones pendientes de revisión: {observations}.", f"- Validación completa de estructura, tipos, fechas, listas, claves y duplicados: aprobada.", "", "## Cambios", ""]
    details.extend(f"- `{key}`: {value:,}." for key, value in sorted(counts.items()))
    details.extend(["", "## Faltantes conservados en la salida", ""])
    details.extend(f"- `{key}`: {value:,}." for key, value in sorted(validation["faltantes"].items()))
    if not validation["faltantes"]:
        details.append("No hay celdas vacías en las columnas de salida.")
    details.extend(["", "## Criterios", ""])
    details.extend(f"- {criterion}" for criterion in report["criterios"])
    markdown_path.write_text("\n".join(details) + "\n", encoding="utf-8")
    print(f"Listo: {output_path} ({written:,} filas)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["games", "reviews", "all"], default="all")
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "data" / "processed")
    parser.add_argument("--report-dir", type=Path, default=PROJECT / "reports" / "cleaning")
    args = parser.parse_args()
    for dataset in CONFIG if args.dataset == "all" else [args.dataset]:
        clean_dataset(dataset, args.input_dir.resolve(), args.output_dir.resolve(), args.report_dir.resolve())


if __name__ == "__main__":
    main()
