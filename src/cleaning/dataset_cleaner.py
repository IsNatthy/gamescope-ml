"""Coordina la limpieza de un dataset completo."""

import csv
import json
from collections import Counter
from decimal import InvalidOperation
from pathlib import Path

from cleaning.config import CONFIG
from cleaning.normalizer import file_hash, row_hash, snake_case
from cleaning.row_cleaner import clean_row
from cleaning.validator import validate_output
from reporting.cleaning_report import build_report, write_reports


class DatasetCleaner:
    def clean(self, dataset: str, input_dir: Path, output_dir: Path, report_dir: Path) -> dict:
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
        original_columns = []
        columns = []
        output_columns = []

        print(f"Limpiando {source_path.name} por separado...", flush=True)
        try:
            with source_path.open(encoding="utf-8-sig", newline="") as source, \
                 staging.open("x", encoding="utf-8", newline="") as target, \
                 issues_path.open("x", encoding="utf-8", newline="") as issues_file:
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

            del seen_raw, seen_clean
            if written + duplicates_raw + duplicates_normalized != rows:
                raise ValueError("No concuerdan las filas de entrada, salida y duplicados")

            print(f"Validando {output_path.name}...", flush=True)
            validation = validate_output(staging, dataset, written, output_columns)
            if file_hash(source_path) != source_sha:
                raise ValueError("El archivo de entrada cambió durante la ejecución")

            staging.replace(output_path)
            report = build_report(
                dataset, source_path, output_path, issues_path, source_sha, rows,
                written, duplicates_raw, duplicates_normalized, original_columns,
                output_columns, counts, missing_before, observations, validation,
            )
            # El mapeo conserva el encabezado original y su versión normalizada.
            report["mapeo_columnas"] = dict(zip(original_columns, columns))
            write_reports(report, report_path, markdown_path, counts, validation)
            print(f"Listo: {output_path} ({written:,} filas)", flush=True)
            return report
        except Exception:
            if staging.exists():
                staging.unlink()
            raise
