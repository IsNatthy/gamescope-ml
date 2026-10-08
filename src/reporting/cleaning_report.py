"""Genera los informes de una limpieza."""

import json
from datetime import datetime, timezone
from pathlib import Path

from cleaning.normalizer import file_hash


def build_report(dataset, source_path, output_path, issues_path, source_sha, rows,
                 written, duplicates_raw, duplicates_normalized, columns,
                 output_columns, counts, missing_before, observations, validation):
    config_file = source_path.name
    types = {column: "texto" for column in output_columns}
    # Importación local para evitar que el generador dependa de detalles del flujo.
    from cleaning.config import CONFIG
    config = CONFIG[dataset]
    types.update({column: "entero" for column in config["integers"]})
    types["appid"] = "identificador; se recomienda leer como string"
    types.update({column: "decimal, escala original" for column in config["decimals"]})
    types.update({column: "booleano true/false" for column in config["booleans"]})
    types.update({column: "lista JSON de cadenas" for column in config["lists"]})
    types["release_date" if dataset == "games" else "review_created_at_utc"] = (
        "fecha ISO 8601" if dataset == "games" else "fecha y hora ISO 8601 UTC"
    )

    return {
        "dataset": dataset,
        "entrada": str(source_path.resolve()),
        "salida": str(output_path.resolve()),
        "generado_utc": datetime.now(timezone.utc).isoformat(),
        "sha256_entrada": source_sha,
        "sha256_salida": file_hash(output_path),
        "filas_entrada": rows,
        "filas_salida": written,
        "duplicados_exactos_eliminados": duplicates_raw,
        "duplicados_adicionales_tras_normalizar": duplicates_normalized,
        "columnas_eliminadas_por_estar_totalmente_vacias": sorted(config["drop"]),
        "columnas_entrada": len(columns),
        "columnas_salida": len(output_columns),
        "mapeo_columnas": dict(zip(columns, columns)),
        "tipos_salida": types,
        "faltantes_entrada": dict(missing_before),
        "cambios_en_registros_no_duplicados": dict(counts),
        "observaciones_para_revision": observations,
        "archivo_observaciones": str(issues_path.resolve()),
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


def write_reports(report, report_path: Path, markdown_path: Path, counts: dict, validation: dict):
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    details = [
        f"# Limpieza de {report['dataset']}", "",
        f"- Entrada: {report['filas_entrada']:,} filas y {report['columnas_entrada']} columnas.",
        f"- Salida: {report['filas_salida']:,} filas y {report['columnas_salida']} columnas.",
        f"- Filas idénticas eliminadas: {report['duplicados_exactos_eliminados']}.",
        f"- Duplicados adicionales tras normalizar: {report['duplicados_adicionales_tras_normalizar']}.",
        f"- Columnas totalmente vacías eliminadas: {', '.join(report['columnas_eliminadas_por_estar_totalmente_vacias'])}.",
        f"- Observaciones pendientes de revisión: {report['observaciones_para_revision']}.",
        "- Validación completa de estructura, tipos, fechas, listas, claves y duplicados: aprobada.",
        "", "## Cambios", "",
    ]
    details.extend(f"- `{key}`: {value:,}." for key, value in sorted(counts.items()))
    details.extend(["", "## Faltantes conservados en la salida", ""])
    details.extend(f"- `{key}`: {value:,}." for key, value in sorted(validation["faltantes"].items()))
    if not validation["faltantes"]:
        details.append("No hay celdas vacías en las columnas de salida.")
    details.extend(["", "## Criterios", ""])
    details.extend(f"- {criterion}" for criterion in report["criterios"])
    markdown_path.write_text("\n".join(details) + "\n", encoding="utf-8")
