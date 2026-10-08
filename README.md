# Gamescope ML

Proyecto para preparar, limpiar y analizar datasets de Steam para un flujo de Machine Learning.

Actualmente el proyecto está en la fase de limpieza, validación y profiling de los datos antes de continuar con análisis más avanzados.

## Estructura real del proyecto

```text
gamescope-ml/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── cleaning/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── dataset_cleaner.py
│   │   ├── header_fixer.py
│   │   ├── normalizer.py
│   │   ├── row_cleaner.py
│   │   └── validator.py
│   ├── data/
│   │   ├── processed/
│   │   │   ├── games_fixed_clean.csv
│   │   │   └── steam_game_reviews_clean.csv
│   │   └── raw/
│   │       ├── games.csv
│   │       ├── games_fixed.csv
│   │       ├── steam_game_reviews.csv
│   │       └── ...
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── cleaning_report.py
│   │   └── report_generator.py
│   ├── reports/
│   │   ├── cleaning/
│   │   │   ├── games_fixed_clean.json
│   │   │   ├── games_fixed_clean.md
│   │   │   ├── games_fixed_clean_observaciones.csv
│   │   │   ├── steam_game_reviews_clean.json
│   │   │   ├── steam_game_reviews_clean.md
│   │   │   └── steam_game_reviews_clean_observaciones.csv
│   │   └── profiling/
│   │       ├── reporte_games_fixed.html
│   │       └── reporte_reviews.html
│   ├── scripts/
│   │   ├── run_cleaning.py
│   │   ├── run_header_fix.py
│   │   └── run_profiling.py
│   └── steam_analyzer/
│       ├── __init__.py
│       ├── config.py
│       └── pipeline.py
└── .venv/   # generado localmente, no versionado
```

## Requisitos

* Python 3.11+
* pip
* Entorno virtual de Python (`venv`)

## Crear el entorno virtual

Desde la raíz del proyecto:

```powershell
python -m venv .venv
```

## Activar el entorno virtual

```powershell
.venv\Scripts\Activate.ps1
```

Una vez activado, en la terminal debería aparecer `(.venv)` al inicio de la línea.

## Instalar dependencias

```powershell
pip install -r requirements.txt
```

## Estructura funcional del código

El proyecto no usa una carpeta `main`, sino que el código fuente vive bajo `src/`:

- `src/cleaning/`: limpieza, normalización y validación de datasets.
- `src/reporting/`: generación de reportes de calidad y profiling.
- `src/scripts/`: puntos de entrada para ejecutar tareas del flujo.
- `src/steam_analyzer/`: lógica de análisis y pipeline del proyecto.
- `src/data/raw/`: datasets originales.
- `src/data/processed/`: datasets ya normalizados y limpios.
- `src/reports/`: informes generados en JSON/Markdown/HTML y CSV de observaciones.

## Ejecutar limpieza de datos

La limpieza se dispara desde `src/scripts/run_cleaning.py`.

### Ejecutar todo el flujo

```powershell
python src/scripts/run_cleaning.py
```

### Limpiar solo un dataset

```powershell
python src/scripts/run_cleaning.py --dataset games
python src/scripts/run_cleaning.py --dataset reviews
```

### Personalizar rutas

```powershell
python src/scripts/run_cleaning.py --input-dir src/data/raw --output-dir src/data/processed --report-dir src/reports/cleaning
```

El script procesa los datasets por separado y genera archivos limpios en `src/data/processed/` junto con informes detallados en `src/reports/cleaning/`.

## Reparación de cabeceras

Si hace falta corregir primero los encabezados del CSV original de juegos, se puede ejecutar:

```powershell
python src/scripts/run_header_fix.py
```

Este paso genera `src/data/raw/games_fixed.csv` a partir de `src/data/raw/games.csv`.

## Generar reportes de profiling

```powershell
python src/scripts/run_profiling.py
```

Esto genera reportes HTML para los datasets principales en `src/reports/profiling/`.

## Notas de la limpieza

El flujo de limpieza actual:

- normaliza encabezados a `snake_case`
- convierte tipos numéricos y booleanos
- recorta espacios en valores de texto
- elimina filas duplicadas completas
- elimina columnas vacías comprobadas (`movies` y `release_date` según el dataset)
- conserva las reseñas de un mismo juego
- valida integridad, fechas, listas JSON y duplicados
- genera reportes de observaciones y resumen de calidad

El entorno virtual `.venv` no debe incluirse en el repositorio y puede recrearse con el archivo `requirements.txt`.
