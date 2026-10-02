# Gamescope ML

Proyecto de preparación y análisis inicial de datasets para un flujo de Machine Learning.

Actualmente el proyecto se encuentra en las etapas de **inspección y limpieza inicial de los datos**.

## Estructura del proyecto

```text
gamescope-ml/
│
├── .venv/
├── .idea/
│
├── data/
│   └── raw/
│       ├── games.csv
│       ├── games_fixed.csv
│       └── steam_game_reviews.csv
│
├── cleaning/
│   └── limpiar_games.py
│
├── reports/
│   ├── generacion_reportes.py
│   ├── reporte_reviews.html
│   └── reporte_games_fixed.html
│
├── main/
│
└── requirements.txt
```

## Requisitos

* Python 3.11.9
* pip
* Entorno virtual de Python (`venv`)

## Crear el entorno virtual

Desde la carpeta raíz del proyecto:

```powershell
python -m venv .venv
```

## Activar el entorno virtual

En la terminal:

```powershell
.venv\Scripts\Activate.ps1
```

Una vez activado, debería aparecer `(.venv)` al inicio de la terminal.

Por ejemplo:

```text
(.venv) PS C:\programacion\repos\gamescope-ml>
```

## Instalar las dependencias

Con el entorno virtual activado:

```powershell
pip install -r requirements.txt
```

## Entorno utilizado

El proyecto es desarrollado utilizando:

```text
Python 3.11.9
```

Las dependencias del proyecto se encuentran especificadas en:

```text
requirements.txt
```

El entorno virtual `.venv` no debe incluirse en el repositorio, ya que puede ser recreado utilizando la versión de Python indicada y el archivo `requirements.txt`.

## Limpieza independiente de los datasets

`cleaning/limpiar_datasets.py` limpia el catálogo y las reseñas por separado. No une
los archivos ni usa datos de uno para completar el otro. Emplea únicamente la
biblioteca estándar de Python, por lo que no requiere instalar dependencias.

Con los CSV originales en `data/raw/`, ejecutar desde `gamescope-ml`:

```powershell
python cleaning/limpiar_datasets.py
```

Si los originales están en la carpeta superior, como en este workspace:

```powershell
py -3.14 cleaning/limpiar_datasets.py --input-dir ..
```

También se puede procesar un único dataset mediante `--dataset games` o
`--dataset reviews`. Las rutas se pueden configurar con `--input-dir`,
`--output-dir` y `--report-dir`. El script rechaza salidas existentes; para una
nueva ejecución se deben usar otros directorios de salida y de reportes.

Resultados:

- `data/processed/games_fixed_clean.csv`
- `data/processed/steam_game_reviews_clean.csv`
- `reports/cleaning/`: reporte Markdown y JSON por dataset, y CSV de observaciones.

La limpieza normaliza encabezados a `snake_case`, números y booleanos, recorta
espacios exteriores y elimina filas completas idénticas. Las reseñas de un mismo
juego se conservan. Solo se eliminan columnas comprobadas como totalmente vacías:
`Movies` del catálogo y `release_date` de las reseñas.

Las fechas del catálogo pasan a `YYYY-MM-DD`. Las reseñas conservan su timestamp
original y añaden `review_created_at_utc`. Las dos columnas de listas de idiomas
se convierten a JSON, reparando las comillas defectuosas observadas en `K'iche`.
No se cambia la escala de los precios, no se imputan faltantes ni se eliminan
ceros o valores extremos. Una fila sin nombre se conserva y se registra para revisión.

Cada salida se relee para validar filas, estructura, tipos, fechas, listas JSON,
duplicados e identificadores. Se comprueba que el original no cambió mediante
SHA-256. El CSV no almacena tipos nativos: el esquema recomendado queda en el
reporte JSON; los valores ausentes se representan con celdas vacías.
