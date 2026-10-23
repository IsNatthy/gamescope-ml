# Gamescope ML

Pipeline modular para preparar, limpiar y analizar datasets de juegos y reseñas de Steam. El proyecto genera datasets reproducibles, particiones sin fuga de información y reportes de calidad y análisis exploratorio como base para futuros modelos de Machine Learning.

## Estructura del proyecto

```text
gamescope-ml/
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── cleaning/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── dataset_cleaner.py
    │   ├── header_fixer.py
    │   ├── normalizer.py
    │   ├── row_cleaner.py
    │   └── validator.py
    ├── data/
    │   ├── raw/
    │   └── processed/
    ├── reporting/
    │   ├── __init__.py
    │   ├── cleaning_report.py
    │   ├── eda_report.py
    │   ├── report_generator.py
    │   └── reviews_eda_report.py
    ├── features/
    │   ├── __init__.py
    │   ├── build_dataset.py
    │   ├── build_reviews_dataset.py
    │   ├── split_dataset.py
    │   └── split_reviews_dataset.py
    ├── reports/
    │   ├── cleaning/
    │   ├── eda/
    │   ├── features/
    │   └── profiling/
    ├── scripts/
    │   ├── __init__.py
    │   ├── run_build_reviews_dataset.py
    │   ├── run_cleaning.py
    │   ├── run_eda.py
    │   ├── run_header_fix.py
    │   ├── run_profiling.py
    │   ├── run_build_dataset.py
    │   ├── run_reviews_eda.py
    │   ├── run_split.py
    │   └── run_split_reviews.py
    └── steam_analyzer/
        ├── __init__.py
        ├── config.py
        └── pipeline.py
```

## Requisitos

- Python 3.13 o una versión compatible con las dependencias del proyecto.
- pip.
- Git, para trabajar con el repositorio.
- Un entorno virtual de Python.

## Instalación y configuración

### 1. Crear el entorno virtual

Desde la raíz del proyecto:

```powershell
py -3.13 -m venv .venv
```

Si Python 3.13 no está instalado, utiliza una versión compatible disponible en tu entorno:

```powershell
python -m venv .venv
```

### 2. Activar el entorno virtual

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

Con el entorno virtual activado y situado en la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
```

## Arquitectura modular

El código se organiza por responsabilidades para separar las reglas de limpieza, el procesamiento de los datos, la generación de informes y los puntos de entrada de la aplicación.

### `src/cleaning/`

Contiene los componentes encargados de limpiar y validar los datasets.

- **`config.py`**: centraliza la configuración de los datasets y las reglas aplicables a cada uno.
- **`normalizer.py`**: contiene funciones auxiliares de normalización y transformación de valores.
- **`row_cleaner.py`**: se encarga del procesamiento y la limpieza de los registros.
- **`validator.py`**: contiene las validaciones de calidad y consistencia de los datos.
- **`dataset_cleaner.py`**: coordina las operaciones necesarias para limpiar un dataset y generar sus resultados.
- **`header_fixer.py`**: corrige el encabezado defectuoso del archivo original de juegos cuando se presenta el problema configurado.

### `src/reporting/`

Agrupa las funcionalidades relacionadas con la generación de informes.

- **`cleaning_report.py`**: genera los informes y registros derivados del proceso de limpieza, incluyendo las observaciones que deban revisarse.
- **`eda_report.py`**: genera gráficos y un resumen del análisis exploratorio utilizando únicamente el conjunto de entrenamiento.
- **`report_generator.py`**: contiene la funcionalidad de generación de reportes de profiling para explorar las características de los datasets.
- **`reviews_eda_report.py`**: genera las figuras y el resumen del análisis exploratorio de reseñas utilizando únicamente el conjunto de entrenamiento.

### `src/features/`

Construye el dataset de modelado de regresión con features disponibles antes del lanzamiento, variables multi-hot de géneros, categorías y etiquetas (tags), y un target numérico: el punto medio del intervalo de propietarios estimados.

- **`build_dataset.py`**: construye el dataset de modelado a partir del dataset de juegos limpio.
- **`split_dataset.py`**: divide el dataset de modelado en conjuntos de entrenamiento y prueba de forma aleatoria y reproducible, sin estratificación.

### `src/scripts/`

Contiene los puntos de entrada que permiten ejecutar las operaciones desde la terminal.

- **`run_header_fix.py`**: inicia la corrección del encabezado del dataset de juegos.
- **`run_cleaning.py`**: ejecuta la limpieza de los datasets. Permite seleccionar los juegos, las reseñas o ambos.
- **`run_profiling.py`**: ejecuta la generación de reportes de profiling configurados en el proyecto.
- **`run_build_dataset.py`**: construye el dataset de modelado a partir del dataset de juegos limpio.
- **`run_build_reviews_dataset.py`**: construye el dataset de modelado de reseñas.
- **`run_split.py`**: divide el dataset de modelado en `train.csv` y `test.csv`.
- **`run_split_reviews.py`**: divide el dataset de reseñas en conjuntos de entrenamiento y prueba sin mezclar juegos entre ambos.
- **`run_eda.py`**: genera el análisis exploratorio y sus figuras a partir de `train.csv`.
- **`run_reviews_eda.py`**: genera el análisis exploratorio y sus figuras a partir de `reviews_train.csv`.

### `src/steam_analyzer/`

Contiene la configuración y los componentes de coordinación general del análisis.

- **`config.py`**: centraliza la configuración correspondiente al flujo general del proyecto.
- **`pipeline.py`**: está destinado a coordinar las distintas etapas del procesamiento.

### `src/data/`

Almacena los archivos utilizados y generados durante el procesamiento.

- **`raw/`**: contiene los datasets originales y los archivos de entrada, como `games.csv`, `games_fixed.csv` y `steam_game_reviews.csv`.
- **`processed/`**: almacena los datasets resultantes de la limpieza.

### `src/reports/`

Almacena los informes generados durante la ejecución.

- **`cleaning/`**: contiene los informes y las observaciones asociados a la limpieza, según la configuración del proceso.
- **`eda/`**: contiene las figuras, el resumen y los análisis de correlación del EDA.
- **`features/`**: contiene los reportes de construcción y división del dataset de modelado.
- **`profiling/`**: contiene los reportes HTML de análisis exploratorio.

## Ejecución del proyecto

Ejecuta los comandos desde la raíz del repositorio. La forma `py -m src.scripts...` mantiene la estructura de imports del proyecto.

### 1. Corregir el encabezado del dataset de juegos

Si el encabezado del archivo original presenta el problema configurado, ejecuta:

```powershell
py -m src.scripts.run_header_fix
```

El proceso utiliza `src/data/raw/games.csv` como entrada y genera `src/data/raw/games_fixed.csv`.

Este paso debe realizarse antes de limpiar los juegos cuando el archivo original requiere esa corrección.

### 2. Ejecutar la limpieza

#### Limpiar ambos datasets

```powershell
py -m src.scripts.run_cleaning
```

#### Limpiar únicamente los juegos

```powershell
py -m src.scripts.run_cleaning --dataset games
```

#### Limpiar únicamente las reseñas

```powershell
py -m src.scripts.run_cleaning --dataset reviews
```

Para consultar los argumentos disponibles:

```powershell
py -m src.scripts.run_cleaning --help
```

El parámetro `--dataset` admite los valores `games`, `reviews` y `all`. Si no se especifica, se utiliza `all`.

La limpieza se ejecuta de manera independiente para cada dataset; no se unen automáticamente los datos de juegos y reseñas.

Los resultados se almacenan en `src/data/processed/` y los informes de limpieza en `src/reports/cleaning/`.

### 3. Generar reportes de profiling

Desde la raíz del proyecto:

```powershell
py -m src.scripts.run_profiling
```

Genera los reportes de profiling definidos en `run_profiling.py` dentro de `src/reports/profiling/`.

### 4. Construir el dataset de features

```powershell
py -m src.scripts.run_build_dataset
```

Genera `src/data/processed/model_dataset.csv` y `src/reports/features/model_dataset.json`.

Para sobrescribir las salidas existentes, añade `--overwrite`:

```powershell
py -m src.scripts.run_build_dataset --overwrite
```

### 5. Dividir el dataset en train y test

```powershell
py -m src.scripts.run_split
```

Divide `src/data/processed/model_dataset.csv` en `train.csv` y `test.csv` con una proporción 80/20 y semilla reproducible. Al ser regresión, no se estratifica por clases. El vocabulario usado para las columnas multi-hot de géneros, categorías y tags se calcula solo con las filas de train y se aplica sin cambios a test; los valores que no aparecieron en train no crean columnas nuevas. El target continuo `owners_midpoint` es el punto medio del intervalo de propietarios estimados; la aplicación puede considerar exitoso un juego cuando la predicción alcance el umbral de 20.000, sin convertir esa regla en el target de entrenamiento. También genera `src/reports/features/split.json` con estadísticas del target, hashes y conteos de cada partición.

En los datos actuales, el punto medio sigue teniendo una distribución asimétrica, aunque alrededor del 20% de los valores son cero (frente a casi el 80% usando el límite inferior). Al implementar el entrenamiento, conviene evaluar entrenar sobre `log1p(owners_midpoint)` e invertir la transformación de las predicciones con `expm1` antes de compararlas con el umbral de 20.000. Esa transformación no se aplica al dataset de entrada.

Para sobrescribir las salidas existentes:

```powershell
py -m src.scripts.run_split --overwrite
```

### 6. Generar el análisis exploratorio

Después de dividir el dataset, ejecuta:

```powershell
py -m src.scripts.run_eda
```

Genera las figuras y el resumen del análisis exploratorio de juegos sobre `src/data/processed/train.csv` en `src/reports/eda/`. Para consultar las rutas de entrada y salida:

```powershell
py -m src.scripts.run_eda --help
```

### 7. Construir y analizar el dataset de reseñas

Después de la limpieza de reseñas, ejecuta desde la raíz del repositorio:

```powershell
py -m src.scripts.run_build_reviews_dataset
py -m src.scripts.run_split_reviews
py -m src.scripts.run_reviews_eda
```

`build_reviews_dataset.py` construye el dataset de modelado a partir de `steam_game_reviews_clean.csv`, conservando el texto crudo, la etiqueta `voted_up` y los campos de trazabilidad. El dataset se filtra, deduplica y balancea por submuestreo de la clase mayoritaria.

`split_reviews_dataset.py` genera una partición 80/20 estratificada por `voted_up` y agrupada por `appid`, garantizando que ningún juego aparezca en train y test. `reviews_eda_report.py` genera el EDA utilizando únicamente `reviews_train.csv`.

Las salidas son:

- `src/data/processed/reviews_model_dataset.csv`, `reviews_train.csv` y `reviews_test.csv`.
- `src/reports/features/reviews_model_dataset.json` y `reviews_split.json`.
- `src/reports/eda/reviews/`, con las figuras y `resumen_eda.md`.

Los scripts de construcción y división aceptan `--overwrite` para sobrescribir salidas existentes.

## Criterios de limpieza

El proceso contempla las reglas definidas en la configuración y en los componentes de limpieza. Entre las operaciones previstas se encuentran:

- Normalización de nombres de columnas.
- Conversión y validación de valores numéricos y booleanos.
- Eliminación de espacios exteriores en los valores.
- Tratamiento de filas duplicadas según las reglas implementadas.
- Eliminación de columnas configuradas para descartarse.
- Tratamiento de valores faltantes sin inventar información.
- Normalización de fechas y listas de idiomas, cuando corresponda.
- Registro de anomalías y observaciones para revisión.
- Validación de la estructura y consistencia de los archivos resultantes.
- Generación de informes de limpieza y profiling.

Las reglas concretas dependen de la configuración de cada dataset. Los archivos originales se mantienen como fuentes de entrada y los resultados se escriben en archivos separados.
