# Gamescope ML

Proyecto de preparación, limpieza, validación y análisis exploratorio de datasets de Steam, orientado al desarrollo de un flujo de Machine Learning para identificar oportunidades de nicho en el mercado de videojuegos.

Actualmente, el proyecto se centra en el procesamiento independiente de los datasets de juegos y reseñas, la validación de su calidad y la generación de informes que permitan revisar los resultados antes de avanzar hacia la ingeniería de características y el modelado predictivo.

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
    │   └── report_generator.py
    ├── features/
    │   ├── __init__.py
    │   └── build_dataset.py
    ├── reports/
    │   ├── cleaning/
    │   └── profiling/
    ├── scripts/
    │   ├── __init__.py
    │   ├── run_cleaning.py
    │   ├── run_header_fix.py
    │   ├── run_profiling.py
    │   └── run_build_dataset.py
    └── steam_analyzer/
        ├── __init__.py
        ├── config.py
        └── pipeline.py
```

Los archivos `__init__.py` permiten organizar los directorios como paquetes de Python. Si alguno de los indicados no existe, debe crearse vacío para mantener esta estructura de paquetes.

Los datasets, los reportes generados y el entorno virtual son archivos locales que no deben versionarse en Git.

## Requisitos

- Python 3.13 o una versión compatible con las dependencias del proyecto.
- pip.
- Git, para trabajar con el repositorio.
- Un entorno virtual de Python.

Se recomienda utilizar la misma versión de Python en los entornos de desarrollo para reducir posibles incompatibilidades entre dependencias.

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

Una vez activado, la terminal mostrará `(.venv)` al inicio de la línea.

### 3. Instalar las dependencias

Con el entorno virtual activado y situado en la raíz del proyecto:

```powershell
python -m pip install -r requirements.txt
```

El archivo `requirements.txt` contiene las dependencias necesarias para ejecutar el proyecto.

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
- **`report_generator.py`**: contiene la funcionalidad de generación de reportes de profiling para explorar las características de los datasets.

### `src/features/`

Construye el dataset de modelado con features pre-lanzamiento, variables multi-hot y el target balanceado.

### `src/scripts/`

Contiene los puntos de entrada que permiten ejecutar las operaciones desde la terminal.

- **`run_header_fix.py`**: inicia la corrección del encabezado del dataset de juegos.
- **`run_cleaning.py`**: ejecuta la limpieza de los datasets. Permite seleccionar los juegos, las reseñas o ambos.
- **`run_profiling.py`**: ejecuta la generación de reportes de profiling configurados en el proyecto.
- **`run_build_dataset.py`**: construye el dataset de modelado a partir del dataset de juegos limpio.

Estos scripts delegan las operaciones a los módulos correspondientes; no deben duplicar la lógica de limpieza ni de generación de informes.

### `src/steam_analyzer/`

Contiene la configuración y los componentes de coordinación general del análisis.

- **`config.py`**: centraliza la configuración correspondiente al flujo general del proyecto.
- **`pipeline.py`**: está destinado a coordinar las distintas etapas del procesamiento.

La existencia de estos componentes no implica que todas las etapas del pipeline estén integradas o automatizadas en la ejecución actual.

### `src/data/`

Almacena los archivos utilizados y generados durante el procesamiento.

- **`raw/`**: contiene los datasets originales y los archivos de entrada, como `games.csv`, `games_fixed.csv` y `steam_game_reviews.csv`.
- **`processed/`**: almacena los datasets resultantes de la limpieza.

### `src/reports/`

Almacena los informes generados durante la ejecución.

- **`cleaning/`**: contiene los informes y las observaciones asociados a la limpieza, según la configuración del proceso.
- **`profiling/`**: contiene los reportes HTML de análisis exploratorio.

## Ejecución del proyecto

**Importante:** ejecuta los siguientes comandos desde la raíz del repositorio, donde se encuentran `README.md` y `.gitignore`. Se utiliza `py -m src.scripts...` para mantener la convención de imports `from src.cleaning...` y `from src.reporting...`.

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

**Nota sobre los archivos de salida:** si una ejecución anterior ya generó un archivo de salida, el proceso puede detenerse para evitar sobrescribirlo. Revisa el comportamiento implementado en `dataset_cleaner.py` antes de volver a ejecutar la limpieza. Conserva o elimina los resultados anteriores según corresponda.

### 3. Generar reportes de profiling

Desde la raíz del proyecto:

```powershell
py -m src.scripts.run_profiling
```

Este comando ejecuta el script de profiling y genera los reportes definidos en `run_profiling.py`. Los resultados se almacenan en `src/reports/profiling/`.

Los reportes de profiling permiten explorar la distribución de los datos, los valores faltantes y otras características relevantes de los datasets.

### 4. Construir el dataset de features

```powershell
py -m src.scripts.run_build_dataset
```

Genera `src/data/processed/model_dataset.csv` y `src/reports/features/model_dataset.json`.

Para sobrescribir las salidas existentes:

```powershell
py -m src.scripts.run_build_dataset --overwrite
```

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

Las reglas concretas dependen de la configuración de cada dataset. Por tanto, esta lista describe las operaciones contempladas por el proceso y no implica que todas se apliquen de forma idéntica a ambos datasets.

Los archivos originales se mantienen como fuentes de entrada. Los resultados de limpieza se escriben en archivos separados para evitar modificar directamente los datasets originales.

## Control de versiones

El archivo `.gitignore` excluye los elementos generados o específicos del entorno local, entre ellos:

- Entornos virtuales.
- Cachés y archivos temporales de Python.
- Datasets CSV originales y procesados.
- Reportes generados.
- Archivos de configuración local que puedan contener secretos.

El código fuente, la documentación y los archivos de configuración necesarios para reproducir el proyecto deben versionarse.

## Estado actual y trabajo futuro

El proyecto cuenta con una estructura modular para separar la limpieza, la validación, la generación de informes y la coordinación del análisis de los datasets de Steam.

Las siguientes etapas del desarrollo podrán incorporar la integración completa del pipeline, la ingeniería de características, el procesamiento de texto de las reseñas y el entrenamiento y evaluación de modelos predictivos.
