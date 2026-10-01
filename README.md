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
