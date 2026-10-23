# Entrenamiento de modelos de Gamescope ML

Este documento explica cómo ejecutar los modelos desde cero. No es necesario
haber entrenado modelos antes. Todos los comandos se ejecutan desde la raíz
del proyecto, es decir, desde la carpeta que contiene `README.md`.

## 1. Qué se va a entrenar

El proyecto tiene dos objetivos:

1. **Predicción de jugadores:** Random Forest Regressor y XGBoost Regressor.
   Ambos predicen una cantidad numérica de jugadores y se comparan para elegir
   el mejor.
2. **Análisis de reseñas:** BERT analiza el sentimiento de las reseñas y
   BERTopic descubre los temas más repetidos. No son modelos que compitan con
   la misma métrica: se complementan para producir recomendaciones.

## 2. Antes de comenzar

Los archivos originales deben estar en `src/data/raw/`:

```text
src/data/raw/games.csv
src/data/raw/games_fixed.csv
src/data/raw/steam_game_reviews.csv
```

Si `games.csv` ya tiene el encabezado correcto pero todavía no existe
`games_fixed.csv`, créalo con PowerShell:

```powershell
Copy-Item .\src\data\raw\games.csv .\src\data\raw\games_fixed.csv
```

No subas los CSV al repositorio. Están excluidos por `.gitignore`.

## 3. Activar el entorno virtual

Desde PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Si todavía no existe `.venv`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 4. Instalar dependencias básicas

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Las dependencias de texto son opcionales y pesadas. Instálalas solamente para
BERT y BERTopic:

```powershell
python -m pip install -r requirements-modeling.txt
```

La instalación puede tardar y ocupar varios GB, especialmente por PyTorch.

## 5. Preparar los datos de juegos

Primero limpia el dataset de juegos:

```powershell
python -m src.scripts.run_cleaning --dataset games
```

Si `src/data/processed/games_fixed_clean.csv` ya existe, no repitas este
comando: el proyecto evita sobrescribir resultados anteriores. Continúa con
`run_build_dataset`.

El resultado esperado es:

```text
src/data/processed/games_fixed_clean.csv
```

Después construye el dataset oficial de regresión:

```powershell
python -m src.scripts.run_build_dataset
python -m src.scripts.run_split
```

Esto crea:

```text
src/data/processed/model_dataset.csv
src/data/processed/train.csv
src/data/processed/test.csv
```

El campo `estimated_owners`, que viene como un rango, se convierte a un
valor numérico aproximado usando el punto medio del rango. El modelo entrena
con `log(1 + jugadores)` porque la cantidad de jugadores tiene valores muy
desiguales.

Si necesitas repetir la preparación completa:

```powershell
python -m src.scripts.run_build_dataset --overwrite
python -m src.scripts.run_split --overwrite
```

## 6. Entrenar Random Forest y XGBoost

Ejecuta primero una versión moderada:

```powershell
python -m src.scripts.run_train_success --threads 2 --trees 100
```

Cuando verifiques que funciona, puedes usar 200 árboles:

```powershell
python -m src.scripts.run_train_success --threads 2 --trees 200
```

El script entrena ambos modelos y muestra:

- MAE: error absoluto medio; menor es mejor.
- RMSE: penaliza más los errores grandes; menor es mejor.
- R²: proporción de variabilidad explicada; mayor es mejor.

Los modelos se guardan en:

```text
src/models/random_forest_regressor.joblib
src/models/xgboost_regressor.joblib
```

Y las métricas en:

```text
src/reports/models/success_models.json
```

El ganador es el que tenga el menor MAE/RMSE y un R² razonable. No se debe
escoger solamente por una métrica sin revisar las demás.

## 7. Preparar las reseñas

El nuevo flujo de reseñas depura el texto, balancea `voted_up` y separa por
`appid`, para evitar que reseñas del mismo juego aparezcan en train y test.

```powershell
python -m src.scripts.run_cleaning --dataset reviews
python -m src.scripts.run_build_reviews_dataset --overwrite
python -m src.scripts.run_split_reviews --overwrite
python -m src.scripts.run_reviews_eda
```

Esto genera `reviews_train.csv`, `reviews_test.csv` y los reportes de reseñas.

## 8. Entrenar BERT

El archivo de reseñas limpio contiene:

```text
review   → texto de la reseña
voted_up → etiqueta positiva o negativa aproximada
```

`voted_up` no es una medición perfecta del sentimiento; se utiliza como proxy
y esa limitación debe aparecer en el informe.

Para proteger la máquina, comienza con 5.000 reseñas:

```powershell
python -m src.scripts.run_train_bert --max-rows 5000 --batch-size 2 --epochs 1
```

Si funciona y la memoria se mantiene estable:

```powershell
python -m src.scripts.run_train_bert --max-rows 20000 --batch-size 4 --epochs 1
```

El script detecta automáticamente la GPU. Si PyTorch detecta tu RTX 3050,
mostrará `Dispositivo: cuda`; de lo contrario mostrará `Dispositivo: cpu`.

Puedes comprobar la detección antes de entrenar:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Los resultados se guardan en:

```text
src/models/bert_sentiment/
src/reports/models/bert_sentiment.json
```

Si aparece un error de memoria GPU, reduce el batch:

```powershell
python -m src.scripts.run_train_bert --max-rows 5000 --batch-size 1 --max-length 96
```

## 9. Entrenar BERTopic

BERTopic encuentra grupos de reseñas con temas similares. Empieza con pocas
reseñas:

```powershell
python -m src.scripts.run_train_bertopic --max-rows 5000 --min-topic-size 40
```

BERTopic usa la GPU, si PyTorch la detecta, para calcular embeddings. El
agrupamiento posterior puede seguir utilizando CPU y RAM.

Después puedes aumentar gradualmente:

```powershell
python -m src.scripts.run_train_bertopic --max-rows 30000 --min-topic-size 50
```

Los resultados se guardan en:

```text
src/models/bertopic_reviews/
src/reports/models/bertopic_topics.csv
src/reports/models/bertopic_review_topics.csv
```

`bertopic_topics.csv` contiene las palabras representativas de cada tema.
`bertopic_review_topics.csv` permite saber qué tema recibió cada reseña.

## 10. Cómo interpretar el resultado de reseñas

BERTopic descubre temas como:

```text
combate, velocidad, ataques, enemigos
armadura, diseño, personalización
errores, rendimiento, crashes
```

BERT puede indicar si una reseña es positiva o negativa. En una etapa
posterior se deben cruzar ambos resultados para calcular, por tema:

- número de menciones;
- proporción positiva;
- proporción negativa;
- juegos similares donde aparece;
- recomendación de diseño.

Ejemplo de interpretación:

```text
Tema: combate
Menciones negativas: 68%
Recomendación: revisar velocidad y respuesta del combate.
```

## 11. Cómo cuidar el equipo

No comiences con todo el dataset de reseñas. Aumenta el límite gradualmente.

- Mantén `--threads 2` al principio.
- Usa `--batch-size 1` o `2` si la GPU tiene poca VRAM.
- Cierra aplicaciones pesadas antes de BERT o BERTopic.
- Vigila RAM, VRAM y temperatura.
- Detén el proceso con `Ctrl+C` si Windows empieza a quedarse sin memoria.
- No ejecutes varios entrenamientos pesados al mismo tiempo.
- Guarda los resultados después de cada modelo.

Usar toda la RAM puede provocar que Windows use el disco como memoria virtual,
lo que vuelve lento o inestable el equipo. Normalmente no destruye físicamente
la RAM, pero sí puede bloquear el sistema o hacer perder trabajo. Por eso los
scripts tienen límites de filas, batch y número de hilos.

## 12. Orden resumido

```powershell
python -m src.scripts.run_cleaning --dataset games
python -m src.scripts.run_build_dataset
python -m src.scripts.run_split
python -m src.scripts.run_train_success --threads 2 --trees 100
python -m src.scripts.run_cleaning --dataset reviews
python -m src.scripts.run_build_reviews_dataset --overwrite
python -m src.scripts.run_split_reviews --overwrite
python -m src.scripts.run_train_bert --max-rows 5000 --batch-size 2 --epochs 1
python -m src.scripts.run_train_bertopic --max-rows 5000 --min-topic-size 40
```

No ejecutes el siguiente paso hasta comprobar que el anterior terminó sin
errores y que la memoria del equipo se mantiene estable.
