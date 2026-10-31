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

Los pipelines entrenados se guardan en:

```text
src/models/random_forest_raw.joblib
src/models/random_forest_log.joblib
src/models/xgboost_raw.joblib
src/models/xgboost_log.joblib
src/models/ridge_log.joblib
```

Y las métricas en:

```text
src/reports/models/success_models.json
```

El ganador es el que tenga el menor MAE/RMSE y un R² razonable. No se debe
escoger solamente por una métrica sin revisar las demás.

## 7. Preparar las reseñas

El flujo de reseñas depura el texto, balancea `voted_up` y separa por `appid`,
para evitar que reseñas del mismo juego aparezcan en train, validación y test.
En la captura actual, la limpieza deja 715.122 reseñas sin balancear: 130.859
negativas (18,30 %) y 584.263 positivas (81,70 %). El constructor equilibra
mediante submuestreo global aleatorio de la clase positiva hasta 130.859 por
clase, con semilla 42. No aplica un máximo por juego; en el dataset balanceado
un juego de los 743 originales queda sin reseñas.
Las proporciones por género principal, calculadas sobre el corpus sin
balancear, están en
`src/results/pendientes_resenas/descriptive_cap_50/real_class_by_primary_genre.csv`.

```powershell
python -m src.scripts.run_cleaning --dataset reviews
python -m src.scripts.run_build_reviews_dataset --overwrite
python -m src.scripts.run_split_reviews --overwrite
python -m src.scripts.run_reviews_eda
```

Esto genera `reviews_train.csv`, `reviews_validation.csv`, `reviews_test.csv` y
los reportes de reseñas. El código construye una partición agrupada y
aproximadamente 80/10/10 por `appid`; la implementación actual usa
`StratifiedGroupKFold` para mantener los juegos separados y aproximar la
distribución de `voted_up`. Los archivos y reportes existentes deben
comprobarse antes de asumir que corresponden a esta versión del split.

## Mejoras del entrenamiento de regresión

El entrenamiento completo incluye:

- líneas base de media y mediana;
- Ridge como referencia lineal;
- variantes del target original y `log1p`;
- validación cruzada de cinco folds;
- búsqueda limitada de hiperparámetros;
- imputación aprendida dentro de cada pipeline;
- MAE, RMSE, MedAE, R² y métricas logarítmicas;
- métricas por segmentos;
- importancia por permutación del modelo ganador.

La regla de 20.000 no se calcula en este script: pertenece a la aplicación.

Ejecuta una búsqueda controlada:

```powershell
python -m src.scripts.run_train_success --threads 2 --trees 100 --cv 5 --search-iterations 2
```

La búsqueda puede tardar porque prueba varios modelos y folds. Para una
prueba rápida:

```powershell
python -m src.scripts.run_train_success --threads 2 --trees 50 --cv 3 --search-iterations 1 --skip-importance
```

Para estudiar si los tags aportan información potencialmente posterior al
lanzamiento, repite el experimento sin ellos:

```powershell
python -m src.scripts.run_train_success --without-tags --threads 2 --trees 100 --cv 5 --search-iterations 2
```

El informe completo se guarda en:

```text
src/reports/models/success_models.json
src/reports/models/success_feature_importance.csv
```

Para evaluar la generalización a juegos recientes, ejecuta además:

```powershell
python -m src.scripts.run_temporal_success --threads 2 --trees 100
```

Su informe se guarda en:

```text
src/reports/models/success_models_temporal.json
```

## 8. Entrenar BERT

El archivo de reseñas limpio contiene:

```text
review   → texto de la reseña
voted_up → etiqueta positiva o negativa aproximada
```

`voted_up` no es una medición perfecta del sentimiento; se utiliza como proxy
y esa limitación debe aparecer en el informe.

El script entrena dos variantes de BERT: estándar y ponderada por frecuencia
de clase. También ejecuta una línea base TF-IDF + regresión logística para
saber si BERT realmente aporta una mejora. La validación se usa para parada
temprana y el test se reserva para la evaluación final.

Para proteger la máquina, comienza con 5.000 reseñas por partición:

```powershell
python -m src.scripts.run_train_bert --max-rows 5000 --batch-size 2 --epochs 2 --patience 1
```

Después entrena la variante ponderada, que puede ayudar cuando hay más
reseñas positivas que negativas:

```powershell
python -m src.scripts.run_train_bert --weighted --max-rows 5000 --batch-size 2 --epochs 2 --patience 1
```

Si la prueba funciona y la memoria se mantiene estable, aumenta gradualmente
`--max-rows` a 10.000 o 20.000 y conserva `--batch-size 2`.

El script detecta automáticamente la GPU. Si PyTorch detecta tu RTX 3050,
mostrará `Dispositivo: cuda`; de lo contrario mostrará `Dispositivo: cpu`.

Puedes comprobar la detección antes de entrenar:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Los resultados se guardan en:

```text
src/models/bert_sentiment/
src/models/bert_sentiment_weighted/
src/reports/models/bert_sentiment_standard.json
src/reports/models/bert_sentiment_weighted.json
```

Cada informe incluye accuracy, precisión/recall/F1 de la clase negativa,
F1 macro, MCC, ROC-AUC, PR-AUC, ECE y el resultado del baseline TF-IDF. Para
comparar las dos variantes, prioriza F1 macro, MCC y PR-AUC; revisa también
la calibración ECE. No elijas por accuracy aislada.

Si aparece un error de memoria GPU, reduce el batch:

```powershell
python -m src.scripts.run_train_bert --max-rows 3000 --batch-size 1 --max-length 96 --no-tfidf
```

## 9. Entrenar BERTopic

BERTopic encuentra grupos de reseñas con temas similares. El script calcula
NPMI, diversidad, porcentaje de outliers y estabilidad entre varias semillas.
También ejecuta LDA y NMF como referencias no neuronales. Empieza con pocas
reseñas:

```powershell
python -m src.scripts.run_train_bertopic --max-rows 5000 --eval-rows 2000 --stability-rows 2000 --baseline-rows 2000 --embedding-batch-size 16 --min-topic-size 40
```

BERTopic usa la GPU, si PyTorch la detecta, para calcular embeddings. El
agrupamiento posterior puede seguir utilizando CPU y RAM.

Después puedes aumentar gradualmente:

```powershell
python -m src.scripts.run_train_bertopic --max-rows 30000 --eval-rows 5000 --stability-rows 5000 --baseline-rows 5000 --embedding-batch-size 16 --min-topic-size 50
```

Para una prueba rápida sin LDA/NMF usa `--skip-baselines`. Si falta VRAM,
reduce `--embedding-batch-size` a 8 o 4. La estabilidad se resume con ARI y
Jaccard de palabras entre semillas; valores más altos indican temas más
reproducibles.

Los resultados se guardan en:

```text
src/models/bertopic_reviews/
src/reports/models/bertopic_train_topics.csv
src/reports/models/bertopic_validation_topics.csv
src/reports/models/bertopic_test_topics.csv
src/reports/models/bertopic_topics.csv
src/reports/models/bertopic_evaluation.json
```

`bertopic_topics.csv` contiene las palabras representativas de cada tema.
Los tres archivos `bertopic_*_topics.csv` permiten saber qué tema recibió cada
reseña en cada partición. `bertopic_evaluation.json` contiene NPMI, diversidad,
outliers, estabilidad y las referencias LDA/NMF.

### Modelo descriptivo sobre reseñas históricas sin balancear

Para analizar temas junto con las recomendaciones observadas, no uses el
dataset balanceado: su proporción 50/50 no representa la distribución histórica.
El análisis reproducible actual toma hasta 50 reseñas por juego, con selección
aleatoria dentro de cada clase y cuotas proporcionales a las clases observadas
en ese juego. La muestra contiene 36.848 reseñas de 743 juegos (18,35 %
negativas y 81,65 % positivas). Es una submuestra limitada por hardware, no el
corpus completo de 715.122 reseñas.

Desde la raíz del proyecto:

```powershell
python src\results\pendientes_resenas\descriptive_corpus_prep.py --max-per-game 50
python src\results\pendientes_resenas\bertopic_descriptive_integration.py
```

En este repositorio los scripts y resultados generados se encuentran bajo
`src/results/pendientes_resenas/`; los resultados están en
`descriptive_cap_50/`. La búsqueda probó
`min_cluster_size` 50, 100 y 200; seleccionó 50 por NPMI más alto (0,3139)
entre configuraciones con al menos cinco temas y ningún tema mayor al 50 % de
los no-outliers. Resultado: 145 temas, diversidad 0,7745 y 43,58 % outliers.
La reducción de outliers por embeddings reasignó 16.059 reseñas y dejó 0 %
outliers. Las integraciones con y sin reducción, métricas y palabras por tema
están en los CSV y JSON de esa carpeta.

El cruce con `voted_up` observado y género principal encuentra señales
descriptivas, no causales. Por ejemplo, sin reducción, el tema de DLC/compra
en Simulation tiene 40,00 % negativas frente a 22,41 % del segmento
(Wilson 95 %: 28,57–52,63 %). No interpretes automáticamente la polaridad
como una oportunidad comercial: revisa las reseñas y la composición de los
juegos del tema.

La plantilla `revision_temas.xlsx` tiene una fila por tema y columnas
independientes para tres revisores. Cada revisor mira palabras y ejemplos y
elige:

- **Interpretable:** idea temática clara respaldada por los ejemplos.
- **Parcialmente:** hay una idea general, pero se mezclan subtemas o varios
  ejemplos/palabras no encajan.
- **No interpretable:** no se identifica una idea común o predominan ruido y
  artefactos.

Cada revisor completa solo su propia columna; no es necesario que una persona
llene las tres. La hoja `Acuerdo` calcula porcentajes por etiqueta y acuerdo
entre revisores (acuerdo total y kappa de Fleiss) cuando los tres han
completado el mismo tema. El nombre propuesto se acuerda después de la
etiquetación independiente.

Para reproducir el archivo Excel se necesita `openpyxl`, incluido en
`requirements-modeling.txt`.

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
python -m src.scripts.run_train_success --threads 2 --trees 50 --cv 3 --search-iterations 1 --skip-importance
python -m src.scripts.run_cleaning --dataset reviews
python -m src.scripts.run_build_reviews_dataset --overwrite
python -m src.scripts.run_split_reviews --overwrite
python -m src.scripts.run_train_bert --max-rows 5000 --batch-size 2 --epochs 2 --patience 1
python -m src.scripts.run_train_bert --weighted --max-rows 5000 --batch-size 2 --epochs 2 --patience 1
python -m src.scripts.run_train_bertopic --max-rows 5000 --eval-rows 2000 --stability-rows 2000 --baseline-rows 2000 --embedding-batch-size 16 --min-topic-size 40
```

No ejecutes el siguiente paso hasta comprobar que el anterior terminó sin
errores y que la memoria del equipo se mantiene estable.
