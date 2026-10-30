# Modelo matemático para la predicción de adopción comercial en GameScope

## Estado del documento

Este documento describe la formulación y evaluación de los modelos.
Random Forest, XGBoost y Ridge ya fueron evaluados. Los resultados nuevos,
las métricas por segmento y los archivos reproducibles están bajo
`src/results/pendientes/`; los reportes originales del entrenamiento siguen
en `src/models/` y `src/reports/`.

La selección reportada se basa en MAE de validación cruzada sobre train. El
test se usó para la evaluación final. Hay una limitación que debe acompañar
los resultados: la validación cruzada parte de las features multi-hot ya
codificadas sobre train completo, por lo que su vocabulario no se reajusta en
cada fold.

## 1. Definición del problema

Se plantea un problema de **regresión supervisada**. El flujo general es:

> datos → variables → modelo matemático → predicción

- **Entrada \(\mathbf{x}\):** características del videojuego, como precio,
  plataformas, idiomas, fecha de lanzamiento, géneros, categorías y tags.
- **Objetivo \(y\):** `owners_midpoint`, punto medio del intervalo
  `estimated_owners` de Steam.

Sea el conjunto de \(n\) videojuegos válidos:

$$
\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^{n},\qquad
\mathbf{x}_i\in\mathbb{R}^{p},\quad y_i\in\mathbb{R}_{\geq 0}.
$$

El objetivo es aprender una función de predicción
\(f:\mathbb{R}^{p}\to\mathbb{R}\). La salida operativa se restringe a valores
no negativos mediante el postprocesamiento descrito en la sección 5:

$$
\hat y_i=\max(0,f(\mathbf{x}_i)).
$$

El modelo no aprende una etiqueta binaria ni predice ingresos, ventas exactas
o rentabilidad. Predice un punto representativo del intervalo de propietarios
estimados en Steam.

## 2. Variable objetivo y distribución observada

Para un intervalo de propietarios \([L_i,U_i]\):

$$
\boxed{y_i=\text{owners\_midpoint}_i=\frac{L_i+U_i}{2}}
$$

Por ejemplo, el intervalo 20.000–50.000 produce un objetivo de 35.000.

El dataset de modelado contiene **125.855 filas**; no se descartaron filas por
intervalos inválidos. `owners_midpoint` toma los valores discretos derivados
de los rangos de Steam. En train (100.684 juegos), 19.629 filas (19,50 %)
tienen target 0, 60.529 (60,12 %) tienen target 10.000 y 20.526 (20,39 %)
tienen target igual o superior a 20.000. En el test de 25.171 filas, 20.126
(79,96 %) quedan bajo el umbral y 5.045 (20,04 %) lo alcanzan.

El valor cero proviene del intervalo original **“0 - 0”** (24.579 filas).
La revisión de `estimated_owners` no encontró intervalos que crucen
estrictamente 20.000; 87.102 intervalos tienen un límite igual a 20.000.

En el conjunto de entrenamiento (**100.684 filas**), los estadísticos
disponibles son:

| Estadístico | `owners_midpoint` |
|---|---:|
| Media | 85.007,05 |
| Mediana | 10.000 |
| Percentil 25 | 10.000 |
| Percentil 75 | 10.000 |
| Percentil 90 | 75.000 |
| Percentil 99 | 1.500.000 |
| Máximo | 150.000.000 |

La mediana y los cuartiles muestran que el target está muy concentrado en
valores bajos, mientras que unos pocos valores muy grandes generan una cola
larga. El conteo exacto de valores de target en train es:

| `owners_midpoint` | Juegos | Porcentaje |
|---:|---:|---:|
| 0 | 19.629 | 19,50 % |
| 10.000 | 60.529 | 60,12 % |
| 35.000 | 9.136 | 9,07 % |
| 75.000 | 4.268 | 4,24 % |
| 150.000 | 2.812 | 2,79 % |
| 350.000 | 2.284 | 2,27 % |
| 750.000 | 930 | 0,92 % |
| 1.500.000 | 595 | 0,59 % |
| 3.500.000 | 320 | 0,32 % |
| 7.500.000 | 101 | 0,10 % |
| 15.000.000 | 42 | 0,04 % |
| 35.000.000 | 27 | 0,03 % |
| 75.000.000 | 8 | 0,01 % |
| 150.000.000 | 3 | <0,01 % |

**Incertidumbre del target.** `owners_midpoint` es una aproximación, no una
medición exacta de los propietarios reales. Si el valor real está dentro del
intervalo, la diferencia absoluta entre este y el punto medio no supera la
mitad del ancho del intervalo. Esto expresa incertidumbre en el objetivo, no
un error mínimo garantizado: la diferencia también puede ser cero.

El punto medio de intervalos muy amplios puede ser poco representativo del
valor real. Esta limitación debe tenerse en cuenta al interpretar los errores,
en especial para la categoría superior de propietarios.

## 3. Variables de entrada y preprocesamiento

El dataset documenta **112 features**, además de las columnas de trazabilidad
`appid` y `name` y del target.

| Grupo | Variables |
|---|---|
| Comerciales | `price` |
| Técnicas | `required_age`, soporte para Windows, macOS y Linux |
| Idiomas | Número de idiomas de interfaz y de idiomas con audio |
| Temporales | Año y mes de lanzamiento |
| Diseño | Cantidad de géneros, categorías y tags |
| Géneros | Multi-hot para los 30 géneros más frecuentes en train |
| Categorías | Multi-hot para las 20 categorías más frecuentes en train |
| Tags | Multi-hot para los 50 tags más frecuentes en train |

El conteo de géneros y tags se calcula a partir de las listas completas del
juego. El conteo de categorías y su vocabulario se calculan después de excluir
`Steam Trading Cards`, `Family Sharing` y `Remote Play Together`.

### Exclusiones

- Trazabilidad, no usadas como predictores: `appid`, `name`.
- Variables excluidas por riesgo de fuga de información o por ser posteriores
  al lanzamiento: `peak_ccu`, reseñas y puntuaciones, recomendaciones, tiempo
  de juego, descuentos y cantidad de DLC.
- Categorías excluidas: `Steam Trading Cards`, `Family Sharing` y
  `Remote Play Together`.

La lista de exclusiones debe mantenerse alineada con la definición real de las
columnas del dataset. Excluir una variable no demuestra por sí mismo que todas
las variables restantes estuvieran disponibles antes del lanzamiento.

### Disponibilidad temporal de las features

El caso de uso se describe como predicción en preproducción. Para que esa
interpretación sea válida, cada feature debe corresponder a información que
estaba disponible en el momento de la predicción. En particular, precio, tags,
categorías y metadatos pueden cambiar después del lanzamiento. La disponibilidad
pre-lanzamiento de esas variables queda **pendiente de comprobar según la fecha
y procedencia de la captura**.

Hasta verificarlo, los resultados deben describirse como predicción a partir
de las variables del dataset, no necesariamente como una simulación fiel de lo
que se habría sabido antes del lanzamiento.

### Ajuste del preprocesamiento

Sea \(\mathbf{z}_i\) el vector de variables originales. El vector que recibe el
modelo es:

$$
\mathbf{x}_i=\phi_{\text{train}}(\mathbf{z}_i),
$$

donde \(\phi_{\text{train}}\) contiene transformaciones aprendidas solamente
con los datos de entrenamiento correspondientes. Esto incluye:

- vocabulario y columnas multi-hot;
- imputación de valores faltantes;
- escalamiento, si el modelo lo requiere.

Una categoría que aparezca solo en validación o test no debe crear una columna
nueva. Para evitar fuga en la validación cruzada, esas transformaciones deben
ajustarse dentro de cada fold mediante un pipeline; no basta con ajustarlas
una vez sobre todo train antes de generar los folds.

En el dataset existente, el vocabulario multi-hot se calcula sobre el conjunto
train completo y luego se usa para las filas de train y test. Si se aplica
validación cruzada después de esta etapa, el vocabulario ya habrá visto los
folds de validación. En la evaluación reportada, el vocabulario no se reajustó
dentro de cada fold; esta es una limitación metodológica que debe corregirse
en una evaluación futura.

Random Forest y XGBoost no suelen requerir escalamiento de las variables. El
modelo lineal Ridge sí puede beneficiarse del escalamiento; imputación,
escalamiento y cualquier otra transformación deben ajustarse solo con train y
dentro de cada fold durante la selección.

## 4. Partición de datos y prevención de fuga

La partición existente es aleatoria y reproducible:

$$
\mathcal{D}=\mathcal{D}_{\text{train}}\cup\mathcal{D}_{\text{test}},
\qquad
|\mathcal{D}_{\text{train}}|=100.684,\quad
|\mathcal{D}_{\text{test}}|=25.171.
$$

Se utilizó `test_size=0.2`, `random_state=42` y no se estratificó el target
continuo. El reporte registra **cero `appid` compartidos** entre train y test.

La ausencia de IDs compartidos no descarta reediciones, ports o juegos casi
idénticos en ambas particiones. Esa comprobación queda **pendiente**.

La evaluación ejecutada usó `KFold(n_splits=5, shuffle=True, random_state=42)`
solo sobre train para seleccionar hiperparámetros, con MAE original como
métrica principal. El conjunto test se reservó para la evaluación final y no
se usó para seleccionar hiperparámetros ni umbrales.

La partición aleatoria mide el rendimiento sobre juegos mezclados de la
distribución observada. Si el propósito principal es predecir juegos futuros,
una partición temporal podría representar mejor ese escenario; se recomienda
considerarla como evaluación adicional.

Se reportarán la media y la desviación estándar de las métricas entre folds.
La desviación estándar describe variación entre particiones, pero no es por sí
sola una medida completa de estabilidad ni un criterio suficiente para elegir
un modelo.

## 5. Parámetros, entrenamiento y función de costo

Los **parámetros** son valores aprendidos por el modelo, como umbrales de
división y valores de hojas en los árboles, o coeficientes en Ridge. Los
**hiperparámetros** se fijan antes de ajustar cada modelo y se seleccionan
mediante validación cruzada, por ejemplo número de árboles, profundidad y
tasa de aprendizaje.

El error residual de una predicción es:

$$
e_i=y_i-\hat y_i.
$$

Por ejemplo, si \(y_i=35.000\) y \(\hat y_i=12.000\), entonces
\(e_i=23.000\).

Para los modelos de regresión con pérdida cuadrática:

$$
\ell(y,\hat y)=(y-\hat y)^2,
$$

y el criterio de entrenamiento idealizado es:

$$
\hat f=\arg\min_{f\in\mathcal{F}}
\frac{1}{n_{\text{train}}}
\sum_{i\in\mathcal{I}_{\text{train}}}
\left(y_i-f(\mathbf{x}_i)\right)^2.
$$

La pérdida cuadrática no permite que errores positivos y negativos se cancelen
y penaliza con mayor intensidad los errores grandes. Por ello, la cola larga
del target puede influir mucho en el ajuste.

**Restricción de no negatividad.** Random Forest con hojas que predicen medias
de objetivos no negativos produce predicciones no negativas. XGBoost, al ser
un modelo aditivo, puede producir una salida negativa; si se desea garantizar
una predicción final no negativa, se aplica:

$$
\hat y_i=\max(0,\hat y_{i,\text{raw}}).
$$

Este mismo postprocesamiento debe utilizarse al calcular métricas y al evaluar
la regla de umbral. Debe distinguirse la salida cruda del modelo de la salida
postprocesada.

**Descenso del gradiente.** Es una idea general de optimización:

$$
\boldsymbol{\theta}\leftarrow
\boldsymbol{\theta}-\eta\nabla_{\boldsymbol{\theta}}J(\boldsymbol{\theta}).
$$

En una regresión lineal puede usarse un método de gradiente para ajustar
coeficientes. XGBoost construye árboles secuencialmente mediante una
optimización funcional basada en derivadas de la pérdida. Random Forest no
ajusta todos sus árboles con descenso de gradiente: los construye mediante
divisiones recursivas y promedia sus predicciones.

## 6. Modelos de referencia

Se recomienda evaluar las siguientes líneas base usando solo train para
estimar sus predicciones:

1. **Mediana de train:** \(\hat y_i=\operatorname{mediana}(y_{\text{train}})\).
   La mediana observada en train es **10.000**.
2. **Media de train:** \(\hat y_i=\bar y_{\text{train}}\). La media observada
   en train es aproximadamente **85.007,05**.
3. **Regresión Ridge** sobre las mismas features:

$$
\hat y_i=\beta_0+\sum_{j=1}^{p}\beta_jx_{ij},
\qquad
\min_{\beta}
\sum_{i\in\mathcal{I}_{\text{train}}}(y_i-\hat y_i)^2
+\alpha\sum_{j=1}^{p}\beta_j^2.
$$

La media es la predicción constante que minimiza el error cuadrático; la
mediana minimiza el error absoluto. Por tanto, cada modelo debe compararse con
la línea base adecuada a la métrica, no exigirse que supere la mediana en
todas las métricas.

En Ridge, \(\beta_j\) describe el cambio asociado a una unidad de la feature
transformada \(x_j\), manteniendo las demás constantes. Si las variables se
escalan, esa unidad es la escala transformada. Los coeficientes describen
asociaciones del modelo y no efectos causales.

## 7. Random Forest Regressor

La predicción de un bosque de \(T\) árboles es el promedio de las predicciones
de los árboles:

$$
\boxed{
\hat y(\mathbf{x})=\frac{1}{T}\sum_{t=1}^{T}h_t(\mathbf{x})
}
$$

En la configuración usual de Random Forest:

- Cada árbol se ajusta con una muestra bootstrap del conjunto de entrenamiento
  (`bootstrap=True`).
- En cada división se evalúa un subconjunto aleatorio de features, controlado
  por `max_features`.
- Los árboles se construyen recursivamente. Con criterio de error cuadrático,
  se elige la división que minimiza la suma de errores cuadráticos dentro de
  sus dos grupos:

$$
\min_{j,s}\left[
\sum_{i\in R_L(j,s)}(y_i-\bar y_{R_L})^2+
\sum_{i\in R_R(j,s)}(y_i-\bar y_{R_R})^2
\right].
$$

- Cada hoja predice la media de los objetivos de entrenamiento que llegan a
  ella.

El bootstrap y la selección aleatoria de features decorrelacionan los árboles;
promediarlos suele reducir varianza en comparación con un árbol individual.
El resultado depende de los hiperparámetros y de los datos, por lo que esta
propiedad no garantiza por sí sola un mejor rendimiento en test.

## 8. XGBoost Regressor

XGBoost construye árboles de forma secuencial. En una formulación aditiva:

$$
\hat y_i^{(0)}=\text{base\_score},
\qquad
\hat y_i^{(t)}=\hat y_i^{(t-1)}+\eta f_t(\mathbf{x}_i),
$$

y después de \(T\) árboles:

$$
\hat y_i=\text{base\_score}+
\sum_{t=1}^{T}\eta f_t(\mathbf{x}_i),
$$

donde \(\eta\) es la tasa de aprendizaje y \(f_t\) es el árbol añadido en la
iteración \(t\).

Una forma estándar del objetivo regularizado es:

$$
\mathcal{J}=
\sum_{i=1}^{n_{\text{train}}}\ell(y_i,\hat y_i)
+\sum_{t=1}^{T}\Omega(f_t),
$$

con una penalización para cada árbol:

$$
\Omega(f_t)=\gamma T_t+
\frac{1}{2}\lambda\sum_{j=1}^{T_t}w_j^2,
$$

donde \(T_t\) es el número de hojas del árbol \(t\), \(w_j\) es el peso de la
hoja \(j\), \(\gamma\) penaliza añadir hojas y \(\lambda\) regulariza sus
pesos.

En cada iteración se aproxima la pérdida alrededor de las predicciones actuales
mediante una expansión de segundo orden. Definiendo:

$$
g_i=\frac{\partial\ell(y_i,\hat y_i)}{\partial\hat y_i},
\qquad
h_i=\frac{\partial^2\ell(y_i,\hat y_i)}{\partial\hat y_i^2},
$$

el peso óptimo regularizado de una hoja \(j\), bajo esa aproximación, es:

$$
w_j^*=-\frac{G_j}{H_j+\lambda},
\qquad
G_j=\sum_{i\in I_j}g_i,\quad
H_j=\sum_{i\in I_j}h_i,
$$

donde \(I_j\) es el conjunto de observaciones que llega a esa hoja.

Con error cuadrático, los gradientes dependen de los residuos de las
predicciones actuales; por eso los árboles añadidos pueden corregir errores
anteriores. XGBoost no es simplemente descenso de gradiente ordinario sobre
coeficientes: la optimización agrega árboles y usa información de gradiente y
curvatura para evaluar hojas y divisiones.

**Diferencia principal:** Random Forest promedia árboles ajustados de manera
independiente (bagging), lo que suele reducir varianza. XGBoost agrega árboles
secuencialmente (boosting), optimizando una función de pérdida regularizada;
su efecto sobre sesgo y varianza depende de la configuración y no es una
garantía automática.

### Hiperparámetros finales seleccionados

| Modelo | Hiperparámetros seleccionados |
|---|---|
| Random Forest | `n_estimators=400`, `max_depth=30`, `max_features=0.7`; `min_samples_leaf=5` (A) o `2` (B) |
| XGBoost | `n_estimators=400`, `max_depth=8`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`, `reg_lambda=10` |

La búsqueda ejecutada probó tres configuraciones por cada modelo. El criterio
de selección del ganador general fue el menor MAE promedio en CV de cinco
folds, medido en escala original:

| Modelo/variante | Configuración seleccionada | MAE CV |
|---|---|---:|
| Random Forest raw (A) | 400 árboles, `max_depth=30`, `min_samples_leaf=5`, `max_features=0.7` | 91.619,08 |
| Random Forest log (B) | 400 árboles, `max_depth=30`, `min_samples_leaf=2`, `max_features=0.7` | 68.122,58 |
| XGBoost raw (A) | 400 árboles, `max_depth=8`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`, `reg_lambda=10` | 97.207,11 |
| XGBoost log (B) | 400 árboles, `max_depth=8`, `learning_rate=0.03`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=10`, `reg_lambda=10` | **67.577,05** |

La búsqueda y los parámetros quedan documentados en
`src/reports/models/success_models.json` y en los artefactos locales de
`src/results/pendientes/`.

## 9. Transformación logarítmica (variantes evaluadas)

Dada la asimetría del target, se pueden comparar dos variantes:

- **Variante A:** ajustar el modelo con \(y\).
- **Variante B:** ajustar el modelo con \(y'=\log(1+y)\) y volver a la escala
  original con:

$$
\hat y_i=\max\left(0,\exp(\hat y_i')-1\right).
$$

La transformación se aplica al target durante el entrenamiento, no modifica
el CSV original. Las métricas en escala original se calculan después de
invertir la transformación.

Con error cuadrático en escala logarítmica, el ajuste apunta a la media
condicional de \(\log(1+y)\). La inversión con `expm1` no equivale, en general,
ni a la media ni a la mediana condicional de \(y\). La transformación puede
reducir la influencia de valores extremos, pero su efecto en la subestimación
de valores altos y en las métricas debe medirse con validación. No se debe
afirmar de antemano que necesariamente mejora o empeora una métrica específica.

## 10. Evaluación de la regresión

Las métricas finales se calculan una sola vez sobre test, en escala original.
Si se ajustó con logaritmo, primero se invierte la transformación y se aplica
el recorte a cero definido en la sección 5.

$$
\operatorname{MAE}=
\frac{1}{n_{\text{test}}}\sum_{i=1}^{n_{\text{test}}}|y_i-\hat y_i|,
\qquad
\operatorname{RMSE}=
\sqrt{\frac{1}{n_{\text{test}}}
\sum_{i=1}^{n_{\text{test}}}(y_i-\hat y_i)^2}.
$$

$$
R^2=
1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y_{\text{test}})^2},
\qquad
\operatorname{MedAE}=
\operatorname{mediana}_i(|y_i-\hat y_i|).
$$

En escala logarítmica:

$$
\operatorname{RMSE}_{\log}=
\sqrt{\frac{1}{n_{\text{test}}}
\sum_i\left[\log(1+y_i)-\log(1+\hat y_i)\right]^2},
$$

$$
\operatorname{MAE}_{\log}=
\frac{1}{n_{\text{test}}}
\sum_i\left|\log(1+y_i)-\log(1+\hat y_i)\right|.
$$

Unos pocos juegos con valores muy altos pueden dominar el RMSE en escala
original. MAE, MedAE y métricas logarítmicas complementan esa lectura, pero
responden a criterios distintos y no deben interpretarse como intercambiables.

Las métricas calculadas sobre test están en la siguiente tabla. Para la
variante B se aplicó `expm1` y después el recorte a cero antes de calcularlas.

| Modelo | MAE | RMSE | \(R^2\) | MedAE | RMSE log | MAE log |
|---|---:|---:|---:|---:|---:|---:|
| Constante: mediana | 71.648,33 | 1.227.517,17 | -0,0031 | 0,00 | 4,2388 | 2,2567 |
| Constante: media | 126.436,32 | 1.225.669,71 | -0,0000 | 75.007,05 | 5,3286 | 3,7091 |
| Ridge (log) | 128.704,05 | 4.128.134,75 | -10,3443 | 8.889,30 | 2,5615 | 1,9642 |
| Random Forest (A) | 83.486,69 | 1.121.916,27 | 0,1621 | 6.415,78 | 2,8075 | 1,5707 |
| Random Forest (B) | 60.866,27 | 1.169.401,19 | 0,0897 | 3.412,85 | 1,7522 | 0,9499 |
| XGBoost (A) | 86.532,39 | 1.096.982,43 | 0,1989 | 9.115,36 | 4,0603 | 2,5300 |
| XGBoost (B) | **59.126,33** | **1.085.764,10** | **0,2152** | **3.343,14** | **1,7284** | **0,9472** |

XGBoost (B) obtuvo el menor MAE CV y el menor MAE de test de los modelos
entrenados; también logró el mejor \(R^2\), RMSE y métricas logarítmicas de
test. Random Forest (B) tuvo un MedAE ligeramente mayor.

La validación cruzada de cinco folds (media ± desviación estándar muestral)
para los modelos se resume aquí:

| Modelo | MAE | RMSE | \(R^2\) | MedAE | RMSE log | MAE log |
|---|---:|---:|---:|---:|---:|---:|
| Constante: mediana | 78.906,15 ± 7.133,53 | 1.264.857,90 ± 264.375,25 | -0,0038 ± 0,0011 | 0,00 ± 0,00 | 4,2269 ± 0,0364 | 2,2520 ± 0,0319 |
| Constante: media | 133.252,71 ± 5.633,41 | 1.262.620,04 ± 264.500,35 | -0,00004 ± 0,00002 | 75.007,05 ± 1.781,58 | 5,3082 ± 0,0437 | 3,6910 ± 0,0391 |
| Ridge (log) | 140.766,53 ± 42.663,18 | 5.837.632,91 ± 6.512.062,93 | -66,9232 ± 126,7373 | 8.966,08 ± 112,15 | 2,5730 ± 0,0126 | 1,9728 ± 0,0068 |
| Random Forest (A) | 91.619,08 ± 4.188,35 | 1.170.434,26 ± 263.985,57 | 0,1438 ± 0,0702 | 6.521,22 ± 211,69 | 2,8046 ± 0,0222 | 1,5757 ± 0,0150 |
| Random Forest (B) | 68.122,58 ± 7.540,09 | 1.214.420,22 ± 275.568,76 | 0,0798 ± 0,0422 | 3.481,07 ± 119,40 | 1,7683 ± 0,0110 | 0,9589 ± 0,0081 |
| XGBoost (A) | 97.207,11 ± 5.347,89 | 1.189.286,41 ± 250.146,24 | 0,1125 ± 0,0411 | 9.452,03 ± 280,77 | 4,0197 ± 0,0579 | 2,4937 ± 0,0461 |
| XGBoost (B) | **67.577,05 ± 6.901,04** | 1.187.521,61 ± 274.628,15 | 0,1208 ± 0,0536 | **3.432,52 ± 106,41** | **1,7487 ± 0,0148** | **0,9591 ± 0,0092** |

Las métricas por segmento y los fold-level scores están en
`src/results/pendientes/test_metrics_by_segment.csv` y
`src/results/pendientes/cv_fold_metrics.csv`.

## 11. Regla del umbral de 20.000 y evaluación como clasificación

El umbral no forma parte del objetivo de entrenamiento. Se define como una
regla de decisión posterior aplicada a la predicción continua:

$$
g(\hat y_i)=
\begin{cases}
1,&\hat y_i\geq20.000,\\
0,&\hat y_i<20.000.
\end{cases}
$$

La clase de referencia para esta evaluación se deriva del target disponible:

$$
g(y_i)=\mathbb{1}[y_i\geq20.000].
$$

Esto evalúa si el punto medio estimado supera el umbral; no demuestra por sí
solo que los propietarios reales hayan superado 20.000.

La comprobación de los 125.855 intervalos originales no encontró rangos que
crucen estrictamente el umbral; el intervalo 20.000–50.000 tiene punto medio
35.000. Sí hay rangos cuyo límite coincide con 20.000, lo que debe distinguirse
de un cruce estricto.

En test, 5.045 juegos (20,04 %) alcanzan el umbral. La evaluación calculada
para los modelos es:

| Modelo | TP | FN | FP | TN | Precisión | Recall | F1 | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Constante: mediana | 0 | 5.045 | 0 | 20.126 | 0,0000 | 0,0000 | 0,0000 | 0,7996 | — | — |
| Constante: media | 5.045 | 0 | 20.126 | 0 | 0,2004 | 1,0000 | 0,3339 | 0,2004 | — | — |
| Ridge (log) | 3.178 | 1.867 | 3.853 | 16.273 | 0,4520 | 0,6299 | 0,5263 | 0,7728 | 0,8194 | 0,5441 |
| Random Forest (A) | 4.547 | 498 | 6.049 | 14.077 | 0,4291 | 0,9013 | 0,5814 | 0,7399 | 0,8884 | 0,6513 |
| Random Forest (B) | 3.450 | 1.595 | 2.083 | 18.043 | 0,6235 | 0,6838 | 0,6523 | 0,8539 | 0,9054 | 0,7270 |
| XGBoost (A) | 4.310 | 735 | 6.382 | 13.744 | 0,4031 | 0,8543 | 0,5478 | 0,7173 | 0,8446 | 0,6022 |
| XGBoost (B) | 3.331 | 1.714 | 1.860 | 18.266 | 0,6417 | 0,6603 | 0,6508 | **0,8580** | **0,9057** | **0,7319** |

Las constantes no tienen ROC-AUC ni PR-AUC porque sus puntuaciones no varían.
Las métricas completas también están en `src/results/pendientes/threshold_metrics.csv`.

La matriz de confusión, con clase positiva igual a “el punto medio alcanza
20.000”, es:

|  | Predicho 1 | Predicho 0 |
|---|---:|---:|
| Real 1 | TP | FN |
| Real 0 | FP | TN |

- **FN:** el punto medio real supera el umbral, pero el modelo predice menos.
  Puede hacer que una idea potencialmente viable no se considere.
- **FP:** el punto medio real está bajo el umbral, pero el modelo predice más.
  Puede generar expectativas comerciales no respaldadas por el target.

La prioridad entre recall y precisión depende del uso de la aplicación y debe
justificarse. Las métricas son:

$$
\operatorname{Precisión}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN},
$$

$$
F1=2\cdot
\frac{\operatorname{Precisión}\cdot\operatorname{Recall}}
{\operatorname{Precisión}+\operatorname{Recall}}.
$$

Si la clase está desbalanceada, la accuracy puede ser engañosa. Se recomienda
reportar precisión, recall, F1 y matriz de confusión; si se informa accuracy,
debe acompañarse de esa advertencia.

La predicción continua puede usarse como puntuación para calcular ROC-AUC y
curva precisión-recall. AUC evalúa la ordenación de ejemplos a través de
distintos umbrales, no el rendimiento de la decisión en el umbral fijo de
20.000. Si se cambia el umbral de decisión, debe seleccionarse usando
validación (o validación cruzada) dentro de train, nunca el test. El test se
reserva para la evaluación final una vez fijado ese umbral.

## 12. Interpretabilidad

La importancia por impureza de Random Forest puede favorecer variables
continuas o variables con muchas categorías. Se recomienda complementarla con
importancia por permutación o SHAP, calculados sobre datos que no se hayan
usado para ajustar el modelo cuando corresponda.

Estas importancias describen cómo se relacionan las variables con las
predicciones del modelo; no prueban causalidad.

## 13. Defensa del modelo

| Pregunta | Respuesta |
|---|---|
| ¿Qué variables utiliza? | Variables comerciales, técnicas, de idioma, temporales y de diseño, incluidas las columnas multi-hot descritas en la sección 3. |
| ¿Qué intenta predecir? | `owners_midpoint`, el punto medio del intervalo de propietarios estimados de Steam. |
| ¿Cómo se entrenó? | Split 80/20 con semilla 42; cinco folds en train; variantes raw/log para RF y XGBoost y Ridge log. |
| ¿Cómo se evaluó? | Métricas en escala original, métricas por segmentos, líneas base y regla de 20.000; test reservado para evaluación final. |
| ¿Qué modelo ganó? | XGBoost (B), por menor MAE de CV (67.577,05); también obtuvo el menor MAE de test (59.126,33). |
| ¿Dónde falla más? | El error absoluto es mucho mayor en juegos con \(y\geq20.000\); las métricas segmentadas están en `src/results/pendientes/test_metrics_by_segment.csv`. |
| ¿Cuáles son las limitaciones? | Target aproximado por punto medio, posibles intervalos amplios, disponibilidad temporal por verificar de algunas features, evaluación no causal y generalización temporal no garantizada por un split aleatorio. |

La tabla resume la metodología y los resultados de la ejecución evaluada.

## 14. Limitaciones

1. El target es el punto medio de un intervalo, no el número exacto de
   propietarios.
2. Los intervalos amplios aportan un target menos preciso; su incertidumbre no
   equivale a un error mínimo garantizado.
3. El modelo predice adopción estimada en Steam, no ingresos ni rentabilidad.
   El umbral de 20.000 es una regla operativa, no una prueba de rentabilidad o
   de atractivo de mercado.
4. La disponibilidad pre-lanzamiento de tags, precio, categorías y otros
   metadatos debe verificarse según la fecha de captura.
5. El vocabulario multi-hot usado en esta evaluación se calculó antes de la
   validación cruzada sobre train completo; los folds no reajustan ese
   vocabulario. Esto puede producir fuga leve de información transductiva y
   debe corregirse en una evaluación futura con un pipeline por fold.
6. La comprobación de IDs compartidos no descarta reediciones ni juegos casi
   duplicados entre particiones.
7. Un split aleatorio no garantiza una estimación de rendimiento en juegos
   futuros; una evaluación temporal puede ser necesaria.
8. La importancia de variables no implica causalidad.
9. Los resultados corresponden a la ejecución registrada en los reportes
   citados, no demuestran causalidad ni generalización temporal.
10. El script de resultados refitió los modelos con parámetros de train-CV.
    El resumen de ejecución advierte que los metadatos de features del modelo
    Random Forest serializado difieren de los del CSV actual
    (`genre_episodic` frente a `genre_tutorial`); por ello, esta evaluación
    refit no debe confundirse con una predicción obtenida directamente del
    artefacto serializado anterior.

## 15. Resumen

| Elemento | Definición |
|---|---|
| Entrada | \(\mathbf{x}_i=\phi_{\text{train}}(\mathbf{z}_i)\in\mathbb{R}^{p}\) |
| Variable objetivo | \(y_i=(L_i+U_i)/2\), punto medio de `estimated_owners` |
| Modelos evaluados | Random Forest, XGBoost y Ridge |
| Líneas base | Predicción constante con mediana y media; Ridge |
| Pérdida principal | Error cuadrático, con evaluación adicional mediante MAE, MedAE y métricas log |
| Selección | MAE original promedio en validación cruzada de cinco folds solo en train |
| Evaluación final | Test reservado, en escala original y con el postprocesamiento definido |
| Regla de aplicación | \(\hat y\geq20.000\), evaluada después de la regresión |
| Ganador de CV | XGBoost log: MAE 67.577,05 ± 6.901,04 |
| Resultado de test | XGBoost log: MAE 59.126,33; RMSE 1.085.764,10; \(R^2=0,2152\) |
| Estado | Evaluación ejecutada; ver tablas y advertencias de la sección 10 |

## Referencias internas de los datos descriptivos

- `src/reports/features/model_dataset.json`: tamaño del dataset, definición del
  target, features, exclusiones y vocabulario multi-hot.
- `src/reports/features/split.json`: tamaños de train y test, configuración de
  la partición, estadísticas del target y verificación de `appid`.
- `src/reports/eda/resumen_eda.md`: estadísticas descriptivas de train y
  análisis exploratorio.

## 16. Análisis complementario de reseñas

Este análisis es descriptivo y distinto del modelo de regresión de
`owners_midpoint`. Sobre 715.122 reseñas limpias sin balancear, hay 130.859
negativas (18,30 %) y 584.263 positivas (81,70 %). Para limitar el coste, la
integración BERTopic usa una muestra proporcional por clase dentro de cada
juego, limitada a 50 reseñas por juego: 36.848 reseñas de 743 juegos.

Se probaron `min_cluster_size` 50, 100 y 200 y se eligió 50 por mayor NPMI
(0,3139), con 145 temas. Antes de reducir outliers, el 43,58 % fue asignado a
outlier; la estrategia de embeddings reasignó 16.059 reseñas. La integración
por género cruza los temas con `voted_up` observado; sus resultados son
asociaciones descriptivas y no evidencia causal.

Las tasas reales por género, tablas de integración, palabras/ejemplos por tema
y plantilla para tres revisores están en `src/results/pendientes_resenas/`.
La plantilla Excel incluye validación de etiquetas y fórmulas de acuerdo
inter-revisor; cada persona debe etiquetar su columna independientemente.
