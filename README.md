# Clasificación de Sentimiento en Redes Sociales
Proyecto para la Clasificación de emociones en posts de Redes Sociales usando técnicas NLP.

## DEFINICIÓN DEL PROBLEMA:
Clasificación supervisada multiclase (3 clases: Positive, Negative, Neutral) a partir del texto de posts en redes sociales.

## DATA:
- Fuente: 
  Nombre: sentimentdataset
  Autor: eshummalik
  Plataforma: Kaggle

- Descripción:
  El dataset es un conjunto de datos diseñado para el análisis de sentimiento/emociones y posibles patrones de engagement en redes sociales.
  Éste contiene publicaciones textuales etiquetadas con categorías emocionales, junto con métricas de interacción como número de likes, retweets y marcas temporales.
  El dataset es adecuado para tareas de análisis de sentimiento mediante el procesado de Lenguaje Natural (NLP) para el estudio del comportamiento en las Redes Sociales.

- Número de muestras:
  El dataset contiene 732 publicaciones y 15 variables
  Tras el análisis exploratorio no se detectaron valores nulos.

- Definición de variables principales:
  Para el problema de clasificación se utilizan:
  Text : contenido textual del post (input).
  Sentiment : etiqueta emocional.

  Dado que el dataset contiene múltiples emociones específicas (191 clases distintas), se decidió agruparlas en tres   categorías generales: Positive, Negative y Neutral.
  Esta decisión se tomó debido al tamaño reducido del dataset (732 muestras)y al fuerte desbalanceo entre clases, con el objetivo de obtener un modelo más estable e interpretable.

- Otras variables disponibles:
  El dataset incluye además variables adicionales que podrían ser utilizadas para análisis complementarios:
  
  Platform : plataforma social (Twitter, Instagram, etc.)
  Likes : número de likes
  Retweets : número de compartidos
  User_Followers : número de seguidores del usuario
  Engagement : puntuación total de interacción
  Country : país de origen
  Timestamp : fecha y hora de la publicación
  Hashtags : hashtags utilizados

## DEFINCIÓN DE MÉTRICAS:
Accuracy, Precision, Recall, F1-score Y Matriz de confusión.

## Resultados ENTREGA2: 
### Métricas por modelo

| Modelo | Nº Parámetros | Acc Train | Acc Val | Acc Test | F1 Train | F1 Val | F1 Test |
|--------|:-------------:|:---------:|:-------:|:--------:|:--------:|:------:|:-------:|
| Logistic Regression | N/A | - | - | - | - | - | - |
| Random Forest | - | - | - | - | - | - | - |
| Simple MLP | - | - | - | - | - | - | - |

### Descripción de modelos

#### 1. Modelo Lineal — `02_linear_model.ipynb`
- **Algoritmo:** Logistic Regression (multinomial, solver LBFGS)
- **Features:** TF-IDF (500 tokens, unigrams+bigrams) + 6 features numéricas
- **Split:** 70% train / 15% val / 15% test

#### 2. Modelo ML — `03_ml_model.ipynb`
- **Algoritmo:** Random Forest (100 árboles, profundidad máx. 10)
- **Features:** TF-IDF (500 tokens) + 6 features numéricas
- **Parámetros:** Número total de nodos en todos los árboles
- **Split:** 70% train / 15% val / 15% test

#### 3. Red Neuronal Simple — `04_neural_network.ipynb`
- **Arquitectura:** `Linear(106, 64) → ReLU → Linear(64, 3)`
- **Features:** TF-IDF (100 tokens) + 6 features numéricas → 106 features de entrada
- **Optimizador:** Adam (lr=1e-3), **Loss:** CrossEntropy
- **Épocas:** 100, **Batch size:** 32
- **Modelo definido en:** `models/simple_nn.py`
- **Split:** 70% train / 15% val / 15% test

## Preprocesado común
1. **Limpieza:** strip de espacios en texto y etiquetas de sentimiento
2. **Agrupación de clases:** los 191 sentimientos finos se mapean a 3 clases (Positivo / Neutro / Negativo)
3. **TF-IDF:** ajustado únicamente sobre el conjunto de entrenamiento
4. **Features numéricas:** Retweets, Likes, Year, Month, Day, Hour
