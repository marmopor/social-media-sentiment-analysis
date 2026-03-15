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
  El dataset contiene 732 publicaciones y 15 variables.
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
Accuracy, F1-score, Precision, Recall y Matriz de confusión.

## Resultados ENTREGA2: 
### Métricas por modelo

| Modelo | Nº Parámetros | Acc Train | Acc Val | Acc Test | F1 Train | F1 Val | F1 Test | Prec Train | Prec Val | Prec Test | Rec Train | Rec Val | Rec Test |
|--------|:-------------:|:---------:|:-------:|:--------:|:--------:|:------:|:-------:|:----------:|:--------:|:-----------:|:-------:|:---------:|:---------:|
| Logistic Regression | N/A | 0.8555 | 0.7182 | 0.7455 | 0.7094 | 0.4492 | 0.5178 | 0.8945 | 0.5149 | 0.8306 | 0.6686 | 0.4521 | 0.4999 |
| Random Forest | 5,236 | 0.8105 | 0.6818 | 0.7455 | 0.6839 | 0.3563 | 0.5068 | 0.9201 | 0.5566 | 0.8779 | 0.6174 | 0.3846 | 0.4846 |
| Simple MLP | 7,043 | 0.6797 | 0.5545 | 0.5455 | 0.6444 | 0.4988 | 0.5055 | 0.6421 | 0.5262 | 0.5496 | 0.7594 | 0.5468 | 0.5579 |

### Descripción de modelos

#### 1. Modelo Lineal - `02_linear_model.ipynb`
- **Algoritmo:** Logistic Regression (multinomial, solver LBFGS)
- **Features:** TF-IDF (500 tokens, unigrams+bigrams) + 6 features numéricas
- **Split:** 70% train / 15% val / 15% test

#### 2. Modelo ML - `03_ml_model.ipynb`
- **Algoritmo:** Random Forest (100 árboles, profundidad máx. 10)
- **Features:** TF-IDF (500 tokens) + 6 features numéricas
- **Parámetros:** Número total de nodos en todos los árboles
- **Split:** 70% train / 15% val / 15% test

#### 3. Red Neuronal Simple - `04_neural_network.ipynb`
- **Arquitectura:** `Linear(106, 64) -> ReLU -> Linear(64, 3)`
- **Features:** TF-IDF (100 tokens) + 6 features numéricas -> 106 features de entrada
- **Optimizador:** Adam (lr=1e-3), **Loss:** CrossEntropy
- **Épocas:** 100, **Batch size:** 32
- **Modelo definido en:** `models/simple_nn.py`
- **Split:** 70% train / 15% val / 15% test

## Preprocesado común
1. **Limpieza:** strip de espacios en texto y etiquetas de sentimiento
2. **Agrupación de clases:** los 191 sentimientos finos se mapean a 3 clases (Positivo / Neutro / Negativo)
3. **TF-IDF:** ajustado únicamente sobre el conjunto de entrenamiento
4. **Features numéricas:** Retweets, Likes, Year, Month, Day, Hour

## Resultados ENTREGA3:    
### Métricas por modelo

| Modelo | Nº Parámetros | Acc Train | Acc Val | Acc Test | F1 Train | F1 Val | F1 Test |
|--------|:-------------:|:---------:|:-------:|:--------:|:--------:|:------:|:-------:|
| Deep MLP | 22,531 | 0.6367 | 0.4727 | 0.5455 | 0.5830 | 0.3840 | 0.5073 |
| TextCNN | 74,851 | 0.9766 | 0.5636 | 0.6636 | 0.9694 | 0.5063 | 0.5914 |
| BiLSTM | 86,275 | 0.6641 | 0.6545 | 0.6364 | 0.3141 | 0.2877 | 0.2593 ||

### Conclusiones de los Modelos Complejos 

#### 1.Deep MLP
- **Análisis:** Sus métricas de Test son un Accuracy de 54.55% y un F1-Score de 0.5073, demostrando un aprendizaje más real y robusto y no solo una memorización del set de entrenamiento.

- **Convergencia:** Las curvas de entrenamiento muestran una convergencia progresiva y relativamente estable. Las curvas de entrenamiento convergen más tempranamente a los verdaderos límites del modelo (el early stopping corta en la época 33). Esto indica que es capaz de capturar patrones relevantes sin degradar excesivamente la capacidad de generalización.
  
- **Predominancia por Plataforma:** Al analizar el gráfico post-entrenamiento por plataformas sociales interactuadas, presenciamos una distribución de predicciones realista: realiza pronósticos coherentes para cada clase (Positivo como principal, y Negativo/Neutro en magnitudes equilibradas).


#### 2.TextCNN
- **Análisis:**. La CNN logra extraer fuertemente las características espaciales de los embeddings de texto (1D).  Consigue las mejores métricas del conjunto: 66.4% de Accuracy y 0.5914 de F1-Macro en Test. Sigue presentando sobreajuste en Train (97%), pero es el que mejor generaliza a datos no vistos.
- **Convergencia:** La curva de validación no es muy errática y detiene su entrenamiento gracias al Max Pooling temporal que aglomera los *features* más representativos sin ruido innecesario, utilizando Early Stopping en la época 91.
- **Predominancia por Plataforma:** A diferencia de iteraciones iniciales, el análisis visual confirma que el modelo ya no predice exclusivamente Neutro en todas las plataformas de forma general. Gracias al correcto rebalanceo de clases y los pesos en la CrossEntropy, ahora identifica correctamente y de forma mayoritaria los sentimientos Positivos y Negativos en las distintas Redes Sociales, adecuándose a los patrones del dataset real.

#### 3.BiLSTM
- **Análisis:** El modelo BiLSTM obtiene un accuracy de 63.64% en test, aparentemente cercano al del MLP. Sin embargo, el análisis del clasificación muestra que el modelo predice casi exclusivamente la clase Positivo.

Esto se refleja en el F1-macro bajo (0.2593) y en métricas negativas como el MCC (-0.0568) y Kappa (-0.0152), lo que indica una capacidad real de clasificación muy limitada.

- **Convergencia:** Las curvas de entrenamiento son relativamente estables y muestran una convergencia temprana alrededor de la época 30–40. No obstante, esta estabilidad se debe en gran medida a que el modelo aprende una estrategia basada en la clase mayoritaria.

- **Predominancia por Plataforma:** El modelo LSTM predice mayoritariamente sentimiento positivo en todas las plataformas analizadas. Esto podría indicar que existe un sesgo de nuestro modelo hacia la clase positiva.

#### 4. Deep MLP - `05_deep_mlp.ipynb`
- **Algoritmo:** Multi-Layer Perceptron Profundo para clasificación multiclase
- **Arquitectura:** Varias capas lineales intercaladas por BatchNorm, ReLU y Dropout (0.5 y 0.2 inicial) para mejorar la estabilidad y erradicar el sobreajuste. Se redujo la dimensionalidad oculta a 128 y 64 neuronas.
- **Features:** TF-IDF (100 tokens) + 6 numéricas -> 106 dimensional vector
- **Optimizador:** Adam (lr=1e-3, weight_decay=1e-4) + ReduceLROnPlateau, **Loss:** CrossEntropy con pesos de clase para compensar el desbalance del dataset
- **Épocas:** 150 (Early Stopping), **Batch size:** 32
- **Modelo definido en:** `models/deep_mlp.py`
- **Split:** 70% train / 15% val / 15% test


#### 5. TextCNN - `06_cnn_text.ipynb`
- **Algoritmo:** Red Neuronal Convolucional 1D 
- **Arquitectura:** Embedding (dim=64), Conv1D paralelas (kernels 2, 3, 4), MaxPool temporal, Dropout (0.5), FC
- **Features:** Texto tokenizado -> secuencias de índices (vocabulario de ~3000 palabras)
- **Optimizador:** Adam (lr=5e-4, weight_decay=1e-3) + ReduceLROnPlateau, **Loss:** CrossEntropy con pesos de clase
- **Épocas:** 150 (Early Stopping), **Seq len:** 50, **Batch size:** 32
- **Modelo definido en:** `models/cnn_text.py`
- **Split:** 70% train / 15% val / 15% test

#### 6. BiLSTM - `07_lstm.ipynb`
- **Algoritmo:** Red neuronal recurrente Bidirectional LSTM para modelado secuencial de texto
- **Arquitectura:** `Embedding(vocab, 128) -> BiLSTM(128, 2 capas) -> Concat hidden -> Dropout(0.5) -> Linear(256, 64) -> ReLU -> Dropout(0.5) -> Linear(64, 3) `
- **Features:** Texto tokenizado -> secuencias de índices (vocabulario de aprox. 3000 palabras)
- **Bidireccional:** Captura contexto pasado y futuro de cada palabra
- **Optimizador:** Adam (lr=1e-3) + ReduceLROnPlateau, **Loss:** CrossEntropy con pesos de clase
- **Gradient clipping:** 1.0 para estabilidad del LSTM
- **Épocas:** 100, **Seq len:** 50, **Batch size:** 32
- **Modelo definido en:** `models/lstm_model.py`
- **Split:** 70% train / 15% val / 15% test

### Métricas detalladas:
Cada notebook de modelo complejo incluye:
- **Accuracy**, **Precision (macro)**, **Recall (macro)**, **F1-Score (macro y weighted)**
- **Cohen's Kappa**, **Matthews Correlation Coefficient (MCC)**
- **AUC-ROC (One-vs-Rest, macro)**
- **Classification Report** completo por clase
- **Matrices de confusión** para Train, Validación y Test
- **Curvas ROC** por clase (Test)
- **Curvas de entrenamiento** (Loss y Accuracy)

## Preprocesado común
1. **Limpieza:** strip de espacios en texto y etiquetas de sentimiento
2. **Agrupación de clases:** los 191 sentimientos finos se mapean a 3 clases (Positivo / Neutro / Negativo)
3. **TF-IDF:** ajustado únicamente sobre el conjunto de entrenamiento (para modelos 1–4)
4. **Tokenización:** vocabulario construido solo sobre train, con padding (para modelos 5–6)
5. **Features numéricas:** Retweets, Likes, Year, Month, Day, Hour (para modelos 1–4)
