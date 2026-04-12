# PPS Quantization Project

Proyecto para:
- Extraer caracteristicas del dataset Fingers.
- Entrenar un MLP para clasificar cantidad de dedos levantados (0 a 5).
- Comparar cuantizacion simetrica signed post-entrenamiento.

## Estructura

- `Fuentes/`: modulos auxiliares.
- `Data_Sets/`: CSV de train/test, resultados de comparacion de error y accuracy, datos de normalizacion, y metadata.
- `Imagenes/Fingers/`: imagenes originales del dataset.
- `Models/`: modelo base y modelos cuantizados (`.tflite`).
- `Test/Test_Fingers.ipynb`: notebook principal del flujo completo.
- `requirements.txt`: dependencias para ejecutar notebook y scripts.
- `setup.py`: empaquetado del modulo `Fuentes`.

## Requisitos

- Python 3.10 o superior.
- Entorno virtual recomendado.

## Instalacion

1. Crear y activar entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. (Opcional) Instalar el paquete local `Fuentes`:

```bash
pip install -e .
```

## Descarga del dataset de imagenes (obligatorio)

Este repositorio no incluye todas las imagenes del dataset para evitar un repositorio demasiado pesado.

Fuente oficial:
- https://www.kaggle.com/koryakinp/fingers

Pasos sugeridos:
1. Descargar el dataset desde Kaggle (ZIP).
2. Extraer el contenido localmente.
3. Copiar las carpetas de imagenes dentro de `Imagenes/Fingers/` respetando la estructura de train/test por clase (`0` a `5`).

La estructura esperada es:
- `Imagenes/Fingers/train/0 ... 5`
- `Imagenes/Fingers/test/0 ... 5`

Si no estan estas carpetas con imagenes, el notebook no podra regenerar correctamente los CSV.

## Uso rapido

### Opcion A: Notebook (recomendado)

Abrir y ejecutar en orden:
- `Test/Test_Fingers.ipynb`

Flujo del notebook:
1. Visualizacion de una imagen y propiedades.
2. Generacion de CSV train/test desde imagenes.
3. Entrenamiento MLP y guardado del modelo base.
4. Cuantizacion simetrica signed y comparacion de metricas:
   - Pipeline con in/out dequantized.
   - Pipeline con in/out quantized.
5. Export de modelos int8 reales a TFLite.

## Pipeline completo de `Test/Test_Fingers.ipynb` 

El notebook esta dividido en dos bloques para evitar recalculo innecesario:

### PARTE 1 - PREPARACION (ejecutar una vez)

1. **Inspeccion inicial del dataset**
  - Carga una imagen ejemplo y muestra propiedades de `regionprops`.

2. **Extraccion de caracteristicas y generacion de CSV base**
  - Recorre `Imagenes/Fingers/train` y `Imagenes/Fingers/test`.
  - Extrae features geometricas normalizadas.
  - Genera:
    - `Data_Sets/fingers_train.csv`
    - `Data_Sets/fingers_test.csv`

3. **Preparacion de datos para entrenamiento**
  - Carga `fingers_train.csv`.
  - Separa features/etiquetas.
  - Aplica split train/val.
  - Ajusta y guarda el normalizador:
    - `Data_Sets/normalizer.pkl`

4. **Entrenamiento del modelo base (float)**
  - Entrena MLP con early stopping.
  - Guarda:
    - `Models/fingers_model_no_quantization.h5`

5. **Cuantizacion de pesos y export de modelos**
  - Construye dos variantes con cuantizacion simetrica signed:
    - por capa
    - por neurona
  - Guarda modelos Keras (carga rapida para evaluacion):
    - `Models/fingers_model_q_signed_symmetric_per_layer.h5`
    - `Models/fingers_model_q_signed_symmetric_per_neuron.h5`
  - Exporta modelos int8 reales a TFLite:
    - `Models/fingers_model_q_signed_symmetric_per_layer_int8.tflite`
    - `Models/fingers_model_q_signed_symmetric_per_neuron_int8.tflite`

6. **Cuantizacion de entradas y metadata exacta**
  - Cuantiza los CSV de entrada en el dominio correcto (features normalizadas).
  - Genera:
    - `Data_Sets/fingers_train_quant8_signed_symmetric.csv`
    - `Data_Sets/fingers_test_quant8_signed_symmetric.csv`
  - Guarda metadata exacta para reproducir dequantizacion en evaluacion:
    - `Data_Sets/quantization_metadata_signed_symmetric.npz`

### PARTE 2 - EVALUACION 

1. **Carga de artefactos precomputados**
  - Modelos base y cuantizados (`.h5`).
  - Normalizador (`normalizer.pkl`).
  - CSV cuantizados precomputados.
  - Metadata de cuantizacion exacta (`.npz`).

2. **Evaluacion baseline**
  - Evalua el modelo float sobre datos float normalizados.

3. **Evaluacion cuantizada sin recalcular cuantizacion**
  - Pipeline I/O quantized: usa directamente los CSV cuantizados.
  - Pipeline I/O dequantized: dequantiza usando las escalas exactas guardadas.

4. **Analisis de error quantized vs dequantized**
  - Calcula MAE, MSE, error maximo y agreement/disagreement top-1.

5. **Persistencia de resultados**
  - Tabla comparativa general:
    - `Data_Sets/quantization_comparison_signed_symmetric.csv`
  - Analisis de error:
    - `Data_Sets/quantized_vs_dequantized_error_signed_symmetric.csv`


## Archivos de salida importantes

- Modelo base (float):
  - `Models/fingers_model_no_quantization.h5`

- Modelos cuantizados reales int8 (TFLite):
  - `Models/fingers_model_q_signed_symmetric_per_layer_int8.tflite`
  - `Models/fingers_model_q_signed_symmetric_per_neuron_int8.tflite`

- Modelos cuantizados Keras (para evaluacion rapida):
  - `Models/fingers_model_q_signed_symmetric_per_layer.h5`
  - `Models/fingers_model_q_signed_symmetric_per_neuron.h5`

- Comparacion de metricas:
  - `Data_Sets/quantization_comparison_signed_symmetric.csv`

- Error entre pipelines quantized/dequantized:
  - `Data_Sets/quantized_vs_dequantized_error_signed_symmetric.csv`

- Metadata exacta de cuantizacion (escalas):
  - `Data_Sets/quantization_metadata_signed_symmetric.npz`

## Notas de cuantizacion

- Se usa cuantizacion simetrica signed de 8 bits: rango entero `[-127, 127]`.
- La cuantizacion puede degradar metricas, especialmente cuando la salida se mantiene cuantizada en enteros.
- En el notebook se comparan ambos escenarios para mostrar esa diferencia de forma explicita.

## Notas de Git/GitHub

- El entorno virtual local debe quedar fuera de control de versiones (`env/` esta ignorado en `.gitignore`).
- Si se suben librerias pesadas del entorno virtual, GitHub puede rechazar el push por limite de tamano de archivo.
