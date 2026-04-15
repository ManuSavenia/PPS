# PPS Quantization Project

Proyecto para:
- Extraer caracteristicas del dataset Fingers.
- Entrenar un MLP para clasificar cantidad de dedos levantados (0 a 5).
- Comparar cuantizacion simetrica signed post-entrenamiento.

## Estructura

- `Fuentes/`: modulos auxiliares.
- `Data_Sets/raw/`: CSV base generados desde las imagenes.
- `Data_Sets/quantized/`: CSV cuantizados de train/test.
- `Data_Sets/metadata/`: normalizador, escalas y metadata exacta de cuantizacion.
- `Data_Sets/reports/`: comparaciones de accuracy, error, niveles usados, diagnosticos capa a capa, clipping, reparacion y validacion TFLite.
- `Imagenes/Fingers/`: imagenes originales del dataset.
- `Models/base/`: modelo base entrenado.
- `Models/quantized_h5/`: modelos cuantizados en formato Keras para evaluacion rapida.
- `Models/quantized_tflite/`: modelos int8 reales para validacion con TFLite.
- `Models/repaired_h5/` y `Models/repaired_tflite/`: modelos ajustados por reparacion y su exportacion int8.
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
5. Depuracion capa a capa para detectar donde aparece la deriva entre cuantizado puro y de-cuantizado.
6. Diagnosticos per-layer vs per-neuron, histogramas y sweep de clipping.
7. Busqueda de reparacion de accuracy con calibracion de activaciones y clipping selectivo.
8. Validacion final con TFLite int8 real.

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
    - `Models/quantized_h5/fingers_model_q_signed_symmetric_per_layer.h5`
    - `Models/quantized_h5/fingers_model_q_signed_symmetric_per_neuron.h5`
  - Exporta modelos int8 reales a TFLite:
    - `Models/quantized_tflite/fingers_model_q_signed_symmetric_per_layer_int8.tflite`
    - `Models/quantized_tflite/fingers_model_q_signed_symmetric_per_neuron_int8.tflite`

6. **Cuantizacion de entradas y metadata exacta**
  - Cuantiza los CSV de entrada en el dominio correcto (features normalizadas).
  - Genera:
    - `Data_Sets/quantized/fingers_train_quant8_signed_symmetric.csv`
    - `Data_Sets/quantized/fingers_test_quant8_signed_symmetric.csv`
  - Guarda metadata exacta para reproducir dequantizacion en evaluacion:
    - `Data_Sets/metadata/quantization_metadata_signed_symmetric.npz`

7. **Analisis y depuracion avanzada**
   - Genera reportes de error, histogramas, clipping, reparacion y validacion TFLite en `Data_Sets/reports/`.

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

5. **Depuracion capa a capa**
   - Compara las salidas intermedias del mismo modelo usando entrada cuantizada vs de-cuantizada.
   - Identifica en qué capa comienza la deriva.

6. **Diagnosticos y reparacion**
   - Compara `per_layer` vs `per_neuron` en accuracy, error de pesos y niveles usados.
   - Genera histogramas por capa y por neurona.
   - Prueba clipping de outliers y calibracion de activaciones.

7. **Persistencia de resultados**
  - Tabla comparativa general:
    - `Data_Sets/reports/quantization_comparison_signed_symmetric.csv`
  - Analisis de error:
    - `Data_Sets/reports/quantized_vs_dequantized_error_signed_symmetric.csv`
   - Depuracion capa a capa:
    - `Data_Sets/reports/layerwise_quantized_vs_dequantized_analysis.csv`
   - Diagnostico per-layer vs per-neuron:
    - `Data_Sets/reports/diagnostic_accuracy_layer_vs_neuron.csv`
    - `Data_Sets/reports/diagnostic_weight_error_layer_vs_neuron.csv`
    - `Data_Sets/reports/diagnostic_levels_layer_vs_neuron.csv`
   - Clipping y reparacion:
    - `Data_Sets/reports/quantized_clipping_analysis_signed_symmetric.csv`
    - `Data_Sets/reports/quantized_accuracy_repair_search.csv`
    - `Data_Sets/reports/quantized_accuracy_repair_best.csv`
    - `Data_Sets/reports/quantized_accuracy_repair_tflite_validation.csv`


## Archivos de salida importantes

- Modelo base (float):
  - `Models/base/fingers_model_no_quantization.h5`

- Modelos cuantizados reales int8 (TFLite):
  - `Models/quantized_tflite/fingers_model_q_signed_symmetric_per_layer_int8.tflite`
  - `Models/quantized_tflite/fingers_model_q_signed_symmetric_per_neuron_int8.tflite`

- Modelos cuantizados Keras (para evaluacion rapida):
  - `Models/quantized_h5/fingers_model_q_signed_symmetric_per_layer.h5`
  - `Models/quantized_h5/fingers_model_q_signed_symmetric_per_neuron.h5`

- Modelos reparados:
  - `Models/repaired_h5/fingers_model_q_signed_symmetric_per_layer_repaired_best.h5`
  - `Models/repaired_h5/fingers_model_q_signed_symmetric_per_neuron_repaired_best.h5`
  - `Models/repaired_tflite/fingers_model_q_signed_symmetric_per_layer_repaired_best_int8.tflite`
  - `Models/repaired_tflite/fingers_model_q_signed_symmetric_per_neuron_repaired_best_int8.tflite`

- Comparacion de metricas:
  - `Data_Sets/reports/quantization_comparison_signed_symmetric.csv`

- Error entre pipelines quantized/dequantized:
  - `Data_Sets/reports/quantized_vs_dequantized_error_signed_symmetric.csv`

- Depuracion y diagnosticos:
  - `Data_Sets/reports/layerwise_quantized_vs_dequantized_analysis.csv`
  - `Data_Sets/reports/diagnostic_accuracy_layer_vs_neuron.csv`
  - `Data_Sets/reports/diagnostic_weight_error_layer_vs_neuron.csv`
  - `Data_Sets/reports/diagnostic_levels_layer_vs_neuron.csv`
  - `Data_Sets/reports/histogram_scope_layer_neuron_summary.csv`
  - `Data_Sets/reports/quantized_clipping_analysis_signed_symmetric.csv`
  - `Data_Sets/reports/quantized_accuracy_repair_search.csv`
  - `Data_Sets/reports/quantized_accuracy_repair_best.csv`
  - `Data_Sets/reports/quantized_accuracy_repair_tflite_validation.csv`

- Metadata exacta de cuantizacion (escalas):
  - `Data_Sets/metadata/quantization_metadata_signed_symmetric.npz`

## Notas de cuantizacion

- Se usa cuantizacion simetrica signed de 8 bits: rango entero `[-127, 127]`.
- La cuantizacion puede degradar metricas, especialmente cuando la salida se mantiene cuantizada en enteros.
- En el notebook se comparan ambos escenarios para mostrar esa diferencia de forma explicita.
- La depuracion capa a capa muestra que la deriva aparece pronto en la primera capa densa y que la calibracion de activaciones es el factor mas sensible.

## Notas de Git/GitHub

- El entorno virtual local debe quedar fuera de control de versiones (`env/` esta ignorado en `.gitignore`).
- Si se suben librerias pesadas del entorno virtual, GitHub puede rechazar el push por limite de tamano de archivo.
