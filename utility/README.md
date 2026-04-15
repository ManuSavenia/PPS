# Documentación del Paquete Utility

Este paquete centraliza todas las funciones reutilizables para el pipeline de cuantización e inferencia en MCU. Cada función se define en su propio archivo para modularidad y claridad.

## Tabla de Contenidos

- [Procesamiento de Imágenes](#procesamiento-de-imágenes)
- [Funciones Core de Cuantización](#funciones-core-de-cuantización)
- [Funciones de Cuantización de Alto Nivel](#funciones-de-cuantización-de-alto-nivel)
- [Exportación C & Orquestación](#exportación-c--orquestación)
- [Funciones de Evaluación](#funciones-de-evaluación)
- [Guía de Importación](#guía-de-importación)

---

## Procesamiento de Imágenes

### `convertir_dataset.py`
**Función:** `convertir_dataset(image_dir, output_csv, scaler=None)`

Convierte un directorio de imágenes PNG de manos en un archivo CSV de características normalizadas.

**Propósito:** Procesar por lotes imágenes de manos desde subdirectorios organizados de entrenamiento/prueba, extraer características morfológicas clave (área, perímetro, excentricidad, solidez, extensión) y normalizarlas para entrada de red neuronal.

**Parámetros:**
- `image_dir` (str): Ruta del directorio que contiene imágenes organizadas en subdirectorios (ej., `Imagenes/Fingers/train/0/`, `.../1/`, etc.)
- `output_csv` (str): Ruta donde se guardará el CSV de características de salida
- `scaler` (StandardScaler, opcional): Escalador pre-ajustado para normalización. Si es None, crea y ajusta uno nuevo

**Retorna:** 
- Tupla: (features_array, scaler_ajustado_o_proporcionado)

**Estructura del CSV de Salida:** Columnas = [área, perímetro, excentricidad, solidez, extensión, etiqueta]

---

### `extraer_caracteristicas.py`
**Función:** `extraer_caracteristicas(image_path)`

Extrae características morfológicas de una única imagen de mano.

**Propósito:** Procesamiento de bajo nivel usando umbralización de Otsu, operaciones morfológicas y análisis de regionprops para identificar la región de la mano y calcular descriptores numéricos.

**Parámetros:**
- `image_path` (str): Ruta al archivo de imagen PNG

**Retorna:** 
- Array: [área, perímetro, excentricidad, solidez, extensión] (5 características morfológicas)

---

## Funciones Core de Cuantización

### `qmax.py`
**Función:** `qmax(n_bits)`

Calcula el valor máximo representable para cuantización de enteros con signo simétrica.

**Propósito:** Ayudante para cálculo del rango de cuantización. Para int8 con signo (n_bits=8), qmax = 2^(8-1) - 1 = 127.

**Parámetros:**
- `n_bits` (int): Ancho de bit (ej., 8 para int8)

**Retorna:** 
- int: Valor máximo representable (ej., 127 para int8)

---

### `quantize_symmetric_signed.py`
**Función:** `quantize_symmetric_signed(x, scale, n_bits=8)`

Aplica cuantización simétrica con signo a valores brutos y proporciona referencia dequantizada.

**Propósito:** Operación de cuantización core para un único tensor con una escala dada. Produce versiones cuantizadas (int32) y dequantizadas (reconstruidas float) para análisis de error.

**Parámetros:**
- `x` (ndarray): Valores de entrada a cuantizar
- `scale` (float): Escala de cuantización (rango / qmax)
- `n_bits` (int): Ancho de bit (por defecto 8)

**Retorna:** 
- Tupla: (quantized_int32, dequantized_float)

---

### `quantize_tensor_per_layer.py`
**Función:** `quantize_tensor_per_layer(tensor, n_bits=8, return_scale=True)`

Cuantiza un tensor completo (capa) usando una **única escala** (cuantización por capa).

**Propósito:** Esquema de cuantización simple donde todos los pesos en una capa comparten una escala. Inferencia más rápida pero puede perder precisión comparado con per-neurona.

**Parámetros:**
- `tensor` (ndarray): Pesos/activaciones de entrada
- `n_bits` (int): Ancho de bit (por defecto 8)
- `return_scale` (bool): Si es True, retorna (quantized, scale); si no, solo quantized

**Retorna:** 
- Tupla: (quantized_int32, scale) o solo quantized_int32

---

### `quantize_tensor_per_neuron.py`
**Función:** `quantize_tensor_per_neuron(tensor, n_bits=8, return_scales=True)`

Cuantiza una matriz de pesos 2D usando **escalas por canal de salida** (cuantización por neurona).

**Propósito:** Método de cuantización de mayor precisión donde cada neurona de salida (columna) tiene su propia escala. Más preciso pero requiere más metadatos.

**Parámetros:**
- `tensor` (ndarray): Matriz de pesos 2D (filas=entradas, cols=neuronas de salida)
- `n_bits` (int): Ancho de bit (por defecto 8)
- `return_scales` (bool): Si es True, retorna (quantized, scales); si no, solo quantized

**Retorna:** 
- Tupla: (quantized_int32, scales_array) o solo quantized_int32
- Forma del array de escalas: (n_neuronas_salida,)

---

### `quantize_dense_weights_symmetric.py`
**Función:** `quantize_dense_weights_symmetric(base_model, mode='per_layer', n_bits=8)`

Clona un modelo Keras, cuantiza todos los pesos de capas densas en modo especificado, dequantiza para inferencia.

**Propósito:** Aplicar cuantización a toda la red neuronal (todas las capas a la vez). Produce un modelo cuantizado-luego-dequantizado para medición de precisión.

**Parámetros:**
- `base_model` (Keras Model): Modelo entrenado original
- `mode` (str): Estrategia de cuantización 'per_layer' o 'per_neuron'
- `n_bits` (int): Ancho de bit (por defecto 8)

**Retorna:** 
- Tupla: (quantized_model, scales_dict)
- scales_dict: Mapea nombres de capas a sus escala(s)

---

## Funciones de Cuantización de Alto Nivel

### `quantize_csv_inputs_with_scale.py`
**Función:** `quantize_csv_inputs_with_scale(csv_path, scaler, mode='per_neuron', n_bits=8, output_csv=None)`

Carga características desde CSV, normaliza vía escalador, cuantiza por neurona, guarda con escalas exactas.

**Propósito:** Preparar conjunto de datos de entrada cuantizado para pipeline de inferencia en C. Produce características int32 cuantizadas y escalas float32 correspondientes en CSVs paralelos.

**Parámetros:**
- `csv_path` (str): Ruta al CSV de características de entrada
- `scaler` (StandardScaler): Escalador ajustado para normalización
- `mode` (str): 'per_neuron' (por defecto) o 'per_layer'
- `n_bits` (int): Ancho de bit (por defecto 8)
- `output_csv` (str): Ruta de salida (si es None, añade '_quantized_int32' al nombre del archivo de entrada)

**Retorna:** 
- Tupla: (quantized_int32_csv_path, scales_csv_path)

---

### `export_full_int8_tflite.py`
**Función:** `export_full_int8_tflite(model, representative_dataset, output_path)`

Convierte modelo Keras a formato TensorFlow Lite int8 usando conjunto de datos representativo.

**Propósito:** Generar modelo TFLite optimizado para MCU con cuantización completa (pesos, activaciones, entrada/salida).

**Parámetros:**
- `model` (Keras Model): Modelo entrenado a convertir
- `representative_dataset` (ndarray): ~100-300 muestras representativas para calibración de cuantización
- `output_path` (str): Donde guardar el archivo .tflite

**Retorna:** 
- str: Ruta al archivo .tflite guardado

---

## Diagnosticos y Reparacion

### `layerwise_quantized_debugging.py`
**Función:** `layerwise_quantized_vs_dequantized_report(model, x_quantized, y_eval, scales, dataset_name="dataset")`

Compara, capa por capa, las salidas del mismo modelo cuando la entrada se mantiene cuantizada versus cuando se de-cuantiza antes de la inferencia.

**Propósito:** detectar en qué capa comienza la deriva numérica que termina degradando la predicción final.

**Retorna:**
- `report`: `DataFrame` con MAE, RMSE, error máximo, error relativo, similitud coseno y agreement final.
- `summary`: diccionario con accuracy cuantizado, accuracy de-cuantizado y agreement final.

### `quantized_model_diagnostics.py`
**Funciones:** `build_accuracy_comparison(...)`, `build_weight_quantization_error_report(...)`, `build_deployed_levels_report(...)`

Agrupa diagnósticos para comparar `per_layer` contra `per_neuron`.

**Propósito:** medir el impacto real de cada esquema sobre accuracy, error de pesos, niveles usados y saturación.

### `quantized_clipping_analysis.py`
**Funciones:** `build_clipped_model_from_source(...)`, `count_levels(...)`, `evaluate_clipping_sweep(...)`

Explora recorte de outliers en pesos cuantizados y mide su efecto sobre accuracy y saturación.

### `quantized_histogram_reports.py`
**Función:** `generate_quantized_histogram_reports(model_name, model_obj, mode, out_dir, bits=8)`

Genera histogramas de pesos cuantizados por tensor, por neurona y globales.

### `quantized_accuracy_repair.py`
**Funciones:** `run_quantized_repair_search(...)`, `calibrate_activation_params(...)`, `predict_with_activation_quantization(...)`

Busca mejoras de accuracy combinando clipping selectivo de pesos, calibración de activaciones y fake-quant de activaciones.

### `evaluate_tflite_int8_from_csv.py`
**Función:** `evaluate_tflite_int8_from_csv(tflite_model_path, x_q_eval, y_eval, scales)`

Evalúa un modelo TFLite int8 real usando el intérprete y entradas reconstruidas desde CSV cuantizado.

---

## Exportación C & Orquestación

### `export_c_assets.py`
**Función:** `export_c_assets(project_root)`

Orquesta la exportación de todos los pesos cuantizados, escalas y conjuntos de datos de entrada para pipeline de inferencia en C.

**Propósito:** Función de exportación de nivel superior. Carga modelo base, cuantiza pesos en modos per-capa y per-neurona, guarda 8 CSVs de pesos + 8 CSVs de escalas, copia conjuntos de datos de entrada, genera metadatos JSON.

**Parámetros:**
- `project_root` (str): Raíz del workspace (ej., `/home/manuel/Documents/Facultad/PPS`)

**Efectos Secundarios:**
- Crea/puebla `C_model/data/` con:
  - `inputs/`:
    - `fingers_train_quant8_signed_symmetric.csv`
    - `fingers_test_quant8_signed_symmetric.csv`
    - `input_scale_train.csv`
    - `input_scale_test.csv`
  - `weights/per_layer/`:
    - `weights_q_per_layer_w0.csv`, `weights_q_per_layer_b0.csv`, `weights_q_per_layer_w1.csv`, `weights_q_per_layer_b1.csv`
  - `weights/per_neuron/`:
    - `weights_q_per_neuron_w0.csv`, `weights_q_per_neuron_b0.csv`, `weights_q_per_neuron_w1.csv`, `weights_q_per_neuron_b1.csv`
  - `scales/`:
    - `weight_scales_per_layer.csv`
    - `weight_scales_per_neuron_w0.csv`, `weight_scales_per_neuron_b0.csv`, `weight_scales_per_neuron_w1.csv`, `weight_scales_per_neuron_b1.csv`
  - `metadata/`:
    - `quantization_metadata_c.json`
  - `reports/`:
    - `quantization_comparison_signed_symmetric.csv` (si existe en `Cuantization_Test/Data_Sets/`)

**Retorna:** 
- None (solo efectos secundarios)

---

## Funciones de Evaluación

### `dequantize_saved_inputs.py`
**Función:** `dequantize_saved_inputs(quantized_csv, scales_csv)`

Reconstruye características float32 desde int32 cuantizado + escalas.

**Propósito:** Operación de cuantización inversa para análisis de precisión. Toma entradas int32 cuantizadas y aplica escalas por elemento para recuperar aproximaciones originals.

**Parámetros:**
- `quantized_csv` (str): Ruta al CSV de características int32 cuantizadas
- `scales_csv` (str): Ruta al CSV de escalas float32 correspondientes

**Retorna:** 
- ndarray: Características float32 dequantizadas (forma coincide con entrada)

---

### `evaluate_baseline.py`
**Función:** `evaluate_baseline(model, X_train, y_train, X_test, y_test)`

Calcula precisión y puntuación F1 para modelo baseline (float).

**Propósito:** Establecer métricas de precisión de referencia antes de efectos de cuantización.

**Parámetros:**
- `model` (Keras Model): Modelo float entrenado
- `X_train`, `y_train`: Características y etiquetas de entrenamiento
- `X_test`, `y_test`: Características y etiquetas de prueba

**Retorna:** 
- Dict: {'train_accuracy': float, 'test_accuracy': float, 'train_f1': float, 'test_f1': float}

---

### `evaluate_pipeline_io_quantized_from_csv.py`
**Función:** `evaluate_pipeline_io_quantized_from_csv(model, quantized_csv, y_true)`

Predice en entradas int32 cuantizadas sin dequantización.

**Propósito:** Medir degradación de precisión cuando el modelo recibe entradas cuantizadas directamente (cuantización más agresiva).

**Parámetros:**
- `model` (Keras Model): Modelo entrenado (espera entradas float normalizadas)
- `quantized_csv` (str): Ruta al CSV de características int32 cuantizadas
- `y_true` (ndarray): Etiquetas verdaderas

**Retorna:** 
- Dict: {'accuracy': float, 'f1': float}

---

### `evaluate_pipeline_io_dequantized_from_csv.py`
**Función:** `evaluate_pipeline_io_dequantized_from_csv(model, quantized_csv, scales_csv, y_true)`

Dequantiza entradas primero, luego predice (recupera pérdida de precisión de cuantización).

**Propósito:** Medir precisión cuando la pérdida por redondeo inducida por cuantización se invierte antes de inferencia. Sirve como punto intermedio entre baseline y completamente cuantizado.

**Parámetros:**
- `model` (Keras Model): Modelo entrenado
- `quantized_csv` (str): Ruta al CSV de características int32 cuantizadas
- `scales_csv` (str): Ruta al CSV de escalas float32
- `y_true` (ndarray): Etiquetas verdaderas

**Retorna:** 
- Dict: {'accuracy': float, 'f1': float}

---

### `get_outputs_io_dequantized_from_csv.py`
**Función:** `get_outputs_io_dequantized_from_csv(model, quantized_csv, scales_csv)`

Retorna logits y probabilidades del modelo después de dequantizar entradas.

**Propósito:** Extraer salidas raw del modelo (antes de argmax) para análisis de error detallado. Variante dequantización.

**Parámetros:**
- `model` (Keras Model): Modelo entrenado
- `quantized_csv` (str): Ruta al CSV de características int32 cuantizadas
- `scales_csv` (str): Ruta al CSV de escalas float32

**Retorna:** 
- Tupla: (logits_ndarray, probabilities_ndarray)
- Forma de logits: (n_muestras, n_clases)
- Forma de probabilidades: (n_muestras, n_clases) [después de softmax]

---

### `get_outputs_io_quantized_from_csv.py`
**Función:** `get_outputs_io_quantized_from_csv(model, quantized_csv)`

Retorna logits y probabilidades del modelo en entradas int32 cuantizadas raw.

**Propósito:** Extraer salidas raw del modelo sin dequantización (escenario más agresivo).

**Parámetros:**
- `model` (Keras Model): Modelo entrenado
- `quantized_csv` (str): Ruta al CSV de características int32 cuantizadas

**Retorna:** 
- Tupla: (logits_ndarray, probabilities_ndarray)

---

### `compare_quant_vs_dequant_outputs_from_csv.py`
**Función:** `compare_quant_vs_dequant_outputs_from_csv(model, quantized_csv, scales_csv, y_true, output_csv=None)`

Compara salidas del modelo entre pipelines de entrada cuantizado y dequantizado. Calcula MAE, MSE, acuerdo de predicción, y deltas de precisión por clase.

**Propósito:** Cuantificar el impacto de cuantización de entrada en el comportamiento del modelo. Identifica qué muestras/clases son más sensibles a cuantización.

**Parámetros:**
- `model` (Keras Model): Modelo entrenado
- `quantized_csv` (str): Ruta al CSV de características int32 cuantizadas
- `scales_csv` (str): Ruta al CSV de escalas float32
- `y_true` (ndarray): Etiquetas verdaderas
- `output_csv` (str, opcional): Ruta para guardar resultados de comparación detallados

**Retorna:** 
- Dict: {
    'mae': float (error absoluto medio en logits),
    'mse': float (error cuadrado medio en logits),
    'prediction_agreement': float (fracción de muestras con mismo argmax),
    'accuracy_delta': float (diferencia de precisión: baseline - cuantizado),
    'per_class_deltas': dict (deltas de precisión por clase)
  }

---

## Guía de Importación

Los helpers nuevos más usados en el notebook principal son:

- `utility.layerwise_quantized_debugging`
- `utility.quantized_model_diagnostics`
- `utility.quantized_clipping_analysis`
- `utility.quantized_histogram_reports`
- `utility.quantized_accuracy_repair`
- `utility.evaluate_tflite_int8_from_csv`

### Importar en Celdas de Notebook

```python
# Procesamiento de imágenes
from utility.convertir_dataset import convertir_dataset
from utility.extraer_caracteristicas import extraer_caracteristicas

# Funciones de cuantización
from utility.qmax import qmax
from utility.quantize_symmetric_signed import quantize_symmetric_signed
from utility.quantize_tensor_per_layer import quantize_tensor_per_layer
from utility.quantize_tensor_per_neuron import quantize_tensor_per_neuron
from utility.quantize_dense_weights_symmetric import quantize_dense_weights_symmetric

# Cuantización de alto nivel
from utility.quantize_csv_inputs_with_scale import quantize_csv_inputs_with_scale
from utility.export_full_int8_tflite import export_full_int8_tflite

# Exportación C
from utility.export_c_assets import export_c_assets

# Evaluación
from utility.dequantize_saved_inputs import dequantize_saved_inputs
from utility.evaluate_baseline import evaluate_baseline
from utility.evaluate_pipeline_io_quantized_from_csv import evaluate_pipeline_io_quantized_from_csv
from utility.evaluate_pipeline_io_dequantized_from_csv import evaluate_pipeline_io_dequantized_from_csv
from utility.get_outputs_io_dequantized_from_csv import get_outputs_io_dequantized_from_csv
from utility.get_outputs_io_quantized_from_csv import get_outputs_io_quantized_from_csv
from utility.compare_quant_vs_dequant_outputs_from_csv import compare_quant_vs_dequant_outputs_from_csv
```

### Usar en Pipeline de Exportación C

```python
import sys
sys.path.insert(0, '/ruta/a/raiz/workspace')
from utility.export_c_assets import export_c_assets

export_c_assets(project_root='/ruta/a/raiz/workspace')
```

---

## Dependencias de Funciones

```
Procesamiento de Imágenes:
  convertir_dataset → extraer_caracteristicas

Cuantización:
  quantize_symmetric_signed → qmax
  quantize_tensor_per_layer → quantize_symmetric_signed
  quantize_tensor_per_neuron → quantize_symmetric_signed
  quantize_dense_weights_symmetric → quantize_tensor_per_layer/per_neuron
  quantize_csv_inputs_with_scale → quantize_tensor_per_neuron

Exportación C:
  export_c_assets → quantize_dense_weights_symmetric, quantize_tensor_per_layer, quantize_tensor_per_neuron

Evaluación:
  evaluate_pipeline_io_dequantized_from_csv → dequantize_saved_inputs
  compare_quant_vs_dequant_outputs_from_csv → get_outputs_io_quantized_from_csv, get_outputs_io_dequantized_from_csv
  get_outputs_io_dequantized_from_csv → dequantize_saved_inputs
```

---

## Organización de Archivos

```
utility/
├── __init__.py
├── README.md                                      (este archivo)
├── convertir_dataset.py
├── extraer_caracteristicas.py
├── qmax.py
├── quantize_symmetric_signed.py
├── quantize_tensor_per_layer.py
├── quantize_tensor_per_neuron.py
├── quantize_dense_weights_symmetric.py
├── quantize_csv_inputs_with_scale.py
├── export_full_int8_tflite.py
├── export_c_assets.py
├── dequantize_saved_inputs.py
├── evaluate_baseline.py
├── evaluate_pipeline_io_quantized_from_csv.py
├── evaluate_pipeline_io_dequantized_from_csv.py
├── get_outputs_io_dequantized_from_csv.py
├── get_outputs_io_quantized_from_csv.py
└── compare_quant_vs_dequant_outputs_from_csv.py
```

---

## Descripción General del Pipeline de Cuantización

```
Dataset (imágenes PNG)
  ↓
extraer_caracteristicas → convertir_dataset
  ↓
CSV de Características Normalizadas
  ↓
quantize_csv_inputs_with_scale
  ↓
CSV INT32 Cuantizado + Escalas CSV
  ↓
[Inferencia en C] OR [Evaluación en Python]
  ↓
Métricas de Precisión/F1
```

**Para Inferencia en C:** Valores cuantizados exportados vía `export_c_assets` → programa C carga y usa directamente con cálculo mínimo.

**Para Evaluación en Python:** Las funciones en la sección "Evaluación" realizan análisis de precisión y métricas de error en pipelines cuantizado vs. dequantizado.
