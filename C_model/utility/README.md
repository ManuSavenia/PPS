# Utility

Esta carpeta contiene utilidades compartidas usadas por el pipeline de inferencia en C.

## Archivos

- `utils.h`: declaraciones de funciones y tipos compartidos.
- `utils.c`: implementacion de activaciones, dequantizacion y carga de CSV.

## Funciones en `utils.c`

- `act_tanh(float x)`:
  aplica la activacion `tanh` para una neurona.

- `act_softmax(const float *in, int n, float *out)`:
  transforma logits a probabilidades normalizadas. Incluye estabilizacion numerica restando el maximo antes de aplicar `exp`.

- `dequantize_value(int32_t q, float scale)`:
  convierte un valor cuantizado a flotante con la formula `q * scale`.

- `argmaxf(const float *x, int n)`:
  devuelve el indice del mayor valor del vector.

- `read_csv_int_matrix(const char *path, int rows, int cols, int32_t *out)`:
  lee un CSV con enteros en formato matriz plana (`rows x cols`).

- `read_csv_float_vector(const char *path, int expected_len, float *out)`:
  lee el primer registro no vacio como vector de `float`.

- `load_quantized_dataset_csv(...)`:
  carga dataset cuantizado desde CSV con formato:
  `feature_1,...,feature_n,label`.
  Devuelve por referencia:
  - matriz de features cuantizadas (`int32_t *`)
  - vector de labels (`int *`)
  - cantidad de muestras (`samples`)

## Relacion con el pipeline Python

Los CSV y metadatos consumidos por estas funciones se generan desde `Cuantization_Test/Data_Sets/` y `Cuantization_Test/Models/`.
En particular, los artefactos relevantes ahora viven en subcarpetas organizadas:

- `Data_Sets/raw/`, `Data_Sets/quantized/`, `Data_Sets/metadata/`, `Data_Sets/reports/`
- `Models/base/`, `Models/quantized_h5/`, `Models/quantized_tflite/`, `Models/repaired_h5/`, `Models/repaired_tflite/`

## Detalles de robustez

- Salta lineas vacias en los CSV.
- Verifica cantidad de columnas y filas esperadas.
- Devuelve codigos de error negativos ante fallos de apertura, parseo o memoria.

## Uso dentro del proyecto

Estas utilidades son consumidas principalmente por `../models/main.c` para:

- reconstruir valores en punto flotante desde enteros cuantizados,
- ejecutar funciones de activacion,
- cargar datasets, pesos y escalas exportados por el pipeline de Python.