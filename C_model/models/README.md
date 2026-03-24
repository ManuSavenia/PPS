# Models

Esta carpeta contiene el archivo principal de ejecucion del pipeline de inferencia en C.

## Archivo principal

- `main.c`: carga modelos cuantizados, ejecuta inferencia sobre train/test y compara resultados de C contra referencias de Python.

## Que hace `main.c`

El programa implementa este flujo:

1. Lee escalas de entrada cuantizada (`input_scale_train.csv` y `input_scale_test.csv`).
2. Carga dos variantes del modelo:
   - cuantizacion por capa (`Q por capa`)
   - cuantizacion por neurona (`Q por neurona`)
3. Carga datasets cuantizados de entrenamiento y prueba.
4. Ejecuta inferencia muestra por muestra:
   - dequantizacion de entradas
   - capa oculta con `tanh`
   - capa de salida con `softmax`
5. Calcula accuracy para train/test en ambos modelos.
6. Si existe reporte Python, calcula diferencias absolutas.
7. Guarda un resumen en `../data/reports/c_inference_comparison.csv`.

## Estructuras clave

- `QModel`: agrupa pesos, biases y escalas de un modelo cuantizado.
- `PythonRefs`: almacena accuracies de referencia leidas desde CSV para comparacion.

## Funciones relevantes

- `load_model_per_layer`: carga pesos/escala de la variante por capa.
- `load_model_per_neuron`: carga pesos/escala de la variante por neurona.
- `infer_one`: ejecuta forward pass para una muestra.
- `evaluate_accuracy`: evalua accuracy sobre un conjunto de muestras.
- `parse_python_refs`: lee accuracies de referencia desde reporte Python.
- `write_results`: exporta el CSV final comparando C vs Python.

## Entradas esperadas

Desde `../data/`:

- `inputs/fingers_train_quant8_signed_symmetric.csv`
- `inputs/fingers_test_quant8_signed_symmetric.csv`
- `inputs/input_scale_train.csv`
- `inputs/input_scale_test.csv`
- Pesos y escalas en subcarpetas `weights/` y `scales/`.

## Salidas

- Impresion por consola de accuracies y diferencias.
- Archivo de reporte:
  - `../data/reports/c_inference_comparison.csv`

## Ejecucion

Desde la carpeta `C_model/`:

```bash
make
make run
```