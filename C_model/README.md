# C_model

Pipeline de inferencia en C para los modelos cuantizados del proyecto Fingers.

## Estructura

- `data/`: artefactos organizados por tipo:
	- `inputs/`: datasets cuantizados y escalas de entrada
	- `weights/per_layer/`: pesos cuantizados por capa
	- `weights/per_neuron/`: pesos cuantizados por neurona
	- `scales/`: escalas de pesos (por capa y por neurona)
	- `metadata/`: metadata JSON para C
	- `reports/`: comparaciones Python/C y resultados de ejecución C
- `utility/`: funciones compartidas (`utils.c`, `utils.h`) para activaciones, dequantización y carga CSV.
- `models/`: `main.c` con inferencia de modelos `per-layer` y `per-neuron`, evaluación de accuracy y comparación con referencia Python.
- `tools/`: script Python para exportar artefactos desde modelos Keras a `data/`.

## Origen de los artefactos

Los datos y modelos que consume `C_model/` se generan desde `Cuantization_Test/Test/Test_Fingers.ipynb` y quedan organizados en:

- `Cuantization_Test/Data_Sets/raw/`
- `Cuantization_Test/Data_Sets/quantized/`
- `Cuantization_Test/Data_Sets/metadata/`
- `Cuantization_Test/Data_Sets/reports/`
- `Cuantization_Test/Models/base/`
- `Cuantization_Test/Models/quantized_h5/`
- `Cuantization_Test/Models/quantized_tflite/`
- `Cuantization_Test/Models/repaired_h5/`
- `Cuantization_Test/Models/repaired_tflite/`

## Activaciones implementadas

- Capa oculta: `tanh`
- Capa salida: `softmax`

## Preparar datos para C

Desde `C_model/`:

```bash
env PYTHONPATH=../Cuantization_Test/env/lib/python3.11/site-packages:.. /usr/bin/python3.11 tools/export_c_assets.py
```

Esto genera/copia en `data/`:

- `inputs/fingers_train_quant8_signed_symmetric.csv`
- `inputs/fingers_test_quant8_signed_symmetric.csv`
- `inputs/input_scale_train.csv`
- `inputs/input_scale_test.csv`
- `weights/per_layer/weights_q_per_layer_w0.csv`
- `weights/per_layer/weights_q_per_layer_b0.csv`
- `weights/per_layer/weights_q_per_layer_w1.csv`
- `weights/per_layer/weights_q_per_layer_b1.csv`
- `weights/per_neuron/weights_q_per_neuron_w0.csv`
- `weights/per_neuron/weights_q_per_neuron_b0.csv`
- `weights/per_neuron/weights_q_per_neuron_w1.csv`
- `weights/per_neuron/weights_q_per_neuron_b1.csv`
- `scales/weight_scales_per_layer.csv`
- `scales/weight_scales_per_neuron_w0.csv`
- `scales/weight_scales_per_neuron_b0.csv`
- `scales/weight_scales_per_neuron_w1.csv`
- `scales/weight_scales_per_neuron_b1.csv`
- `metadata/quantization_metadata_c.json`
- `reports/quantization_comparison_signed_symmetric.csv` (si existe en `Cuantization_Test/Data_Sets/`)
- `reports/quantized_vs_dequantized_error_signed_symmetric.csv` (si existe en `Cuantization_Test/Data_Sets/reports/`)
- `reports/layerwise_quantized_vs_dequantized_analysis.csv` (si existe)
- `reports/diagnostic_accuracy_layer_vs_neuron.csv` (si existe)
- `reports/quantized_accuracy_repair_best.csv` y validacion TFLite asociada (si existen)

Nota:
El script `tools/export_c_assets.py` delega en la utilidad compartida de nivel superior `../utility/export_c_assets.py` para evitar duplicación de lógica.

## Build y ejecución

Desde `C_model/`:

```bash
make
make run
```

Salida esperada:

- Accuracy train/test para `Q por capa` y `Q por neurona`.
- Diferencia contra referencia Python (si está disponible).
- Archivo `data/reports/c_inference_comparison.csv` con resultados.
