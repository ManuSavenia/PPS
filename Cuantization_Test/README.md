# PPS Quantization Project

Proyecto para:
- Extraer caracteristicas del dataset Fingers.
- Entrenar un MLP para clasificar cantidad de dedos levantados (0 a 5).
- Comparar cuantizacion simetrica signed post-entrenamiento.

## Estructura

- `Fuentes/`: modulos auxiliares.
- `Data_Sets/`: CSV de train/test y resultados de comparacion.
- `Imagenes/Fingers/`: imagenes originales del dataset.
- `Models/`: modelo base y modelos cuantizados (`.tflite`).
- `Test/10_Fingers.ipynb`: notebook principal del flujo completo.
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

### Opcion B: Regenerar CSV de entradas cuantizadas int8

```bash
python Fuentes/quantize_signed_symmetric_inputs.py
```

Genera:
- `Data_Sets/fingers_train_quant8_signed_symmetric.csv`
- `Data_Sets/fingers_test_quant8_signed_symmetric.csv`

## Archivos de salida importantes

- Modelo base (float):
  - `Models/fingers_model_no_quantization.h5`

- Modelos cuantizados reales int8 (TFLite):
  - `Models/fingers_model_q_signed_symmetric_per_layer_int8.tflite`
  - `Models/fingers_model_q_signed_symmetric_per_neuron_int8.tflite`

- Comparacion de metricas:
  - `Data_Sets/quantization_comparison_signed_symmetric.csv`

## Notas de cuantizacion

- Se usa cuantizacion simetrica signed de 8 bits: rango entero `[-127, 127]`.
- La cuantizacion puede degradar metricas, especialmente cuando la salida se mantiene cuantizada en enteros.
- En el notebook se comparan ambos escenarios para mostrar esa diferencia de forma explicita.
