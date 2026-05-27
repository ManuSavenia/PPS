# Cambios Totales realizados en el proyecto EmbedIA (resumen técnico)

Este documento resume, en español, los cambios realizados para reparar la ruta FULL_QUANT8 y alinear el generador con el runtime. Está pensado para pasar al mantenedor del proyecto y facilita la revisión de los parches.

Archivos documentados:

- `embedia/libraries/mcu/generic/full_quant8/neural_net.h`
- `embedia/libraries/mcu/generic/full_quant8/neural_net.c`
- `embedia/layers/activation/activation.py`
- `embedia/libraries/model/model.c`
- `embedia/model_generator/generate_files.py`
- `embedia/layers/dense/dense.py`

---

## 1) `embedia/libraries/mcu/generic/full_quant8/neural_net.h`

Propósito:
- Declarar las interfaces del runtime FULL_QUANT8 (capas densa, activaciones, helpers de cuantización).

Cambios realizados:
- Se actualizó la firma de las funciones de activación para aceptar `qparam_t` como parámetro adicional cuando procede. Ejemplo de firma modificada:

```c
void relu_activation(quant8 *data, uint32_t length, qparam_t qp);
void softmax_activation(quant8 *data, uint32_t length, qparam_t qp);
```

Motivo:
- Las activaciones deben conocer los parámetros de cuantización (escala/zero-point) para des/recuantizar correctamente cuando se trabaja en dominio quantizado.

Impacto esperado:
- Evita desajustes entre la escala usada por la capa y la que asumía la activación; prepara el terreno para implementaciones qparam-aware en C.

---

## 2) `embedia/libraries/mcu/generic/full_quant8/neural_net.c`

Propósito:
- Implementación en C del runtime FULL_QUANT8: evaluación de capas Dense y funciones de activación en el camino de inferencia.

Cambios realizados:
- Implementaciones de activaciones reescritas para operar con `qparam_t`: cada activación descuantiza el bloque (o algunos valores), aplica la operación en punto flotante y vuelve a cuantizar usando `QUANTIZE/DEQUANTIZE` y `qparam` de salida.
- La función `dense_layer` ahora usa `layer->weights_qparam` y `layer->output_qparam` para convertir correctamente entre representaciones y escribir la salida cuantizada.
- Softmax se implementó: descuantiza entradas a float, calcula exponentes/probabilidades en float y re-quantiza el vector de salida con el `qparam` de la capa.

Fragmento ilustrativo (simplificado):

```c
// Descuantizar, aplicar softmax en float, requantizar
float *tmp = malloc(sizeof(float)*length);
for (i=0;i<length;i++) tmp[i] = DEQUANTIZE(data[i], qp_in);
// calcular softmax en tmp[] → out_float[]
for (i=0;i<length;i++) data[i] = QUANTIZE(out_float[i], qp_out);
free(tmp);
```

Motivo:
- Alinear la implementación con la lógica de generación: el runtime debe respetar los `qparams` emitidos por el generador para evitar errores de escala.

Impacto esperado:
- Salidas de capa coherentes con las qparams emitidas, mejor precisión y ausencia de desbordes por malas suposiciones de escala.

---

## 3) `embedia/layers/activation/activation.py`

Propósito:
- Generador de código (template) que emite las llamadas C a funciones de activación desde la descripción del modelo.

Cambios realizados:
- Cuando el modelo está cuantizado (`model.is_data_quantized`), la generación de la llamada a la activación ahora pasa el `qparam` de salida del tensor. Ejemplo emitido:

```c
softmax_activation(output.data, size, output.qparam);
```

Motivo:
- Garantizar que la activación reciba la información de cuantización correcta (escala/zero-point) para su evaluación.

Impacto esperado:
- Código C generado que invoca activaciones qparam-aware, consistente con las firmas del runtime.

---

## 4) `embedia/libraries/model/model.c`

Propósito:
- Plantilla C para el archivo `model.c` generado; contiene includes, inicializadores, datos estáticos y la función `predict`.

Cambios realizados:
- Se corrigieron llaves/llaves dobles que interferían con `str.format()` en Python (escape de `{`/`}`) para evitar `KeyError` en la renderización de la plantilla.
- Se aseguró que `#include "common.h"` y la definición `EMBEDIA_MODEL_STORAGE` (si el generador la inyecta) aparezcan antes de las declaraciones de datos estáticos, evitando errores de compilación por macro indefinida.

Motivo:
- La plantilla causaba fallos de formateo y errores de compilación cuando el generator inyectaba macros antes de las arrays `static const`.

Impacto esperado:
- Generación de `model.c` libre de errores de formateo y compilable sin requerir parches manuales.

---

## 5) `embedia/model_generator/generate_files.py`

Propósito:
- Orquestar la generación de archivos C del modelo: componer includes, defines, estructuras, inicializadores y el `predict` generado.

Cambios realizados:
- Prepend `#include "common.h"\n` y permitir que `EMBEDIA_MODEL_STORAGE` (si procede) se inserte antes de los datos estáticos.
- Corrección en la generación de `predict_class` para producir `return argmax(*results);` en el caso de salida por probabilidades.
- Se añadió una pasada de calibración (best-effort) cuando hay ejemplos disponibles, para calcular y asignar `output_qparam` por capa antes de emitir los datos.
- Mejorada la búsqueda recursiva de archivos fuente del runtime para copiar correctamente los archivos `full_quant8` en el proyecto generado.

Fragmento relevante (conceptual):

```python
text_model_c = text_model_c.format(includes=model_storage_define + includes, ...)
# model_storage_define se inyecta antes de los arrays estáticos
```

Motivo:
- Evitar plantillas mal renderizadas, asegurar definición de macros, y emitir qparams por capa para que el runtime tenga los parámetros necesarios.

Impacto esperado:
- Proyectos generados que contienen las definiciones de qparams necesarias y que compilan usando los archivos runtime copiados.

---

## 6) `embedia/layers/dense/dense.py`

Propósito:
- Generador de la capa Dense: emite arrays de pesos, biases, parámetros de layer y la estructura `dense_layer_t`/inicializadores.

Cambios realizados:
- Se emitieron `weights_qparam` y `output_qparam` como parte de la estructura de la capa en el C generado.
- Los biases se emiten en tipo cuantizado (cuando procede) y se respeta la cuantización de pesos.
- Se añadió la posibilidad de que la pasada de calibración sobreescriba `output_qparam` con valores medidos a partir de ejemplos reales.

Fragmento ilustrativo (simplificado):

```c
static const qparam_t layer0_weights_qparam = { .scale_q = 12345, .zero_point = -2 };
static const qparam_t layer0_output_qparam = { .scale_q = 23456, .zero_point = 0 };
dense_layer_t layer0 = { .weights = layer0_weights, .weights_qparam = layer0_weights_qparam, .output_qparam = layer0_output_qparam, ... };
```

Motivo:
- Sin `output_qparam` emitido, el runtime no puede cuantizar correctamente la salida de la capa, lo que genera errores de escala acumulativa.

Impacto esperado:
- Capas Dense generadas incluyen los qparams necesarios para que el runtime aplique `QUANTIZE`/`DEQUANTIZE` de forma consistente.

---

## Recomendaciones para el mantenedor

- Revisar los cambios en las firmas de activación y sincronizar cualquier otra implementación de runtime que dependa de las firmas antiguas.
- Ejecutar una batería adicional de pruebas sobre otros backends (por ejemplo, quantized TFLite, si aplica) para asegurar que no haya regresiones en rutas no ejercitadas.
- Considerar la adición de tests unitarios de generación → compilación automática para detectar desincronizaciones de plantilla/headers en el futuro.

---

Si necesita los diffs completos (parches) para cada archivo o fragmentos más extensos para enviar al repositorio upstream, puedo generar los parches (`git diff`) o crear archivos `.patch` listos para aplicar.
