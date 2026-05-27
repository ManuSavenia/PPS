# PPS / EmbedIA — Notas de Exportación

Esta carpeta contiene la exportación regenerada de EmbedIA en modo FULL_QUANT8 para el proyecto de clasificación de dedos (PPS), además de la ejecución de validación a gran escala utilizada para verificar las correcciones del firmware y del generador.

## Correcciones realizadas en el firmware y en el generador

El firmware C generado y las plantillas de EmbedIA se actualizaron para que la tubería FULL_QUANT8 funcione correctamente con tensores cuantizados, en lugar de tratar las activaciones como si fueran valores float o simples buffers de `int8`.

### Archivos principales modificados

- `embedia/libraries/mcu/generic/full_quant8/neural_net.h`
- `embedia/libraries/mcu/generic/full_quant8/neural_net.c`
- `embedia/layers/activation/activation.py`
- `embedia/libraries/model/model.c`
- `embedia/model_generator/generate_files.py`
- `embedia/layers/dense/dense.py`

### Cambios funcionales

- Los prototipos de activación para FULL_QUANT8 se actualizaron para aceptar `qparam_t`, de modo que las activaciones se evalúan en el dominio cuantizado correcto.
- ReLU, ReLU6, tanh, sigmoid, softsign, softplus y softmax ahora son conscientes de `qparam` (qparam-aware).
- Softmax ahora descuantiza, calcula probabilidades en punto flotante y vuelve a cuantizarlas usando los `qparams` de la capa.
- La exportación de la capa Dense ahora preserva los parámetros de cuantización de pesos y la cuantización de salida.
- Las llamadas a funciones de activación generadas ahora pasan los `qparams` de salida cuando el modelo está cuantizado.
- Se arregló la plantilla del modelo para que el generador pueda renderizar correctamente la fuente C.

## Script de exportación usado para este proyecto

La exportación se ejecutó mediante `utility/export_embedia_project.py`.

Resumen de pasos que realiza el script:

1. Carga el modelo Keras entrenado desde `Cuantization_Test/Models/base/fingers_model_no_quantization.h5`.
2. Lee los CSV de entrenamiento y prueba desde `Cuantization_Test/Data_Sets/raw/`.
3. Construye un conjunto balanceado de ejemplos con la cantidad solicitada por clase.
4. Aplica el normalizador guardado en `Cuantization_Test/Data_Sets/metadata/normalizer.pkl`.
5. Configura EmbedIA para un proyecto C en modo FULL_QUANT8.
6. Exporta el proyecto a la carpeta de salida solicitada.
7. Copia los archivos del runtime FULL_QUANT8 dentro del proyecto generado.

El script se ejecutó con el conjunto completo de prueba para producir la exportación de validación a gran escala:

```bash
/home/manuel/Documents/Facultad/PPS/Cuantization_Test/env/bin/python utility/export_embedia_project.py \
  --output-folder /home/manuel/Documents/Facultad/PPS/EmbedIA \
  --project-name fingers_full_quant8_large \
  --train-per-class 0 \
  --test-per-class 600
```

## Resultado de la validación

El proyecto generado se compiló y ejecutó correctamente contra el conjunto completo de prueba.

- Muestras de prueba: 3600
- Predicciones correctas: 3570
- Precisión: 99.17%

Consulte `large_scale_test_result.md` para el resumen guardado.

## Artefactos generados en esta carpeta

- `fingers_full_quant8_large/` — proyecto EmbedIA generado
- `fingers_full_quant8_large_raw_examples.csv` — conjunto de ejemplos normalizados exportado
- `large_scale_test_result.md` — informe con el resultado de precisión

## Notas

Los cambios se verificaron en la ruta de exportación FULL_QUANT8. No se observaron regresiones en el flujo exportar/compilar/ejecutar que aquí se comprobó, aunque no se probaron exhaustivamente otros objetivos de EmbedIA no relacionados.
