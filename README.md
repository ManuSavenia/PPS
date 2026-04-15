# PPS - Cuantización de modelos Neuronales

Este repositorio contiene avances y pruebas de la Practica Profesional Supervisada
realizada en la Facultad Nacional de La Plata (UNLP).
Se busca el pasaje de modelos neuronales en Python a C, para su ejecucion en MCU's y MTU's, utilizando
Cuantización simetrica con signo para ahorrar espacio y tiempo de ejecucion al evitar usar numeros de punto flotante.

## Proyectos principales

Proyecto de pruebas de cuantizacion en Python:

- [Cuantization_Test](Cuantization_Test)

Pipeline de inferencia en C para los modelos cuantizados:

- [C_model](C_model)

## Documentacion

Documentacion general:

- [Cuantization_Test/README.md](Cuantization_Test/README.md): descripcion del pipeline en Python, entrenamiento, cuantizacion, depuracion capa a capa, reparacion y validacion TFLite.
- [C_model/README.md](C_model/README.md): vista global del pipeline de inferencia en C, estructura de datos y ejecucion.
- [utility/README.md](utility/README.md): catalogo de funciones Python reutilizables para cuantizacion, exportacion, evaluacion y diagnosticos.

Documentacion de componentes C:

- [C_model/models/README.md](C_model/models/README.md): explicacion de `main.c`, flujo de inferencia, calculo de accuracy y comparacion contra Python.
- [C_model/utility/README.md](C_model/utility/README.md): referencia de `utils.c` y `utils.h` con activaciones, dequantizacion y lectura de CSV.

## Estructura actual del flujo Python

- `Cuantization_Test/Data_Sets/raw/`: CSV base generados desde imagenes.
- `Cuantization_Test/Data_Sets/quantized/`: CSV cuantizados de train/test.
- `Cuantization_Test/Data_Sets/metadata/`: escalas y metadata de cuantizacion.
- `Cuantization_Test/Data_Sets/reports/`: comparaciones, diagnosticos, reparacion y validacion TFLite.
- `Cuantization_Test/Models/base/`: modelo float entrenado.
- `Cuantization_Test/Models/quantized_h5/`: variantes cuantizadas para evaluacion rapida.
- `Cuantization_Test/Models/quantized_tflite/`: variantes int8 reales.
- `Cuantization_Test/Models/repaired_h5/` y `Cuantization_Test/Models/repaired_tflite/`: modelos reparados y su validacion final.
