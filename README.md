# PPS

Este repositorio contiene avances y pruebas de la Practica Profesional Supervisada
realizada en la Facultad Nacional de La Plata (UNLP).

## Proyectos principales

Proyecto de pruebas de cuantizacion en Python:

- [Cuantization_Test](Cuantization_Test)

Pipeline de inferencia en C para los modelos cuantizados:

- [C_model](C_model)

## Documentacion

Documentacion general:

- [Cuantization_Test/README.md](Cuantization_Test/README.md): descripcion general del pipeline en Python, entrenamiento, cuantizacion y evaluaciones.
- [C_model/README.md](C_model/README.md): vista global del pipeline de inferencia en C, estructura de datos y ejecucion.
- [utility/README.md](utility/README.md): catalogo de funciones Python reutilizables para cuantizacion, exportacion y evaluacion.

Documentacion de componentes C:

- [C_model/models/README.md](C_model/models/README.md): explicacion de `main.c`, flujo de inferencia, calculo de accuracy y comparacion contra Python.
- [C_model/utility/README.md](C_model/utility/README.md): referencia de `utils.c` y `utils.h` con activaciones, dequantizacion y lectura de CSV.
