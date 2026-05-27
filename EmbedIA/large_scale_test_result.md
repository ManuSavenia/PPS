# Resultado de la prueba a gran escala — FULL_QUANT8

- Fecha: 2026-05-26
- Proyecto: fingers_full_quant8_large
- Ruta de exportación: /home/manuel/Documents/Facultad/PPS/EmbedIA/fingers_full_quant8_large
- Conjunto de datos usado: partición completa de `fingers_test.csv`
- Tamaño de la prueba: 3600 ejemplos
- Precisión: 3570 / 3600 = 99.17%

## Validación

- La exportación se completó correctamente usando el generador FULL_QUANT8.
- La compilación tuvo éxito con `gcc -std=c11 -I. main.c embedia/*.c -lm -o fingers_full_quant8_large_test`.
- La prueba de ejecución (smoke test) finalizó correctamente con código de salida 0.

## Observaciones

- Las correcciones recientes afectan la ruta FULL_QUANT8: activaciones compatibles con `qparam`, formato de la plantilla del modelo y la salida del generador.
- No se observaron regresiones adicionales en el flujo exportar/compilar/ejecutar verificado en esta validación.
