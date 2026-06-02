# Resultado de la prueba a gran escala — FULL_QUANT8

- Fecha: 2026-06-02
- Proyecto: fingers_full_quant8_large_repeat
- Ruta de exportación: /home/manuel/Documents/Facultad/PPS/EmbedIA_exports/fingers_full_quant8_large_repeat
- Conjunto de datos usado: partición completa de `fingers_test.csv`
- Tamaño solicitado de la prueba: 3600 ejemplos
- Tamaño evaluado de la prueba: 3606 ejemplos
- Precisión: 3580 / 3606 = 99.28%

## Validación

- La exportación se completó correctamente usando el generador FULL_QUANT8.
- La compilación tuvo éxito con `gcc -std=c11 -O2 -I. main.c embedia/*.c -lm -o fingers_full_quant8_large_repeat_test`.
- La prueba de ejecución (smoke test) finalizó correctamente con código de salida 0.

## Observaciones

- Las correcciones recientes afectan la ruta FULL_QUANT8: activaciones compatibles con `qparam`, calibración de Dense sobre la salida pre-activación y formato correcto de la plantilla del modelo.
- La repetición de 3600 muestras confirma que el flujo exportar/compilar/ejecutar sigue estable con el estado actual del runtime y del generador.
