import glob
import math
import os

import pandas as pd
from skimage import io

from utility.extraer_caracteristicas import extraer_caracteristicas


def convertir_dataset(dir_orig, dir_arch_dest):
    columnas = [
        "AreaNorm",
        "PerimNorm",
        "RazonEjes",
        "Excentricidad",
        "Solidez",
        "Extension",
        "CantDedos",
    ]

    archivos = glob.glob(os.path.join(dir_orig, "**/*.png"), recursive=True)
    data = []

    total = len(archivos)
    for nro, archivo in enumerate(archivos, start=1):
        if nro % 200 == 0 or nro == total:
            print(f"\rProcesando {nro}/{total} imágenes ({100*nro/total:.2f}%)", end="")

        imagen = io.imread(archivo)
        cant_dedos = int(os.path.basename(os.path.dirname(archivo)))

        props, _ = extraer_caracteristicas(imagen)

        area = props.filled_area
        ej_mayor = props.major_axis_length
        ej_menor = props.minor_axis_length
        perim = props.perimeter
        excentr = props.eccentricity
        solidez = props.solidity
        extension = props.extent

        area = area / (ej_mayor * ej_menor)
        perim = perim / math.sqrt(ej_mayor * ej_menor)
        razon_ej = ej_menor / ej_mayor

        data.append([area, perim, razon_ej, excentr, solidez, extension, cant_dedos])

    df = pd.DataFrame(columns=columnas, data=data)
    df.to_csv(dir_arch_dest, index=False)
    print(f"\n✅ Guardado en {dir_arch_dest}")
