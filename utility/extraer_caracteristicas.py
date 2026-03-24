from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, footprint_rectangle
from skimage.segmentation import clear_border


def extraer_caracteristicas(imagen):
    umbral = threshold_otsu(imagen)
    imagen_bn = (imagen > umbral) * 1
    imagen_bn = closing(imagen_bn, footprint_rectangle((3, 3)))
    imagen_lista = clear_border(imagen_bn)
    regiones = regionprops(label(imagen_lista))
    return regiones[0], imagen_lista
