#estadisticas_histograma.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

media = np.mean(img)
desviacion = np.std(img)
varianza = np.var(img)
min_val, max_val, _, _ = cv2.minMaxLoc(img)

print(f"Estadísticas de la imagen:")
print(f"Media de intensidad: {media:.2f}")
print(f"Desviación Estándar: {desviacion:.2f}")
print(f"Varianza: {varianza:.2f}")
print(f"Rango de valores: [{min_val}, {max_val}]")
