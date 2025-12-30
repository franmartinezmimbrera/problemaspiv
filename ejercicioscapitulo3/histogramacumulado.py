#histogramacumulado.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen en escala de grises
img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)

# Calcular el histograma
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# Convertir a histograma acumulado
cum_hist = hist.cumsum()

# Normalizar el histograma acumulado
cum_hist_normalized = cum_hist / float(cum_hist.max())

# Mostrar el histograma acumulado
plt.figure()
plt.title("Histograma Acumulado")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(cum_hist_normalized)
plt.xlim([0, 256])
plt.show()
