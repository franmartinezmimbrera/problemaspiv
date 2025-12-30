#comparahistograma.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('imagenbordes.jpg', cv2.IMREAD_GRAYSCALE)

# Calcular los histogramas
hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
# Normalizar los histogramas
cv2.normalize(hist1, hist1)
cv2.normalize(hist2, hist2)

# Comparar los histogramas usando Correlación
comparison = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
print(f'Correlación: {comparison}')
