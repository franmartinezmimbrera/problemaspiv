#comparacion_bhattacharyya.py
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('imagen.jpg')
img2 = cv2.imread('imagencopia.jpg') # O una imagen similar

# Convertir a HSV y calcular histogramas
h1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
h2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])

cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)

# 0 indica coincidencia perfecta en Bhattacharyya
distancia = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
print(f"Distancia de Bhattacharyya: {distancia:.4f}")
