#rastreo_shi_tomasi.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#  ALGORITMO SHI-TOMASI 
# maxCorners: Queremos solo los 100 mejores puntos
# qualityLevel: Calidad mínima (0.01 es permisivo, 0.3 es estricto)
# minDistance: Distancia mínima entre esquinas (para no agruparlas todas)
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                  qualityLevel=0.01, 
                                  minDistance=10)
# Convertir a enteros para poder dibujar
corners = np.int64(corners)
# Dibujar los puntos encontrados
vis = img.copy()
for i in corners:
    x, y = i.ravel()
    # Dibuja un círculo pequeño verde en cada esquina detectada
    cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi: Detección de características rastreables')
plt.axis('off')
plt.show()
