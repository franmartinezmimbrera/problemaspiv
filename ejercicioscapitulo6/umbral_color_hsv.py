#umbral_color_hsv.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg')

# Convertir a espacio de color HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definir el rango para detectar color azul
# Los valores HSV en OpenCV van: H(0-179), S(0-255), V(0-255)
bajo = np.array([100, 50, 50])
alto = np.array([130, 255, 255])

# Crear la máscara (los píxeles en rango serán blancos, el resto negros)
mask = cv2.inRange(hsv, bajo, alto)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagen Original (Color)')
ax1.axis('off')

ax2.imshow(mask, cmap='gray')
ax2.set_title('Máscara: Color Azul Detectado')
ax2.axis('off')

plt.tight_layout()
plt.show() 
