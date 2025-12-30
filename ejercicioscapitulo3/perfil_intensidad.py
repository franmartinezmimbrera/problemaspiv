#perfil_intensidad.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', cv2.IMREAD_GRAYSCALE)
h, w = img.shape

# Definir una línea horizontal a la mitad de la imagen
fila = h // 2
perfil = img[fila, :]

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.axhline(y=fila, color='r') # Dibujar línea de referencia
plt.title('Imagen con línea de perfil')

plt.subplot(122)
plt.plot(perfil)
plt.title('Perfil de intensidad (Fila Central)')
plt.xlabel('Columna')
plt.ylabel('Intensidad')
plt.show()
