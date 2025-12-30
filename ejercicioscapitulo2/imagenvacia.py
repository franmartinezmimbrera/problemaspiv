#imagenvacia.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Crear una imagen vacía
img = np.zeros((512, 512, 3), dtype=np.uint8)

# Dibujar un círculo en la imagen
cv2.circle(img, (256, 256), 100, (255, 0, 0), -1)

# Dibujar una línea en la imagen
cv2.line(img, (100, 100), (400, 400), (0, 255, 0), 5)

# Dibujar un rectángulo en la imagen
cv2.rectangle(img, (200, 200), (300, 300), (0, 0, 255), -1)

# Mostrar la imagen
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
