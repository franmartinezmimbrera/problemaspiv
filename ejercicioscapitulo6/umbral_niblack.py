# umbral_niblack.py
import cv2
import matplotlib.pyplot as plt
import numpy as np # IMPORTANTE: faltaba esta línea
from skimage.filters import threshold_niblack

# 1. Cargamos en color para el panel 1 y en gris para procesar
img_color = cv2.imread('imagentexto.png')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 2. Aplicar el umbral de Niblack
# El cálculo se basa en la media y desviación estándar local:
# T = m + k * s
thresh = threshold_niblack(img_gray, window_size=25, k=0.8)
res = (img_gray > thresh).astype(np.uint8) * 255

# 3. Visualización de 2 paneles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Panel 1: Original (Color real)
ax1.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagen Original (Color)')
ax1.axis('off')

# Panel 2: Resultado de Niblack
# Cambiamos 'mask' por 'res' que es tu variable correcta
ax2.imshow(res, cmap='gray')
ax2.set_title('Umbralizado Niblack (Local)')
ax2.axis('off')

plt.tight_layout()
plt.show()
