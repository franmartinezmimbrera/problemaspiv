# filtro_paso_banda.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)
rows, cols = img.shape

# Crear la malla de frecuencias y calcular distancias al centro
x, y = np.meshgrid(np.linspace(-cols/2, cols/2, cols), np.linspace(-rows/2, rows/2, rows))
d = np.sqrt(x**2 + y**2)

# Definir parámetros y Máscara de Paso de Banda
d0, w = 50, 20  # d0 es el radio central, w es el ancho de la banda
# La máscara es True solo dentro del anillo definido por w
mask = np.logical_and(d >= (d0 - w/2), d <= (d0 + w/2)).astype(float)

# Procesar en el dominio de la frecuencia
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
f_filtrado = fshift * mask

#  Volver al dominio del espacio (IFFT)
f_ishift = np.fft.ifftshift(f_filtrado)
img_back = np.abs(np.fft.ifft2(f_ishift))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(mask, cmap='gray')
ax2.set_title(f'Máscara Paso Banda (D0={d0}, W={w})')
ax2.axis('off')

ax3.imshow(img_back, cmap='gray')
ax3.set_title('Resultado (Paso de Banda)')
ax3.axis('off')

plt.tight_layout()
plt.show()
