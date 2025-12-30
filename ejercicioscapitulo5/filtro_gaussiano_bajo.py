#filtro_gaussiano_bajo.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)
rows, cols = img.shape

# Crear la malla de frecuencias y calcular distancias al centro
x, y = np.meshgrid(np.linspace(-cols/2, cols/2, cols), np.linspace(-rows/2, rows/2, rows))
distancias_cuadrado = x**2 + y**2

# Definir la Máscara Gaussiana de Paso Bajo
d0 = 30 # Radio de corte (desviación estándar)
# Fórmula: H(u,v) = exp(-D^2(u,v) / (2 * D0^2))
mask = np.exp(-distancias_cuadrado / (2 * d0**2))

# Procesar en el dominio de la frecuencia
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
f_filtrado = fshift * mask

# Volver al dominio del espacio (IFFT)
img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtrado)))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(mask, cmap='gray')
ax2.set_title(f'Máscara Gaussiana (D0={d0})')
ax2.axis('off')

ax3.imshow(img_back, cmap='gray')
ax3.set_title('Resultado (Paso Bajo)')
ax3.axis('off')

plt.tight_layout()
plt.show()
