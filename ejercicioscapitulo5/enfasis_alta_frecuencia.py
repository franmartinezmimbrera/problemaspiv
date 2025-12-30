#enfasis_alta_frecuencia.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar imagen en gris
img = cv2.imread('imagen.jpg', 0)
rows, cols = img.shape

# Crear malla de frecuencias y Filtro Paso Alto Gaussiano (HP)
x, y = np.meshgrid(np.linspace(-cols/2, cols/2, cols), np.linspace(-rows/2, rows/2, rows))
d0 = 30
hp = 1 - np.exp(-(x**2 + y**2) / (2 * d0**2))

#  Máscara de Énfasis: a + b * HP
# a=0.5 (mantiene parte de la base), b=1.5 (potencia los bordes)
a, b = 0.5, 1.5
mask = a + b * hp

#  Procesar en frecuencia
f = np.fft.fftshift(np.fft.fft2(img))
f_filtrado = f * mask

#  Reconstruir (IFFT)
# Usamos np.real para mayor estabilidad en la reconstrucción
res = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtrado)))

#  Normalizar correctamente para visualización
res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
res_final = np.uint8(res_norm)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(mask, cmap='gray')
ax2.set_title(f'Máscara Énfasis ({a} + {b}*HP)')
ax2.axis('off')

ax3.imshow(res_final, cmap='gray')
ax3.set_title('Resultado (Nitidez aumentada)')
ax3.axis('off')

plt.tight_layout()
plt.show() 
