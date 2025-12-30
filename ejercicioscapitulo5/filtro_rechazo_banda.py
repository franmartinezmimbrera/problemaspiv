#filtro_rechazo_banda.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)
rows, cols = img.shape

# Crear malla de distancias al centro
x, y = np.meshgrid(np.linspace(-cols/2, cols/2, cols), np.linspace(-rows/2, rows/2, rows))
d = np.sqrt(x**2 + y**2)

# Definir máscara de Rechazo de Banda
d0, w = 50, 10 # Radio central y ancho de la banda
# Deja pasar lo que está fuera del anillo (menor que d0-w/2 o mayor que d0+w/2)
mask = np.logical_or(d < (d0-w/2), d > (d0+w/2)).astype(float)

# Procesar en frecuencia
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
f_filtrado = fshift * mask

#  Reconstruir (IFFT)
img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtrado)))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(mask, cmap='gray')
ax2.set_title(f'Máscara Rechazo (D0={d0}, W={w})')
ax2.axis('off')

ax3.imshow(img_back, cmap='gray')
ax3.set_title('Resultado (Filtro Rechazo)')
ax3.axis('off')

plt.tight_layout()
plt.show()
