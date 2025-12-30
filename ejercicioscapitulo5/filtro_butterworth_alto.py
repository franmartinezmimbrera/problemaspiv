# filtro_butterworth_alto.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)
rows, cols = img.shape

# Crear la malla de frecuencias y calcular distancias al centro
x, y = np.meshgrid(np.linspace(-cols/2, cols/2, cols), np.linspace(-rows/2, rows/2, rows))
d = np.sqrt(x**2 + y**2)

#  Definir parámetros del filtro
d0 = 30 # Radio de corte
n = 2   # Orden del filtro

# Ecuación del filtro Butterworth de Paso Alto:
# Se añade 1e-5 para evitar la división por cero en el centro exacto
mask = 1 / (1 + (d0 / (d + 1e-5))**(2*n))

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
ax2.set_title(f'Máscara Paso Alto (n={n})')
ax2.axis('off')

ax3.imshow(img_back, cmap='gray')
ax3.set_title('Resultado (Paso Alto)')
ax3.axis('off')

plt.tight_layout()
plt.show()
