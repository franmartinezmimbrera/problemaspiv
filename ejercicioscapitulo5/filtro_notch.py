#filtro_notch.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)
rows, cols = img.shape

# Ir al dominio de la frecuencia
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Crear el Espectro de Magnitud para visualizar los picos de ruido
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Aplicar el Filtro de Muesca (Anular frecuencias específicas)
# Aquí estamos poniendo a cero dos pequeños cuadrados que representan el ruido
fshift[100:105, 100:105] = 0
fshift[rows-105:rows-100, cols-105:cols-100] = 0

#  Volver al dominio del espacio (IFFT)
f_ishift = np.fft.ifftshift(fshift)
img_back = np.abs(np.fft.ifft2(f_ishift))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.set_title('Espectro (Notch aplicado)')
ax2.axis('off')

ax3.imshow(img_back, cmap='gray')
ax3.set_title('Reconstrucción IFFT')
ax3.axis('off')

plt.tight_layout()
plt.show()
