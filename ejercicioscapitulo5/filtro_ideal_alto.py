#filtro_ideal_alto.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagenbordes.jpg', 0)
rows, cols = img.shape
crow, ccol = rows // 2 , cols // 2

# Ir al dominio de la frecuencia
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
#  Crear la Máscara Ideal de Paso ALTO
mask = np.ones((rows, cols), np.uint8)
d0 = 30 
# Dibujamos un círculo de CEROS en el centro (negro/bloquea el centro)
cv2.circle(mask, (ccol, crow), d0, 0, -1)
# Aplicar máscara y volver al dominio del espacio
fshift_filtrado = fshift * mask
img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtrado)))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(mask, cmap='gray')
ax2.set_title(f'Máscara Paso Alto (D0={d0})')
ax2.axis('off')
ax3.imshow(img_back, cmap='gray')
ax3.set_title('Imagen Filtrada (Solo Bordes)')
ax3.axis('off')

plt.tight_layout()
plt.show() 
