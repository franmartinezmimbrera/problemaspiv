#ifft_reconstruccion.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', 0)
# Calcular FFT e ir al dominio de la frecuencia
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# Calcular el espectro de magnitud (logarítmico para verlo bien)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
# Calcular la IFFT para volver al dominio de la imagen
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.set_title('Espectro Magnitud (log)')
ax2.axis('off')
ax3.imshow(img_back, cmap='gray')
ax3.set_title('Reconstrucción IFFT')
ax3.axis('off')

plt.tight_layout()
plt.show()
