#fft_magnitud.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', 0)

# Cálculo de la FFT 2D
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.set_title('Espectro de Magnitud (Frecuencias)')
ax2.axis('off')

plt.tight_layout()
plt.show()
