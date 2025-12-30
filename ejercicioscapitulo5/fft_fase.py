#fft_fase.py
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrum = np.angle(fshift)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')
ax2.imshow(phase_spectrum, cmap='gray')
ax2.set_title('Espectro de Fase')
ax2.axis('off')

plt.tight_layout()
plt.show()
