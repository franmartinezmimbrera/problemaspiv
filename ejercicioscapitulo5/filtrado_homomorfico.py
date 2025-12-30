# filtrado_homomorfico.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)
# Evitamos log(0) y trabajamos en float64
img_float = np.float64(img) + 1.0
img_log = np.log(img_float)

rows, cols = img.shape
x, y = np.meshgrid(np.linspace(-cols/2, cols/2, cols), 
                   np.linspace(-rows/2, rows/2, rows))
d2 = x**2 + y**2

#  Filtro Homomórfico (Parámetros ajustados)
rh, rl, c, d0 = 1.5, 0.5, 1.0, 50  # d0 un poco más grande
mask = (rh - rl) * (1 - np.exp(-c * (d2 / (d0**2)))) + rl

#  FFT y Filtrado
f = np.fft.fftshift(np.fft.fft2(img_log))
res_f = f * mask

#  Inversa y Exponencial
res_ifft = np.fft.ifft2(np.fft.ifftshift(res_f))
# Usamos np.real para evitar problemas de fase
res_exp = np.exp(np.real(res_ifft)) - 1.0

#  Normalización ROBUSTA
# Eliminamos posibles valores infinitos o Nan
res_exp = np.nan_to_num(res_exp)
# Re-escalado manual para asegurar visibilidad
res_final = cv2.normalize(res_exp, None, 0, 255, cv2.NORM_MINMAX)
res_final = np.uint8(np.clip(res_final, 0, 255))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(mask, cmap='gray')
ax2.set_title('Máscara Filtro')
ax2.axis('off')

ax3.imshow(res_final, cmap='gray')
ax3.set_title('Resultado Corregido')
ax3.axis('off')

plt.tight_layout()
plt.show()
