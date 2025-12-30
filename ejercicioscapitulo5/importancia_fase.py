#importancia_fase.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar y redimensionar imágenes en gris
# Ambas deben tener el mismo tamaño para que sus espectros coincidan
img1 = cv2.resize(cv2.imread('imagen.jpg', 0), (300, 300))
img2 = cv2.resize(cv2.imread('imagenbordes.jpg', 0), (300, 300))

# 2. Calcular la FFT de ambas
f1 = np.fft.fft2(img1)
f2 = np.fft.fft2(img2)

# 3. Combinar: Magnitud de img1 + Fase de img2
# La fase (angle) determina dónde se colocan los componentes (la estructura)
combined = np.multiply(np.abs(f1), np.exp(1j * np.angle(f2)))

# 4. Reconstruir (IFFT)
# Usamos np.abs para obtener la intensidad final de los números complejos
res = np.abs(np.fft.ifft2(combined))

# Normalizar el resultado para que se vea correctamente (0-255)
res_final = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 5. Visualización de 3 paneles
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

ax1.imshow(img1, cmap='gray')
ax1.set_title('Imagen 1 (Aporta Magnitud)')
ax1.axis('off')

ax2.imshow(img2, cmap='gray')
ax2.set_title('Imagen 2 (Aporta Fase)')
ax2.axis('off')

ax3.imshow(res_final, cmap='gray')
ax3.set_title('Resultado: Magnitud 1 + Fase 2')
ax3.axis('off')

plt.tight_layout()
plt.show()
