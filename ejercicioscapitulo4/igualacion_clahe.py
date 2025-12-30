#igualacion_clahe.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

cl1 = clahe.apply(img)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Bajo Contraste)')
ax1.axis('off')

ax2.imshow(cl1, cmap='gray')
ax2.set_title('Resultado CLAHE (Adaptativo)')
ax2.axis('off')

plt.tight_layout()
plt.show()
