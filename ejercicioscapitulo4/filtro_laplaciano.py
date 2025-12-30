#filtro_laplaciano.py
import cv2
import matplotlib.pyplot as plt

# Cargar en escala de grises
img = cv2.imread('imagenbordes.jpg', 0)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(laplacian, cmap='gray')
ax2.set_title('Filtro Laplaciano (Bordes)')
ax2.axis('off')

plt.tight_layout()
plt.show()
