# umbral_otsu.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', 0)

#  Aplicar umbral simple (Binary Thresholding) + Otsu
val, res = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(res, cmap='gray')
ax2.set_title('Umbral Otsu')
ax2.axis('off')

plt.tight_layout()
plt.show()
