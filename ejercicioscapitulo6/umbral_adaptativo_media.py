# umbral_adaptativo_media.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', 0)

t_adapt_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(t_adapt_mean, cmap='gray')
ax2.set_title('Umbral adaptativo media')
ax2.axis('off')

plt.tight_layout()
plt.show()
