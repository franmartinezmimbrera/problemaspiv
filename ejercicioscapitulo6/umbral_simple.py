# umbral_simple.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg', 0)

#  Aplicar umbral simple (Binary Thresholding)
# Los valores menores a 127 se vuelven negros (0) y los mayores blancos (255)
_, res = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(res, cmap='gray')
ax2.set_title('Umbral Simple (127)')
ax2.axis('off')

plt.tight_layout()
plt.show()
