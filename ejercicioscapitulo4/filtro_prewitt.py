#filtro_prewitt.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagenbordes.jpg', 0)

kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

prewitt_x = cv2.filter2D(img, -1, kernelx)
prewitt_y = cv2.filter2D(img, -1, kernely)

res = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(res, cmap='gray')
ax2.set_title('Filtro Prewitt (Bordes)')
ax2.axis('off')

plt.tight_layout()
plt.show()
