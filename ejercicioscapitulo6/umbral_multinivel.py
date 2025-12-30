#umbral_multinivel.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('imagen.jpg', 0)
res = np.zeros_like(img)
res[img > 85] = 127
res[img > 170] = 255


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(res, cmap='gray')
ax2.set_title('Método umbralización multinivel')
ax2.axis('off')

plt.tight_layout()
plt.show()


