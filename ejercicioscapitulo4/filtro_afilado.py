#filtro_afilado.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg')
kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
sharp = cv2.filter2D(img, -1, kernel)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
ax2.set_title('Imagen Afilada')
ax2.axis('off')

plt.tight_layout()
plt.show()
