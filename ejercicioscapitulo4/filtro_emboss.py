#filtro_emboss.py
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('imagen.jpg', 0)

# Definición del kernel de Emboss (Realce/Relieve)
kernel = np.array([[-2,-1,0], 
                   [-1, 1,1], 
                   [ 0, 1,2]])

emboss = cv2.filter2D(img, -1, kernel) + 128

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(emboss, cmap='gray')
ax2.set_title('Efecto Emboss (Relieve)')
ax2.axis('off')

plt.tight_layout()
plt.show()
