#filtro_gaussiano.py
#filtro_gaussiano.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')

gauss = cv2.GaussianBlur(img, (5, 5), 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagen Original (Nítida)')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB))
ax2.set_title('Suavizado Gaussiano (5x5)')
ax2.axis('off')

plt.tight_layout()
plt.show()
