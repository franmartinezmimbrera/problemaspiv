#filtro_media.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagen.jpg')
# El kernel (35,35) suma todos los píxeles en esa área y los divide por 25
blur = cv2.blur(img, (35, 35))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Imagen Original')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
ax2.set_title('Suavizado (Filtro de Media 5x5)')
ax2.axis('off')

plt.tight_layout()
plt.show()
