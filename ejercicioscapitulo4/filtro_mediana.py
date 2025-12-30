import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagenbordes.jpg',0)

median = cv2.medianBlur(img, 35)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
ax2.set_title('Filtro de Mediana (Sal y Pimienta)')
ax2.axis('off')

plt.tight_layout()
plt.show()
