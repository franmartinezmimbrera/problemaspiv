import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagenbordes.jpg', 0)

bilateral = cv2.bilateralFilter(img, 35, 75, 75)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(bilateral, cmap='gray')
ax2.set_title('Bilateral (Bordes preservados)')
ax2.axis('off')

plt.tight_layout()
plt.show()
