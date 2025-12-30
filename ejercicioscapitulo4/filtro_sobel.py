#filtro_sobel.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagenbordes.jpg', 0)

sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

sobel_combined = cv2.magnitude(sobelx, sobely)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(sobel_combined, cmap='gray')
ax2.set_title('Detección de Bordes Sobel')
ax2.axis('off')

plt.tight_layout()
plt.show()
