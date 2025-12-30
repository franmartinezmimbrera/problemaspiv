#unsharp_masking.py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('imagenbordes.jpg', 0)

gaussian = cv2.GaussianBlur(img, (9,9), 10.0)

unsharp = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.imshow(img, cmap='gray')
ax1.set_title('Original (Gris)')
ax1.axis('off')

ax2.imshow(unsharp, cmap='gray')
ax2.set_title('Unsharp Masking (Nitidez)')
ax2.axis('off')

plt.tight_layout()
plt.show()
